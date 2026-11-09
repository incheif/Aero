"""
Multi-Object Tracker for Tabletop Manipulation.
Maintains persistent track IDs, position filtering, and occlusion handling.
"""

import math
from typing import List, Dict, Optional, Tuple
from .models import ObjectDetection, TrackedObject
from simulation.constants import CUBE_SIZE


class MultiObjectTracker:
    """
    Temporal tracker maintaining stable identity for blocks across frames.
    Handles temporary occlusion and prevents identity switches.
    """
    def __init__(
        self,
        distance_threshold: float = 0.065, # meters
        smoothing_alpha: float = 0.65,
        max_missed_frames: int = 5
    ):
        self.distance_threshold = distance_threshold
        self.smoothing_alpha = smoothing_alpha
        self.max_missed_frames = max_missed_frames
        self.next_track_num = 1
        self.tracks: Dict[str, TrackedObject] = {}

    def update(self, detections: List[ObjectDetection]) -> List[TrackedObject]:
        """
        Updates active tracks with fresh detections from the current frame.
        """
        unmatched_detections = list(detections)
        matched_track_ids = set()

        # Match detections to existing active/occluded tracks
        for track_id, track in list(self.tracks.items()):
            best_det_idx = None
            min_dist = float("inf")

            for idx, det in enumerate(unmatched_detections):
                if det.position_base is None:
                    continue

                dx = det.position_base[0] - track.position[0]
                dy = det.position_base[1] - track.position[1]
                dz = det.position_base[2] - track.position[2]
                dist = math.hypot(dx, dz) # horizontal distance on tabletop

                # Strict color consistency check if color is known
                if track.color_name and det.color_name and track.color_name != det.color_name:
                    continue

                if dist < self.distance_threshold and dist < min_dist:
                    min_dist = dist
                    best_det_idx = idx

            if best_det_idx is not None:
                det = unmatched_detections.pop(best_det_idx)
                matched_track_ids.add(track_id)

                # Update track with smoothed position
                alpha = self.smoothing_alpha
                new_pos = [
                    alpha * det.position_base[0] + (1.0 - alpha) * track.position[0],
                    alpha * det.position_base[1] + (1.0 - alpha) * track.position[1],
                    alpha * det.position_base[2] + (1.0 - alpha) * track.position[2],
                ]
                # Compute velocity estimate
                vel = [
                    new_pos[0] - track.position[0],
                    new_pos[1] - track.position[1],
                    new_pos[2] - track.position[2],
                ]

                track.position = new_pos
                track.velocity = vel
                track.bbox_xyxy = det.bbox_xyxy
                track.confidence = min(0.99, track.confidence * 0.95 + det.confidence * 0.05 + 0.02)
                track.hits += 1
                track.age_ticks += 1
                track.time_since_update = 0
                track.state = "ACTIVE"
            else:
                # Track was not matched in this frame
                track.time_since_update += 1
                track.age_ticks += 1
                track.confidence = max(0.20, track.confidence * 0.85)

                if track.time_since_update <= self.max_missed_frames:
                    track.state = "OCCLUDED"
                else:
                    track.state = "LOST"

        # Create new tracks for unmatched detections
        for det in unmatched_detections:
            if det.position_base is None:
                continue

            track_id = f"track_{self.next_track_num}"
            self.next_track_num += 1

            new_track = TrackedObject(
                track_id=track_id,
                class_name=det.class_name,
                color_name=det.color_name,
                color_rgb=det.color_rgb,
                position=list(det.position_base),
                bbox_xyxy=det.bbox_xyxy,
                dimensions=det.dimensions or (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
                confidence=det.confidence,
                age_ticks=1,
                hits=1,
                time_since_update=0,
                state="ACTIVE"
            )
            self.tracks[track_id] = new_track

        # Prune completely lost tracks
        self.tracks = {tid: t for tid, t in self.tracks.items() if t.state != "LOST"}

        return list(self.tracks.values())
