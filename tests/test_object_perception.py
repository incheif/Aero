"""
Unit tests for Perception Pipeline: Detection, 3D Localisation & Multi-Object Tracking.
Verifies Stage 2 & 3 perception requirements.
"""

import math
import numpy as np
import pytest

from simulation.physics_engine import RobotArmSimulation
from simulation.camera import SimulationCamera
from simulation.constants import BLOCK_SPAWNS, CUBE_SIZE, TABLE_TOP_Y, CUBE_HALF
from perception.detector import BlockColorDetector
from perception.localisation import Object3DLocalizer
from perception.tracker import MultiObjectTracker
from perception.models import ObjectDetection


class TestObjectPerceptionAndTracking:
    def test_01_image_based_detector_identifies_all_blocks(self):
        """Validates that BlockColorDetector segments all 6 colored blocks from raw CameraObservation."""
        sim = RobotArmSimulation()
        sim.reset()

        cam = SimulationCamera(sim, size=128)
        obs = cam.capture()

        detector = BlockColorDetector(min_cluster_pixels=4)
        detections = detector.detect(obs)

        # We expect 6 blocks in standard spawn: red, blue, yellow, green, orange, purple
        detected_colors = {d.color_name for d in detections if d.color_name}
        expected_colors = {"red", "blue", "yellow", "green", "orange", "purple"}

        assert expected_colors.issubset(detected_colors), f"Missing colors: {expected_colors - detected_colors}"
        assert len(detections) == 6

        # Check confidence and bounding boxes
        for det in detections:
            assert det.confidence >= 0.90
            assert det.pixel_area >= 4
            u1, v1, u2, v2 = det.bbox_xyxy
            assert u2 >= u1
            assert v2 >= v1

    def test_02_localization_accuracy_vs_ground_truth(self):
        """
        Evaluates 3D localisation error of image+depth estimates against true physics block coordinates.
        Note: Ground truth is used purely here as test evaluation metric!
        """
        sim = RobotArmSimulation()
        sim.reset()

        cam = SimulationCamera(sim, size=128)
        obs = cam.capture()

        detector = BlockColorDetector()
        localizer = Object3DLocalizer()

        dets = detector.detect(obs)
        localizer.localize(dets, obs)

        # Match each detection to true block by color
        gt_blocks = {b["color"]: b for b in sim.get_snapshot()["blocks"]}
        color_map = {
            "red": "#FF1E27",
            "blue": "#1E60FF",
            "yellow": "#FFC300",
            "green": "#00B341",
            "orange": "#FF5500",
            "purple": "#9333EA",
        }

        for det in dets:
            assert det.position_base is not None
            hex_color = color_map[det.color_name]
            gt = gt_blocks[hex_color]["position"]

            # Localisation error in meters
            err_x = abs(det.position_base[0] - gt[0])
            err_y = abs(det.position_base[1] - gt[1])
            err_z = abs(det.position_base[2] - gt[2])
            horizontal_err = math.hypot(err_x, err_z)

            # 128x128 resolution -> pixel size is ~ 0.010m. Horizontal error should be <= 12mm
            assert horizontal_err < 0.015, f"Horizontal error {horizontal_err:.4f}m exceeds 15mm for {det.color_name}"
            # Vertical altitude error should be <= 10mm
            assert err_y < 0.012, f"Altitude error {err_y:.4f}m exceeds 12mm for {det.color_name}"

    def test_03_temporal_tracker_preserves_identities(self):
        """Verifies multi-object tracker preserves track IDs across frames and handles occlusion."""
        sim = RobotArmSimulation()
        sim.reset()

        cam = SimulationCamera(sim, size=128)
        detector = BlockColorDetector()
        localizer = Object3DLocalizer()
        tracker = MultiObjectTracker(distance_threshold=0.065, max_missed_frames=3)

        # Frame 1
        obs1 = cam.capture()
        dets1 = localizer.localize(detector.detect(obs1), obs1)
        tracks1 = tracker.update(dets1)
        assert len(tracks1) == 6
        initial_ids = {t.track_id for t in tracks1}
        assert len(initial_ids) == 6

        # Frame 2: repeated detection
        obs2 = cam.capture()
        dets2 = localizer.localize(detector.detect(obs2), obs2)
        tracks2 = tracker.update(dets2)
        assert len(tracks2) == 6
        assert {t.track_id for t in tracks2} == initial_ids, "Track IDs must not change on static scene"

        # Frame 3: Simulate temporary occlusion by omitting one detection (e.g. red cube)
        dets3_occluded = [d for d in dets2 if d.color_name != "red"]
        tracks3 = tracker.update(dets3_occluded)
        assert len(tracks3) == 6  # Track is retained!
        red_tracks = [t for t in tracks3 if t.color_name == "red"]
        assert len(red_tracks) == 1
        assert red_tracks[0].state == "OCCLUDED"
        assert red_tracks[0].time_since_update == 1

        # Frame 4: Red cube re-appears
        tracks4 = tracker.update(dets2)
        red_tracks_recovered = [t for t in tracks4 if t.color_name == "red"]
        assert len(red_tracks_recovered) == 1
        assert red_tracks_recovered[0].state == "ACTIVE"
        assert red_tracks_recovered[0].track_id == red_tracks[0].track_id, "Track ID must persist through occlusion"

        # Frame 5+: Multiple consecutive missed frames causes track to become LOST
        for _ in range(4):
            tracker.update(dets3_occluded)
        tracks_lost = tracker.update(dets3_occluded)
        assert len([t for t in tracks_lost if t.color_name == "red"]) == 0, "Stale track must be pruned when lost"
