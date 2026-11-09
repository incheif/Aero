"""
Data models and contracts for Perception, Detection, and Multi-Object Tracking.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any


@dataclass
class ObjectDetection:
    """
    Single frame detection from raw camera observation.
    Contains no hidden simulator flags or true block IDs.
    """
    class_name: str                    # "block", "pad", "obstacle"
    color_name: Optional[str]          # "red", "blue", "yellow", "green", "orange", "purple"
    color_rgb: Optional[Tuple[int, int, int]]
    bbox_xyxy: Tuple[int, int, int, int]   # Image space bounding box (u1, v1, u2, v2)
    centroid_uv: Tuple[float, float]       # Image space centroid (u, v)
    pixel_area: int                        # Area in pixels
    position_camera: Optional[Tuple[float, float, float]] = None
    position_base: Optional[Tuple[float, float, float]] = None # Metric (X, Y, Z) in robot base frame
    dimensions: Optional[Tuple[float, float, float]] = None     # Metric (dx, dy, dz)
    confidence: float = 0.95
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_name": self.class_name,
            "color_name": self.color_name,
            "color_rgb": self.color_rgb,
            "bbox_xyxy": self.bbox_xyxy,
            "centroid_uv": [round(c, 2) for c in self.centroid_uv],
            "pixel_area": self.pixel_area,
            "position_base": [round(p, 4) for p in self.position_base] if self.position_base else None,
            "dimensions": [round(d, 4) for d in self.dimensions] if self.dimensions else None,
            "confidence": round(self.confidence, 3),
            "timestamp": self.timestamp,
        }


@dataclass
class TrackedObject:
    """
    Temporally tracked entity maintained across consecutive observations.
    Provides persistent track IDs, position smoothing, and occlusion handling.
    """
    track_id: str                      # Persistent identifier, e.g. "track_1"
    class_name: str                    # "block", "pad"
    color_name: Optional[str]
    color_rgb: Optional[Tuple[int, int, int]]
    position: List[float]              # Filtered [X, Y, Z] in robot base frame
    velocity: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    bbox_xyxy: Tuple[int, int, int, int] = (0, 0, 0, 0)
    dimensions: Tuple[float, float, float] = (0.055, 0.055, 0.055)
    confidence: float = 0.90
    age_ticks: int = 1
    hits: int = 1
    time_since_update: int = 0
    state: str = "ACTIVE"              # "ACTIVE", "OCCLUDED", "LOST"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "color_name": self.color_name,
            "color_rgb": self.color_rgb,
            "position": [round(p, 4) for p in self.position],
            "velocity": [round(v, 4) for v in self.velocity],
            "bbox_xyxy": self.bbox_xyxy,
            "dimensions": [round(d, 4) for d in self.dimensions],
            "confidence": round(self.confidence, 3),
            "age_ticks": self.age_ticks,
            "hits": self.hits,
            "time_since_update": self.time_since_update,
            "state": self.state,
        }

