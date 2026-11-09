"""
AERO Perception Package.
Provides pure observation-based object detection, 3D localisation, and tracking.
"""

from .models import ObjectDetection, TrackedObject
from .detector import BlockColorDetector
from .localisation import Object3DLocalizer
from .tracker import MultiObjectTracker

__all__ = [
    "ObjectDetection",
    "TrackedObject",
    "BlockColorDetector",
    "Object3DLocalizer",
    "MultiObjectTracker",
]

