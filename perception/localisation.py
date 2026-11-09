"""
3D Localisation module for workspace objects.
Transforms image-space detections into calibrated 3D metric coordinates in robot_base_frame.
"""

from typing import List, Tuple, Optional
import numpy as np

from simulation.constants import TABLE_TOP_Y, CUBE_SIZE, CUBE_HALF
from simulation.sensor_interfaces import CameraCalibration, CameraObservation
from .models import ObjectDetection


class Object3DLocalizer:
    """
    Localizes 2D bounding boxes and centroids to 3D metric coordinates in robot_base_frame.
    Uses calibrated orthographic projection and synthetic/sensor depth.
    """
    def __init__(self, table_top_y: float = TABLE_TOP_Y):
        self.table_top_y = table_top_y

    def localize(
        self,
        detections: List[ObjectDetection],
        observation: CameraObservation
    ) -> List[ObjectDetection]:
        """
        Computes 3D position_base and dimensions for each detection.
        Modifies detections in place and returns the list.
        """
        calib = observation.calibration
        depth_map = observation.depth

        for det in detections:
            u, v = det.centroid_uv
            # 1. Horizontal coordinates (X, Z) on table plane
            x_m, z_m = calib.pixel_to_world(u, v)

            # 2. Vertical coordinate (Y)
            if depth_map is not None:
                iu = int(round(u))
                iv = int(round(v))
                # Sample local median in 3x3 window around centroid
                u_min = max(0, iu - 1)
                u_max = min(observation.width - 1, iu + 1)
                v_min = max(0, iv - 1)
                v_max = min(observation.height - 1, iv + 1)

                depth_patch = depth_map[v_min:v_max + 1, u_min:u_max + 1]
                median_depth = float(np.median(depth_patch))

                # Camera looks down from camera_height onto top face of cube
                top_y = calib.depth_to_y(median_depth)
                # Cube center is CUBE_HALF below top face
                center_y = top_y - CUBE_HALF

                # Fallback safeguard if depth is noisy or out of bounds
                if center_y < self.table_top_y:
                    center_y = self.table_top_y + CUBE_HALF
            else:
                # Default tabletop assumption
                center_y = self.table_top_y + CUBE_HALF

            det.position_base = (float(x_m), float(center_y), float(z_m))
            det.dimensions = (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)

        return detections
