"""
Coordinate Frame Systems and Transform Utilities for AERO Workcell.
Defines canonical 3D reference frames:
- robot_base_frame: Origin at base of robot arm.
- table_frame: Origin at table surface center (0.0, TABLE_TOP_Y, 0.0).
- camera_frame: Origin at overhead camera looking downward onto workspace.
- tcp_frame: Tool Center Point at gripper center.
"""

import math
from typing import Tuple, List, Optional, Union
import numpy as np

from .constants import TABLE_TOP_Y

# Default camera mounting height above robot base
OVERHEAD_CAMERA_HEIGHT: float = 1.60

def euler_to_rot_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Construct 3x3 rotation matrix from intrinsic Euler angles (roll, pitch, yaw) in radians.
    Convention: R = Rz(yaw) * Ry(pitch) * Rx(roll).
    """
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    R_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cr, -sr],
        [0.0, sr, cr]
    ], dtype=np.float64)

    R_y = np.array([
        [cp, 0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp]
    ], dtype=np.float64)

    R_z = np.array([
        [cy, -sy, 0.0],
        [sy, cy, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    return R_z @ R_y @ R_x


def make_transform(
    translation: Union[List[float], Tuple[float, float, float], np.ndarray],
    rotation: Optional[Union[np.ndarray, Tuple[float, float, float]]] = None
) -> np.ndarray:
    """
    Create a 4x4 homogeneous transformation matrix from translation and rotation.
    Rotation can be a 3x3 rotation matrix or (roll, pitch, yaw) tuple in radians.
    """
    T = np.eye(4, dtype=np.float64)
    t = np.asarray(translation, dtype=np.float64).reshape(3)
    T[0:3, 3] = t

    if rotation is not None:
        if isinstance(rotation, np.ndarray) and rotation.shape == (3, 3):
            T[0:3, 0:3] = rotation
        elif len(rotation) == 3:
            T[0:3, 0:3] = euler_to_rot_matrix(rotation[0], rotation[1], rotation[2])
    return T


def inverse_transform(T: np.ndarray) -> np.ndarray:
    """
    Compute rigid inverse of a 4x4 transformation matrix:
    [R  t]^-1 = [R^T  -R^T * t]
    [0  1]      [0         1  ]
    """
    T_inv = np.eye(4, dtype=np.float64)
    R = T[0:3, 0:3]
    t = T[0:3, 3]
    R_t = R.T
    T_inv[0:3, 0:3] = R_t
    T_inv[0:3, 3] = -R_t @ t
    return T_inv


def transform_point(
    point: Union[List[float], Tuple[float, float, float], np.ndarray],
    T: np.ndarray
) -> np.ndarray:
    """
    Transform a 3D point using a 4x4 homogeneous matrix.
    p_dst = (T @ [x, y, z, 1]^T)[0:3]
    """
    p_hom = np.array([point[0], point[1], point[2], 1.0], dtype=np.float64)
    res = T @ p_hom
    return res[0:3]


def transform_vector(
    vector: Union[List[float], Tuple[float, float, float], np.ndarray],
    T: np.ndarray
) -> np.ndarray:
    """
    Transform a 3D direction vector (ignores translation).
    v_dst = R @ v
    """
    v = np.asarray(vector, dtype=np.float64).reshape(3)
    return T[0:3, 0:3] @ v


class CoordinateFrameManager:
    """
    Central manager for coordinate transforms across workcell frames.
    """
    def __init__(self, camera_height: float = OVERHEAD_CAMERA_HEIGHT):
        self.camera_height = camera_height

        # 1. Base to Table transform:
        # Table surface origin is at (0.0, TABLE_TOP_Y, 0.0) in robot base frame.
        self.T_table_to_base = make_transform([0.0, TABLE_TOP_Y, 0.0])
        self.T_base_to_table = inverse_transform(self.T_table_to_base)

        # 2. Camera Frame:
        # Looking down: Camera optical axis Z_cam points along -Y_base.
        # X_cam points along +X_base.
        # Y_cam points along -Z_base.
        # R_cam_to_base maps [x_cam, y_cam, z_cam] to base frame.
        R_cam_to_base = np.array([
            [1.0,  0.0,  0.0],
            [0.0,  0.0, -1.0],
            [0.0, -1.0,  0.0]
        ], dtype=np.float64)
        t_cam_in_base = np.array([0.0, camera_height, 0.0], dtype=np.float64)

        self.T_cam_to_base = np.eye(4, dtype=np.float64)
        self.T_cam_to_base[0:3, 0:3] = R_cam_to_base
        self.T_cam_to_base[0:3, 3] = t_cam_in_base
        self.T_base_to_cam = inverse_transform(self.T_cam_to_base)

    def camera_to_base(self, pt_cam: Union[Tuple[float, float, float], np.ndarray]) -> Tuple[float, float, float]:
        """Convert point from camera optical frame to robot base frame."""
        pt = transform_point(pt_cam, self.T_cam_to_base)
        return float(pt[0]), float(pt[1]), float(pt[2])

    def base_to_camera(self, pt_base: Union[Tuple[float, float, float], np.ndarray]) -> Tuple[float, float, float]:
        """Convert point from robot base frame to camera optical frame."""
        pt = transform_point(pt_base, self.T_base_to_cam)
        return float(pt[0]), float(pt[1]), float(pt[2])

    def table_to_base(self, pt_table: Union[Tuple[float, float, float], np.ndarray]) -> Tuple[float, float, float]:
        """Convert point on table surface coordinates to robot base frame."""
        pt = transform_point(pt_table, self.T_table_to_base)
        return float(pt[0]), float(pt[1]), float(pt[2])

    def base_to_table(self, pt_base: Union[Tuple[float, float, float], np.ndarray]) -> Tuple[float, float, float]:
        """Convert point in robot base frame to table surface coordinates."""
        pt = transform_point(pt_base, self.T_base_to_table)
        return float(pt[0]), float(pt[1]), float(pt[2])
