"""
Sensor interfaces and observational data contracts for AERO robotics workcell.
Provides clean boundaries isolating perception and planning from internal simulation state.
"""

from dataclasses import dataclass, field
from typing import Protocol, Tuple, Optional, Any, List
import numpy as np
from PIL import Image

@dataclass
class CameraCalibration:
    """
    Overhead orthographic and perspective camera calibration parameters.
    """
    model_type: str  # "orthographic" or "perspective"
    width: int
    height: int
    # Coordinate boundaries in meters for orthographic view
    bounds_x: Tuple[float, float]  # (x_min, x_max)
    bounds_z: Tuple[float, float]  # (z_min, z_max)
    camera_height: float  # camera Y in robot base frame (m)
    # Extrinsic transform: camera_frame to robot_base_frame (4x4 matrix)
    cam_to_base_matrix: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))
    # Near and far clipping planes for depth
    near: float = 0.1
    far: float = 2.0

    def world_to_pixel(self, x: float, z: float) -> Tuple[int, int]:
        """Convert robot base X, Z coordinates to camera pixel (u, v)."""
        x_min, x_max = self.bounds_x
        z_min, z_max = self.bounds_z
        u = ((x - x_min) / (x_max - x_min)) * (self.width - 1)
        v = ((z_max - z) / (z_max - z_min)) * (self.height - 1)
        u_clamped = int(max(0, min(self.width - 1, round(u))))
        v_clamped = int(max(0, min(self.height - 1, round(v))))
        return u_clamped, v_clamped

    def pixel_to_world(self, u: float, v: float) -> Tuple[float, float]:
        """Convert camera pixel (u, v) to robot base X, Z coordinates."""
        x_min, x_max = self.bounds_x
        z_min, z_max = self.bounds_z
        denom_u = max(1.0, float(self.width - 1))
        denom_v = max(1.0, float(self.height - 1))
        x = x_min + (u / denom_u) * (x_max - x_min)
        z = z_max - (v / denom_v) * (z_max - z_min)
        return float(x), float(z)

    def depth_to_y(self, depth_val: float) -> float:
        """
        Convert camera depth (distance below camera plane along -Y)
        to robot base frame Y coordinate: Y_base = camera_height - depth.
        """
        return float(self.camera_height - depth_val)

    def y_to_depth(self, y_val: float) -> float:
        """Convert robot base Y coordinate to depth below camera."""
        return float(self.camera_height - y_val)


@dataclass
class CameraObservation:
    """
    Standard camera observation packet delivered to the perception stack.
    Contains no hidden simulation metadata.
    """
    timestamp: float
    tick: int
    rgb: Image.Image
    depth: Optional[np.ndarray]  # (H, W) float32 depth map in meters from camera
    calibration: CameraCalibration
    width: int
    height: int
    is_valid: bool = True
    metadata: dict = field(default_factory=dict)


class CameraInterface(Protocol):
    """Protocol for workcell camera devices (synthetic simulator or physical)."""

    def capture(self) -> CameraObservation:
        """Capture a fresh camera observation."""
        ...

    def get_calibration(self) -> CameraCalibration:
        """Return the current camera calibration."""
        ...

    def is_available(self) -> bool:
        """Check whether sensor hardware / pipeline is operational."""
        ...


@dataclass
class RobotStateObservation:
    """
    Robot internal telemetry exposed to perception and controller.
    Contains joint measurements and gripper status, but strictly no object ground-truth.
    """
    timestamp: float
    joint_positions: List[float]  # 5-DOF joint angles in radians
    joint_velocities: List[float]  # 5-DOF joint velocities in rad/s
    gripper_position: float        # Gripper gap opening (0.0=closed, 0.08=open)
    gripper_effort: float          # Gripper motor effort / applied torque
    tcp_pose_estimated: Optional[Tuple[float, float, float, float, float, float]] = None # (x, y, z, roll, pitch, yaw)
    is_e_stopped: bool = False


class RobotStateInterface(Protocol):
    """Protocol for reading robot telemetry."""

    def get_state(self) -> RobotStateObservation:
        """Return latest joint and gripper measurements."""
        ...
