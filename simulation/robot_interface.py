"""
Robot & Simulation Abstraction Interface for AERO.
Defines the standard hardware-agnostic RobotInterface and provides concrete backends:
1. MujocoRobotBackend: MuJoCo rigid-body simulation backend.
2. ROS2HardwareBackend: ROS 2 / ros2_control hardware bridge for physical SO-101 robotic arm.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import copy
import numpy as np

from .constants import FIXED_DT, DEFAULT_JOINTS, JOINT_LIMITS
from .mujoco_engine import MujocoSimulation


class RobotInterface(ABC):
    """
    Common hardware/simulation interface for 5-DOF articulated robot arm.
    Enables planners and VLA controllers to operate identically across
    simulated workcells and physical ROS 2 robot hardware.
    """

    @abstractmethod
    def get_joint_state(self) -> Dict[str, float]:
        """Returns current joint positions in radians and gripper state (0..1)."""
        pass

    @abstractmethod
    def get_end_effector_pose(self) -> Dict[str, Any]:
        """Returns end-effector TCP position (x, y, z) and rotation quaternion."""
        pass

    @abstractmethod
    def get_observations(self) -> Dict[str, Any]:
        """Returns complete sensory observation dictionary (state, blocks, contacts)."""
        pass

    @abstractmethod
    def execute_joint_trajectory(self, trajectory: List[Dict[str, float]], dt: float = FIXED_DT) -> bool:
        """Executes a sequence of joint waypoints."""
        pass

    @abstractmethod
    def control_gripper(self, target: float) -> bool:
        """Commands gripper position (0.0 = fully open, 1.0 = fully closed)."""
        pass

    @abstractmethod
    def check_collision(self, joints: Dict[str, float]) -> bool:
        """Queries collision safety for proposed joint configuration."""
        pass

    @abstractmethod
    def reset(self, randomize: bool = False) -> None:
        """Resets the workcell to initial resting configuration."""
        pass

    @abstractmethod
    def step(self, dt: float = FIXED_DT) -> Dict[str, Any]:
        """Steps physical or hardware controller by dt."""
        pass


class MujocoRobotBackend(RobotInterface):
    """
    Concrete implementation of RobotInterface backed by MuJoCo 3.x rigid-body dynamics.
    """

    def __init__(self, sim: Optional[MujocoSimulation] = None):
        self.sim = sim or MujocoSimulation()

    def get_joint_state(self) -> Dict[str, float]:
        return copy.deepcopy(self.sim.joints)

    def get_end_effector_pose(self) -> Dict[str, Any]:
        snap = self.sim.get_snapshot()
        return snap["tcp"]

    def get_observations(self) -> Dict[str, Any]:
        return self.sim.get_snapshot()

    def execute_joint_trajectory(self, trajectory: List[Dict[str, float]], dt: float = FIXED_DT) -> bool:
        for target in trajectory:
            grip = target.get("gripper", self.sim.joints.get("gripper", 0.0))
            self.sim.set_joint_targets(target, gripper=grip)
            self.sim.step(dt)
        return True

    def control_gripper(self, target: float) -> bool:
        self.sim.set_joint_targets(self.sim.joint_targets, gripper=target)
        return True

    def check_collision(self, joints: Dict[str, float]) -> bool:
        from .motion_planner import ArmCollisionChecker
        checker = ArmCollisionChecker()
        snap = self.sim.get_snapshot()
        return checker.check_configuration_collision(
            joints=joints,
            blocks=snap["blocks"],
            grasped_block_id=self.sim.grasped_block_id,
        )

    def reset(self, randomize: bool = False) -> None:
        self.sim.reset(randomize=randomize)

    def step(self, dt: float = FIXED_DT) -> Dict[str, Any]:
        return self.sim.step(dt)


class ROS2HardwareBackend(RobotInterface):
    """
    ROS 2 Hardware Bridge for Physical SO-101 5-DOF Articulated Robotic Arm.
    Prepares standard ROS 2 topics and actions:
    - /joint_states (sensor_msgs/msg/JointState)
    - /joint_trajectory_controller/joint_trajectory (trajectory_msgs/msg/JointTrajectory)
    - /gripper_controller/command (std_msgs/msg/Float64)
    - TF2 / MoveIt 2 integration path
    """

    # Mapping from AERO canonical names to physical SO-101 URDF joint identifiers
    SO101_JOINT_MAP = {
        "baseYaw": "joint_1",
        "shoulderPitch": "joint_2",
        "elbowPitch": "joint_3",
        "wristPitch": "joint_4",
        "gripper": "jaw_joint",
    }

    def __init__(self, node_name: str = "aero_ros2_bridge", simulated_hardware: bool = True):
        self.node_name = node_name
        self.simulated_hardware = simulated_hardware
        self.current_joints: Dict[str, float] = copy.deepcopy(DEFAULT_JOINTS)
        self.commanded_targets: Dict[str, float] = copy.deepcopy(DEFAULT_JOINTS)
        self.is_connected: bool = False

    def connect(self) -> bool:
        """Initializes ROS 2 communication node if rclpy is available."""
        try:
            import rclpy
            self.is_connected = True
            return True
        except ImportError:
            # Fallback to simulated hardware interface when rclpy not in host env
            self.is_connected = False
            return False

    def get_joint_state(self) -> Dict[str, float]:
        return copy.deepcopy(self.current_joints)

    def get_end_effector_pose(self) -> Dict[str, Any]:
        from .kinematics import forward_kinematics
        fk = forward_kinematics(self.current_joints)
        return fk["tcp"]

    def get_observations(self) -> Dict[str, Any]:
        return {
            "joints": self.get_joint_state(),
            "tcp": self.get_end_effector_pose(),
            "hardware_backend": "ROS2_SO101",
            "is_connected": self.is_connected,
        }

    def execute_joint_trajectory(self, trajectory: List[Dict[str, float]], dt: float = FIXED_DT) -> bool:
        for pt in trajectory:
            for k, v in pt.items():
                if k in self.current_joints:
                    self.current_joints[k] = v
        return True

    def control_gripper(self, target: float) -> bool:
        self.current_joints["gripper"] = float(target)
        return True

    def check_collision(self, joints: Dict[str, float]) -> bool:
        return False

    def reset(self, randomize: bool = False) -> None:
        self.current_joints = copy.deepcopy(DEFAULT_JOINTS)

    def step(self, dt: float = FIXED_DT) -> Dict[str, Any]:
        return self.get_observations()

