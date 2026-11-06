"""
Trajectory Controller & Plan Execution Engine.
Translates Google VLA waypoints and actions into continuous joint commands
using analytical inverse kinematics and smooth velocity profiling.
"""

import math
from typing import Dict, List, Any, Optional, Tuple
from simulation.constants import (
    FIXED_DT,
    SAFE_HOVER_Y,
    DEFAULT_JOINTS,
    MAX_TCP_STEP,
    GRASP_CLOSE_THRESHOLD,
    GRASP_OPEN_THRESHOLD,
)
from simulation.kinematics import inverse_kinematics

class TrajectoryController:
    """
    Executes high-level VLA plan steps on the simulated robot arm.
    Interpolates TCP positions along straight Cartesian lines, handles dwell durations,
    and generates continuous joint commands via analytical inverse kinematics.
    """
    def __init__(self):
        self.current_plan: List[Dict[str, Any]] = []
        self.step_idx: int = 0
        self.dwell_counter: int = 0
        self.is_active: bool = False
        self.current_action_desc: str = "Idle"
        self.tcp_trail: List[Tuple[float, float, float]] = []
        self.max_trail_points: int = 120
        self.current_setpoint: Optional[List[float]] = None

    def load_plan(self, plan: List[Dict[str, Any]]):
        self.current_plan = list(plan)
        self.step_idx = 0
        self.dwell_counter = 0
        self.is_active = len(plan) > 0
        self.current_action_desc = self.current_plan[0]["desc"] if self.is_active else "Idle"
        self.current_setpoint = None

    def stop(self):
        self.is_active = False
        self.current_plan.clear()
        self.step_idx = 0
        self.current_action_desc = "Stopped"
        self.tcp_trail.clear()
        self.current_setpoint = None

    def step(self, current_tcp_pos: Tuple[float, float, float], current_joints: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """
        Calculates joint targets for the current simulation tick.
        Uses Cartesian setpoint linear interpolation to guarantee the TCP follows a clean,
        non-dipping 3D trajectory segment between waypoints.
        Returns: Dict with {"joints": ..., "gripper": ..., "step_desc": ..., "is_done": ...}
        """
        # Append TCP position to trail
        self.tcp_trail.append(current_tcp_pos)
        if len(self.tcp_trail) > self.max_trail_points:
            self.tcp_trail.pop(0)

        if not self.is_active or self.step_idx >= len(self.current_plan):
            self.is_active = False
            self.current_action_desc = "Task Completed"
            self.current_setpoint = None
            return None

        current_step = self.current_plan[self.step_idx]
        target_pos = current_step["target"]
        desired_gripper = current_step.get("gripper", 0.0)
        self.current_action_desc = current_step.get("desc", f"Executing step {self.step_idx + 1}")

        # Initialize Cartesian setpoint to current TCP if starting fresh
        if self.current_setpoint is None:
            self.current_setpoint = [float(current_tcp_pos[0]), float(current_tcp_pos[1]), float(current_tcp_pos[2])]

        # Advance Cartesian setpoint linearly toward target_pos only when tracking lag is bounded.
        # If joints lag behind (e.g. accelerating under load), hold setpoint stationary to let arm catch up,
        # rather than destructively dragging the setpoint backward or down into table blocks.
        dx = target_pos[0] - self.current_setpoint[0]
        dy = target_pos[1] - self.current_setpoint[1]
        dz = target_pos[2] - self.current_setpoint[2]
        dist_setpoint = math.hypot(dx, dy, dz)

        lag = math.hypot(
            self.current_setpoint[0] - current_tcp_pos[0],
            self.current_setpoint[1] - current_tcp_pos[1],
            self.current_setpoint[2] - current_tcp_pos[2]
        )

        max_step = 0.010  # 10mm per tick at 60Hz (~0.60 m/s, matches physical joint actuator speed limit)
        if lag <= 0.050:
            if dist_setpoint <= max_step:
                self.current_setpoint = [float(target_pos[0]), float(target_pos[1]), float(target_pos[2])]
            else:
                scale = max_step / dist_setpoint
                self.current_setpoint[0] += dx * scale
                self.current_setpoint[1] += dy * scale
                self.current_setpoint[2] += dz * scale

        # Safe altitude floor: never allow setpoint altitude below clearance plane during transit/hover
        if target_pos[1] >= SAFE_HOVER_Y - 0.01:
            self.current_setpoint[1] = max(self.current_setpoint[1], SAFE_HOVER_Y - 0.005)

        # Candidate pitches for IK: step-specified candidates, or continuity from current pitch
        step_pitches = current_step.get("candidate_pitches")
        if step_pitches is None and current_step.get("world_pitch") is not None:
            step_pitches = [current_step["world_pitch"]]
        cur_pitch = current_joints.get("shoulderPitch", 0.0) + current_joints.get("elbowPitch", 0.0) + current_joints.get("wristPitch", 0.0)
        if step_pitches is None:
            step_pitches = [cur_pitch, -1.25, -1.0, -1.45, -0.60, -0.12]
        else:
            # Preserve pitch continuity across ticks to avoid mid-trajectory joint branch flipping
            step_pitches = sorted(step_pitches, key=lambda p: abs(p - cur_pitch))

        # Compute IK targets for the interpolated setpoint along the 3D line segment
        joint_targets = inverse_kinematics(tuple(self.current_setpoint), gripper=desired_gripper, candidate_pitches=step_pitches)

        # Distance from actual current TCP to final step target
        actual_dx = target_pos[0] - current_tcp_pos[0]
        actual_dy = target_pos[1] - current_tcp_pos[1]
        actual_dz = target_pos[2] - current_tcp_pos[2]
        actual_dist = math.hypot(actual_dx, actual_dy, actual_dz)

        # Gripper physical state convergence check
        current_grip = current_joints.get("gripper", 0.0)
        action_name = str(current_step.get("action", "")).upper()
        gripper_settled = True
        if desired_gripper >= 0.85 or "GRIP" in action_name or "GRASP" in action_name:
            # Grasp step: jaws must physically close past threshold before advancing
            if current_grip < GRASP_CLOSE_THRESHOLD:
                gripper_settled = False
        elif desired_gripper <= 0.65 or "RELEASE" in action_name:
            # Release step: jaws must physically open below threshold before advancing
            if current_grip > GRASP_OPEN_THRESHOLD:
                gripper_settled = False
        else:
            # Partial / gesture aperture setpoint (e.g. 0.5 for wave fingers)
            if abs(current_grip - desired_gripper) > 0.12:
                gripper_settled = False

        # Check if setpoint reached target AND actual TCP is within arrival threshold AND gripper converged
        arrival_threshold = 0.018  # 18mm acceptance radius
        setpoint_dist = math.hypot(
            target_pos[0] - self.current_setpoint[0],
            target_pos[1] - self.current_setpoint[1],
            target_pos[2] - self.current_setpoint[2]
        )
        setpoint_at_target = setpoint_dist < 0.003

        if setpoint_at_target and actual_dist <= arrival_threshold and gripper_settled:
            self.dwell_counter += 1
            # Action-specific minimum dwell to guarantee physical contact stabilization
            if desired_gripper >= 0.85 or "GRIP" in action_name or "GRASP" in action_name:
                min_dwell = 18
            elif "RELEASE" in action_name:
                min_dwell = 20
            elif "DESCEND" in action_name or "PLACE" in action_name:
                min_dwell = 14
            else:
                min_dwell = 10
            required_dwell = max(min_dwell, current_step.get("dwell_ticks", min_dwell))
            if self.dwell_counter >= required_dwell:
                # Advance to next waypoint
                self.dwell_counter = 0
                self.step_idx += 1
                self.current_setpoint = None
                if self.step_idx >= len(self.current_plan):
                    self.is_active = False
                    self.current_action_desc = "Task Completed Successfully"
                    self.current_setpoint = None
        else:
            self.dwell_counter = 0

        return {
            "joints": joint_targets,
            "gripper": desired_gripper,
            "current_step": self.step_idx + 1,
            "total_steps": len(self.current_plan),
            "step_desc": self.current_action_desc,
            "is_active": self.is_active,
        }

    def get_progress(self) -> Dict[str, Any]:
        return {
            "is_active": self.is_active,
            "step": self.step_idx + 1 if self.is_active else 0,
            "total_steps": len(self.current_plan),
            "pct": round((self.step_idx / max(1, len(self.current_plan))) * 100.0, 1),
            "action": self.current_action_desc,
        }
