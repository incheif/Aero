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
    Interpolates TCP positions, handles dwell durations, and generates joint commands.
    """
    def __init__(self):
        self.current_plan: List[Dict[str, Any]] = []
        self.step_idx: int = 0
        self.dwell_counter: int = 0
        self.is_active: bool = False
        self.current_action_desc: str = "Idle"
        self.tcp_trail: List[Tuple[float, float, float]] = []
        self.max_trail_points: int = 120

    def load_plan(self, plan: List[Dict[str, Any]]):
        self.current_plan = list(plan)
        self.step_idx = 0
        self.dwell_counter = 0
        self.is_active = len(plan) > 0
        self.current_action_desc = self.current_plan[0]["desc"] if self.is_active else "Idle"

    def stop(self):
        self.is_active = False
        self.current_plan.clear()
        self.step_idx = 0
        self.current_action_desc = "Stopped"

    def step(self, current_tcp_pos: Tuple[float, float, float], current_joints: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """
        Calculates joint targets for the current simulation tick.
        Returns: Dict with {"joints": ..., "gripper": ..., "step_desc": ..., "is_done": ...}
        """
        # Append TCP position to trail
        self.tcp_trail.append(current_tcp_pos)
        if len(self.tcp_trail) > self.max_trail_points:
            self.tcp_trail.pop(0)

        if not self.is_active or self.step_idx >= len(self.current_plan):
            self.is_active = False
            self.current_action_desc = "Task Completed"
            return None

        current_step = self.current_plan[self.step_idx]
        target_pos = current_step["target"]
        desired_gripper = current_step.get("gripper", 0.0)
        self.current_action_desc = current_step.get("desc", f"Executing step {self.step_idx + 1}")

        # Check distance to target TCP
        dx = target_pos[0] - current_tcp_pos[0]
        dy = target_pos[1] - current_tcp_pos[1]
        dz = target_pos[2] - current_tcp_pos[2]
        dist = math.hypot(dx, dy, dz)

        # Compute IK targets for target position
        joint_targets = inverse_kinematics(target_pos, gripper=desired_gripper)

        # Gripper physical state convergence check
        current_grip = current_joints.get("gripper", 0.0)
        gripper_settled = True
        if desired_gripper >= 0.5:
            # Grasp step: jaws must physically close past threshold before advancing
            if current_grip < GRASP_CLOSE_THRESHOLD:
                gripper_settled = False
        else:
            # Release step: jaws must physically open below threshold before advancing
            if current_grip > GRASP_OPEN_THRESHOLD:
                gripper_settled = False

        # Check if we have arrived close enough to target AND gripper has converged
        arrival_threshold = 0.015  # 15mm acceptance radius
        if dist <= arrival_threshold and gripper_settled:
            self.dwell_counter += 1
            # Action-specific minimum dwell to guarantee physical contact stabilization
            action_name = str(current_step.get("action", "")).upper()
            if desired_gripper >= 0.5 or "GRIP" in action_name or "GRASP" in action_name:
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
                if self.step_idx >= len(self.current_plan):
                    self.is_active = False
                    self.current_action_desc = "Task Completed Successfully"
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
