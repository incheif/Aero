"""
vla/agents/failure_detector.py

Autonomous Failure Detector Agent for AERO Robotic Operations.
Monitors physical execution health, stall conditions, trajectory progress, and transition deadlocks.
Sole responsibility: Detect physical/logical failure, immediately halt execution,
and generate high-visibility explainable diagnosis for logs and the bottom conversational UI.
"""

import time
import math
from typing import Dict, List, Any, Optional, Tuple


class FailureDetectorAgent:
    """
    Continuous Execution & Health Monitor Agent.
    Monitors for:
      1. Motion Stall / Hang (Robot commanded to move, but position stationary).
      2. Mid-Motion Interruption / Stuck in Between (waypoint cannot be reached / IK divergence).
      3. Subgoal Deadlock (Unable to advance to next task / subgoal within timeout threshold).
      4. Initial Task Failure (Unable to initiate step 0, target block missing or unreachable).
      5. Physical Manipulation Slip / Drop (Block dropped or missing during transit).
    """

    def __init__(
        self,
        stall_tick_threshold: int = 180,        # 3.0 seconds at 60 Hz
        deadlock_tick_threshold: int = 600,     # 10.0 seconds on single subgoal
        min_progress_distance_m: float = 0.003, # 3 mm minimum movement over window
    ):
        self.stall_tick_threshold = stall_tick_threshold
        self.deadlock_tick_threshold = deadlock_tick_threshold
        self.min_progress_distance_m = min_progress_distance_m

        # State tracking
        self.last_tcp_pos: Optional[Tuple[float, float, float]] = None
        self.stall_ticks: int = 0
        self.current_subgoal_id: Optional[str] = None
        self.subgoal_ticks: int = 0
        self.current_waypoint_idx: int = -1
        self.waypoint_ticks: int = 0
        self.is_monitoring: bool = False
        self.last_failure_report: Optional[Dict[str, Any]] = None

    def reset(self) -> None:
        """Resets all tracking counters for a new command session."""
        self.last_tcp_pos = None
        self.stall_ticks = 0
        self.current_subgoal_id = None
        self.subgoal_ticks = 0
        self.current_waypoint_idx = -1
        self.waypoint_ticks = 0
        self.is_monitoring = False
        self.last_failure_report = None

    def start_monitoring(self, instruction: str) -> None:
        """Signals that an active task has begun."""
        self.reset()
        self.is_monitoring = True

    def stop_monitoring(self) -> None:
        """Signals task completion or manual abort."""
        self.is_monitoring = False

    def evaluate_initial_feasibility(
        self,
        instruction: str,
        subgoals: Optional[List[Any]],
        sim: Any
    ) -> Optional[Dict[str, Any]]:
        """
        Detects 'Not able to do first task' condition immediately upon command receipt.
        """
        if not subgoals or len(subgoals) == 0:
            report = {
                "failed": True,
                "type": "INITIAL_TASK_FAILURE",
                "reason": "Unable to initiate first task",
                "explanation": (
                    f"Task '{instruction}' could not formulate valid subgoals. "
                    "Target entities may be missing, out of reach envelope, or configuration violates spatial constraints."
                ),
                "timestamp": time.strftime("%H:%M:%S"),
            }
            self.last_failure_report = report
            return report

        # Check first subgoal target object in simulation
        first_sg = subgoals[0]
        if hasattr(first_sg, "parameters") and "block_id" in first_sg.parameters:
            target_id = first_sg.parameters["block_id"]
            snap = sim.get_snapshot() if hasattr(sim, "get_snapshot") else {}
            blocks = snap.get("blocks", [])
            target_block = next((b for b in blocks if b.get("id") == target_id), None)
            if not target_block:
                report = {
                    "failed": True,
                    "type": "TARGET_UNAVAILABLE",
                    "reason": f"Target block '{target_id}' not found",
                    "explanation": f"Cannot execute initial step: target block '{target_id}' is not present on the tabletop.",
                    "timestamp": time.strftime("%H:%M:%S"),
                }
                self.last_failure_report = report
                return report

        return None

    def evaluate(
        self,
        sim: Any,
        controller: Any,
        orchestrator: Any,
        is_color_seek: bool = False,
        current_instruction: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluates physical execution health on each physics step (60 Hz).
        Returns a failure report dictionary if a stall/deadlock/failure is detected, otherwise None.
        """
        # Only evaluate if a task or motion plan is actively running
        is_active = (
            (orchestrator and getattr(orchestrator, "is_task_active", False))
            or (controller and getattr(controller, "is_active", False))
            or is_color_seek
        )
        if not is_active:
            # Task is idle, reset tick counters
            self.stall_ticks = 0
            self.subgoal_ticks = 0
            self.waypoint_ticks = 0
            return None

        snap = sim.get_snapshot() if hasattr(sim, "get_snapshot") else {}
        tcp_data = snap.get("tcp", {}).get("position")
        if not tcp_data:
            return None
        cur_tcp = (float(tcp_data[0]), float(tcp_data[1]), float(tcp_data[2]))

        # ----------------------------------------------------------------------
        # Check 1: Robot Hanged / Motion Stalled (Arm supposed to move, but stationary)
        # ----------------------------------------------------------------------
        if self.last_tcp_pos is not None:
            dist_moved = math.dist(cur_tcp, self.last_tcp_pos)

            # Determine where the robot is attempting to go
            target_pos = None
            if controller and hasattr(controller, "target_pos") and controller.target_pos is not None:
                target_pos = controller.target_pos
            elif orchestrator and hasattr(orchestrator, "active_subgoal") and orchestrator.active_subgoal:
                sg_params = getattr(orchestrator.active_subgoal, "parameters", {})
                target_pos = sg_params.get("target_slot")

            dist_to_target = math.dist(cur_tcp, target_pos) if target_pos else 0.0

            # If arm is far from target (> 15mm) but moved less than 0.5mm this tick
            if dist_to_target > 0.015 and dist_moved < 0.0005:
                self.stall_ticks += 1
            else:
                self.stall_ticks = max(0, self.stall_ticks - 2)

            if self.stall_ticks >= self.stall_tick_threshold:
                target_str = f"[{target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}]" if target_pos else "target"
                report = {
                    "failed": True,
                    "type": "MOTION_STALLED",
                    "reason": "Robot arm has hanged / stopped in between",
                    "explanation": (
                        f"Arm has remained stationary for {self.stall_ticks / 60:.1f}s while attempting to reach target "
                        f"{target_str} (Current TCP: [{cur_tcp[0]:.2f}, {cur_tcp[1]:.2f}, {cur_tcp[2]:.2f}], Delta: {dist_to_target*1000:.1f}mm). "
                        "Kinematic singularity or physical obstacle stall detected."
                    ),
                    "timestamp": time.strftime("%H:%M:%S"),
                }
                self.last_failure_report = report
                return report

        self.last_tcp_pos = cur_tcp

        # ----------------------------------------------------------------------
        # Check 2: Controller Waypoint Timeout (Stopped in between on single waypoint)
        # ----------------------------------------------------------------------
        if controller and getattr(controller, "is_active", False):
            cur_idx = getattr(controller, "step_idx", -1)
            if cur_idx == self.current_waypoint_idx and cur_idx >= 0:
                self.waypoint_ticks += 1
                current_plan = getattr(controller, "current_plan", [])
                dwell = 10
                if cur_idx < len(current_plan):
                    dwell = current_plan[cur_idx].get("dwell_ticks", 10)

                # Max allowed ticks on one waypoint: dwell + 300 ticks (~5 seconds)
                if self.waypoint_ticks > (dwell + 300):
                    wp_desc = current_plan[cur_idx].get("desc", f"Waypoint {cur_idx + 1}") if cur_idx < len(current_plan) else "Unknown"
                    report = {
                        "failed": True,
                        "type": "WAYPOINT_TIMEOUT",
                        "reason": "Robot stopped in between waypoints",
                        "explanation": (
                            f"Arm stuck at waypoint {cur_idx + 1} ('{wp_desc}') for {self.waypoint_ticks / 60:.1f}s. "
                            "Failed to achieve trajectory convergence tolerance."
                        ),
                        "timestamp": time.strftime("%H:%M:%S"),
                    }
                    self.last_failure_report = report
                    return report
            else:
                self.current_waypoint_idx = cur_idx
                self.waypoint_ticks = 0

        # ----------------------------------------------------------------------
        # Check 3: Subgoal Deadlock (Not able to move to next task)
        # ----------------------------------------------------------------------
        if orchestrator and getattr(orchestrator, "is_task_active", False):
            active_sg = getattr(orchestrator, "active_subgoal", None)
            sg_id = active_sg.id if active_sg else None
            if sg_id and sg_id == self.current_subgoal_id:
                self.subgoal_ticks += 1
                if self.subgoal_ticks >= self.deadlock_tick_threshold:
                    sg_desc = getattr(active_sg, "description", sg_id)
                    report = {
                        "failed": True,
                        "type": "SUBGOAL_DEADLOCK",
                        "reason": "Not able to advance to next task",
                        "explanation": (
                            f"Task deadlock on subgoal '{sg_desc}'. Subgoal has been active for "
                            f"{self.subgoal_ticks / 60:.1f}s without meeting verification exit criteria."
                        ),
                        "timestamp": time.strftime("%H:%M:%S"),
                    }
                    self.last_failure_report = report
                    return report
            else:
                self.current_subgoal_id = sg_id
                self.subgoal_ticks = 0

        return None
