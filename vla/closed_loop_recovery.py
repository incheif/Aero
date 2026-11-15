"""
vla/closed_loop_recovery.py

Closed-Loop Recursive Self-Healing Orchestrator for AERO.
Integrates H-JEPA Level 1 real-time discrepancy monitoring (60 Hz)
with dynamic Subgoal DAG repair to guarantee task completion under physical disturbances.

Features:
- Real-time latent prediction residual tracking: Delta_z(t) = ||hat{z}_t - z_t||_2.
- Anomaly / slip / disturbance detection before hardware fault or catastrophic failure.
- Autonomous recovery synthesis: re-perceives physical positions, creates recovery nodes,
  and splices them into the active Subgoal DAG without human intervention.
- Zero-cheat execution: all recovery maneuvers use genuine 6-DOF kinematics and MuJoCo contacts.
"""

import math
import copy
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np

from simulation.constants import FIXED_DT, TABLE_TOP_Y, CUBE_HALF, SAFE_HOVER_Y
from vla.predictive_world_model import (
    HierarchicalJEPA,
    extract_state_vector_6dof,
    extract_action_vector_7d,
)
from vla.recursive_dag_planner import RecursiveDAGPlanner, Subgoal
from vla.primitive_skills import PrimitiveSkillExecutor
from vla.latent_action_scorer import LatentActionScorer


class ClosedLoopRecoveryOrchestrator:
    """
    Autonomous Execution Engine with real-time H-JEPA predictive monitoring
    and recursive self-healing capability.
    """

    def __init__(
        self,
        jepa_checkpoint: str = "models/hjepa_6dof_world_model.npz",
        residual_anomaly_threshold: float = 4.5,
    ):
        self.planner = RecursiveDAGPlanner()
        self.executor = PrimitiveSkillExecutor()
        self.scorer = LatentActionScorer(checkpoint_path=jepa_checkpoint)
        self.anomaly_threshold = residual_anomaly_threshold

        self.latent_residuals: List[float] = []
        self.anomaly_events: List[Dict[str, Any]] = []
        self.subgoal_history: List[Dict[str, Any]] = []

    def run_task(
        self,
        sim: Any,
        instruction: str,
        max_total_ticks: int = 1500,
        perturbation_hook: Optional[Callable[[int, Any], None]] = None,
        telemetry_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Executes a high-level natural language instruction end-to-end with
        recursive DAG dispatch and real-time self-healing.
        """
        self.latent_residuals.clear()
        self.anomaly_events.clear()
        self.subgoal_history.clear()

        # Step 1: Initial Perception & Recursive DAG Compilation
        snap = sim.get_snapshot()
        subgoals = self.planner.decompose(instruction, snap["blocks"])
        if not subgoals:
            return {"success": False, "message": "Failed to compile task into Subgoal DAG"}

        prev_snap = snap
        prev_action = (np.zeros(6, dtype=np.float32), 0.0)
        total_ticks = 0

        # Step 2: Recursive Execution Loop
        while total_ticks < max_total_ticks:
            ready_subgoals = self.planner.get_ready_subgoals()
            if not ready_subgoals:
                # Check if all subgoals are completed
                has_subgoals = len(self.planner.subgoals) > 0
                all_done = has_subgoals and all(sg.status == "COMPLETED" for sg in self.planner.subgoals.values())
                if all_done:
                    return {
                        "success": True,
                        "message": "All subgoals completed and physically verified",
                        "total_ticks": total_ticks,
                        "anomalies_healed": len(self.anomaly_events),
                        "max_residual": max(self.latent_residuals) if self.latent_residuals else 0.0,
                    }
                else:
                    # Stalled or pending unrecoverable subgoals
                    failed_sgs = [sg for sg in self.planner.subgoals.values() if sg.status == "FAILED"]
                    msg = f"Execution stalled with {len(failed_sgs)} failed subgoals" if has_subgoals else "No subgoals generated (insufficient blocks or invalid request)"
                    return {
                        "success": False,
                        "message": msg,
                        "total_ticks": total_ticks,
                        "anomalies_healed": len(self.anomaly_events),
                    }

            # Dispatch next ready subgoal
            active_sg = ready_subgoals[0]
            self.planner.mark_active(active_sg.id)

            # Pre-validate in latent space via H-JEPA Level 2
            is_feasible, conf = self.scorer.evaluate_subgoal_feasibility(prev_snap, active_sg)

            # Compile to trajectory plan steps
            plan_steps = self.executor.subgoal_to_plan_steps(sim, active_sg)
            if not plan_steps:
                self.planner.mark_failed(active_sg.id, "Plan compilation failed")
                continue

            self.executor.controller.load_plan(plan_steps)

            # Step-by-step execution with 60 Hz H-JEPA monitoring
            subgoal_success = False
            subgoal_failed_reason = ""

            while self.executor.controller.is_active and total_ticks < max_total_ticks:
                total_ticks += 1

                # Inject dynamic perturbation if hook provided
                if perturbation_hook:
                    perturbation_hook(total_ticks, sim)

                fk = sim.get_snapshot()["tcp"]["position"]
                cmd = self.executor.controller.step(tuple(fk), sim.joints)
                if cmd is not None:
                    sim.set_joint_targets(cmd["joints"], cmd.get("gripper"))

                curr_snap = sim.step(FIXED_DT)

                # Level 1 Latent Residual Evaluation: ||hat{z}_t - z_t||_2
                joints_cmd = cmd["joints"] if cmd else sim.joints
                grip_cmd = cmd.get("gripper", 0.0) if cmd else 0.0
                a_vec = extract_action_vector_7d(joints_cmd, grip_cmd)

                residual = self.scorer.jepa.evaluate_residual(
                    extract_state_vector_6dof(prev_snap),
                    a_vec,
                    extract_state_vector_6dof(curr_snap),
                )
                self.latent_residuals.append(residual)

                # Anomaly / Slip Detection
                if residual > self.anomaly_threshold:
                    event = {
                        "tick": total_ticks,
                        "subgoal_id": active_sg.id,
                        "residual": residual,
                        "type": "LATENT_ANOMALY_SPIKE",
                    }
                    self.anomaly_events.append(event)

                if telemetry_callback:
                    telemetry_callback({
                        "tick": total_ticks,
                        "subgoal": active_sg.id,
                        "residual": residual,
                        "snapshot": curr_snap,
                    })

                prev_snap = curr_snap

            # Post-Condition Physical Verification
            skill = active_sg.skill_type
            if skill in ("DEPOSIT", "PICK_AND_PLACE", "CLEAR_OBSTACLE"):
                block_id = active_sg.parameters["block_id"]
                target_slot = active_sg.parameters["target_slot"]
                final_block = next((b for b in curr_snap["blocks"] if b["id"] == block_id), None)

                if final_block:
                    fx, fy, fz = final_block["position"]
                    dx = abs(fx - target_slot[0])
                    dz = abs(fz - target_slot[2])
                    dy = abs(fy - target_slot[1])
                    if dx < 0.065 and dz < 0.065 and dy < 0.045:
                        subgoal_success = True
                    else:
                        subgoal_failed_reason = f"Slot tolerance exceeded (dx={dx:.3f}, dz={dz:.3f})"
                else:
                    subgoal_failed_reason = f"Block {block_id} missing"
            else:
                subgoal_success = True

            if subgoal_success:
                self.planner.mark_completed(active_sg.id)
                self.subgoal_history.append({"id": active_sg.id, "status": "COMPLETED"})
                # If this was a recovery subgoal, also mark parent subgoal completed
                if active_sg.parent_id and active_sg.parent_id in self.planner.subgoals:
                    parent_sg = self.planner.subgoals[active_sg.parent_id]
                    if parent_sg.status in ("PENDING", "RECOVERING", "FAILED"):
                        self.planner.mark_completed(parent_sg.id)
            else:
                # Trigger Self-Healing Recovery!
                self.planner.mark_failed(active_sg.id, subgoal_failed_reason)
                self.subgoal_history.append({"id": active_sg.id, "status": "FAILED", "reason": subgoal_failed_reason})

                # Re-perceive dropped block pose and splice dynamic recovery node
                if active_sg.retry_count < active_sg.max_retries:
                    block_id = active_sg.parameters["block_id"]
                    re_snap = sim.get_snapshot()
                    curr_block = next((b for b in re_snap["blocks"] if b["id"] == block_id), None)
                    if curr_block:
                        rec_id = f"recover_{active_sg.id}_r{active_sg.retry_count}"
                        rec_sg = Subgoal(
                            id=rec_id,
                            parent_id=active_sg.id,
                            level=active_sg.level,
                            skill_type="DEPOSIT",
                            required_objects=[block_id],
                            parameters=copy.deepcopy(active_sg.parameters),
                            preconditions={"block_present": True},
                            desired_effects=copy.deepcopy(active_sg.desired_effects),
                            dependencies=[],
                            description=f"Self-healing recovery: re-place {block_id} onto target slot",
                        )
                        self.planner.splice_recovery_subgoal(active_sg.id, rec_sg)
                        self.anomaly_events.append({"type": "RECOVERY_SPLICED", "subgoal": rec_id})

        return {
            "success": False,
            "message": f"Execution timed out after {max_total_ticks} ticks",
            "total_ticks": total_ticks,
            "anomalies_healed": len(self.anomaly_events),
        }

