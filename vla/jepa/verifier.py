"""
vla/jepa/verifier.py

Outcome Verifier and Surprise Detector for Closed-Loop Hierarchical VL-JEPA.
Implements the dual-gated verification mechanism:
1. Physical Resting Stability & Velocity Hold Verification (PhysicsEngine / StabilityValidator)
2. Latent Prediction Error (Surprise Detector: e_pred = || \hat{z}^(1) - z^(1) ||_2)
3. Latent Subgoal Energy Convergence (e_goal = || P_{1->2}(z^(1)) - z^(2)_goal ||_2)
4. Structural Integrity Guard (detects tower collapse or base displacement to trigger L2 replan)
"""

import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from simulation.constants import (
    STACK_ORIGIN,
    TABLE_TOP_Y,
    CUBE_SIZE,
    STACK_TOLERANCE,
)
from simulation.scoring import StabilityValidator, vec_dist
from .schemas import SubgoalSpecification, VerificationReport


class OutcomeVerifier:
    """
    Closed-loop outcome verifier combining physical stability gating
    with latent predictive surprise and goal convergence.
    """
    def __init__(
        self,
        tau_pred: float = 0.45,
        tau_goal: float = 0.35,
        required_hold_ticks: int = 18,
        max_l1_retries: int = 2,
        topple_displacement_threshold: float = 0.055,
    ):
        self.tau_pred = tau_pred
        self.tau_goal = tau_goal
        self.max_l1_retries = max_l1_retries
        self.topple_displacement_threshold = topple_displacement_threshold

        self.stability_validator = StabilityValidator(required_ticks=required_hold_ticks)
        self.l1_retry_count: int = 0
        self.completed_subgoal_targets: Dict[str, Tuple[float, float, float]] = {}

    def reset(self):
        """Resets verifier internal tracking state for a new task."""
        self.stability_validator.reset()
        self.l1_retry_count = 0
        self.completed_subgoal_targets.clear()

    def record_subgoal_completion(self, block_id: str, target_xyz: Tuple[float, float, float]):
        """Records a verified block position to guard tower structural integrity."""
        self.completed_subgoal_targets[block_id] = target_xyz
        self.l1_retry_count = 0

    def check_tower_intact(self, blocks: List[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
        """
        Verifies whether all previously stacked blocks remain in their designated slots.
        If any prior block fell to the table or moved > threshold, the tower has toppled.
        """
        block_map = {b.get("id"): b for b in blocks}
        for b_id, expected_xyz in self.completed_subgoal_targets.items():
            b = block_map.get(b_id)
            if not b:
                return False, f"Previously placed block '{b_id}' missing from workspace."
            pos = b.get("position", [0.0, 0.0, 0.0])
            dist = math.hypot(pos[0] - expected_xyz[0], pos[1] - expected_xyz[1], pos[2] - expected_xyz[2])
            if dist > self.topple_displacement_threshold:
                return False, (
                    f"Structural collapse: block '{b_id}' displaced by {dist:.3f}m "
                    f"from target slot [{expected_xyz[0]:.3f}, {expected_xyz[1]:.3f}, {expected_xyz[2]:.3f}]."
                )
        return True, None

    def verify_step(
        self,
        subgoal: SubgoalSpecification,
        phase: str,
        current_tcp: Tuple[float, float, float],
        blocks: List[Dict[str, Any]],
        grasped_block_id: Optional[str],
        pred_z1: Optional[np.ndarray] = None,
        actual_z1: Optional[np.ndarray] = None,
        proj_z2: Optional[np.ndarray] = None,
    ) -> VerificationReport:
        """
        Evaluates post-action outcome for the active micro/macro phase.
        Emits VerificationReport directing the orchestrator to advance, retry, or replan.
        """
        # 1. Check structural integrity of earlier blocks
        tower_intact, collapse_reason = self.check_tower_intact(blocks)
        if not tower_intact:
            return VerificationReport(
                subgoal_id=subgoal.subgoal_id,
                phase_index=0,
                latent_prediction_error=1.0,
                subgoal_energy_distance=1.0,
                physical_contact_verified=False,
                tower_intact=False,
                status="ESCALATE_L2_REPLAN",
                failure_reason=collapse_reason,
            )

        # 2. Latent surprise & goal energy metrics
        e_pred = 0.0
        if pred_z1 is not None and actual_z1 is not None:
            e_pred = float(np.linalg.norm(pred_z1 - actual_z1))

        e_goal = 0.0
        if proj_z2 is not None and subgoal.target_latent_l2 is not None:
            e_goal = float(np.linalg.norm(proj_z2 - subgoal.target_latent_l2))

        # Find target block
        target_b = None
        for b in blocks:
            if b.get("id") == subgoal.source_label:
                target_b = b
                break

        # 3. Micro-Phase Verification Logic
        if phase in ("HOVER_PICK", "DESCEND_PICK"):
            return VerificationReport(
                subgoal_id=subgoal.subgoal_id,
                phase_index=1,
                latent_prediction_error=e_pred,
                subgoal_energy_distance=e_goal,
                physical_contact_verified=False,
                tower_intact=True,
                status="IN_PROGRESS",
            )

        elif phase == "GRASP":
            # Grasp verification: did gripper successfully capture the intended cube?
            is_grasped = (grasped_block_id == subgoal.source_label) or (
                target_b is not None and target_b.get("is_grasped", False)
            )
            if not is_grasped and target_b is not None:
                # Check proximity
                pos = target_b.get("position", [0.0, 0.0, 0.0])
                dist = math.hypot(current_tcp[0] - pos[0], current_tcp[1] - pos[1], current_tcp[2] - pos[2])
                if dist > 0.045:
                    # Missed grasp or slip
                    self.l1_retry_count += 1
                    if self.l1_retry_count <= self.max_l1_retries:
                        return VerificationReport(
                            subgoal_id=subgoal.subgoal_id,
                            phase_index=2,
                            latent_prediction_error=max(e_pred, 0.5),
                            subgoal_energy_distance=e_goal,
                            physical_contact_verified=False,
                            tower_intact=True,
                            status="RETRY_L1",
                            failure_reason=f"Grasp missed for '{subgoal.source_label}' (dist={dist:.3f}m). Retrying L1.",
                        )
                    else:
                        return VerificationReport(
                            subgoal_id=subgoal.subgoal_id,
                            phase_index=2,
                            latent_prediction_error=max(e_pred, 0.8),
                            subgoal_energy_distance=e_goal,
                            physical_contact_verified=False,
                            tower_intact=True,
                            status="ESCALATE_L2_REPLAN",
                            failure_reason=f"Grasp failed after {self.max_l1_retries} retries for '{subgoal.source_label}'.",
                        )

            return VerificationReport(
                subgoal_id=subgoal.subgoal_id,
                phase_index=2,
                latent_prediction_error=e_pred,
                subgoal_energy_distance=e_goal,
                physical_contact_verified=is_grasped,
                tower_intact=True,
                status="IN_PROGRESS",
            )

        elif phase in ("LIFT", "HOVER_PLACE", "DESCEND_PLACE", "RELEASE"):
            return VerificationReport(
                subgoal_id=subgoal.subgoal_id,
                phase_index=3,
                latent_prediction_error=e_pred,
                subgoal_energy_distance=e_goal,
                physical_contact_verified=(grasped_block_id == subgoal.source_label),
                tower_intact=True,
                status="IN_PROGRESS",
            )

        elif phase == "VERIFY_HOLD":
            # Physical stability hold validation
            is_verified, hold_ticks, stability_status = self.stability_validator.step(
                blocks=blocks,
                grasped_block_id=grasped_block_id
            )

            # Spatial accuracy check for the newly placed block
            target_xyz = subgoal.target_xyz
            b_pos = target_b.get("position", [0.0, 0.0, 0.0]) if target_b else [0.0, 0.0, 0.0]
            dist_to_slot = math.hypot(
                b_pos[0] - target_xyz[0],
                b_pos[1] - target_xyz[1],
                b_pos[2] - target_xyz[2]
            )

            tol = subgoal.position_tolerance_m or STACK_TOLERANCE
            placed_accurately = dist_to_slot <= (tol + 0.025)

            # Dual-gate check: Both physical resting stability verified AND accurate slot landing
            if is_verified and placed_accurately:
                self.record_subgoal_completion(subgoal.source_label, subgoal.target_xyz)
                return VerificationReport(
                    subgoal_id=subgoal.subgoal_id,
                    phase_index=4,
                    latent_prediction_error=e_pred,
                    subgoal_energy_distance=e_goal,
                    physical_contact_verified=True,
                    tower_intact=True,
                    status="SUBGOAL_COMPLETED",
                    hold_ticks=hold_ticks,
                    stability_score=1.0,
                )

            # Check if block rolled or fell away
            if not placed_accurately and hold_ticks > 5:
                # Block slipped off target platform
                self.l1_retry_count += 1
                if self.l1_retry_count <= self.max_l1_retries:
                    return VerificationReport(
                        subgoal_id=subgoal.subgoal_id,
                        phase_index=4,
                        latent_prediction_error=max(e_pred, 0.6),
                        subgoal_energy_distance=max(e_goal, 0.5),
                        physical_contact_verified=False,
                        tower_intact=True,
                        status="RETRY_L1",
                        failure_reason=f"Block '{subgoal.source_label}' settled off-target (dist={dist_to_slot:.3f}m). Retrying L1.",
                        hold_ticks=hold_ticks,
                    )
                else:
                    return VerificationReport(
                        subgoal_id=subgoal.subgoal_id,
                        phase_index=4,
                        latent_prediction_error=max(e_pred, 0.8),
                        subgoal_energy_distance=max(e_goal, 0.8),
                        physical_contact_verified=False,
                        tower_intact=True,
                        status="ESCALATE_L2_REPLAN",
                        failure_reason=f"Block placement repeatedly off-target after {self.max_l1_retries} attempts.",
                        hold_ticks=hold_ticks,
                    )

            return VerificationReport(
                subgoal_id=subgoal.subgoal_id,
                phase_index=4,
                latent_prediction_error=e_pred,
                subgoal_energy_distance=e_goal,
                physical_contact_verified=False,
                tower_intact=True,
                status="IN_PROGRESS",
                hold_ticks=hold_ticks,
            )

        return VerificationReport(
            subgoal_id=subgoal.subgoal_id,
            phase_index=0,
            latent_prediction_error=e_pred,
            subgoal_energy_distance=e_goal,
            physical_contact_verified=True,
            tower_intact=True,
            status="IN_PROGRESS",
        )
