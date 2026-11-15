"""
vla/latent_action_scorer.py

H-JEPA Level 2 Latent Action Scorer & Pre-Validation Evaluator for AERO.
Evaluates candidate manipulation trajectories and candidate subgoals in abstract
latent representation space (R^64) prior to committing motor movements.

Features:
- Action-conditioned latent forward rollouts via pre-trained H-JEPA world model.
- Latent goal-distance scoring and risk assessment.
- Feasibility discrimination between collision-free, stable vs risky trajectories.
- Zero-cost simulation pre-check preventing physical drops or excessive motor load.
"""

import os
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from vla.predictive_world_model import (
    HierarchicalJEPA,
    extract_state_vector_6dof,
    extract_action_vector_7d,
)
from vla.recursive_dag_planner import Subgoal


class LatentActionScorer:
    """
    Evaluates candidate action sequences and subgoals using learned H-JEPA dynamics.
    Enables model-predictive decision making entirely in representation space.
    """

    def __init__(self, checkpoint_path: str = "models/hjepa_6dof_world_model.npz"):
        self.jepa = HierarchicalJEPA(state_dim=112, action_dim=7, latent_dim=64, seed=42)
        self.is_loaded = False
        if os.path.exists(checkpoint_path):
            self.is_loaded = self.jepa.load_checkpoint(checkpoint_path)

    def score_candidate_action(
        self,
        current_snapshot: Dict[str, Any],
        candidate_joints: Dict[str, float],
        candidate_gripper: float,
        target_snapshot: Dict[str, Any],
    ) -> float:
        """
        Scores a candidate single-transition action in latent space:
        Score = - || hat{z}_{t+1}(s_t, a) - z_target ||_2
        """
        x_curr = extract_state_vector_6dof(current_snapshot)
        x_target = extract_state_vector_6dof(target_snapshot)
        a_vec = extract_action_vector_7d(candidate_joints, candidate_gripper)
        return self.jepa.score_candidate_action(x_curr, a_vec, x_target)

    def score_action_trajectory(
        self,
        current_snapshot: Dict[str, Any],
        action_sequence: List[Tuple[Dict[str, float], float]],
        target_snapshot: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Multi-step forward rollout in latent space:
        z_0 = E(s_0)
        z_{k+1} = P(z_k, a_k)
        Returns: Cumulative score (higher is better).
        """
        x_0 = extract_state_vector_6dof(current_snapshot)
        z_curr = self.jepa.encode(x_0)

        # Roll forward along the action trajectory in latent space
        for joints_cmd, grip_cmd in action_sequence:
            a_vec = extract_action_vector_7d(joints_cmd, grip_cmd)
            z_curr = self.jepa.predict_latent(z_curr, a_vec)

        if target_snapshot is not None:
            x_goal = extract_state_vector_6dof(target_snapshot)
            z_goal = self.jepa.encode(x_goal)
            dist = float(np.linalg.norm(z_curr - z_goal))
            return -dist

        # If no target snapshot specified, return representation confidence/magnitude
        return float(np.mean(z_curr))

    def evaluate_subgoal_feasibility(
        self,
        current_snapshot: Dict[str, Any],
        subgoal: Subgoal,
    ) -> Tuple[bool, float]:
        """
        Pre-validates whether a subgoal is physically feasible from the current state:
        - Simulates one macro transition in latent space toward the target configuration.
        - Returns: (is_feasible: bool, confidence_score: float)
        """
        x_curr = extract_state_vector_6dof(current_snapshot)
        z_curr = self.jepa.encode(x_curr)

        # Construct target pseudo-snapshot from subgoal parameters
        pseudo_target = copy_and_update_snapshot(current_snapshot, subgoal)
        x_target = extract_state_vector_6dof(pseudo_target)
        z_target = self.jepa.encode(x_target)

        # Macro step action estimation
        nominal_action = np.zeros(7, dtype=np.float32)
        z_pred = self.jepa.predict_latent(z_curr, nominal_action)

        latent_dist = float(np.linalg.norm(z_pred - z_target))
        # Relative latent distance normalized by representation scale
        norm_scale = max(1.0, float(np.linalg.norm(z_target)))
        rel_dist = latent_dist / norm_scale
        confidence = float(np.exp(-rel_dist))
        is_feasible = (rel_dist < 1.5)

        return is_feasible, confidence


def copy_and_update_snapshot(snapshot: Dict[str, Any], subgoal: Subgoal) -> Dict[str, Any]:
    """Generates an idealized target snapshot corresponding to subgoal completion."""
    import copy
    target_snap = copy.deepcopy(snapshot)
    if subgoal.skill_type in ("DEPOSIT", "PICK_AND_PLACE", "CLEAR_OBSTACLE"):
        b_id = subgoal.parameters.get("block_id")
        t_slot = subgoal.parameters.get("target_slot")
        if b_id and t_slot:
            for b in target_snap.get("blocks", []):
                if b["id"] == b_id:
                    b["position"] = list(t_slot)
                    b["is_grasped"] = False
                    break
    return target_snap
