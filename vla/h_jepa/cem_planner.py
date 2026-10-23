"""
vla/h_jepa/cem_planner.py

Decoupled Cross-Entropy Method (CEM) Planners for 2-Level H-JEPA:
- Level 1 Micro-CEM Planner: High-frequency action optimization (H=4, N=64, K=8)
  evaluating candidate physical trajectories via Level 1 predictor F^(1)
  and lifting into Level 2 latent space via P^(1->2).
- Level 2 Macro-CEM Planner: Macro-step sub-goal scheduler against z_goal^(2).
Runs in sub-15ms using vectorized NumPy matrix computations.
"""

import time
import math
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from simulation.constants import (
    TABLE_HALF_EXTENTS,
    TABLE_TOP_Y,
    SAFE_HOVER_Y,
    PICK_Y,
)
from .encoders import (
    Level1ObservationEncoder,
    Level1ActionEncoder,
    ProjectionHead1to2,
)
from .predictors import Level1Predictor, Level2Predictor

class Level1MicroCEMPlanner:
    """
    Level 1 Micro-CEM Trajectory Optimizer.
    Finds the optimal short-horizon action trajectory (H=4 steps) that minimizes
    the distance in Level 2 latent space to the active sub-goal embedding z_sub^(2).
    """
    def __init__(
        self,
        obs_encoder_l1: Optional[Level1ObservationEncoder] = None,
        act_encoder_l1: Optional[Level1ActionEncoder] = None,
        predictor_l1: Optional[Level1Predictor] = None,
        projector_1to2: Optional[ProjectionHead1to2] = None,
        horizon: int = 4,
        num_samples: int = 48,
        num_elites: int = 6,
        num_iterations: int = 2,
    ):
        self.encoder_obs = obs_encoder_l1 or Level1ObservationEncoder()
        self.encoder_act = act_encoder_l1 or Level1ActionEncoder()
        self.predictor = predictor_l1 or Level1Predictor()
        self.projector = projector_1to2 or ProjectionHead1to2()
        
        self.H = horizon
        self.N = num_samples
        self.K = num_elites
        self.n_iters = num_iterations
        
        # Action space: [dx, dy, dz, gripper_cmd]
        self.action_dim = 4
        # Cartesian step limits (max 18mm per micro-step)
        self.max_step_xyz = 0.018

    def plan(
        self,
        current_tcp: List[float],
        target_pos: List[float],
        active_subgoal_z2: np.ndarray,
        gripper_aperture: float = 0.0,
        current_vel: Optional[List[float]] = None,
        desired_gripper: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Executes CEM optimization loop in latent space.
        Returns:
          - 'optimal_action': first action to execute [dx, dy, dz, gripper]
          - 'planned_trajectory': list of H Cartesian waypoints
          - 'latent_distance': best evaluated Euclidean distance to z_sub^(2)
          - 'latency_ms': computation time in milliseconds
        """
        t_start = time.perf_counter()

        # 1. Encode current physical state into Level 1 latent space z_init^(1)
        z_init = self.encoder_obs.encode(
            tcp_pos=current_tcp,
            target_pos=target_pos,
            gripper_aperture=gripper_aperture,
            tcp_vel=current_vel or [0.0, 0.0, 0.0],
            micro_step=0
        )
        # Broadcast initial state across N candidate particles: shape (N, 64)
        z_init_batch = np.tile(z_init, (self.N, 1))

        # 2. Prior mean towards target vector
        dx_ideal = (target_pos[0] - current_tcp[0]) / max(1, self.H)
        dy_ideal = (target_pos[1] - current_tcp[1]) / max(1, self.H)
        dz_ideal = (target_pos[2] - current_tcp[2]) / max(1, self.H)
        
        # Clamp prior mean within physical velocity envelope
        scale = max(1.0, math.hypot(dx_ideal, dy_ideal, dz_ideal) / self.max_step_xyz)
        dx_ideal /= scale
        dy_ideal /= scale
        dz_ideal /= scale

        # Action distribution mean: shape (H, 4)
        mu = np.zeros((self.H, self.action_dim), dtype=np.float32)
        mu[:, 0] = dx_ideal
        mu[:, 1] = dy_ideal
        mu[:, 2] = dz_ideal
        mu[:, 3] = desired_gripper

        # Action distribution std: shape (H, 4)
        sigma = np.zeros((self.H, self.action_dim), dtype=np.float32)
        sigma[:, 0:3] = 0.008  # 8mm exploratory spread
        sigma[:, 3] = 0.05    # tight gripper spread

        best_cost = float("inf")
        best_trajectory = None
        best_action = None

        # 3. Iterative Cross-Entropy Refinement
        for iteration in range(self.n_iters):
            # Sample candidate action sequences: shape (N, H, 4)
            samples = np.random.normal(mu, sigma, size=(self.N, self.H, self.action_dim)).astype(np.float32)
            
            # Clamp actions to kinematic safety bounds (Euclidean norm <= max_step_xyz)
            mags = np.linalg.norm(samples[:, :, 0:3], axis=-1, keepdims=True)
            norm_scale = np.maximum(1.0, mags / self.max_step_xyz)
            samples[:, :, 0:3] = samples[:, :, 0:3] / norm_scale
            samples[:, :, 3] = np.clip(samples[:, :, 3], 0.0, 1.0)

            # Fast vectorized encoding of actions into Level 1 latent action vectors: shape (N, H, 16)
            act_encoded = self.encoder_act.encode_batch(samples)

            # 4. Latent World Model Rollout via F^(1)
            z_final_batch = self.predictor.rollout(z_init_batch, act_encoded)  # shape (N, 64)

            # 5. Cross-Level Projection via P^(1->2) into Level 2 Latent Space: shape (N, 128)
            z_lifted = self.projector.project(z_final_batch)

            # 6. Evaluate Latent Costs against active sub-goal z_sub^(2)
            # Euclidean distance in R^128
            latent_diffs = z_lifted - active_subgoal_z2.reshape(1, -1)
            latent_costs = np.linalg.norm(latent_diffs, axis=-1)  # shape (N,)

            # Smoothness penalty (acceleration / jerk regularization)
            accel = np.diff(samples[:, :, 0:3], axis=1)
            smoothness_penalty = np.sum(accel ** 2, axis=(1, 2)) * 50.0

            # Table collision avoidance penalty (altitude clamp)
            # Integrate TCP positions
            cum_disp = np.cumsum(samples[:, :, 1], axis=1)
            predicted_y = current_tcp[1] + cum_disp
            collision_penalty = np.sum(np.maximum(0.0, TABLE_TOP_Y - predicted_y) * 500.0, axis=1)

            total_costs = latent_costs + smoothness_penalty + collision_penalty

            # 7. Select Top-K Elites
            elite_indices = np.argsort(total_costs)[:self.K]
            elites = samples[elite_indices]

            if total_costs[elite_indices[0]] < best_cost:
                best_cost = float(total_costs[elite_indices[0]])
                best_action = elites[0, 0, :].tolist()
                # Compute integrated trajectory waypoints
                waypoints = []
                c_pos = list(current_tcp)
                for step_t in range(self.H):
                    c_pos[0] += float(elites[0, step_t, 0])
                    c_pos[1] += float(elites[0, step_t, 1])
                    c_pos[2] += float(elites[0, step_t, 2])
                    waypoints.append(list(c_pos))
                best_trajectory = waypoints

            # Refit Gaussian distribution to elites
            mu = np.mean(elites, axis=0)
            sigma = np.std(elites, axis=0) + 1e-4

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0

        return {
            "optimal_action": best_action or [dx_ideal, dy_ideal, dz_ideal, desired_gripper],
            "planned_trajectory": best_trajectory or [],
            "latent_distance": round(best_cost, 4),
            "latency_ms": round(latency_ms, 2),
            "num_samples_evaluated": self.N * self.n_iters,
        }


class Level2MacroCEMPlanner:
    """
    Level 2 Macro-CEM Task & Sub-Goal Planner.
    Schedules macro-action sequences and monitors phase convergence towards z_goal^(2).
    """
    def __init__(self, predictor_l2: Optional[Level2Predictor] = None):
        self.predictor = predictor_l2 or Level2Predictor()

    def evaluate_subgoal_progress(
        self,
        current_z2: np.ndarray,
        subgoal_z2: np.ndarray,
        goal_z2: np.ndarray
    ) -> Dict[str, float]:
        """
        Measures progress towards active sub-goal and final task goal in Level 2 latent space.
        """
        sub_dist = float(np.linalg.norm(current_z2 - subgoal_z2))
        goal_dist = float(np.linalg.norm(current_z2 - goal_z2))
        return {
            "subgoal_latent_distance": round(sub_dist, 4),
            "final_goal_latent_distance": round(goal_dist, 4),
        }
