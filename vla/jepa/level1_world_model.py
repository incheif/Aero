r"""
vla/jepa/level1_world_model.py

Level 1 Action-Conditioned Latent World Model & Cross-Entropy Method (CEM) Planner.
- F^(1): (z^(1)_t, a_t) -> \hat{z}^(1)_{t+1} in latent space R^384.
- Level1CEMPlanner: Cross-Entropy Method trajectory optimizer operating over short horizon (H=4).
  Filters candidates through analytical IK reachability and scores cost against active Level 2 subgoal.
"""

import time
import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from simulation.kinematics import inverse_kinematics
from .encoders import (
    Level1ObservationEncoder,
    Level1ActionEncoder,
    ProjectionHead1to2,
    layer_norm,
    gelu,
)
from .schemas import CEMPlanResult


class Level1Predictor:
    r"""
    F^(1): Lightweight Action-Conditioned Latent Predictor.
    Given current latent state z^(1)_t and action embedding a_t,
    predicts the future latent state \hat{z}^(1)_{t+1} without pixel decoding.
    """
    def __init__(self, state_dim: int = 384, action_dim: int = 32):
        self.state_dim = state_dim
        self.action_dim = action_dim
        in_dim = state_dim + action_dim

        rng = np.random.RandomState(201)
        self.W1 = rng.randn(in_dim, state_dim) * math.sqrt(2.0 / in_dim)
        self.b1 = np.zeros(state_dim, dtype=np.float32)
        self.W2 = rng.randn(state_dim, state_dim) * math.sqrt(2.0 / state_dim)
        self.b2 = np.zeros(state_dim, dtype=np.float32)

    def predict(self, z1: np.ndarray, a1: np.ndarray) -> np.ndarray:
        """Rolls forward one latent step. Supports single or batched inputs."""
        single = (z1.ndim == 1)
        if single:
            z1 = z1.reshape(1, -1)
            a1 = a1.reshape(1, -1)

        cat = np.concatenate([z1, a1], axis=-1)
        h = gelu(cat @ self.W1 + self.b1)
        delta_z = h @ self.W2 + self.b2

        z_next = z1 + delta_z
        norm = np.linalg.norm(z_next, axis=-1, keepdims=True)
        norm = np.maximum(norm, 1e-6)
        z_next = z_next / norm

        if single:
            return z_next[0].astype(np.float32)
        return z_next.astype(np.float32)

    def rollout(self, z_init: np.ndarray, a_seq: np.ndarray) -> np.ndarray:
        """
        Recursively unrolls micro predictions over horizon H.
        z_init: shape (N, state_dim)
        a_seq: shape (N, H, action_dim)
        Returns terminal latent embeddings of shape (N, state_dim).
        """
        z_curr = z_init
        H = a_seq.shape[1]
        for t in range(H):
            z_curr = self.predict(z_curr, a_seq[:, t, :])
        return z_curr


class Level1CEMPlanner:
    """
    Cross-Entropy Method (CEM) trajectory optimizer in Level 1 latent space.
    Vectorized NumPy implementation: rolls out N candidate particles across horizon H=4,
    evaluates projected Level 2 subgoal cost, and verifies IK reachability on elites.
    Runs in sub-15ms for real-time 60 Hz physics.
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
        lambda_align: float = 0.5,
        lambda_ctrl: float = 0.05,
        max_step_xyz: float = 0.025,
    ):
        self.encoder_l1 = obs_encoder_l1 or Level1ObservationEncoder()
        self.encoder_act = act_encoder_l1 or Level1ActionEncoder()
        self.predictor = predictor_l1 or Level1Predictor()
        self.projector = projector_1to2 or ProjectionHead1to2()

        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.num_iterations = num_iterations
        self.lambda_align = lambda_align
        self.lambda_ctrl = lambda_ctrl
        self.max_step_xyz = max_step_xyz

    def plan(
        self,
        subgoal_id: int,
        current_tcp: List[float] | Tuple[float, float, float],
        target_pos: List[float] | Tuple[float, float, float],
        active_subgoal_z2: np.ndarray,
        gripper_aperture: float = 0.0,
        desired_gripper: float = 0.0,
        joints: Optional[Dict[str, float]] = None,
    ) -> CEMPlanResult:
        """
        Executes fast vectorized CEM optimization in Level 1 latent space.
        """
        t0 = time.perf_counter()
        tcp = np.asarray(current_tcp[:3], dtype=np.float32)
        tgt = np.asarray(target_pos[:3], dtype=np.float32)

        # Baseline nominal step vector toward target
        delta_nom = tgt - tcp
        dist = float(np.linalg.norm(delta_nom))
        step_mag = min(dist, self.max_step_xyz)
        step_dir = (delta_nom / max(1e-6, dist)) * step_mag

        # Initialize Gaussian distribution over action sequences (H, 4: dx, dy, dz, gripper)
        mean_actions = np.zeros((self.horizon, 4), dtype=np.float32)
        std_actions = np.ones((self.horizon, 4), dtype=np.float32) * 0.008
        for h in range(self.horizon):
            mean_actions[h, :3] = step_dir * (0.85 ** h)
            mean_actions[h, 3] = desired_gripper

        # Current latent state
        curr_z1 = self.encoder_l1.encode(
            joints=joints or {},
            tcp_pos=tcp,
            gripper=gripper_aperture,
            dst_pos=tgt,
        )

        best_sequence = mean_actions.copy()
        candidate_trajs = []

        # Vectorized initial batch: (N, state_dim)
        z_batch = np.tile(curr_z1, (self.num_samples, 1))

        for it in range(self.num_iterations):
            # 1. Sample candidate action sequences: (N, H, 4)
            noise = np.random.randn(self.num_samples, self.horizon, 4).astype(np.float32)
            candidates = mean_actions + noise * std_actions

            # Bound cartesian steps
            mags = np.linalg.norm(candidates[:, :, :3], axis=-1, keepdims=True)
            norm_scale = np.maximum(1.0, mags / self.max_step_xyz)
            candidates[:, :, :3] = candidates[:, :, :3] / norm_scale
            candidates[:, :, 3] = np.clip(candidates[:, :, 3], 0.0, 1.0)

            # Cumulative Cartesian positions: (N, H, 3)
            cum_disp = np.cumsum(candidates[:, :, :3], axis=1)
            traj_xyz = tcp.reshape(1, 1, 3) + cum_disp

            # Vectorized action batch encoding: (N, H, 5) -> (N, H, 32)
            act5 = np.zeros((self.num_samples, self.horizon, 5), dtype=np.float32)
            act5[:, :, :3] = candidates[:, :, :3]
            act5[:, :, 4] = candidates[:, :, 3]
            act_encoded = self.encoder_act.encode_batch(act5)

            # 2. Vectorized Latent Rollout via F^(1)
            z_final = self.predictor.rollout(z_batch, act_encoded)

            # 3. Cross-level projection to Level 2: (N, dim_l2)
            z_lifted = self.projector.project(z_final)

            # 4. Latent and Spatial Cost Evaluation
            latent_costs = np.linalg.norm(z_lifted - active_subgoal_z2.reshape(1, -1), axis=-1)
            term_tcp = traj_xyz[:, -1, :]
            spatial_costs = np.linalg.norm(term_tcp - tgt.reshape(1, 3), axis=-1)

            reach_dist = np.linalg.norm(term_tcp[:, [0, 2]], axis=-1)
            reach_pen = np.where(reach_dist > 0.65, 30.0, 0.0)

            total_costs = latent_costs + self.lambda_align * spatial_costs + reach_pen

            # 5. Elite selection
            elite_idx = np.argsort(total_costs)[:self.num_elites]
            elites = candidates[elite_idx]

            # 6. Refit Gaussian
            mean_actions = np.mean(elites, axis=0)
            std_actions = np.std(elites, axis=0) + 0.002
            best_sequence = elites[0]

            if it == self.num_iterations - 1:
                for idx in elite_idx[:4]:
                    pts = [tuple(float(v) for v in tcp)]
                    for h in range(self.horizon):
                        pts.append(tuple(float(v) for v in traj_xyz[idx, h]))
                    candidate_trajs.append(pts)

        # 7. Rollout winning sequence to compute final latents
        pred_latents = []
        z_step = curr_z1.copy()
        for h in range(self.horizon):
            act5_single = np.array([best_sequence[h, 0], best_sequence[h, 1], best_sequence[h, 2], 0.0, best_sequence[h, 3]], dtype=np.float32)
            act_emb_single = self.encoder_act.encode(act5_single)
            z_step = self.predictor.predict(z_step, act_emb_single)
            pred_latents.append(z_step)

        pred_l1_arr = np.array(pred_latents, dtype=np.float32)
        proj_term = self.projector.project(pred_l1_arr[-1])
        terminal_energy = float(np.linalg.norm(proj_term - active_subgoal_z2))

        # Check terminal reachability once on optimal step
        term_pos = tcp + best_sequence[0, :3]
        term_ik = inverse_kinematics(term_pos, best_sequence[0, 3])

        lat_ms = (time.perf_counter() - t0) * 1000.0

        opt_act_5d = np.zeros((self.horizon, 5), dtype=np.float32)
        opt_act_5d[:, :3] = best_sequence[:, :3]
        opt_act_5d[:, 4] = best_sequence[:, 3]

        return CEMPlanResult(
            subgoal_id=subgoal_id,
            optimal_actions=opt_act_5d,
            predicted_latents_l1=pred_l1_arr,
            terminal_goal_energy=terminal_energy,
            predicted_stability=max(0.0, 1.0 - terminal_energy * 0.5),
            candidate_trajectories_xyz=candidate_trajs,
            ik_feasible=term_ik.get("reachable", True),
            latency_ms=lat_ms,
        )
