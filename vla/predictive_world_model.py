"""
Action-Conditioned Predictive World Model (H-JEPA Level 1 & 2) for AERO.

Implements Yann LeCun's Joint-Embedding Predictive Architecture (JEPA):
- 112-dimensional continuous physical state vector extracted from the 6-DOF SO-ARM100
  multibody twin and Feetech STS3215 actuator physics in MuJoCo.
- 7-dimensional continuous action space (6 joint increments + gripper command).
- 64-dimensional latent embedding space regularized with VICReg (Variance-Invariance-Covariance).
- Context Encoder E_theta(x_t) -> z_t in R^64.
- Target EMA Encoder E_ema(x_{t+1}) -> bar{z}_{t+1} (stop-gradient with EMA updates).
- Residual Predictor P_phi(z_t, a_t) -> hat{z}_{t+1} in R^64.
- Zero-collapse guarantee via analytical VICReg gradient backpropagation and Adam optimizer.
- Latent rollouts, energy-based candidate action scoring, and real-time residual tracking.
"""

import os
import math
import copy
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from simulation.constants import (
    TABLE_TOP_Y,
    CUBE_SIZE,
    CUBE_HALF,
    DEFAULT_JOINTS,
    JOINT_LIMITS,
)
from simulation.actuator_model import STS3215_SPECS


def gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit (GELU) activation function."""
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x ** 3))))


def gelu_backward(x: np.ndarray, grad_out: np.ndarray) -> np.ndarray:
    """Analytical gradient of GELU activation."""
    tanh_arg = math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x ** 3))
    t = np.tanh(tanh_arg)
    dt = (1.0 - t ** 2) * math.sqrt(2.0 / math.pi) * (1.0 + 3.0 * 0.044715 * (x ** 2))
    d_gelu = 0.5 * (1.0 + t) + 0.5 * x * dt
    return grad_out * d_gelu


class LinearLayer:
    """Fully-connected layer with analytical forward and backward passes and Adam optimizer."""

    def __init__(self, in_features: int, out_features: int, rng: Optional[np.random.RandomState] = None):
        self.in_features = in_features
        self.out_features = out_features
        if rng is None:
            rng = np.random.RandomState()
        limit = math.sqrt(6.0 / (in_features + out_features))
        self.W = rng.uniform(-limit, limit, (in_features, out_features)).astype(np.float32)
        self.b = np.zeros(out_features, dtype=np.float32)

        # Gradients
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

        # Adam moments
        self.m_W = np.zeros_like(self.W)
        self.v_W = np.zeros_like(self.W)
        self.m_b = np.zeros_like(self.b)
        self.v_b = np.zeros_like(self.b)

        self.last_x: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.last_x = x
        return x @ self.W + self.b

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self.last_x is None:
            raise RuntimeError("Backward called before forward.")
        if self.last_x.ndim == 1:
            x = self.last_x.reshape(1, -1)
            g = grad_out.reshape(1, -1)
        else:
            x = self.last_x
            g = grad_out

        self.grad_W += x.T @ g
        self.grad_b += np.sum(g, axis=0)
        grad_in = g @ self.W.T

        if self.last_x.ndim == 1:
            return grad_in.reshape(-1)
        return grad_in

    def step(self, lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8, t: int = 1):
        self.m_W = beta1 * self.m_W + (1.0 - beta1) * self.grad_W
        self.v_W = beta2 * self.v_W + (1.0 - beta2) * (self.grad_W ** 2)
        m_hat = self.m_W / (1.0 - beta1 ** t)
        v_hat = self.v_W / (1.0 - beta2 ** t)
        self.W -= lr * m_hat / (np.sqrt(v_hat) + eps)

        self.m_b = beta1 * self.m_b + (1.0 - beta1) * self.grad_b
        self.v_b = beta2 * self.v_b + (1.0 - beta2) * (self.grad_b ** 2)
        m_hat_b = self.m_b / (1.0 - beta1 ** t)
        v_hat_b = self.v_b / (1.0 - beta2 ** t)
        self.b -= lr * m_hat_b / (np.sqrt(v_hat_b) + eps)

        self.grad_W.fill(0.0)
        self.grad_b.fill(0.0)


class MLP:
    """Multi-Layer Perceptron with GELU non-linearities and Adam optimization."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, rng: Optional[np.random.RandomState] = None):
        self.fc1 = LinearLayer(in_dim, hidden_dim, rng=rng)
        self.fc2 = LinearLayer(hidden_dim, out_dim, rng=rng)
        self.last_h1: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        h1 = self.fc1.forward(x)
        self.last_h1 = h1
        h1_act = gelu(h1)
        out = self.fc2.forward(h1_act)
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        grad_h1_act = self.fc2.backward(grad_out)
        if self.last_h1 is None:
            raise RuntimeError("Backward called before forward.")
        grad_h1 = gelu_backward(self.last_h1, grad_h1_act)
        grad_in = self.fc1.backward(grad_h1)
        return grad_in

    def step(self, lr: float, t: int):
        self.fc1.step(lr, t=t)
        self.fc2.step(lr, t=t)


def extract_state_vector_6dof(snapshot: Dict[str, Any]) -> np.ndarray:
    """
    Extracts a structured 112-dimensional physical state vector from MuJoCo simulation telemetry.
    Dimensions:
      1. SO-ARM100 6-DOF Joint Angles (6):
         [baseYaw, shoulderPitch, elbowPitch, wristPitch, wristRoll, gripper]
      2. Joint Velocities (6):
         [d_baseYaw, d_shoulderPitch, d_elbowPitch, d_wristPitch, d_wristRoll, d_gripper]
      3. STS3215 Actuator Currents / Torques (6):
         Normalized load / torque for each bus servo
      4. TCP Metric Pose & Velocity (10):
         pos (3), quat (4), vel (3)
      5. Gripper Contact State (4):
         jaw_sep, contact_force, grasped_flag, target_verified
      6. Tabletop Rigid Bodies (up to 7 blocks * 11 = 77):
         pos (3), quat (4), lin_vel (3), is_grasped (1)
      7. Global Task & Workcell Metrics (3):
         table_clearance, active_progress, stability_index
    Total = 6 + 6 + 6 + 10 + 4 + 77 + 3 = 112
    """
    vec = []
    j_names = ["baseYaw", "shoulderPitch", "elbowPitch", "wristPitch", "wristRoll", "gripper"]

    # 1. Joint angles
    j = snapshot.get("joints", DEFAULT_JOINTS)
    for name in j_names:
        vec.append(float(j.get(name, 0.0)))

    # 2. Joint velocities
    jv = snapshot.get("joint_velocities", {})
    for name in j_names:
        vec.append(float(jv.get(name, 0.0)))

    # 3. STS3215 Torques / Currents
    actuators = snapshot.get("actuators", {})
    torques = actuators.get("torques", {})
    for name in j_names:
        t_val = float(torques.get(name, 0.0))
        vec.append(t_val)

    # 4. TCP Pose & Velocity
    tcp = snapshot.get("tcp", {})
    tcp_pos = tcp.get("position", [0.0, 0.82, 0.0])
    tcp_rot = tcp.get("rotation", [0.0, 0.0, 0.0, 1.0])
    tcp_vel = tcp.get("velocity", [0.0, 0.0, 0.0])
    vec.extend([float(tcp_pos[0]), float(tcp_pos[1]), float(tcp_pos[2])])
    vec.extend([float(tcp_rot[0]), float(tcp_rot[1]), float(tcp_rot[2]), float(tcp_rot[3])])
    vec.extend([float(tcp_vel[0]), float(tcp_vel[1]), float(tcp_vel[2])])

    # 5. Gripper Contact State
    jaw_sep = float(j.get("gripper", 0.035))
    contact_force = float(snapshot.get("contact_force", 0.0))
    grasped_flag = 1.0 if snapshot.get("grasped_block_id") else 0.0
    verified_flag = 1.0 if snapshot.get("scores", {}).get("stability", {}).get("is_verified") else 0.0
    vec.extend([jaw_sep, contact_force, grasped_flag, verified_flag])

    # 6. Tabletop Blocks (7 blocks * 11 dimensions)
    blocks = sorted(snapshot.get("blocks", []), key=lambda b: b.get("id", ""))
    for b in blocks[:7]:
        p = b.get("position", [0.0, TABLE_TOP_Y + CUBE_HALF, 0.0])
        r = b.get("rotation", [0.0, 0.0, 0.0, 1.0])
        v = b.get("velocity", [0.0, 0.0, 0.0])
        g = 1.0 if b.get("is_grasped") else 0.0
        vec.extend([float(p[0]), float(p[1]), float(p[2])])
        vec.extend([float(r[0]), float(r[1]), float(r[2]), float(r[3])])
        vec.extend([float(v[0]), float(v[1]), float(v[2])])
        vec.append(g)

    # Zero-padding for missing blocks
    missing_blocks = 7 - len(blocks[:7])
    for _ in range(missing_blocks):
        vec.extend([0.0] * 11)

    # 7. Global Workcell Metrics (3)
    vec.append(float(snapshot.get("table_clearance", 1.0)))
    vec.append(float(snapshot.get("step_progress", 0.0)))
    stability_idx = float(snapshot.get("scores", {}).get("stability", {}).get("stability_score", 1.0))
    vec.append(stability_idx)

    arr = np.array(vec, dtype=np.float32)
    if len(arr) != 112:
        # Fallback pad or trim to ensure strict dimension invariance
        if len(arr) < 112:
            arr = np.pad(arr, (0, 112 - len(arr)))
        else:
            arr = arr[:112]
    return arr


def extract_action_vector_7d(joints_cmd: Dict[str, float], gripper_cmd: float) -> np.ndarray:
    """
    Extracts 7-dimensional continuous action vector:
    [baseYaw, shoulderPitch, elbowPitch, wristPitch, wristRoll, gripper, gripper_vel]
    """
    j_names = ["baseYaw", "shoulderPitch", "elbowPitch", "wristPitch", "wristRoll"]
    act = [float(joints_cmd.get(k, 0.0)) for k in j_names]
    act.append(float(gripper_cmd))
    act.append(float(joints_cmd.get("gripper_velocity", 0.0)))
    return np.array(act, dtype=np.float32)


class HierarchicalJEPA:
    """
    Hierarchical Joint-Embedding Predictive Architecture (H-JEPA).
    Operates across two temporal / semantic abstractions:
      - Level 1: Micro action-conditioned transition predictor (60 Hz control verification)
      - Level 2: Macro latent state rollout & action scoring (subgoal feasibility)
    """

    def __init__(
        self,
        state_dim: int = 112,
        action_dim: int = 7,
        latent_dim: int = 64,
        ema_tau: float = 0.996,
        seed: int = 42,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.ema_tau = ema_tau
        self.rng = np.random.RandomState(seed)

        # Context Encoder E_theta: R^112 -> R^64
        self.context_encoder = MLP(state_dim, 128, latent_dim, rng=self.rng)

        # Target Encoder E_ema (stop-gradient EMA copy)
        self.target_encoder = MLP(state_dim, 128, latent_dim, rng=self.rng)
        self._sync_target_encoder()

        # Action-Conditioned Predictor P_phi: R^(64 + 7) -> R^64
        self.predictor = MLP(latent_dim + action_dim, 128, latent_dim, rng=self.rng)

        self.step_count: int = 0
        self.loss_history: List[Dict[str, float]] = []

    def _sync_target_encoder(self):
        self.target_encoder.fc1.W = self.context_encoder.fc1.W.copy()
        self.target_encoder.fc1.b = self.context_encoder.fc1.b.copy()
        self.target_encoder.fc2.W = self.context_encoder.fc2.W.copy()
        self.target_encoder.fc2.b = self.context_encoder.fc2.b.copy()

    def update_target_encoder(self):
        """Exponential Moving Average (EMA) update for target encoder parameters."""
        tau = self.ema_tau
        self.target_encoder.fc1.W = tau * self.target_encoder.fc1.W + (1.0 - tau) * self.context_encoder.fc1.W
        self.target_encoder.fc1.b = tau * self.target_encoder.fc1.b + (1.0 - tau) * self.context_encoder.fc1.b
        self.target_encoder.fc2.W = tau * self.target_encoder.fc2.W + (1.0 - tau) * self.context_encoder.fc2.W
        self.target_encoder.fc2.b = tau * self.target_encoder.fc2.b + (1.0 - tau) * self.context_encoder.fc2.b

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encodes high-dimensional physical state vector into latent representation z in R^64."""
        single = (x.ndim == 1)
        if single:
            x_in = x.reshape(1, -1)
        else:
            x_in = x
        z = self.context_encoder.forward(x_in)
        return z[0] if single else z

    def predict_latent(self, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        """
        Rolls forward in latent space using action-conditioned residual predictor:
        hat{z}_{t+1} = z_t + P_phi(z_t, a_t)
        """
        single = (z.ndim == 1)
        if single:
            z_in = z.reshape(1, -1)
            a_in = a.reshape(1, -1)
        else:
            z_in = z
            a_in = a

        za = np.concatenate([z_in, a_in], axis=-1)
        delta_z = self.predictor.forward(za)
        z_next = z_in + delta_z

        return z_next[0] if single else z_next

    def compute_loss_and_grad(
        self,
        x_curr: np.ndarray,
        a_curr: np.ndarray,
        x_next: np.ndarray,
        gamma: float = 1.0,
        var_weight: float = 4.0,
        cov_weight: float = 1.0,
    ) -> Tuple[float, float, float, float]:
        """
        Computes VICReg multi-term objective:
        1. Invariance MSE: ||hat{z}_{t+1} - bar{z}_{t+1}||_2^2
        2. Variance Hinge: max(0, gamma - std(z_curr))
        3. Covariance Decorrelation: off-diagonal(Cov(z_curr))^2
        """
        B = x_curr.shape[0] if x_curr.ndim > 1 else 1
        if x_curr.ndim == 1:
            x_curr = x_curr.reshape(1, -1)
            a_curr = a_curr.reshape(1, -1)
            x_next = x_next.reshape(1, -1)

        # 1. Forward passes
        z_curr = self.context_encoder.forward(x_curr)
        za = np.concatenate([z_curr, a_curr], axis=-1)
        delta_z = self.predictor.forward(za)
        z_pred = z_curr + delta_z

        # Target encoder forward (stop-gradient target)
        z_target = self.target_encoder.forward(x_next)

        # Invariance Loss (MSE)
        diff = z_pred - z_target
        pred_loss = float(np.mean(diff ** 2))
        grad_z_pred = (2.0 / (B * self.latent_dim)) * diff

        # Variance & Covariance Regularization on z_curr
        var_loss = 0.0
        cov_loss = 0.0
        grad_reg = np.zeros_like(z_curr)

        if B > 1:
            # Variance hinge
            std = np.sqrt(np.var(z_curr, axis=0) + 1e-4)
            var_loss = float(np.mean(np.maximum(0.0, gamma - std)))
            mean = np.mean(z_curr, axis=0, keepdims=True)
            mask = (std < gamma).astype(np.float32)
            grad_var = -var_weight * (mask / (self.latent_dim * B)) * (z_curr - mean) / (std + 1e-4)

            # Covariance off-diagonal penalty
            z_centered = z_curr - mean
            cov = (z_centered.T @ z_centered) / (B - 1)
            np.fill_diagonal(cov, 0.0)
            cov_loss = float(np.sum(cov ** 2) / self.latent_dim)
            grad_cov = (4.0 * cov_weight / (self.latent_dim * (B - 1))) * (z_centered @ cov)

            grad_reg = grad_var + grad_cov

        total_loss = pred_loss + var_weight * var_loss + cov_weight * cov_loss

        # Backward pass through Predictor
        grad_za = self.predictor.backward(grad_z_pred)
        grad_z_from_pred = grad_za[:, :self.latent_dim]

        # Backward pass through Context Encoder
        grad_z_total = grad_z_pred + grad_z_from_pred + grad_reg
        self.context_encoder.backward(grad_z_total)

        return total_loss, pred_loss, var_loss, cov_loss

    def train_step(self, x_curr: np.ndarray, a_curr: np.ndarray, x_next: np.ndarray, lr: float = 1e-3) -> Dict[str, float]:
        """Performs a single Adam optimization step and EMA update."""
        self.step_count += 1
        total_l, pred_l, var_l, cov_l = self.compute_loss_and_grad(x_curr, a_curr, x_next)
        self.predictor.step(lr, self.step_count)
        self.context_encoder.step(lr, self.step_count)
        self.update_target_encoder()

        metrics = {
            "total_loss": total_l,
            "pred_loss": pred_l,
            "var_loss": var_l,
            "cov_loss": cov_l,
            "step": self.step_count,
        }
        self.loss_history.append(metrics)
        return metrics

    def score_candidate_action(
        self,
        current_state_vec: np.ndarray,
        candidate_action_vec: np.ndarray,
        target_state_vec: np.ndarray,
    ) -> float:
        """
        Scores candidate action in latent space:
        Score = - || hat{z}_{t+1} - z_{target} ||_2
        Higher score represents higher predicted likelihood of successfully reaching target state.
        """
        z_curr = self.encode(current_state_vec)
        z_pred = self.predict_latent(z_curr, candidate_action_vec)
        z_goal = self.encode(target_state_vec)

        dist = float(np.linalg.norm(z_pred - z_goal))
        score = -dist
        return score

    def evaluate_residual(
        self,
        current_state_vec: np.ndarray,
        action_executed: np.ndarray,
        observed_next_state_vec: np.ndarray,
    ) -> float:
        """
        Calculates Level 1 real-time physical prediction residual:
        Residual = || hat{z}_{t+1} - z_{t+1}^{observed} ||_2
        A sharp rise indicates an unmodeled physical disturbance (e.g. slip, collision, topple).
        """
        z_curr = self.encode(current_state_vec)
        z_pred = self.predict_latent(z_curr, action_executed)
        z_obs = self.encode(observed_next_state_vec)
        return float(np.linalg.norm(z_pred - z_obs))

    def save_checkpoint(self, path: str):
        """Saves trained model weights to compressed NPZ archive."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(
            path,
            enc_w1=self.context_encoder.fc1.W,
            enc_b1=self.context_encoder.fc1.b,
            enc_w2=self.context_encoder.fc2.W,
            enc_b2=self.context_encoder.fc2.b,
            pred_w1=self.predictor.fc1.W,
            pred_b1=self.predictor.fc1.b,
            pred_w2=self.predictor.fc2.W,
            pred_b2=self.predictor.fc2.b,
            tgt_w1=self.target_encoder.fc1.W,
            tgt_b1=self.target_encoder.fc1.b,
            tgt_w2=self.target_encoder.fc2.W,
            tgt_b2=self.target_encoder.fc2.b,
            step_count=self.step_count,
        )

    def load_checkpoint(self, path: str) -> bool:
        """Loads trained model weights from archive."""
        if not os.path.exists(path):
            return False
        data = np.load(path)
        self.context_encoder.fc1.W = data["enc_w1"]
        self.context_encoder.fc1.b = data["enc_b1"]
        self.context_encoder.fc2.W = data["enc_w2"]
        self.context_encoder.fc2.b = data["enc_b2"]
        self.predictor.fc1.W = data["pred_w1"]
        self.predictor.fc1.b = data["pred_b1"]
        self.predictor.fc2.W = data["pred_w2"]
        self.predictor.fc2.b = data["pred_b2"]
        self.target_encoder.fc1.W = data["tgt_w1"]
        self.target_encoder.fc1.b = data["tgt_b1"]
        self.target_encoder.fc2.W = data["tgt_w2"]
        self.target_encoder.fc2.b = data["tgt_b2"]
        self.step_count = int(data.get("step_count", 0))
        return True
