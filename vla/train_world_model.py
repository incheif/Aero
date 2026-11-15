"""
Dataset Collection and Training Pipeline for AERO 6-DOF Action-Conditioned H-JEPA World Model.
Collects genuine transitions from MuJoCo physics with Feetech STS3215 actuator dynamics,
performs analytical gradient descent updates with VICReg regularization, evaluates validation loss,
and saves a trained checkpoint to models/hjepa_6dof_world_model.npz.
"""

import os
import sys
import time
import random
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation.mujoco_engine import MujocoSimulation
from simulation.constants import FIXED_DT, DEFAULT_JOINTS, JOINT_LIMITS
from simulation.kinematics import solve_ik
from vla.predictive_world_model import (
    HierarchicalJEPA,
    extract_state_vector_6dof,
    extract_action_vector_7d,
)


class TrajectoryCollector6DOF:
    """Collects structured 6-DOF transitions from MuJoCo workcell."""

    def __init__(self):
        self.transitions: List[Dict[str, np.ndarray]] = []

    def record_transition(
        self,
        snapshot_t: Dict[str, Any],
        joints_cmd: Dict[str, float],
        gripper_cmd: float,
        snapshot_next: Dict[str, Any],
    ):
        x_t = extract_state_vector_6dof(snapshot_t)
        a_t = extract_action_vector_7d(joints_cmd, gripper_cmd)
        x_tp1 = extract_state_vector_6dof(snapshot_next)
        self.transitions.append({
            "x_curr": x_t,
            "a_curr": a_t,
            "x_next": x_tp1,
        })

    def get_dataset(self, train_ratio: float = 0.8) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        if not self.transitions:
            raise ValueError("No transitions recorded.")
        X = np.stack([t["x_curr"] for t in self.transitions])
        A = np.stack([t["a_curr"] for t in self.transitions])
        Y = np.stack([t["x_next"] for t in self.transitions])

        N = len(self.transitions)
        indices = np.arange(N)
        np.random.RandomState(42).shuffle(indices)

        split = int(N * train_ratio)
        train_idx = indices[:split]
        val_idx = indices[split:]

        train_data = {"X": X[train_idx], "A": A[train_idx], "Y": Y[train_idx]}
        val_data = {"X": X[val_idx], "A": A[val_idx], "Y": Y[val_idx]}
        return train_data, val_data


def collect_simulation_trajectories(num_episodes: int = 15, episode_length: int = 35) -> TrajectoryCollector6DOF:
    """
    Collects rich manipulation transitions from MuJoCo physics:
    - Directed pick and lift movements
    - Horizontal transit and descending placements
    - Random joint exploratory perturbations and gripper actuations
    """
    sim = MujocoSimulation()
    collector = TrajectoryCollector6DOF()

    for ep in range(num_episodes):
        sim.reset(randomize=(ep % 2 == 1))
        # Sequence of target waypoints for arm to execute
        waypoints = [
            (0.20, 0.90, 0.0, 0.038),
            (0.24, 0.76, 0.10, 0.038),
            (0.24, 0.76, 0.10, 0.008),
            (0.24, 0.92, 0.10, 0.008),
            (0.32, 0.92, -0.12, 0.008),
            (0.32, 0.76, -0.12, 0.038),
            (0.32, 0.90, -0.12, 0.038),
            (0.20, 0.88, 0.0, 0.038),
        ]
        wp_idx = 0
        tgt_x, tgt_y, tgt_z, tgt_grip = waypoints[wp_idx]

        for step in range(episode_length):
            snap_t = sim.get_snapshot()

            if step % 7 == 0 and step > 0:
                wp_idx = (wp_idx + 1) % len(waypoints)
                tgt_x, tgt_y, tgt_z, tgt_grip = waypoints[wp_idx]

            # 6-DOF IK solve
            ik_res = solve_ik((tgt_x, tgt_y, tgt_z), gripper=tgt_grip)
            target_joints = dict(ik_res.joints)

            # Add exploratory noise
            for k in ["shoulderPitch", "elbowPitch", "wristPitch", "wristRoll"]:
                if k in target_joints:
                    target_joints[k] += random.uniform(-0.012, 0.012)

            sim.set_joint_targets(target_joints, tgt_grip)
            snap_next = sim.step(FIXED_DT)

            collector.record_transition(
                snapshot_t=snap_t,
                joints_cmd=target_joints,
                gripper_cmd=tgt_grip,
                snapshot_next=snap_next,
            )

    return collector


def train_hjepa_6dof_world_model(
    checkpoint_path: str = "models/hjepa_6dof_world_model.npz",
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 2e-3,
) -> Tuple[HierarchicalJEPA, Dict[str, List[float]]]:
    """Trains the 6-DOF H-JEPA world model on collected MuJoCo transitions."""
    print("Collecting genuine physical transitions from 6-DOF MuJoCo workcell...")
    collector = collect_simulation_trajectories(num_episodes=15, episode_length=35)
    train_data, val_data = collector.get_dataset(train_ratio=0.8)

    N_train = train_data["X"].shape[0]
    N_val = val_data["X"].shape[0]
    print(f"Dataset assembled: {N_train} training samples, {N_val} validation samples.")

    model = HierarchicalJEPA(state_dim=112, action_dim=7, latent_dim=64, ema_tau=0.99, seed=42)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_mse": [],
    }

    print(f"Beginning training over {epochs} epochs (batch_size={batch_size}, lr={lr})...")
    rng = np.random.RandomState(42)

    for epoch in range(1, epochs + 1):
        perm = rng.permutation(N_train)
        epoch_losses = []

        for start_idx in range(0, N_train, batch_size):
            end_idx = min(start_idx + batch_size, N_train)
            batch_indices = perm[start_idx:end_idx]
            if len(batch_indices) < 2:
                continue

            xb = train_data["X"][batch_indices]
            ab = train_data["A"][batch_indices]
            yb = train_data["Y"][batch_indices]

            res = model.train_step(xb, ab, yb, lr=lr)
            epoch_losses.append(res["total_loss"])

        avg_train = float(np.mean(epoch_losses)) if epoch_losses else 0.0

        # Evaluate validation loss
        z_val = model.encode(val_data["X"])
        z_pred_val = model.predict_latent(z_val, val_data["A"])
        z_target_val = model.target_encoder.forward(val_data["Y"])
        # normalize z_target_val to unit sphere for fair metric
        norm_t = np.linalg.norm(z_target_val, axis=-1, keepdims=True)
        z_target_val_norm = z_target_val / np.maximum(norm_t, 1e-6)
        val_mse = float(np.mean((z_pred_val - z_target_val_norm) ** 2))

        history["train_loss"].append(avg_train)
        history["val_loss"].append(val_mse)
        history["val_mse"].append(val_mse)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:2d}/{epochs} | Train Loss: {avg_train:.5f} | Val MSE: {val_mse:.5f}")

    # Save trained model checkpoint
    model.save_checkpoint(checkpoint_path)
    print(f"Training complete! Genuine trained checkpoint saved to: {checkpoint_path}")

    return model, history


if __name__ == "__main__":
    train_hjepa_6dof_world_model()

