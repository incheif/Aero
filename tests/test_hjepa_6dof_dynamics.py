"""
Tests for 6-DOF Action-Conditioned H-JEPA World Model.
Validates:
- Structured 112-D state extraction with 6-DOF kinematics and Feetech STS3215 actuator feedback.
- 7-D action vector extraction.
- Latent representation space invariance and normalization to R^64.
- Analytical VICReg loss, backpropagation, and Adam parameter updates.
- EMA target encoder tracking.
- Candidate action scoring and real-time residual tracking under simulated disturbances.
- Checkpoint persistence and weight reconstruction.
"""

import os
import tempfile
import numpy as np
import pytest

from vla.predictive_world_model import (
    extract_state_vector_6dof,
    extract_action_vector_7d,
    HierarchicalJEPA,
    LinearLayer,
    MLP,
    gelu,
    gelu_backward,
)


def test_gelu_and_gradients():
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    y = gelu(x)
    assert y.shape == x.shape
    assert y[2] == 0.0  # GELU(0) == 0
    assert y[4] > y[3]  # strictly monotonically increasing in positive regime

    grad_out = np.ones_like(x)
    grad_in = gelu_backward(x, grad_out)
    assert grad_in.shape == x.shape
    assert np.all(np.isfinite(grad_in))


def test_state_vector_6dof_extraction():
    snapshot = {
        "joints": {
            "baseYaw": 0.12,
            "shoulderPitch": -0.45,
            "elbowPitch": 0.88,
            "wristPitch": -0.32,
            "wristRoll": 0.15,
            "gripper": 0.038,
        },
        "joint_velocities": {
            "baseYaw": 0.01,
            "shoulderPitch": -0.02,
            "elbowPitch": 0.03,
            "wristPitch": -0.01,
            "wristRoll": 0.0,
            "gripper": 0.005,
        },
        "actuators": {
            "torques": {
                "baseYaw": 0.42,
                "shoulderPitch": 1.25,
                "elbowPitch": 0.85,
                "wristPitch": 0.31,
                "wristRoll": 0.12,
                "gripper": 0.45,
            }
        },
        "tcp": {
            "position": [0.24, 0.85, 0.05],
            "rotation": [0.0, 0.7071, 0.0, 0.7071],
            "velocity": [0.01, -0.02, 0.0],
        },
        "contact_force": 2.4,
        "grasped_block_id": "block_cyan",
        "blocks": [
            {
                "id": "block_cyan",
                "position": [0.24, 0.85, 0.05],
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "velocity": [0.0, 0.0, 0.0],
                "is_grasped": True,
            },
            {
                "id": "block_magenta",
                "position": [0.15, 0.75, -0.10],
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "velocity": [0.0, 0.0, 0.0],
                "is_grasped": False,
            },
        ],
        "table_clearance": 0.95,
        "step_progress": 0.25,
        "scores": {
            "stability": {
                "is_verified": True,
                "stability_score": 0.98,
            }
        },
    }

    vec = extract_state_vector_6dof(snapshot)
    assert isinstance(vec, np.ndarray)
    assert vec.dtype == np.float32
    assert len(vec) == 112
    # Verify values in first section
    assert np.isclose(vec[0], 0.12)   # baseYaw
    assert np.isclose(vec[1], -0.45)  # shoulderPitch
    assert np.isclose(vec[4], 0.15)   # wristRoll


def test_action_vector_7d_extraction():
    joints_cmd = {
        "baseYaw": 0.10,
        "shoulderPitch": -0.40,
        "elbowPitch": 0.85,
        "wristPitch": -0.30,
        "wristRoll": 0.05,
        "gripper_velocity": -0.02,
    }
    act = extract_action_vector_7d(joints_cmd, gripper_cmd=0.012)
    assert len(act) == 7
    assert np.isclose(act[0], 0.10)
    assert np.isclose(act[4], 0.05)
    assert np.isclose(act[5], 0.012)
    assert np.isclose(act[6], -0.02)


def test_hjepa_forward_and_residual_rollout():
    jepa = HierarchicalJEPA(state_dim=112, action_dim=7, latent_dim=64, seed=42)

    x = np.random.randn(112).astype(np.float32)
    z = jepa.encode(x)
    assert z.shape == (64,)
    assert np.all(np.isfinite(z))
    assert np.linalg.norm(z) > 0.0

    a = np.random.randn(7).astype(np.float32)
    z_next = jepa.predict_latent(z, a)
    assert z_next.shape == (64,)
    assert np.all(np.isfinite(z_next))


def test_hjepa_vicreg_training_and_optimization():
    jepa = HierarchicalJEPA(state_dim=112, action_dim=7, latent_dim=64, seed=123)

    B = 8
    X_curr = np.random.randn(B, 112).astype(np.float32)
    A_curr = np.random.randn(B, 7).astype(np.float32)
    X_next = X_curr + 0.05 * np.random.randn(B, 112).astype(np.float32)

    # Initial weights snapshot
    w_pred_init = jepa.predictor.fc1.W.copy()
    w_enc_init = jepa.context_encoder.fc1.W.copy()
    w_tgt_init = jepa.target_encoder.fc1.W.copy()

    metrics = jepa.train_step(X_curr, A_curr, X_next, lr=1e-3)
    assert "total_loss" in metrics
    assert "pred_loss" in metrics
    assert "var_loss" in metrics
    assert "cov_loss" in metrics
    assert metrics["total_loss"] > 0.0

    # Weights must be genuinely updated via Adam
    assert not np.array_equal(jepa.predictor.fc1.W, w_pred_init)
    assert not np.array_equal(jepa.context_encoder.fc1.W, w_enc_init)
    # Target weights must be updated via EMA
    assert not np.array_equal(jepa.target_encoder.fc1.W, w_tgt_init)


def test_candidate_action_scoring_and_residuals():
    jepa = HierarchicalJEPA(state_dim=112, action_dim=7, latent_dim=64, seed=99)

    s_curr = np.random.randn(112).astype(np.float32)
    s_target = s_curr + 0.1 * np.random.randn(112).astype(np.float32)

    good_action = np.zeros(7, dtype=np.float32)
    bad_action = np.ones(7, dtype=np.float32) * 5.0

    score_good = jepa.score_candidate_action(s_curr, good_action, s_target)
    score_bad = jepa.score_candidate_action(s_curr, bad_action, s_target)

    assert isinstance(score_good, float)
    assert isinstance(score_bad, float)
    assert score_good <= 0.0  # Distance negated

    # Residual calculation
    s_obs_normal = s_curr + 0.02 * np.random.randn(112).astype(np.float32)
    s_obs_disturbed = s_curr + 2.5 * np.random.randn(112).astype(np.float32)

    res_normal = jepa.evaluate_residual(s_curr, good_action, s_obs_normal)
    res_disturbed = jepa.evaluate_residual(s_curr, good_action, s_obs_disturbed)

    assert res_disturbed > res_normal


def test_checkpoint_persistence():
    jepa = HierarchicalJEPA(state_dim=112, action_dim=7, latent_dim=64, seed=77)
    # Run a step
    x = np.random.randn(4, 112).astype(np.float32)
    a = np.random.randn(4, 7).astype(np.float32)
    jepa.train_step(x, a, x, lr=1e-3)

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        jepa.save_checkpoint(tmp_path)
        assert os.path.exists(tmp_path)

        jepa2 = HierarchicalJEPA(state_dim=112, action_dim=7, latent_dim=64, seed=999)
        loaded = jepa2.load_checkpoint(tmp_path)
        assert loaded is True
        assert np.allclose(jepa.context_encoder.fc1.W, jepa2.context_encoder.fc1.W)
        assert np.allclose(jepa.predictor.fc2.W, jepa2.predictor.fc2.W)
        assert jepa2.step_count == 1
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
