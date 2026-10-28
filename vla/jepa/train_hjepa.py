"""
vla/jepa/train_hjepa.py

Self-Supervised VICReg and Multi-Step Latent Prediction Trainer for H-JEPA.
Implements the four-part collapse-prevention training objective from Section 9.3:
1. Level 1 Multi-Step Latent Prediction Loss (L_pred_1)
2. Level 2 Subtask Boundary & Text Alignment Loss (L_pred_2)
3. Cross-Level Consistency Loss (L_hier)
4. VICReg Variance-Covariance Regularization Loss (L_VIC)
5. Auxiliary Tower Stability Loss (L_stab)
"""

import math
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from .encoders import (
    Level1ObservationEncoder,
    Level1ActionEncoder,
    Level2ObservationEncoder,
    ProjectionHead1to2,
    VLTextGoalEncoder,
)
from .level1_world_model import Level1Predictor
from .level2_world_model import Level2Predictor, StabilityProbe


def vicreg_loss(
    Z: np.ndarray,
    gamma: float = 1.0,
    eps: float = 1e-4,
    lambda_var: float = 15.0,
    lambda_cov: float = 1.0,
) -> Tuple[float, float, float]:
    """
    Computes VICReg Variance and Covariance regularization loss on a batch of embeddings Z (B, D).
    Prevents representation collapse without negative samples.
    """
    B, D = Z.shape
    if B < 2:
        return 0.0, 0.0, 0.0

    # 1. Variance hinge loss: encourages std across batch to stay >= gamma
    mean = np.mean(Z, axis=0, keepdims=True)
    std = np.sqrt(np.var(Z, axis=0) + eps)
    var_loss = np.mean(np.maximum(0.0, gamma - std))

    # 2. Covariance loss: penalizes off-diagonal correlations to prevent feature collapse
    Z_centered = Z - mean
    cov = (Z_centered.T @ Z_centered) / (B - 1)
    np.fill_diagonal(cov, 0.0)
    cov_loss = np.sum(cov ** 2) / D

    total_vic = lambda_var * var_loss + lambda_cov * cov_loss
    return float(total_vic), float(var_loss), float(cov_loss)


def compute_hjepa_losses(
    pred_z1: np.ndarray,
    target_z1: np.ndarray,
    pred_z2: np.ndarray,
    target_z2: np.ndarray,
    proj_z2: np.ndarray,
    pred_stab: float,
    true_stab: float,
    batch_z: np.ndarray,
) -> Dict[str, float]:
    """Computes all H-JEPA loss components."""
    # L1 latent prediction error
    l1_loss = float(np.mean((pred_z1 - target_z1) ** 2))

    # L2 macro prediction error
    l2_loss = float(np.mean((pred_z2 - target_z2) ** 2))

    # Cross-level projection consistency
    l_hier = float(np.mean((proj_z2 - target_z2) ** 2))

    # Stability BCE
    p = max(1e-6, min(1.0 - 1e-6, pred_stab))
    l_stab = -float(true_stab * math.log(p) + (1.0 - true_stab) * math.log(1.0 - p))

    # VICReg regularization
    l_vic, var_l, cov_l = vicreg_loss(batch_z)

    total_loss = l1_loss + l2_loss + 0.5 * l_hier + 0.2 * l_stab + 0.1 * l_vic

    return {
        "total_loss": total_loss,
        "l1_loss": l1_loss,
        "l2_loss": l2_loss,
        "l_hier": l_hier,
        "l_stab": l_stab,
        "l_vic": l_vic,
        "var_loss": var_l,
        "cov_loss": cov_l,
    }


class HJEPATrainer:
    """Trainer orchestrating H-JEPA weights updates and evaluation."""
    def __init__(self):
        self.encoder_l1 = Level1ObservationEncoder()
        self.encoder_l2 = Level2ObservationEncoder()
        self.projector_1to2 = ProjectionHead1to2()
        self.predictor_l1 = Level1Predictor()
        self.predictor_l2 = Level2Predictor()
        self.stability_probe = StabilityProbe()

    def train_step(self, sample: Dict[str, Any]) -> Dict[str, float]:
        """Runs a forward loss computation over a transition sample."""
        # Simulated states
        z1_curr = np.random.randn(self.encoder_l1.latent_dim).astype(np.float32)
        z1_curr = z1_curr / np.linalg.norm(z1_curr)

        z2_curr = np.random.randn(self.encoder_l2.latent_dim).astype(np.float32)
        z2_curr = z2_curr / np.linalg.norm(z2_curr)

        act_emb = np.random.randn(32).astype(np.float32)
        act_emb = act_emb / np.linalg.norm(act_emb)

        u_macro = np.random.randn(16).astype(np.float32)

        # Forward predictions
        pred_z1 = self.predictor_l1.predict(z1_curr, act_emb)
        pred_z2 = self.predictor_l2.predict(z2_curr, u_macro)
        proj_z2 = self.projector_1to2.project(pred_z1)

        # Simulated targets
        target_z1 = pred_z1 + np.random.normal(0, 0.05, size=pred_z1.shape).astype(np.float32)
        target_z2 = pred_z2 + np.random.normal(0, 0.05, size=pred_z2.shape).astype(np.float32)

        pred_stab = self.stability_probe.predict_stability(pred_z2)
        true_stab = 0.0 if sample.get("toppled", False) else 1.0

        batch_z = np.random.randn(16, self.encoder_l1.latent_dim).astype(np.float32)

        losses = compute_hjepa_losses(
            pred_z1=pred_z1,
            target_z1=target_z1,
            pred_z2=pred_z2,
            target_z2=target_z2,
            proj_z2=proj_z2,
            pred_stab=pred_stab,
            true_stab=true_stab,
            batch_z=batch_z,
        )
        return losses
