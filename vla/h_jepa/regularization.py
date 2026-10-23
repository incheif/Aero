"""
vla/h_jepa/regularization.py

SIGReg / VICReg Regularization Suite for Joint-Embedding Predictive Architectures.
Prevents representation collapse (informational collapse and dimensional collapse)
without requiring negative contrastive pairs.
Reference: LeCun et al. (VICReg / I-JEPA / V-JEPA).
"""

import numpy as np
from typing import Dict, Tuple

def invariance_loss(z_pred: np.ndarray, z_target: np.ndarray) -> float:
    """
    Mean Squared Error / L2 distance between predicted latent state and ground truth target.
    z_pred, z_target: shape (N, D) or (D,)
    """
    diff = z_pred - z_target
    return float(np.mean(diff ** 2))

def variance_loss(z: np.ndarray, gamma: float = 1.0, eps: float = 1e-4) -> float:
    """
    Variance regularization term: forces standard deviation of each feature dimension
    to be at least gamma, preventing representation collapse to a constant.
    z: shape (N, D)
    """
    if z.ndim == 1:
        z = z.reshape(1, -1)
    if z.shape[0] <= 1:
        return 0.0
    
    std = np.sqrt(np.var(z, axis=0) + eps)
    # Hinge loss: max(0, gamma - std)
    loss = np.mean(np.maximum(0.0, gamma - std))
    return float(loss)

def covariance_loss(z: np.ndarray) -> float:
    """
    Covariance regularization term: decorrelates different feature dimensions
    by penalizing off-diagonal elements of the covariance matrix.
    Prevents dimensional collapse where dimensions encode redundant information.
    z: shape (N, D)
    """
    if z.ndim == 1:
        z = z.reshape(1, -1)
    n, d = z.shape
    if n <= 1:
        return 0.0

    z_centered = z - np.mean(z, axis=0, keepdims=True)
    cov = (z_centered.T @ z_centered) / (n - 1)
    
    # Zero out diagonal
    np.fill_diagonal(cov, 0.0)
    loss = np.sum(cov ** 2) / float(d)
    return float(loss)

def sigreg_loss(
    z_pred: np.ndarray,
    z_target: np.ndarray,
    sim_weight: float = 1.0,
    var_weight: float = 0.5,
    cov_weight: float = 0.05,
    gamma: float = 1.0,
) -> Tuple[float, Dict[str, float]]:
    """
    Computes total SIGReg / VICReg loss with individual metric breakdowns.
    """
    l_sim = invariance_loss(z_pred, z_target)
    l_var = variance_loss(z_pred, gamma=gamma)
    l_cov = covariance_loss(z_pred)

    total = sim_weight * l_sim + var_weight * l_var + cov_weight * l_cov
    metrics = {
        "loss_sim": round(l_sim, 5),
        "loss_var": round(l_var, 5),
        "loss_cov": round(l_cov, 5),
        "total_sigreg": round(total, 5),
    }
    return float(total), metrics
