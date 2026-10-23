"""
vla/h_jepa/predictors.py

Latent Transition Predictors for the 2-Level H-JEPA Architecture:
- Level 2 Macro Predictor F^(2): (z_t^(2), a_macro^(2)) -> z_{t+Delta T}^(2)
- Level 1 Micro Predictor F^(1): (z_k^(1), a_k^(1)) -> z_{k+1}^(1)
"""

import math
from typing import Optional, List
import numpy as np

def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x ** 3))))

def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)

class Level2Predictor:
    """
    F^(2): Macro-level latent world model predictor.
    Forecasts multi-phase state evolution across macroscopic temporal jumps:
      (z_t^(2) in R^128, a_macro^(2) in R^32) -> hat{z}_{t+Delta T}^(2) in R^128
    Uses residual connections to preserve topological stability across long jumps.
    """
    def __init__(self, state_dim: int = 128, action_dim: int = 32):
        self.state_dim = state_dim
        self.action_dim = action_dim
        in_dim = state_dim + action_dim

        rng = np.random.RandomState(50)
        self.W1 = rng.randn(in_dim, state_dim) * math.sqrt(2.0 / in_dim)
        self.b1 = np.zeros(state_dim)
        self.W2 = rng.randn(state_dim, state_dim) * math.sqrt(2.0 / state_dim)
        self.b2 = np.zeros(state_dim)

    def predict(self, z2: np.ndarray, a2: np.ndarray) -> np.ndarray:
        """
        Rolls forward one macro step in Level 2 latent space.
        Supports single vectors or batched inputs (B, D).
        """
        single = (z2.ndim == 1)
        if single:
            z2 = z2.reshape(1, -1)
            a2 = a2.reshape(1, -1)

        cat = np.concatenate([z2, a2], axis=-1)
        h = gelu(cat @ self.W1 + self.b1)
        h = layer_norm(h)
        delta_z = h @ self.W2 + self.b2
        
        # Residual update: z_next = z_curr + delta_z
        z_next = z2 + delta_z
        # Normalize to unit sphere
        norm = np.linalg.norm(z_next, axis=-1, keepdims=True)
        norm = np.maximum(norm, 1e-6)
        z_next = z_next / norm

        if single:
            return z_next[0].astype(np.float32)
        return z_next.astype(np.float32)


class Level1Predictor:
    """
    F^(1): Micro-level latent world model predictor.
    Forecasts single-tick physical motion in lightweight embedding space:
      (z_k^(1) in R^64, a_k^(1) in R^16) -> hat{z}_{k+1}^(1) in R^64
    Operates at sub-millisecond execution times for fast CEM trajectory evaluation.
    """
    def __init__(self, state_dim: int = 64, action_dim: int = 16):
        self.state_dim = state_dim
        self.action_dim = action_dim
        in_dim = state_dim + action_dim

        rng = np.random.RandomState(51)
        self.W1 = rng.randn(in_dim, state_dim) * math.sqrt(2.0 / in_dim)
        self.b1 = np.zeros(state_dim)
        self.W2 = rng.randn(state_dim, state_dim) * math.sqrt(2.0 / state_dim)
        self.b2 = np.zeros(state_dim)

    def predict(self, z1: np.ndarray, a1: np.ndarray) -> np.ndarray:
        """
        Rolls forward one micro step in Level 1 latent space.
        Supports batched evaluation (N_samples, state_dim).
        """
        single = (z1.ndim == 1)
        if single:
            z1 = z1.reshape(1, -1)
            a1 = a1.reshape(1, -1)

        cat = np.concatenate([z1, a1], axis=-1)
        h = gelu(cat @ self.W1 + self.b1)
        delta_z = h @ self.W2 + self.b2
        
        # Residual update
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
        z_init: shape (N, 64)
        a_seq: shape (N, H, 16)
        Returns: z_final shape (N, 64)
        """
        z_curr = z_init
        H = a_seq.shape[1]
        for t in range(H):
            a_t = a_seq[:, t, :]
            z_curr = self.predict(z_curr, a_t)
        return z_curr
