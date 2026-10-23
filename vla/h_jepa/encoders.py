"""
vla/h_jepa/encoders.py

Observation and Action Encoders for the 2-Level H-JEPA Architecture:
- Level 2 Observation Encoder E^(2): Global tabletop scene state -> z^(2) in R^128
- Level 2 Action Encoder A^(2): Macro-action -> a_macro^(2) in R^32
- Level 1 Observation Encoder E^(1): High-rate micro state (TCP + local target) -> z^(1) in R^64
- Level 1 Action Encoder A^(1): Delta-TCP and gripper state -> a^(1) in R^16
- Cross-Level Projection Head P^(1->2): Projects z^(1) -> z^(2) for sub-goal distance evaluation
"""

import math
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from simulation.constants import (
    STACK_ORIGIN,
    TABLE_TOP_Y,
    CUBE_SIZE,
    VLA_IMAGE_SIZE,
)

def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Standard LayerNorm across the last dimension."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)

def gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit."""
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x ** 3))))

class Level2ObservationEncoder:
    """
    E^(2): Encodes global scene state into macro latent embedding z^(2) in R^128.
    Captures 3D block coordinates, relative distances to target slots,
    spatial layer occupancy, and grasp status.
    """
    def __init__(self, latent_dim: int = 128, max_blocks: int = 7):
        self.latent_dim = latent_dim
        self.max_blocks = max_blocks
        
        # Raw block feature dimension per block:
        # pos (3) + rot (4) + in_pad (1) + is_grasped (1) + target_dist (1) + layer (1) = 11
        # Tabletop global context: TCP pos (3) + pad origin (3) + active task blocks count (1) = 7
        # Total raw input dim = 7 * 11 + 7 = 84
        self.raw_dim = max_blocks * 11 + 7
        
        # Deterministic orthogonal projection weights
        rng = np.random.RandomState(42)
        W = rng.randn(self.raw_dim, self.latent_dim)
        # Gram-Schmidt / SVD orthogonalization for stable representation dispersion
        u, _, vt = np.linalg.svd(W, full_matrices=False)
        self.W1 = u @ vt
        self.b1 = np.zeros(self.latent_dim)
        
        # Second projection layer
        self.W2 = rng.randn(self.latent_dim, self.latent_dim) * math.sqrt(2.0 / self.latent_dim)
        self.b2 = np.zeros(self.latent_dim)

    def extract_scene_features(
        self,
        blocks: List[Dict[str, Any]],
        tcp_pos: Optional[List[float]] = None,
        target_origin: Optional[List[float]] = None,
        grasped_id: Optional[str] = None
    ) -> np.ndarray:
        origin = target_origin or STACK_ORIGIN
        tcp = tcp_pos or [0.0, 0.8, 0.0]
        
        feats = []
        # Tabletop global context
        feats.extend([tcp[0], tcp[1], tcp[2]])
        feats.extend([origin[0], origin[1], origin[2]])
        feats.append(float(len(blocks)))
        
        # Deterministic sorting of blocks by ID
        sorted_blocks = sorted(blocks, key=lambda b: b.get("id", ""))
        for i in range(self.max_blocks):
            if i < len(sorted_blocks):
                b = sorted_blocks[i]
                pos = b.get("position", [0.0, TABLE_TOP_Y, 0.0])
                rot = b.get("rotation", [0.0, 0.0, 0.0, 1.0])
                in_pad = 1.0 if b.get("in_pad") else 0.0
                is_grasped = 1.0 if (b.get("id") == grasped_id or b.get("is_grasped")) else 0.0
                
                # Distance to target origin
                dx = pos[0] - origin[0]
                dz = pos[2] - origin[2]
                dist_pad = math.hypot(dx, dz)
                
                # Estimated layer index (0 for table, 1 for stack tier 1, etc.)
                rel_y = max(0.0, pos[1] - TABLE_TOP_Y)
                layer = round(rel_y / max(0.001, CUBE_SIZE))
                
                feats.extend([
                    pos[0], pos[1], pos[2],
                    rot[0], rot[1], rot[2], rot[3],
                    in_pad, is_grasped, dist_pad, float(layer)
                ])
            else:
                # Padding for inactive slots
                feats.extend([0.0] * 11)

        return np.array(feats, dtype=np.float32)

    def encode(
        self,
        blocks: List[Dict[str, Any]],
        tcp_pos: Optional[List[float]] = None,
        target_origin: Optional[List[float]] = None,
        grasped_id: Optional[str] = None
    ) -> np.ndarray:
        raw = self.extract_scene_features(blocks, tcp_pos, target_origin, grasped_id)
        # Two-layer MLP with LayerNorm & GELU
        h = gelu(raw @ self.W1 + self.b1)
        h = layer_norm(h)
        z2 = h @ self.W2 + self.b2
        # Normalize onto unit sphere to stabilize JEPA metric space
        norm = np.linalg.norm(z2)
        if norm > 1e-6:
            z2 = z2 / norm
        return z2.astype(np.float32)


class Level2ActionEncoder:
    """
    A^(2): Encodes macro actions (which block, target slot [x,y,z], tier)
    into a continuous macro-action latent vector a_macro^(2) in R^32.
    """
    def __init__(self, action_dim: int = 32):
        self.action_dim = action_dim
        # Macro raw action: target_xyz (3) + block_idx_onehot (7) + tier_onehot (4) + phase_type (2) = 16
        self.raw_dim = 16
        rng = np.random.RandomState(43)
        self.W = rng.randn(self.raw_dim, self.action_dim) * math.sqrt(2.0 / self.raw_dim)
        self.b = np.zeros(self.action_dim)

    def encode(
        self,
        target_xyz: List[float],
        block_index: int = 0,
        tier: int = 0,
        action_type: str = "PICK_AND_PLACE"
    ) -> np.ndarray:
        raw = np.zeros(self.raw_dim, dtype=np.float32)
        raw[0:3] = target_xyz[:3]
        if 0 <= block_index < 7:
            raw[3 + block_index] = 1.0
        if 0 <= tier < 4:
            raw[10 + tier] = 1.0
        raw[14] = 1.0 if action_type == "PICK_AND_PLACE" else 0.0
        raw[15] = 1.0 if action_type == "RELEASE_AND_VERIFY" else 0.0

        a2 = gelu(raw @ self.W + self.b)
        return a2.astype(np.float32)


class Level1ObservationEncoder:
    """
    E^(1): Encodes high-rate micro state into z^(1) in R^64.
    Encodes current TCP position, gripper jaw opening, delta to active block/target,
    and instantaneous velocity vector.
    """
    def __init__(self, latent_dim: int = 64):
        self.latent_dim = latent_dim
        # Raw micro features:
        # tcp_pos (3) + tcp_vel (3) + gripper_aperture (1) + target_offset (3) + dist (1) + phase_step (1) = 12
        self.raw_dim = 12
        rng = np.random.RandomState(44)
        W = rng.randn(self.raw_dim, self.latent_dim)
        u, _, vt = np.linalg.svd(W, full_matrices=False)
        self.W1 = u @ vt
        self.b1 = np.zeros(self.latent_dim)
        self.W2 = rng.randn(self.latent_dim, self.latent_dim) * math.sqrt(2.0 / self.latent_dim)
        self.b2 = np.zeros(self.latent_dim)

    def encode(
        self,
        tcp_pos: List[float],
        target_pos: List[float],
        gripper_aperture: float = 0.0,
        tcp_vel: Optional[List[float]] = None,
        micro_step: int = 0
    ) -> np.ndarray:
        vel = tcp_vel or [0.0, 0.0, 0.0]
        dx = target_pos[0] - tcp_pos[0]
        dy = target_pos[1] - tcp_pos[1]
        dz = target_pos[2] - tcp_pos[2]
        dist = math.hypot(dx, dy, dz)

        raw = np.array([
            tcp_pos[0], tcp_pos[1], tcp_pos[2],
            vel[0], vel[1], vel[2],
            gripper_aperture,
            dx, dy, dz,
            dist,
            float(micro_step)
        ], dtype=np.float32)

        h = gelu(raw @ self.W1 + self.b1)
        z1 = h @ self.W2 + self.b2
        norm = np.linalg.norm(z1)
        if norm > 1e-6:
            z1 = z1 / norm
        return z1.astype(np.float32)


class Level1ActionEncoder:
    """
    A^(1): Encodes micro actions (delta_tcp [dx, dy, dz], gripper_cmd) into a^(1) in R^16.
    """
    def __init__(self, action_dim: int = 16):
        self.action_dim = action_dim
        self.raw_dim = 4  # dx, dy, dz, gripper
        rng = np.random.RandomState(45)
        self.W = rng.randn(self.raw_dim, self.action_dim) * math.sqrt(2.0 / self.raw_dim)
        self.b = np.zeros(self.action_dim)

    def encode(self, delta_tcp: Any, gripper_cmd: float = 0.0) -> np.ndarray:
        if isinstance(delta_tcp, np.ndarray):
            if delta_tcp.ndim == 1 and delta_tcp.shape[0] == 4:
                return gelu(delta_tcp @ self.W + self.b).astype(np.float32)
            elif delta_tcp.ndim >= 2:
                return gelu(delta_tcp @ self.W + self.b).astype(np.float32)
        raw = np.array([delta_tcp[0], delta_tcp[1], delta_tcp[2], gripper_cmd], dtype=np.float32)
        a1 = gelu(raw @ self.W + self.b)
        return a1.astype(np.float32)

    def encode_batch(self, actions: np.ndarray) -> np.ndarray:
        """Fast vectorized batch encoding: shape (..., 4) -> shape (..., 16) in sub-millisecond."""
        return gelu(actions @ self.W + self.b).astype(np.float32)


class ProjectionHead1to2:
    """
    P^(1->2): Cross-level projector lifting Level 1 micro latent z^(1) in R^64
    into Level 2 macro latent space z^(2) in R^128.
    Allows evaluating micro action trajectories directly against Level 2 sub-goals.
    """
    def __init__(self, in_dim: int = 64, out_dim: int = 128):
        self.in_dim = in_dim
        self.out_dim = out_dim
        rng = np.random.RandomState(46)
        W = rng.randn(in_dim, out_dim)
        u, _, vt = np.linalg.svd(W, full_matrices=False)
        self.W1 = u @ vt
        self.b1 = np.zeros(out_dim)
        self.W2 = rng.randn(out_dim, out_dim) * math.sqrt(2.0 / out_dim)
        self.b2 = np.zeros(out_dim)

    def project(self, z1: np.ndarray) -> np.ndarray:
        h = gelu(z1 @ self.W1 + self.b1)
        z2_lifted = h @ self.W2 + self.b2
        norm = np.linalg.norm(z2_lifted, axis=-1, keepdims=True)
        norm = np.maximum(norm, 1e-6)
        return (z2_lifted / norm).astype(np.float32)
