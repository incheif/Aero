"""
vla/jepa/encoders.py

Observation and Action Encoders for the Two-Level Hierarchical VL-JEPA Architecture:
1. Level 1 Context Encoder E^(1): Fuses visual patch representations (DINOv3 / State+Patch)
   with proprioceptive arm kinematics (q_t, x_ee, g_t) into z^(1) in R^384.
2. Level 1 Action Encoder A^(1): Maps 5-DOF delta commands into action embeddings.
3. Level 2 Context Encoder E^(2): Object-centric scene graph encoder into z^(2) in R^512.
4. Projection Head P_{1->2}: Projects Level 1 states into Level 2 latent space.
5. VL-JEPA Language Alignment Head E_txt: Embeds symbolic task prompts into z_goal^(2).
"""

import math
import hashlib
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from simulation.constants import (
    STACK_ORIGIN,
    TABLE_TOP_Y,
    CUBE_SIZE,
    VLA_IMAGE_SIZE,
)
from .schemas import WorkspaceObservation, ObjectLatentState


def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Standard LayerNorm across the last dimension."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit."""
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x ** 3))))


class Level1ObservationEncoder:
    """
    E^(1): Encodes visual observation + proprioceptive joint/end-effector state
    into fine-grained spatial latent vector z^(1) in R^{dim_l1} (default 384, compatible with DINOv3).
    """
    def __init__(self, latent_dim: int = 384, state_input_dim: int = 18):
        self.latent_dim = latent_dim
        self.state_input_dim = state_input_dim

        rng = np.random.RandomState(101)
        # Proprioceptive MLP: R^18 -> R^128
        self.W_state1 = rng.randn(state_input_dim, 128) * math.sqrt(2.0 / state_input_dim)
        self.b_state1 = np.zeros(128, dtype=np.float32)
        self.W_state2 = rng.randn(128, 128) * math.sqrt(2.0 / 128)
        self.b_state2 = np.zeros(128, dtype=np.float32)

        # Visual patch feature projection (or State+Crop projection): R^128 -> R^256
        self.W_vis = rng.randn(128, 256) * math.sqrt(2.0 / 128)
        self.b_vis = np.zeros(256, dtype=np.float32)

        # Fusion layer: [vis (256) + state (128)] = 384 -> 384
        self.W_fuse = rng.randn(384, latent_dim) * math.sqrt(2.0 / 384)
        self.b_fuse = np.zeros(latent_dim, dtype=np.float32)

    def extract_proprioceptive_vector(
        self,
        joints: Dict[str, float] | List[float],
        tcp_pos: List[float] | Tuple[float, float, float],
        gripper: float,
        src_pos: Optional[List[float]] = None,
        dst_pos: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Constructs the 18-dim kinematic & relative target vector."""
        vec = np.zeros(self.state_input_dim, dtype=np.float32)

        # 1. 5 joint angles
        if isinstance(joints, dict):
            j_list = [
                joints.get("baseYaw", 0.0),
                joints.get("shoulderPitch", 0.0),
                joints.get("elbowPitch", 0.0),
                joints.get("wristPitch", 0.0),
                joints.get("gripper", 0.0),
            ]
        else:
            j_list = list(joints)[:5]
            while len(j_list) < 5:
                j_list.append(0.0)
        vec[0:5] = j_list

        # 2. TCP 3D position
        vec[5:8] = tcp_pos[:3]
        # 3. Gripper aperture
        vec[8] = gripper

        # 4. Relative vector to source object (p_src - x_ee)
        src = tcp_pos if src_pos is None else src_pos
        vec[9:12] = [src[0] - tcp_pos[0], src[1] - tcp_pos[1], src[2] - tcp_pos[2]]

        # 5. Relative vector to destination (p_dst - p_src)
        dst = STACK_ORIGIN if dst_pos is None else dst_pos
        vec[12:15] = [dst[0] - src[0], dst[1] - src[1], dst[2] - src[2]]

        # 6. Relative distance to destination from TCP
        vec[15:18] = [dst[0] - tcp_pos[0], dst[1] - tcp_pos[1], dst[2] - tcp_pos[2]]

        return vec

    def extract_lightweight_visual_patch(
        self,
        rgb_frame: Optional[np.ndarray],
        tcp_pos: List[float] | Tuple[float, float, float],
        target_pos: Optional[List[float]] = None,
    ) -> np.ndarray:
        """
        Lightweight State-Plus-Patch feature extractor running in <0.5 ms on CPU.
        Extracts color histogram / spatial centroid descriptors when DINOv3 is running in CPU mode.
        """
        vis_feat = np.zeros(128, dtype=np.float32)
        if rgb_frame is not None and rgb_frame.size > 0:
            # Downsampled spatial grid pooling
            h, w = rgb_frame.shape[:2]
            # 4x4 spatial pooled quadrants
            grid_h = max(1, h // 4)
            grid_w = max(1, w // 4)
            idx = 0
            for r in range(4):
                for c in range(4):
                    if idx + 3 <= 128:
                        patch = rgb_frame[r*grid_h:(r+1)*grid_h, c*grid_w:(c+1)*grid_w]
                        if patch.size > 0:
                            m = np.mean(patch, axis=(0, 1)) / 255.0
                            vis_feat[idx:idx+3] = m[:3]
                        idx += 3
        else:
            # Synthetic patch encoding from spatial targets
            tgt = STACK_ORIGIN if target_pos is None else target_pos
            vis_feat[0:3] = tcp_pos[:3]
            vis_feat[3:6] = tgt[:3]
            vis_feat[6] = math.hypot(tcp_pos[0] - tgt[0], tcp_pos[2] - tgt[2])

        return vis_feat

    def encode(
        self,
        joints: Dict[str, float] | List[float],
        tcp_pos: List[float] | Tuple[float, float, float],
        gripper: float = 0.0,
        src_pos: Optional[List[float]] = None,
        dst_pos: Optional[List[float]] = None,
        rgb_frame: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Encodes state + visual input into z^(1) in R^{latent_dim}."""
        s_vec = self.extract_proprioceptive_vector(joints, tcp_pos, gripper, src_pos, dst_pos)
        v_vec = self.extract_lightweight_visual_patch(rgb_frame, tcp_pos, dst_pos)

        # Proprioceptive MLP
        h_s = gelu(s_vec @ self.W_state1 + self.b_state1)
        h_s = gelu(h_s @ self.W_state2 + self.b_state2)

        # Visual projection
        h_v = gelu(v_vec @ self.W_vis + self.b_vis)

        # Fusion
        fused = np.concatenate([h_v, h_s], axis=-1)
        z1 = gelu(fused @ self.W_fuse + self.b_fuse)
        z1 = layer_norm(z1)

        # Unit sphere normalization
        norm = np.linalg.norm(z1)
        if norm > 1e-6:
            z1 = z1 / norm
        return z1.astype(np.float32)


class Level1ActionEncoder:
    """A^(1): Encodes continuous 5-DOF delta action (dx, dy, dz, dpitch, gripper) into R^32."""
    def __init__(self, action_dim: int = 5, latent_dim: int = 32):
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        rng = np.random.RandomState(102)
        self.W = rng.randn(action_dim, latent_dim) * math.sqrt(2.0 / action_dim)
        self.b = np.zeros(latent_dim, dtype=np.float32)

    def encode(self, action: np.ndarray | List[float]) -> np.ndarray:
        act = np.asarray(action, dtype=np.float32)
        if act.shape[-1] < self.action_dim:
            pad_shape = list(act.shape[:-1]) + [self.action_dim - act.shape[-1]]
            act = np.concatenate([act, np.zeros(pad_shape, dtype=np.float32)], axis=-1)
        elif act.shape[-1] > self.action_dim:
            act = act[..., :self.action_dim]

        a_emb = gelu(act @ self.W + self.b)
        norm = np.linalg.norm(a_emb, axis=-1, keepdims=True)
        norm = np.maximum(norm, 1e-6)
        return (a_emb / norm).astype(np.float32)

    def encode_batch(self, actions: np.ndarray) -> np.ndarray:
        return self.encode(actions)



class Level2ObservationEncoder:
    """
    E^(2): Object-centric scene graph encoder into Level 2 Macro Latent Embedding z^(2) in R^{latent_dim} (default 512).
    Encodes positions, layer tiers, stability geometry, and relative clearances of all blocks.
    """
    def __init__(self, latent_dim: int = 512, max_blocks: int = 7):
        self.latent_dim = latent_dim
        self.max_blocks = max_blocks
        self.raw_dim = max_blocks * 11 + 7  # 7 * 11 + 7 = 84

        rng = np.random.RandomState(103)
        self.W1 = rng.randn(self.raw_dim, 256) * math.sqrt(2.0 / self.raw_dim)
        self.b1 = np.zeros(256, dtype=np.float32)
        self.W2 = rng.randn(256, latent_dim) * math.sqrt(2.0 / 256)
        self.b2 = np.zeros(latent_dim, dtype=np.float32)

    def extract_scene_vector(
        self,
        blocks: List[Dict[str, Any]],
        tcp_pos: Optional[List[float]] = None,
        target_origin: Optional[List[float]] = None,
        grasped_id: Optional[str] = None,
    ) -> np.ndarray:
        origin = target_origin or STACK_ORIGIN
        tcp = tcp_pos or [0.0, 0.8, 0.0]

        feats = [tcp[0], tcp[1], tcp[2], origin[0], origin[1], origin[2], float(len(blocks))]

        sorted_blocks = sorted(blocks, key=lambda b: str(b.get("id", "")))
        for i in range(self.max_blocks):
            if i < len(sorted_blocks):
                b = sorted_blocks[i]
                pos = b.get("position", [0.0, TABLE_TOP_Y, 0.0])
                rot = b.get("rotation", [0.0, 0.0, 0.0, 1.0])
                in_pad = 1.0 if b.get("in_pad") else 0.0
                is_grasped = 1.0 if (b.get("id") == grasped_id or b.get("is_grasped")) else 0.0
                dx = pos[0] - origin[0]
                dz = pos[2] - origin[2]
                dist_pad = math.hypot(dx, dz)
                layer = round(max(0.0, pos[1] - TABLE_TOP_Y) / max(0.001, CUBE_SIZE))

                feats.extend([
                    pos[0], pos[1], pos[2],
                    rot[0], rot[1], rot[2], rot[3],
                    in_pad, is_grasped, dist_pad, float(layer)
                ])
            else:
                feats.extend([0.0] * 11)

        return np.asarray(feats, dtype=np.float32)

    def encode(
        self,
        blocks: List[Dict[str, Any]],
        tcp_pos: Optional[List[float]] = None,
        target_origin: Optional[List[float]] = None,
        grasped_id: Optional[str] = None,
    ) -> np.ndarray:
        raw_vec = self.extract_scene_vector(blocks, tcp_pos, target_origin, grasped_id)
        h = gelu(raw_vec @ self.W1 + self.b1)
        z2 = gelu(h @ self.W2 + self.b2)
        z2 = layer_norm(z2)
        norm = np.linalg.norm(z2)
        if norm > 1e-6:
            z2 = z2 / norm
        return z2.astype(np.float32)


class ProjectionHead1to2:
    """
    P_{1->2}: Cross-Level Projection Head mapping Level 1 latent state z^(1)
    into Level 2 subgoal space z^(2) for distance and energy evaluation.
    """
    def __init__(self, dim_l1: int = 384, dim_l2: int = 512):
        self.dim_l1 = dim_l1
        self.dim_l2 = dim_l2

        rng = np.random.RandomState(104)
        self.W = rng.randn(dim_l1, dim_l2) * math.sqrt(2.0 / dim_l1)
        self.b = np.zeros(dim_l2, dtype=np.float32)

    def project(self, z1: np.ndarray) -> np.ndarray:
        single = (z1.ndim == 1)
        if single:
            z1 = z1.reshape(1, -1)

        z2 = gelu(z1 @ self.W + self.b)
        norm = np.linalg.norm(z2, axis=-1, keepdims=True)
        norm = np.maximum(norm, 1e-6)
        z2 = z2 / norm

        if single:
            return z2[0].astype(np.float32)
        return z2.astype(np.float32)


class VLTextGoalEncoder:
    """
    E_txt: VL-JEPA Language Alignment Head.
    Maps structured task description, target layer, and color attributes into Level 2 goal embedding space R^{dim_l2}.
    """
    def __init__(self, latent_dim: int = 512):
        self.latent_dim = latent_dim
        rng = np.random.RandomState(105)
        self.W = rng.randn(64, latent_dim) * math.sqrt(2.0 / 64)
        self.b = np.zeros(latent_dim, dtype=np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        text_lower = text.lower().strip()
        feats = np.zeros(64, dtype=np.float32)

        # Structure intent flags
        feats[0] = 1.0 if any(w in text_lower for w in ["tower", "stack", "vertical"]) else 0.0
        feats[1] = 1.0 if any(w in text_lower for w in ["pyramid", "stepped", "triangle"]) else 0.0
        feats[2] = 1.0 if any(w in text_lower for w in ["train", "line", "row"]) else 0.0
        feats[3] = 1.0 if any(w in text_lower for w in ["1", "one"]) else 0.0
        feats[4] = 1.0 if any(w in text_lower for w in ["2", "two"]) else 0.0
        feats[5] = 1.0 if any(w in text_lower for w in ["3", "three"]) else 0.0
        feats[6] = 1.0 if any(w in text_lower for w in ["4", "four"]) else 0.0
        feats[7] = 1.0 if any(w in text_lower for w in ["5", "five"]) else 0.0
        feats[8] = 1.0 if any(w in text_lower for w in ["6", "six"]) else 0.0
        feats[9] = 1.0 if any(w in text_lower for w in ["7", "seven", "all"]) else 0.0

        # Color flags
        colors = ["red", "blue", "green", "yellow", "cyan", "purple", "orange", "pink", "magenta", "emerald"]
        for idx, col in enumerate(colors):
            if col in text_lower:
                feats[10 + idx] = 1.0

        # N-gram hash slots
        tokens = text_lower.split()
        for tok in tokens:
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            slot = 20 + (h % 44)
            feats[slot] += 1.0

        norm = np.linalg.norm(feats)
        if norm > 1e-6:
            feats = feats / norm

        z_txt = gelu(feats @ self.W + self.b)
        norm_z = np.linalg.norm(z_txt)
        if norm_z > 1e-6:
            z_txt = z_txt / norm_z
        return z_txt.astype(np.float32)
