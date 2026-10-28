"""
vla/jepa/schemas.py

Typed data contracts and schemas for the Two-Level Hierarchical VL-JEPA Architecture.
Defines representation interfaces between Level 2 (Macro Subgoal Planner),
Level 1 (Action-Conditioned Latent World Model & CEM Planner),
OutcomeVerifier, and the robotic execution stack.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Any
import numpy as np


@dataclass
class ObjectLatentState:
    """Grounded 3D object state fused with visual patch embedding."""
    label: str
    color: str
    position_xyz: Tuple[float, float, float]
    half_extents: Tuple[float, float, float]
    confidence: float
    support_object_label: Optional[str] = None       # None if resting on table (y == 0)
    visual_embedding: Optional[np.ndarray] = None    # Shape: (D_vis,) e.g. (384,) from DINOv3
    is_grasped: bool = False
    in_pad: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "color": self.color,
            "position_xyz": list(self.position_xyz),
            "half_extents": list(self.half_extents),
            "confidence": float(self.confidence),
            "support_object_label": self.support_object_label,
            "is_grasped": self.is_grasped,
            "in_pad": self.in_pad,
            "visual_embedding_dim": len(self.visual_embedding) if self.visual_embedding is not None else 0,
        }


@dataclass
class WorkspaceObservation:
    """Complete multi-modal observation passed to Level 1 and Level 2 encoders."""
    step_id: int
    sim_time: float
    overhead_rgb: Optional[np.ndarray] = None                  # Shape: (H, W, 3), uint8
    gripper_rgb: Optional[np.ndarray] = None                   # Shape: (H, W, 3), uint8
    joint_angles_rad: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    end_effector_xyz: Tuple[float, float, float] = (0.0, 0.8, 0.0)
    gripper_open: bool = True
    gripper_aperture: float = 0.0
    grasped_object_label: Optional[str] = None
    objects: Dict[str, ObjectLatentState] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "sim_time": self.sim_time,
            "joint_angles_rad": list(self.joint_angles_rad),
            "end_effector_xyz": list(self.end_effector_xyz),
            "gripper_open": self.gripper_open,
            "gripper_aperture": self.gripper_aperture,
            "grasped_object_label": self.grasped_object_label,
            "objects": {k: v.to_dict() for k, v in self.objects.items()},
        }


@dataclass
class SubgoalSpecification:
    """Contract emitted by Level 2 and consumed by Level 1."""
    subgoal_id: int
    action_type: Literal["pick_and_place", "stack", "push", "clear_obstacle"]
    source_label: str
    source_color: str
    target_xyz: Tuple[float, float, float]
    support_label: Optional[str] = None              # Expected object directly underneath after placement
    target_latent_l2: Optional[np.ndarray] = None    # z^(2)_{goal, k} in R^{512} or R^{128}
    predicted_stability_prob: float = 1.0            # h_stab(z^(2)_{goal, k}) in [0, 1]
    position_tolerance_m: float = 0.038
    description: str = ""

    def __getitem__(self, item: str) -> Any:
        if item in ("block_id", "source_label"):
            return self.source_label
        elif item in ("target_slot", "target_xyz"):
            return self.target_xyz
        elif item in ("z_sub", "target_latent_l2"):
            return self.target_latent_l2
        elif item in ("subgoal_index", "subgoal_id"):
            return self.subgoal_id
        elif hasattr(self, item):
            return getattr(self, item)
        raise KeyError(item)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subgoal_id": self.subgoal_id,
            "action_type": self.action_type,
            "source_label": self.source_label,
            "source_color": self.source_color,
            "target_xyz": list(self.target_xyz),
            "support_label": self.support_label,
            "predicted_stability_prob": float(self.predicted_stability_prob),
            "position_tolerance_m": float(self.position_tolerance_m),
            "description": self.description,
        }


@dataclass
class CEMPlanResult:
    """Output of Level 1 Action-Conditioned World Model + CEM Planner."""
    subgoal_id: int
    optimal_actions: np.ndarray               # Shape: (H, 5) -> [dx, dy, dz, pitch, gripper]
    predicted_latents_l1: np.ndarray          # Shape: (H, D1) -> \hat{z}^(1)_{t+1:t+H}
    terminal_goal_energy: float               # || P_{1->2}(\hat{z}^(1)_{t+H}) - z^(2)_goal ||_2
    predicted_stability: float                # Predicted post-release stability in [0, 1]
    candidate_trajectories_xyz: List[List[Tuple[float, float, float]]] = field(default_factory=list)
    ik_feasible: bool = True
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subgoal_id": self.subgoal_id,
            "optimal_action_step0": [float(v) for v in self.optimal_actions[0]] if len(self.optimal_actions) > 0 else [],
            "terminal_goal_energy": float(self.terminal_goal_energy),
            "predicted_stability": float(self.predicted_stability),
            "ik_feasible": self.ik_feasible,
            "latency_ms": float(self.latency_ms),
            "num_candidates": len(self.candidate_trajectories_xyz),
        }


@dataclass
class VerificationReport:
    """Closed-loop outcome check emitted by OutcomeVerifier after action execution."""
    subgoal_id: int
    phase_index: int
    latent_prediction_error: float            # e_pred = || \hat{z}^(1)_{t+1} - z^{(1)}_{t+1} ||_2
    subgoal_energy_distance: float            # e_goal = || P_{1->2}(z^{(1)}_{t+1}) - z^(2)_goal ||_2
    physical_contact_verified: bool           # True if block is grasped or resting stably on target
    tower_intact: bool                        # False if any previously placed block toppled
    status: Literal["IN_PROGRESS", "SUBGOAL_COMPLETED", "RETRY_L1", "ESCALATE_L2_REPLAN"]
    failure_reason: Optional[str] = None
    hold_ticks: int = 0
    stability_score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subgoal_id": self.subgoal_id,
            "phase_index": self.phase_index,
            "latent_prediction_error": float(self.latent_prediction_error),
            "subgoal_energy_distance": float(self.subgoal_energy_distance),
            "physical_contact_verified": self.physical_contact_verified,
            "tower_intact": self.tower_intact,
            "status": self.status,
            "failure_reason": self.failure_reason,
            "hold_ticks": self.hold_ticks,
            "stability_score": float(self.stability_score),
        }
