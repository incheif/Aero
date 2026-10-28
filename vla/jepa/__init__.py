"""
vla/jepa package

Two-Level Hierarchical Joint-Embedding Predictive Architecture (H-VL-JEPA).
Implements the predictive world model, Level 2 macro planner, Level 1 micro CEM,
closed-loop outcome verifier, and stability prediction.
"""

from .schemas import (
    ObjectLatentState,
    WorkspaceObservation,
    SubgoalSpecification,
    CEMPlanResult,
    VerificationReport,
)
from .encoders import (
    Level1ObservationEncoder,
    Level1ActionEncoder,
    Level2ObservationEncoder,
    ProjectionHead1to2,
    VLTextGoalEncoder,
)
from .level1_world_model import Level1Predictor, Level1CEMPlanner
from .level2_world_model import Level2Predictor, StabilityProbe, Level2TaskPlanner
from .verifier import OutcomeVerifier
from .data_collector import HJEPADataCollector
from .train_hjepa import vicreg_loss, compute_hjepa_losses, HJEPATrainer
from .orchestrator import HierarchicalVLJEPAOrchestrator

__all__ = [
    "ObjectLatentState",
    "WorkspaceObservation",
    "SubgoalSpecification",
    "CEMPlanResult",
    "VerificationReport",
    "Level1ObservationEncoder",
    "Level1ActionEncoder",
    "Level2ObservationEncoder",
    "ProjectionHead1to2",
    "VLTextGoalEncoder",
    "Level1Predictor",
    "Level1CEMPlanner",
    "Level2Predictor",
    "StabilityProbe",
    "Level2TaskPlanner",
    "OutcomeVerifier",
    "HJEPADataCollector",
    "vicreg_loss",
    "compute_hjepa_losses",
    "HJEPATrainer",
    "HierarchicalVLJEPAOrchestrator",
]
