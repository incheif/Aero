"""
vla/h_jepa package

Two-Level Hierarchical Joint-Embedding Predictive Architecture (H-JEPA)
with VL-JEPA Language Grounding Layer and Closed-Loop Skill Execution.
"""

from .regularization import sigreg_loss, invariance_loss, variance_loss, covariance_loss
from .encoders import (
    Level2ObservationEncoder,
    Level2ActionEncoder,
    Level1ObservationEncoder,
    Level1ActionEncoder,
    ProjectionHead1to2,
)
from .predictors import Level2Predictor, Level1Predictor
from .vl_grounder import VLJEPAGrounder
from .cem_planner import Level1MicroCEMPlanner, Level2MacroCEMPlanner
from .orchestrator import HierarchicalVLJEPAOrchestrator
from .motor_controller import PrecisionCartesianController
from .skills import BaseSkill, PickSkill, PlaceSkill, StackSkill
from .task_analyzer import TaskFeasibilityAnalyzer

__all__ = [
    "sigreg_loss",
    "invariance_loss",
    "variance_loss",
    "covariance_loss",
    "Level2ObservationEncoder",
    "Level2ActionEncoder",
    "Level1ObservationEncoder",
    "Level1ActionEncoder",
    "ProjectionHead1to2",
    "Level2Predictor",
    "Level1Predictor",
    "VLJEPAGrounder",
    "Level1MicroCEMPlanner",
    "Level2MacroCEMPlanner",
    "HierarchicalVLJEPAOrchestrator",
    "PrecisionCartesianController",
    "TaskFeasibilityAnalyzer",
    "BaseSkill",
    "PickSkill",
    "PlaceSkill",
    "StackSkill",
]
