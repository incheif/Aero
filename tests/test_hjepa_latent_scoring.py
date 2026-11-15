"""
Tests for H-JEPA Level 2 Latent Action Scoring and Feasibility Evaluator.
Validates:
- Loading pre-trained 6-DOF H-JEPA model weights.
- Multi-step latent forward rollouts.
- Discrimination between goal-directed vs divergent action trajectories.
- Subgoal feasibility pre-validation in latent representation space.
"""

import pytest
import numpy as np

from simulation.mujoco_engine import MujocoSimulation
from simulation.constants import STACK_ORIGIN, TABLE_TOP_Y, CUBE_HALF
from vla.latent_action_scorer import LatentActionScorer
from vla.recursive_dag_planner import Subgoal


@pytest.fixture
def scorer():
    return LatentActionScorer(checkpoint_path="models/hjepa_6dof_world_model.npz")


@pytest.fixture
def sim():
    sim = MujocoSimulation()
    sim.reset(randomize=False)
    return sim


def test_scorer_checkpoint_loaded(scorer):
    assert scorer.is_loaded is True
    assert scorer.jepa.step_count > 0


def test_candidate_action_scoring_discrimination(scorer, sim):
    snap_0 = sim.get_snapshot()

    joints_cmd_good = {"baseYaw": 0.10, "shoulderPitch": -0.45, "elbowPitch": 0.85, "wristPitch": -0.35, "wristRoll": 0.0}
    joints_cmd_bad = {"baseYaw": -1.20, "shoulderPitch": 0.60, "elbowPitch": -0.50, "wristPitch": 0.50, "wristRoll": 1.0}

    sim_target = MujocoSimulation()
    sim_target.reset(randomize=False)
    sim_target.set_joint_targets(joints_cmd_good, 0.038)
    target_snap = sim_target.step(0.016)

    score_good = scorer.score_candidate_action(snap_0, joints_cmd_good, 0.038, target_snap)
    score_bad = scorer.score_candidate_action(snap_0, joints_cmd_bad, 0.038, target_snap)

    assert isinstance(score_good, float)
    assert isinstance(score_bad, float)
    assert score_good > score_bad  # Good action brings state closer to target in latent space


def test_subgoal_feasibility_evaluation(scorer, sim):
    snap = sim.get_snapshot()

    # Feasible subgoal
    feasible_sg = Subgoal(
        id="sg_feasible",
        parent_id="root",
        level=1,
        skill_type="DEPOSIT",
        required_objects=["block_cyan"],
        parameters={
            "block_id": "block_cyan",
            "target_slot": [STACK_ORIGIN[0], TABLE_TOP_Y + CUBE_HALF, STACK_ORIGIN[2]],
        },
        preconditions={},
        desired_effects={},
        dependencies=[],
    )

    is_feasible, conf = scorer.evaluate_subgoal_feasibility(snap, feasible_sg)
    assert is_feasible is True
    assert 0.0 < conf <= 1.0
