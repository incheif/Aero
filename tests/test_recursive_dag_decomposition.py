"""
Tests for Recursive Subgoal DAG Decomposition & Task Decomposer.
Validates:
- Complex task decomposition across diverse topological architectures (bridge, pyramid, tower, cantilever).
- Automatic physical support dependency derivation.
- Obstacle occlusion detection & prerequisite clearance generation.
- Graph acyclicity guarantee via Kahn's algorithm.
- Ready subgoal retrieval and dependency resolution.
- Dynamic runtime recovery subgoal splicing.
"""

import pytest
import numpy as np

from vla.recursive_dag_planner import RecursiveDAGPlanner, Subgoal
from simulation.constants import TABLE_TOP_Y, CUBE_SIZE, CUBE_HALF, STACK_ORIGIN


@pytest.fixture
def sample_blocks():
    return [
        {"id": "block_magenta", "position": [0.15, 0.75, -0.10], "is_grasped": False},
        {"id": "block_cyan", "position": [0.25, 0.75, 0.05], "is_grasped": False},
        {"id": "block_yellow", "position": [0.30, 0.75, -0.15], "is_grasped": False},
        {"id": "block_emerald", "position": [0.18, 0.75, 0.12], "is_grasped": False},
        {"id": "block_orange", "position": [0.22, 0.75, -0.05], "is_grasped": False},
        {"id": "block_purple", "position": [0.28, 0.75, 0.0], "is_grasped": False},
    ]


def test_bridge_arch_decomposition(sample_blocks):
    planner = RecursiveDAGPlanner()
    subgoals = planner.decompose("Build a bridge with two pillars and a top span", sample_blocks)

    assert len(subgoals) == 3
    assert planner.is_acyclic()

    # Check that there are 2 base subgoals and 1 top subgoal
    levels = [s.level for s in subgoals]
    assert levels.count(1) == 2  # 2 pillars at level 1
    assert levels.count(2) == 1  # 1 spanning block at level 2

    # Top spanning block must depend on the pillars
    top_sg = [s for s in subgoals if s.level == 2][0]
    pillar_ids = [s.id for s in subgoals if s.level == 1]
    for p_id in pillar_ids:
        assert p_id in top_sg.dependencies

    # Ready subgoals initially should be only the 2 pillars
    ready = planner.get_ready_subgoals()
    assert len(ready) == 2
    assert set(s.id for s in ready) == set(pillar_ids)


def test_cantilever_decomposition(sample_blocks):
    planner = RecursiveDAGPlanner()
    subgoals = planner.decompose("Build a cantilever structure", sample_blocks)

    assert len(subgoals) == 2
    assert planner.is_acyclic()

    base_sg = subgoals[0]
    top_sg = subgoals[1]

    assert base_sg.level == 1
    assert top_sg.level == 2
    assert base_sg.id in top_sg.dependencies


def test_6_block_pyramid_hierarchy(sample_blocks):
    planner = RecursiveDAGPlanner()
    subgoals = planner.decompose("Assemble a 6 block stepped pyramid", sample_blocks)

    assert len(subgoals) == 6
    assert planner.is_acyclic()

    # Verify bottom-up layer ordering
    layers = [s.parameters["layer"] for s in subgoals]
    # In a 6-block pyramid: 3 base (layer 0), 2 middle (layer 1), 1 apex (layer 2)
    assert layers.count(0) == 3
    assert layers.count(1) == 2
    assert layers.count(2) == 1

    # Apex must depend on middle layer blocks
    apex = [s for s in subgoals if s.parameters["layer"] == 2][0]
    assert len(apex.dependencies) > 0


def test_obstacle_clearance_generation(sample_blocks):
    planner = RecursiveDAGPlanner()

    # Place an obstructing block (block_orange) directly at the stack origin slot
    ox, oz = STACK_ORIGIN[0], STACK_ORIGIN[2]
    oy = TABLE_TOP_Y + CUBE_HALF
    # Set block_orange position to target slot
    for b in sample_blocks:
        if b["id"] == "block_orange":
            b["position"] = [ox, oy, oz]

    subgoals = planner.decompose("Stack magenta and cyan blocks onto the target pad", sample_blocks)
    assert planner.is_acyclic()

    # A clear_obstacle subgoal must have been generated for block_orange
    clear_sgs = [s for s in subgoals if s.skill_type == "CLEAR_OBSTACLE"]
    assert len(clear_sgs) >= 1
    assert "block_orange" in clear_sgs[0].required_objects

    # The first deposit subgoal must depend on the clearance subgoal
    deposit_sgs = [s for s in subgoals if s.skill_type == "DEPOSIT"]
    assert clear_sgs[0].id in deposit_sgs[0].dependencies


def test_dynamic_recovery_splicing(sample_blocks):
    planner = RecursiveDAGPlanner()
    subgoals = planner.decompose("Stack 3 blocks into a column", sample_blocks)

    ready = planner.get_ready_subgoals()
    first_sg = ready[0]

    # Simulate a slip / failure on first subgoal
    planner.mark_failed(first_sg.id, reason="Slip detected during transit")
    assert first_sg.status == "RECOVERING"

    # Create and splice a dynamic recovery subgoal
    rec_sg = Subgoal(
        id=f"recover_{first_sg.id}",
        parent_id=first_sg.id,
        level=first_sg.level,
        skill_type="APPROACH_GRASP",
        required_objects=first_sg.required_objects,
        parameters={"re_localize": True},
        preconditions={"block_localized": True},
        desired_effects={"re_grasped": True},
        dependencies=[],
        description="Re-align and re-grasp dropped block",
    )
    planner.splice_recovery_subgoal(first_sg.id, rec_sg)

    assert planner.is_acyclic()
    assert rec_sg.id in planner.subgoals
    assert rec_sg.id in first_sg.dependencies

    # The recovery subgoal is now ready to execute
    ready_after = planner.get_ready_subgoals()
    assert any(s.id == rec_sg.id for s in ready_after)
