"""
tests/test_hjepa_edge_cases.py

Edge Case and Robustness Test Suite for H-JEPA + VLA Architecture:
- Empty scenes, missing objects, and vanished blocks.
- Unrecognized or malformed natural language instructions.
- Excessive block counts requested beyond physical availability.
- Acyclicity rejection under circular dependency injection.
- Retry budget preservation during dynamic DAG repair.
- State and action vector dimension invariance under missing telemetry.
- Zero-tick execution boundary conditions.
"""

import pytest
import numpy as np
from typing import Dict, Any

from simulation.constants import TABLE_TOP_Y, CUBE_HALF, SAFE_STAGING_SLOTS
from simulation.mujoco_engine import MujocoSimulation
from vla.predictive_world_model import (
    extract_state_vector_6dof,
    extract_action_vector_7d,
    HierarchicalJEPA,
)
from vla.recursive_dag_planner import RecursiveDAGPlanner, Subgoal
from vla.latent_action_scorer import LatentActionScorer, copy_and_update_snapshot
from vla.primitive_skills import PrimitiveSkillExecutor
from vla.closed_loop_recovery import ClosedLoopRecoveryOrchestrator


class TestHJEPAEdgeCases:
    """Verifies edge-case resilience and non-crashing boundary behaviors."""

    def test_01_empty_scene_handling(self):
        """Planner and orchestrator handle scenes with 0 blocks gracefully."""
        planner = RecursiveDAGPlanner()
        sgs = planner.decompose("Build a tower of 3 blocks", current_blocks=[])
        assert len(sgs) == 0

        sim = MujocoSimulation()
        sim.reset(randomize=False)
        sim.blocks.clear()  # Empty all blocks

        orch = ClosedLoopRecoveryOrchestrator()
        res = orch.run_task(sim, "Build a tower of 3 blocks", max_total_ticks=100)
        assert res["success"] is False
        msg = res["message"].lower()
        assert "failed to compile" in msg or "insufficient blocks" in msg or "no subgoals" in msg

    def test_02_unrecognized_instruction_fallback(self):
        """Unrecognized instructions degrade gracefully to default single pick-place."""
        sim = MujocoSimulation()
        sim.reset(randomize=False)
        blocks = sim.get_snapshot()["blocks"]

        planner = RecursiveDAGPlanner()
        sgs = planner.decompose("Execute something totally novel and abstract", current_blocks=blocks)
        assert len(sgs) == 1
        assert sgs[0].skill_type == "DEPOSIT"
        assert sgs[0].parameters["block_id"] in [b["id"] for b in blocks]

    def test_03_excessive_count_bounding(self):
        """Requests for more blocks than available are bounded safely."""
        sim = MujocoSimulation()
        sim.reset(randomize=False)
        blocks = sim.get_snapshot()["blocks"][:2]  # Only 2 blocks available

        planner = RecursiveDAGPlanner()
        sgs = planner.decompose("Build a train of six blocks", current_blocks=blocks)
        assert len(sgs) == 2  # Bounded to available count

    def test_04_vanished_block_execution(self):
        """Attempting to manipulate a missing block fails gracefully without exception."""
        sim = MujocoSimulation()
        sim.reset(randomize=False)
        executor = PrimitiveSkillExecutor()

        dummy_sg = Subgoal(
            id="subgoal_ghost",
            parent_id="root",
            level=1,
            skill_type="DEPOSIT",
            required_objects=["ghost_block"],
            parameters={
                "block_id": "ghost_block",
                "target_slot": [0.4, 0.75, 0.0],
                "layer": 0,
            },
            preconditions={"block_present": True},
            desired_effects={},
            dependencies=[],
        )

        ok, msg, data = executor.execute_subgoal(sim, dummy_sg)
        assert ok is False
        assert "Failed to generate plan" in msg or "missing" in msg.lower() or "disappeared" in msg.lower()

    def test_05_clear_obstacle_pseudo_target_update(self):
        """copy_and_update_snapshot correctly handles staging_slot for CLEAR_OBSTACLE."""
        snap = {
            "blocks": [
                {"id": "block_orange", "position": [0.34, 0.75, -0.06], "is_grasped": False}
            ]
        }
        clear_sg = Subgoal(
            id="clear_orange",
            parent_id="root",
            level=1,
            skill_type="CLEAR_OBSTACLE",
            required_objects=["block_orange"],
            parameters={
                "block_id": "block_orange",
                "staging_slot": [0.20, 0.7475, 0.30],
            },
            preconditions={},
            desired_effects={},
            dependencies=[],
        )

        target_snap = copy_and_update_snapshot(snap, clear_sg)
        b = next(x for x in target_snap["blocks"] if x["id"] == "block_orange")
        assert np.allclose(b["position"], [0.20, 0.7475, 0.30])

    def test_06_zero_ticks_timeout(self):
        """max_total_ticks=0 terminates cleanly without hanging."""
        sim = MujocoSimulation()
        sim.reset(randomize=False)
        orch = ClosedLoopRecoveryOrchestrator()
        res = orch.run_task(sim, "Build a cantilever structure", max_total_ticks=0)
        assert res["success"] is False
        assert "timed out" in res["message"].lower() or "stalled" in res["message"].lower()

    def test_07_circular_dependency_rejection(self):
        """Kahn's algorithm detects and rejects injected circular dependency cycles."""
        planner = RecursiveDAGPlanner()
        sg1 = Subgoal(id="sg1", parent_id="root", level=1, skill_type="DEPOSIT",
                      required_objects=[], parameters={}, preconditions={}, desired_effects={}, dependencies=["sg2"])
        sg2 = Subgoal(id="sg2", parent_id="root", level=1, skill_type="DEPOSIT",
                      required_objects=[], parameters={}, preconditions={}, desired_effects={}, dependencies=["sg1"])
        planner.subgoals = {"sg1": sg1, "sg2": sg2}
        assert planner.is_acyclic() is False

    def test_08_retry_count_budget_enforcement(self):
        """Recovery splicing preserves single-increment retry counting."""
        planner = RecursiveDAGPlanner()
        sg = Subgoal(id="sg1", parent_id="root", level=1, skill_type="DEPOSIT",
                     required_objects=[], parameters={}, preconditions={}, desired_effects={}, dependencies=[], max_retries=2)
        planner.subgoals = {"sg1": sg}

        # First failure
        planner.mark_failed("sg1", "Failed step 1")
        assert sg.retry_count == 1
        assert sg.status == "RECOVERING"

        rec_sg = Subgoal(id="rec_sg1_r1", parent_id="sg1", level=1, skill_type="DEPOSIT",
                         required_objects=[], parameters={}, preconditions={}, desired_effects={}, dependencies=[])
        planner.splice_recovery_subgoal("sg1", rec_sg)
        assert sg.retry_count == 1  # Not double incremented!

        # Second failure (exhausted budget)
        planner.mark_failed("sg1", "Failed step 2")
        assert sg.retry_count == 2
        assert sg.status == "FAILED"

    def test_09_state_vector_dimension_invariance(self):
        """extract_state_vector_6dof maintains exact 112-D shape under degraded inputs."""
        empty_snap: Dict[str, Any] = {}
        vec1 = extract_state_vector_6dof(empty_snap)
        assert isinstance(vec1, np.ndarray)
        assert vec1.shape == (112,)
        assert not np.any(np.isnan(vec1))

        sparse_snap = {
            "joints": {"baseYaw": 0.1},
            "blocks": [{"id": "b1", "position": [0.1, 0.2, 0.3]}],
        }
        vec2 = extract_state_vector_6dof(sparse_snap)
        assert vec2.shape == (112,)
        assert not np.any(np.isnan(vec2))

    def test_10_action_vector_dimension_invariance(self):
        """extract_action_vector_7d maintains exact 7-D shape under partial commands."""
        act = extract_action_vector_7d({}, 0.5)
        assert isinstance(act, np.ndarray)
        assert act.shape == (7,)
        assert act[5] == 0.5
