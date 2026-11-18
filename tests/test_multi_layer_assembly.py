"""
tests/test_multi_layer_assembly.py

Comprehensive tests for multi-layer structure assembly and unstacking continuation:
- Natural language parsing of multi-tier requests ("2 blocks in the first layer and 2 blocks on top of it")
- Recursive DAG decomposition with physical prerequisite support dependencies
- Physical execution without premature completion or zero-action false positives
"""

import pytest
from simulation.physics_engine import RobotArmSimulation
from simulation.constants import TABLE_TOP_Y, CUBE_SIZE, CUBE_HALF
from vla.spatial_geometry import parse_multi_layer_spec, SpatialGeometryEngine
from vla.instruction_interpreter import InstructionInterpreter
from vla.recursive_dag_planner import RecursiveDAGPlanner
from vla.closed_loop_recovery import ClosedLoopRecoveryOrchestrator
from vla.google_vla_agent import GoogleVLAAgent


class TestMultiLayerNLU:
    """Verifies that natural multi-tier structure commands are parsed accurately."""

    @pytest.mark.parametrize("cmd,expected_counts", [
        ("make a structure with 2 blocks in the first layer and 2 blocks on top of it", [2, 2]),
        ("make a structure with 2 blocks in the 1st layer and 2 blocks on top of it", [2, 2]),
        ("build a structure using 4 blocks, 1st layer with 2 blocks and 2nd layer with 2 blocks", [2, 2]),
        ("2 blocks on bottom layer and 2 blocks on top layer", [2, 2]),
        ("2 blocks in the bottom layer and 2 blocks on top", [2, 2]),
        ("2 blocks on the bottom and 2 blocks on top", [2, 2]),
        ("build 2 layers of 2 blocks each", [2, 2]),
        ("make a 2x2 wall of blocks", [2, 2]),
        ("make a 2 by 2 structure of blocks", [2, 2]),
        ("structure with 3 blocks in the first layer and 1 block on top of it", [3, 1]),
        ("structure with 3 blocks in the bottom layer and 2 blocks on top", [3, 2]),
    ])
    def test_parse_multi_layer_spec_phrasings(self, cmd, expected_counts):
        counts = parse_multi_layer_spec(cmd)
        assert counts == expected_counts, f"Failed for {cmd!r}: got {counts}, expected {expected_counts}"

    def test_interpreter_identifies_multi_layer_action_type(self):
        interp = InstructionInterpreter()
        res = interp.interpret("make a structure with 2 blocks in the first layer and 2 blocks on top of it")
        assert res["is_possible"] is True
        assert res["status"] == "OK"
        assert res["action_type"] == "MULTI_LAYER_STRUCTURE"
        assert res["layer_counts"] == [2, 2]
        assert "2-layer structure" in res["thought"]


class TestMultiLayerDAGPlanner:
    """Verifies that RecursiveDAGPlanner synthesizes 4 distinct, collision-free 3D slots with dependencies."""

    def test_four_block_dag_decomposition(self):
        planner = RecursiveDAGPlanner()
        blocks = [
            {"id": "block_magenta", "position": [0.26, 0.7535, 0.12]},
            {"id": "block_cyan", "position": [0.24, 0.7535, -0.18]},
            {"id": "block_yellow", "position": [0.21, 0.7535, -0.05]},
            {"id": "block_emerald", "position": [0.22, 0.7535, 0.24]},
            {"id": "block_orange", "position": [0.34, 0.7535, -0.06]},
            {"id": "block_purple", "position": [0.2, 0.7535, -0.27]},
        ]
        cmd = "make a structure with 2 blocks in the first layer and 2 blocks on top of it"
        subgoals = planner.decompose(cmd, blocks)

        assert len(subgoals) == 4
        # Two base blocks (layer 0) on table plane
        base_sgs = [sg for sg in subgoals if abs(sg.parameters["target_slot"][1] - (TABLE_TOP_Y + CUBE_HALF)) < 0.01]
        top_sgs = [sg for sg in subgoals if abs(sg.parameters["target_slot"][1] - (TABLE_TOP_Y + CUBE_HALF + CUBE_SIZE)) < 0.01]

        assert len(base_sgs) == 2, f"Expected 2 base layer subgoals, got {len(base_sgs)}"
        assert len(top_sgs) == 2, f"Expected 2 top layer subgoals, got {len(top_sgs)}"

        # Base blocks have no dependencies
        for b_sg in base_sgs:
            assert len(b_sg.dependencies) == 0

        # Top blocks depend on base blocks
        for t_sg in top_sgs:
            assert len(t_sg.dependencies) > 0
            # Every dependency must reference a base subgoal
            for dep in t_sg.dependencies:
                assert dep in [b.id for b in base_sgs]

    def test_four_block_assembly_from_existing_six_block_tower(self):
        """When an existing 6-block tower is present, blocks must be unstacked first before assembling the 2x2."""
        planner = RecursiveDAGPlanner()
        blocks = [
            {"id": "block_magenta", "position": [0.45, 0.7475, 0.2]},
            {"id": "block_cyan", "position": [0.45, 0.8025, 0.2]},
            {"id": "block_yellow", "position": [0.45, 0.8575, 0.2]},
            {"id": "block_emerald", "position": [0.45, 0.9125, 0.2]},
            {"id": "block_orange", "position": [0.45, 0.9675, 0.2]},
            {"id": "block_purple", "position": [0.45, 1.0225, 0.2]},
        ]
        cmd = "make a structure with 2 blocks in the first layer and 2 blocks on top of it"
        subgoals = planner.decompose(cmd, blocks)

        # Must have clearance subgoals for occluded blocks
        clear_sgs = [sg for sg in subgoals if sg.skill_type == "CLEAR_OBSTACLE"]
        deposit_sgs = [sg for sg in subgoals if sg.skill_type == "DEPOSIT"]

        assert len(clear_sgs) > 0, "Expected obstacle clearance subgoals for stacked blocks"
        assert len(deposit_sgs) == 4, "Expected 4 deposit subgoals for the 2x2 structure"


class TestMultiLayerPhysicalExecution:
    """Verifies that orchestrator compiles and steps multi-layer plans without premature completion."""

    def test_orchestrator_initializes_four_subgoals(self):
        sim = RobotArmSimulation()
        orch = ClosedLoopRecoveryOrchestrator()
        cmd = "make a structure with 2 blocks in the first layer and 2 blocks on top of it"
        started = orch.start_task(sim, cmd)
        assert started is True
        assert len(orch.planner.subgoals) == 4

        # Verify not immediately completed (did not falsely claim success)
        assert orch.is_task_active is True
        assert orch.task_success is False

    def test_google_vla_agent_parses_multi_layer_plan(self):
        agent = GoogleVLAAgent()
        snap = {
            "block_magenta": {"id": "block_magenta", "name": "Red Cube", "color": "#FF1E27", "position": [0.26, 0.7535, 0.12]},
            "block_cyan": {"id": "block_cyan", "name": "Blue Cube", "color": "#1E60FF", "position": [0.24, 0.7535, -0.18]},
            "block_yellow": {"id": "block_yellow", "name": "Yellow Cube", "color": "#FFD700", "position": [0.21, 0.7535, -0.05]},
            "block_emerald": {"id": "block_emerald", "name": "Green Cube", "color": "#00FF66", "position": [0.22, 0.7535, 0.24]},
            "block_orange": {"id": "block_orange", "name": "Orange Cube", "color": "#FF5500", "position": [0.34, 0.7535, -0.06]},
            "block_purple": {"id": "block_purple", "name": "Purple Cube", "color": "#800080", "position": [0.2, 0.7535, -0.27]},
        }
        cmd = "make a structure with 2 blocks in the first layer and 2 blocks on top of it"
        res = agent.parse_instruction(cmd, snapshot=snap)
        assert res["is_possible"] is True
        assert res["status"] == "OK"
        # Grounding should contain 4 structural blocks across 2 layers (plus any staging clearances)
        grounding = res.get("grounding", [])
        structure_blocks = [g for g in grounding if g.get("role") != "Step Clearance Staging"]
        assert len(structure_blocks) == 4
        layers = [g.get("target_layer") for g in structure_blocks]
        assert layers.count(0) == 2
        assert layers.count(1) == 2
