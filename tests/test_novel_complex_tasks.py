"""
tests/test_novel_complex_tasks.py

Exhaustive verification of Open-Ended Complex Task Execution
using H-JEPA World Model and Recursive Closed-Loop Subgoal DAGs.

Tests unhurd-of, novel complex tasks:
1. Letter 'D' formation ("make a letter D with blocks")
2. L-Shape formation ("arrange blocks in an L shape")
3. Triangle formation ("form a triangle of blocks on the table")
4. Cross / Plus formation ("make a plus sign with blocks")
5. Spatial chromatic sorting ("sort warm colors to the right and cool colors to the left")
6. Edge case: In-slot obstacle clearance prerequisite generation
7. Edge case: Real-time disturbance self-healing recovery splicing
8. Edge case: Impossible command cognitive rejection
"""

import pytest
import numpy as np
from typing import Dict, Any

from simulation.physics_engine import RobotArmSimulation
from simulation.constants import TABLE_TOP_Y, CUBE_SIZE, CUBE_HALF
from vla.spatial_geometry import SpatialGeometryEngine, is_within_reach, clamp_to_reach
from vla.instruction_interpreter import InstructionInterpreter
from vla.recursive_dag_planner import RecursiveDAGPlanner, Subgoal
from vla.closed_loop_recovery import ClosedLoopRecoveryOrchestrator


class TestSpatialGeometryEngine:
    """Verifies that the Open-Ended Spatial Geometry Engine synthesizes valid 3D layouts."""

    def test_letter_d_slots_geometry(self):
        slots = SpatialGeometryEngine.get_letter_slots("D", num_blocks=6)
        assert len(slots) == 6
        # All slots must be within physical reach
        for s in slots:
            assert is_within_reach(s[0], s[1], s[2]), f"Slot {s} out of reach"
            assert abs(s[1] - (TABLE_TOP_Y + CUBE_HALF)) < 0.01

        # Check that blocks do not overlap (min distance >= 40mm)
        for i in range(len(slots)):
            for j in range(i + 1, len(slots)):
                dist = math_dist(slots[i], slots[j])
                assert dist >= 0.040, f"Slots {i} and {j} overlap: dist={dist:.4f}m"

    def test_letter_l_slots_geometry(self):
        slots = SpatialGeometryEngine.get_letter_slots("L", num_blocks=5)
        assert len(slots) == 5
        for s in slots:
            assert is_within_reach(s[0], s[1], s[2])

    def test_shape_triangle_geometry(self):
        slots = SpatialGeometryEngine.get_shape_slots("triangle", num_blocks=3)
        assert len(slots) == 3
        for s in slots:
            assert is_within_reach(s[0], s[1], s[2])

    def test_shape_cross_geometry(self):
        slots = SpatialGeometryEngine.get_shape_slots("cross", num_blocks=5)
        assert len(slots) == 5
        for s in slots:
            assert is_within_reach(s[0], s[1], s[2])

    def test_warm_cool_sorting_geometry(self):
        sim = RobotArmSimulation()
        snap = sim.get_snapshot()
        slots, ordered_blocks = SpatialGeometryEngine.get_sorting_slots("warm_cool", snap["blocks"])
        assert len(slots) == 6
        assert len(ordered_blocks) == 6
        # Warm blocks must have positive Z (+0.08 to +0.25)
        for b_id, s in zip(ordered_blocks, slots):
            if b_id in ("block_magenta", "block_orange", "block_yellow"):
                assert s[2] > 0.0, f"Warm block {b_id} should have positive Z, got {s[2]}"
            elif b_id in ("block_cyan", "block_emerald", "block_purple"):
                assert s[2] < 0.0, f"Cool block {b_id} should have negative Z, got {s[2]}"


class TestOpenEndedInstructionParsing:
    """Verifies that the VLA Cognitive Interpreter correctly parses novel commands."""

    def setup_method(self):
        self.interpreter = InstructionInterpreter()

    def test_parse_letter_d_command(self):
        cmd = "make a letter D with blocks"
        res = self.interpreter.interpret(cmd)
        assert res["is_possible"] is True
        assert res["status"] == "OK"
        assert res["action_type"] == "BUILD_LETTER"
        assert res["letter"] == "D"

    def test_parse_letter_l_command(self):
        cmd = "arrange blocks in an L shape"
        res = self.interpreter.interpret(cmd)
        assert res["is_possible"] is True
        assert res["status"] == "OK"
        assert res["action_type"] == "BUILD_LETTER"
        assert res["letter"] == "L"

    def test_parse_triangle_shape_command(self):
        cmd = "form an equilateral triangle of blocks on the table"
        res = self.interpreter.interpret(cmd)
        assert res["is_possible"] is True
        assert res["status"] == "OK"
        assert res["action_type"] == "BUILD_SHAPE"
        assert res["shape_name"] == "triangle"

    def test_parse_cross_shape_command(self):
        cmd = "make a plus sign with blocks"
        res = self.interpreter.interpret(cmd)
        assert res["is_possible"] is True
        assert res["status"] == "OK"
        assert res["action_type"] == "BUILD_SHAPE"
        assert res["shape_name"] in ("plus", "cross")

    def test_parse_chromatic_sorting_command(self):
        cmd = "sort warm colors to the right and cool colors to the left"
        res = self.interpreter.interpret(cmd)
        assert res["is_possible"] is True
        assert res["status"] == "OK"
        assert res["action_type"] == "SPATIAL_SORT"
        assert res["sort_mode"] == "warm_cool"

    def test_reject_impossible_command(self):
        cmd = "crush the red block into dust"
        res = self.interpreter.interpret(cmd)
        assert res["is_possible"] is False
        assert res["status"] == "IMPOSSIBLE"


class TestRecursiveDAGExecutionNovelTasks:
    """Verifies DAG decomposition and execution of novel complex tasks."""

    def test_dag_decomposes_letter_d(self):
        sim = RobotArmSimulation()
        snap = sim.get_snapshot()
        planner = RecursiveDAGPlanner()
        subgoals = planner.decompose("make a letter D with blocks", snap["blocks"])

        # Must have exactly 6 deposit subgoals for the 6 blocks forming letter D
        deposit_sgs = [sg for sg in subgoals if sg.skill_type == "DEPOSIT"]
        assert len(deposit_sgs) == 6
        subgoal_ids = [sg.id for sg in subgoals]
        assert len(set(subgoal_ids)) == len(subgoals)
        # Initial ready subgoals should be available
        ready = planner.get_ready_subgoals()
        assert len(ready) > 0

    def test_dag_decomposes_cross_pattern(self):
        sim = RobotArmSimulation()
        snap = sim.get_snapshot()
        planner = RecursiveDAGPlanner()
        subgoals = planner.decompose("make a plus sign with blocks", snap["blocks"])
        deposit_sgs = [sg for sg in subgoals if sg.skill_type == "DEPOSIT"]
        assert len(deposit_sgs) == 6
        for sg in deposit_sgs:
            assert "target_slot" in sg.parameters

    def test_dag_obstacle_clearance_prerequisite(self):
        """
        Edge Case: Target slot is occupied by another unassigned block.
        The planner must generate a CLEAR_OBSTACLE prerequisite subgoal first.
        """
        sim = RobotArmSimulation()
        snap = sim.get_snapshot()
        # Find where block_magenta currently sits
        mag_block = next(b for b in snap["blocks"] if b["id"] == "block_magenta")
        mag_pos = mag_block["position"]

        planner = RecursiveDAGPlanner()
        # Request placing block_cyan exactly at block_magenta's current coordinates
        custom_slots = [(mag_pos[0], mag_pos[1], mag_pos[2])]
        subgoals = planner.decompose(
            "custom placement",
            snap["blocks"],
            custom_slots=custom_slots,
            preferred_block_ids=["block_cyan"],
        )

        # Must have generated a CLEAR_OBSTACLE subgoal for block_magenta
        clear_sgs = [sg for sg in subgoals if sg.skill_type == "CLEAR_OBSTACLE"]
        assert len(clear_sgs) >= 1
        assert clear_sgs[0].required_objects == ["block_magenta"]
        # The cyan deposit must depend on clearing magenta
        deposit_cyan = next(sg for sg in subgoals if "block_cyan" in sg.id)
        assert clear_sgs[0].id in deposit_cyan.dependencies

    def test_orchestrator_tick_by_tick_lifecycle(self):
        """Verifies start_task() and step_task() on the ClosedLoopRecoveryOrchestrator."""
        sim = RobotArmSimulation()
        orch = ClosedLoopRecoveryOrchestrator()
        started = orch.start_task(sim, "make a letter D with blocks")
        assert started is True
        assert orch.is_task_active is True
        deposit_sgs = [sg for sg in orch.planner.subgoals.values() if sg.skill_type == "DEPOSIT"]
        assert len(deposit_sgs) == 6

        # Step a few ticks
        for _ in range(15):
            res = orch.step_task(sim)
            assert res["is_active"] is True
            assert isinstance(res["residual"], float)

        orch.stop_task()
        assert orch.is_task_active is False


def math_dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

import math
