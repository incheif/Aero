"""
tests/test_autonomous_deconstruction.py

Comprehensive test suite for Autonomous Deconstruction and Sequential Task Continuation.
Validates:
1. NLU Intent Disambiguation for unstacking/dismantling commands.
2. Top-down extraction sequence (highest Y descending) to safe perimeter staging slots.
3. Multi-turn task continuity: constructing a tower followed immediately by building a train/pyramid
   without manual workcell resets, requiring the arm to autonomously unstack occluding blocks first.
4. Recursive DAG Planner occlusion detection and dependency compilation.
"""

import pytest
import math
from typing import Dict, List, Any

from simulation.constants import (
    TABLE_TOP_Y,
    CUBE_HALF,
    CUBE_SIZE,
    STACK_ORIGIN,
    SAFE_STAGING_SLOTS,
)
from vla.instruction_interpreter import InstructionInterpreter
from vla.recursive_dag_planner import RecursiveDAGPlanner, Subgoal
from vla.google_vla_agent import GoogleVLAAgent


class TestAutonomousDeconstructionNLU:
    """Validates natural language understanding of deconstruction commands."""

    def setup_method(self):
        self.interpreter = InstructionInterpreter()
        self.interpreter.execution_mode = "deterministic"

    @pytest.mark.parametrize("instruction", [
        "unstack the tower",
        "bring the blocks down from the tower",
        "take down the tower",
        "dismantle the tower",
        "deconstruct the tower",
        "disassemble the structure",
        "bring the blocks down",
        "take the blocks down",
        "unstack the pyramid",
        "dismantle the pyramid",
        "tear down the blocks",
        "remove blocks from tower",
    ])
    def test_unstack_phrasings_classified_as_unstack_all(self, instruction: str):
        res = self.interpreter.interpret(instruction)
        assert res["is_possible"] is True, f"Failed for '{instruction}': marked impossible"
        assert res["action_type"] == "UNSTACK_ALL", f"Failed for '{instruction}': got {res.get('action_type')}"
        assert "unstack" in res["thought"].lower() or "deconstruct" in res["thought"].lower()


class TestRecursiveDAGPlannerUnstacking:
    """Validates recursive occlusion detection and top-down subgoal generation."""

    def setup_method(self):
        self.planner = RecursiveDAGPlanner()

    def test_get_blocks_above_tower(self):
        # 4-block tower: Cyan (0), Orange (1), Magenta (2), Yellow (3)
        blocks = [
            {"id": "block_cyan", "position": [0.45, TABLE_TOP_Y + CUBE_HALF, 0.20], "is_grasped": False},
            {"id": "block_orange", "position": [0.45, TABLE_TOP_Y + CUBE_HALF + CUBE_SIZE, 0.20], "is_grasped": False},
            {"id": "block_magenta", "position": [0.45, TABLE_TOP_Y + CUBE_HALF + 2 * CUBE_SIZE, 0.20], "is_grasped": False},
            {"id": "block_yellow", "position": [0.45, TABLE_TOP_Y + CUBE_HALF + 3 * CUBE_SIZE, 0.20], "is_grasped": False},
        ]

        # Cyan should have Yellow, Magenta, Orange above it in descending order
        above_cyan = self.planner.get_blocks_above("block_cyan", blocks)
        assert above_cyan == ["block_yellow", "block_magenta", "block_orange"]

        # Magenta should have Yellow above it
        above_mag = self.planner.get_blocks_above("block_magenta", blocks)
        assert above_mag == ["block_yellow"]

        # Yellow (topmost) should have no blocks above it
        above_yel = self.planner.get_blocks_above("block_yellow", blocks)
        assert above_yel == []

    def test_pure_unstack_dag_generation(self):
        # 3-block tower
        blocks = [
            {"id": "block_cyan", "position": [0.45, TABLE_TOP_Y + CUBE_HALF, 0.20], "is_grasped": False},
            {"id": "block_orange", "position": [0.45, TABLE_TOP_Y + CUBE_HALF + CUBE_SIZE, 0.20], "is_grasped": False},
            {"id": "block_magenta", "position": [0.45, TABLE_TOP_Y + CUBE_HALF + 2 * CUBE_SIZE, 0.20], "is_grasped": False},
        ]
        subgoals = self.planner.decompose("unstack the tower", blocks)
        # 2 elevated blocks (magenta at layer 2, orange at layer 1) need unstacking
        assert len(subgoals) == 2
        assert subgoals[0].required_objects == ["block_magenta"]
        assert subgoals[1].required_objects == ["block_orange"]
        # Orange must depend on Magenta being cleared first
        assert subgoals[0].id in subgoals[1].dependencies
        assert self.planner.is_acyclic()

    def test_sequential_train_from_existing_tower(self):
        # Existing 3-block tower: Cyan (base), Orange (mid), Magenta (top)
        blocks = [
            {"id": "block_cyan", "position": [0.45, TABLE_TOP_Y + CUBE_HALF, 0.20], "is_grasped": False},
            {"id": "block_orange", "position": [0.45, TABLE_TOP_Y + CUBE_HALF + CUBE_SIZE, 0.20], "is_grasped": False},
            {"id": "block_magenta", "position": [0.45, TABLE_TOP_Y + CUBE_HALF + 2 * CUBE_SIZE, 0.20], "is_grasped": False},
        ]
        subgoals = self.planner.decompose("make a block train with 3 blocks", blocks)
        # Should contain clearance subgoals for Magenta and Orange before Cyan is moved
        clearance_sgs = [sg for sg in subgoals if sg.skill_type == "CLEAR_OBSTACLE"]
        assert len(clearance_sgs) >= 2
        deposit_sgs = [sg for sg in subgoals if sg.skill_type == "DEPOSIT"]
        assert len(deposit_sgs) == 3
        # Cyan deposit must depend on clearance subgoals
        cyan_sg = next(sg for sg in deposit_sgs if sg.required_objects == ["block_cyan"])
        for clr in clearance_sgs:
            if "for_block_cyan" in clr.id:
                assert clr.id in cyan_sg.dependencies
        assert self.planner.is_acyclic()


class TestGoogleVLAAutonomousDeconstruction:
    """Validates end-to-end plan generation in GoogleVLAAgent with unstacking."""

    def setup_method(self):
        self.agent = GoogleVLAAgent()
        self.agent.execution_mode = "deterministic"

    def _create_6_block_tower_snapshot(self) -> Dict[str, Any]:
        all_ids = ["block_cyan", "block_orange", "block_magenta", "block_yellow", "block_emerald", "block_purple"]
        sim_blocks = []
        for i, bid in enumerate(all_ids):
            sim_blocks.append({
                "id": bid,
                "name": bid.replace("block_", "").title() + " Cube",
                "color": "#ffffff",
                "position": [STACK_ORIGIN[0], TABLE_TOP_Y + CUBE_HALF + i * CUBE_SIZE, STACK_ORIGIN[2]]
            })
        return {"blocks": sim_blocks}

    def test_unstack_6_block_tower(self):
        snap = self._create_6_block_tower_snapshot()
        res = self.agent.parse_instruction("unstack the tower", snapshot=snap)
        assert res["is_possible"] is True
        assert res["status"] == "OK"
        # 5 elevated blocks unstacked
        unstack_groundings = [g for g in res["grounding"] if g.get("role") == "Deconstruction Unstack"]
        assert len(unstack_groundings) == 5

        # Verify top-down order: Purple (top) -> Emerald -> Yellow -> Magenta -> Orange
        unstacked_ids = [g["entity"] for g in unstack_groundings]
        assert unstacked_ids == ["block_purple", "block_emerald", "block_yellow", "block_magenta", "block_orange"]

        # Verify each unstack target is on the tabletop plane
        for g in unstack_groundings:
            assert abs(g["target_coords"][1] - (TABLE_TOP_Y + CUBE_HALF)) < 0.005

    def test_train_from_6_block_tower_without_reset(self):
        snap = self._create_6_block_tower_snapshot()
        res = self.agent.parse_instruction("make a block train with all boxes in one line", snapshot=snap)
        assert res["is_possible"] is True
        assert res["status"] == "OK"

        # Verify occluding blocks were staged out of the tower first
        clearance_groundings = [g for g in res["grounding"] if "Clearance" in g.get("role", "")]
        assert len(clearance_groundings) >= 5

        # Topmost block (purple) must be cleared first
        first_cleared = clearance_groundings[0]["entity"]
        assert first_cleared == "block_purple"

        # All 6 blocks placed into train slots
        train_groundings = [g for g in res["grounding"] if "Clearance" not in g.get("role", "")]
        assert len(train_groundings) == 6

    def test_3_block_pyramid_from_6_block_tower_without_reset(self):
        snap = self._create_6_block_tower_snapshot()
        res = self.agent.parse_instruction("build a 3-block stepped pyramid", snapshot=snap)
        assert res["is_possible"] is True
        assert res["status"] == "OK"
        # High blocks cleared from the stack pad column
        clearance_groundings = [g for g in res["grounding"] if "Clearance" in g.get("role", "")]
        assert len(clearance_groundings) >= 3
