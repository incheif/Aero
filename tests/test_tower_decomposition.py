"""
tests/test_tower_decomposition.py

Unit tests for compositional task decomposition and dual-system intent routing.
Validates:
1. "build a tower of five blocks" decomposes into 5 ordered SubgoalSpecifications.
2. Progressive vertical stack heights y_k = TABLE_TOP_Y + (k + 0.5) * CUBE_SIZE.
3. 3-block and 5-block pyramid decomposition.
4. Block train decomposition.
5. Dual-system routing: System 1 (fast reflex) vs System 2 (H-JEPA world model).
"""

import sys
import os
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from simulation.constants import TABLE_TOP_Y, CUBE_SIZE, STACK_ORIGIN
from vla.instruction_interpreter import InstructionInterpreter
from vla.jepa.schemas import SubgoalSpecification


class TestTowerDecomposition(unittest.TestCase):
    def setUp(self):
        self.interpreter = InstructionInterpreter()
        self.sample_blocks = [
            {"id": "block_magenta", "name": "Red Cube", "color": "#FF1E27", "position": [0.15, TABLE_TOP_Y + 0.02, 0.16]},
            {"id": "block_cyan", "name": "Blue Cube", "color": "#1E60FF", "position": [0.22, TABLE_TOP_Y + 0.02, 0.22]},
            {"id": "block_yellow", "name": "Yellow Cube", "color": "#FFC300", "position": [0.12, TABLE_TOP_Y + 0.02, -0.15]},
            {"id": "block_emerald", "name": "Green Cube", "color": "#00B341", "position": [0.28, TABLE_TOP_Y + 0.02, -0.18]},
            {"id": "block_orange", "name": "Orange Cube", "color": "#FF5500", "position": [-0.14, TABLE_TOP_Y + 0.02, 0.16]},
            {"id": "block_purple", "name": "Purple Cube", "color": "#9333EA", "position": [-0.18, TABLE_TOP_Y + 0.02, -0.14]},
            {"id": "block_pink", "name": "Pink Cube", "color": "#FF2E93", "position": [0.24, TABLE_TOP_Y + 0.02, 0.0]},
        ]

    def test_01_five_block_tower_decomposition(self):
        """Validates that 'build a tower of five blocks' produces 5 ordered subgoals with progressive heights."""
        subgoals = self.interpreter.parse_sequence(
            prompt="Build a tower of five blocks: red, blue, yellow, green, orange",
            current_blocks=self.sample_blocks
        )
        self.assertEqual(len(subgoals), 5)
        for idx, sg in enumerate(subgoals):
            self.assertIsInstance(sg, SubgoalSpecification)
            self.assertEqual(sg.subgoal_id, idx)
            # Verify vertical stack height increases monotonically
            expected_y = TABLE_TOP_Y + CUBE_SIZE * (idx + 0.5)
            self.assertAlmostEqual(sg.target_xyz[1], expected_y, places=3)
            # Check origin X, Z
            self.assertAlmostEqual(sg.target_xyz[0], STACK_ORIGIN[0], places=3)
            self.assertAlmostEqual(sg.target_xyz[2], STACK_ORIGIN[2], places=3)
            # Layer 0 is on table, layer >= 1 has support
            if idx == 0:
                self.assertEqual(sg.support_label, "table")
            else:
                self.assertIsNotNone(sg.support_label)

    def test_02_three_block_pyramid_decomposition(self):
        """Validates 3-block pyramid decomposition into 2 base blocks + 1 apex."""
        subgoals = self.interpreter.parse_sequence(
            prompt="Build a 3-block pyramid",
            current_blocks=self.sample_blocks
        )
        self.assertEqual(len(subgoals), 3)
        # First 2 should be at table height (tier 0)
        base_y = TABLE_TOP_Y + CUBE_SIZE * 0.5
        self.assertAlmostEqual(subgoals[0].target_xyz[1], base_y, places=3)
        self.assertAlmostEqual(subgoals[1].target_xyz[1], base_y, places=3)
        # Apex should be at tier 1
        apex_y = TABLE_TOP_Y + CUBE_SIZE * 1.5
        self.assertAlmostEqual(subgoals[2].target_xyz[1], apex_y, places=3)

    def test_03_block_train_decomposition(self):
        """Validates linear block train assembly."""
        subgoals = self.interpreter.parse_sequence(
            prompt="Assemble a block train of 4 boxes in a row",
            current_blocks=self.sample_blocks
        )
        self.assertEqual(len(subgoals), 4)
        # All train blocks must rest on table
        base_y = TABLE_TOP_Y + CUBE_SIZE * 0.5
        for sg in subgoals:
            self.assertAlmostEqual(sg.target_xyz[1], base_y, places=3)
            self.assertEqual(sg.support_label, "table")

    def test_04_dual_system_routing(self):
        """Validates intent router distinguishes atomic fast-path from compositional deep-path."""
        # Atomic commands -> System 1 Fast Path
        self.assertEqual(self.interpreter.route_task("pick the red cube"), "SYSTEM_1_FAST_PATH")
        self.assertEqual(self.interpreter.route_task("open gripper"), "SYSTEM_1_FAST_PATH")
        self.assertEqual(self.interpreter.route_task("wave high"), "SYSTEM_1_FAST_PATH")
        self.assertEqual(self.interpreter.route_task("return home"), "SYSTEM_1_FAST_PATH")

        # Compositional commands -> System 2 H-JEPA
        self.assertEqual(self.interpreter.route_task("build a tower of five blocks"), "SYSTEM_2_H_JEPA")
        self.assertEqual(self.interpreter.route_task("build a 3-block pyramid"), "SYSTEM_2_H_JEPA")
        self.assertEqual(self.interpreter.route_task("stack all cubes"), "SYSTEM_2_H_JEPA")
        self.assertEqual(self.interpreter.route_task("assemble a train of 4 blocks"), "SYSTEM_2_H_JEPA")
        self.assertEqual(self.interpreter.route_task("pick red then place on blue"), "SYSTEM_2_H_JEPA")

        # Explicit mode overrides
        self.assertEqual(self.interpreter.route_task("build a tower", mode="fast"), "SYSTEM_1_FAST_PATH")
        self.assertEqual(self.interpreter.route_task("pick red", mode="hjepa"), "SYSTEM_2_H_JEPA")

    def test_05_multi_tower_clause_color_assignment(self):
        """Validates that distinct multi-tower requests with clauses assign the exact specified colors."""
        cmd = "Make 2 distinct towers of 2 blocks each. one tower with pink and purple blocks next tower with green and red blocks"
        res = self.interpreter.interpret(cmd)
        self.assertEqual(res["action_type"], "STACK_DUAL_TOWERS")
        self.assertEqual(res["num_towers"], 2)
        self.assertEqual(len(res["tower_blocks_list"]), 2)
        # Tower 1 must have pink and purple
        self.assertEqual(res["tower_blocks_list"][0], ["block_pink", "block_purple"])
        # Tower 2 must have green and red
        self.assertEqual(res["tower_blocks_list"][1], ["block_emerald", "block_magenta"])


if __name__ == "__main__":
    unittest.main()
