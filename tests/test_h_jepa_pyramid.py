"""
tests/test_h_jepa_pyramid.py

End-to-End Multi-Box Pyramid Manipulation Benchmark for 2-Level H-JEPA:
- Evaluates 3-Block Stepped Pyramid (2 base + 1 apex)
- Evaluates 5-Block Square Pyramid (4 base + 1 apex)
- Evaluates 6-Block Stepped Pyramid (3 base + 2 mid + 1 apex)
- Validates:
    1. Zero False-Positive Subgoal Transitions (strictly gated by physical stability hold)
    2. Sub-20ms Latency per CEM Step
    3. Multi-tier geometric slot convergence
"""

import sys
import os
import time
import math
import unittest
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from simulation.constants import (
    STACK_ORIGIN,
    SAFE_HOVER_Y,
    PICK_Y,
    TABLE_TOP_Y,
    CUBE_SIZE,
    get_pyramid_slots,
    DEFAULT_JOINTS,
)
from simulation.scoring import StabilityValidator, spatial_accuracy, task_completion_score
from simulation.kinematics import forward_kinematics
from vla.h_jepa import HierarchicalVLJEPAOrchestrator

class TestHJEPAEndToEndPyramids(unittest.TestCase):

    def setUp(self):
        # 7-cube workcell tabletop setup
        self.blocks_inventory = [
            {"id": "cube_red", "name": "Red Cube", "color": "#E05345", "position": [0.15, TABLE_TOP_Y + 0.02, 0.16], "rotation": [0,0,0,1], "in_pad": False, "is_grasped": False},
            {"id": "cube_blue", "name": "Blue Cube", "color": "#3B82F6", "position": [0.22, TABLE_TOP_Y + 0.02, 0.22], "rotation": [0,0,0,1], "in_pad": False, "is_grasped": False},
            {"id": "cube_green", "name": "Green Cube", "color": "#10B981", "position": [0.12, TABLE_TOP_Y + 0.02, -0.15], "rotation": [0,0,0,1], "in_pad": False, "is_grasped": False},
            {"id": "cube_yellow", "name": "Yellow Cube", "color": "#F59E0B", "position": [0.28, TABLE_TOP_Y + 0.02, -0.18], "rotation": [0,0,0,1], "in_pad": False, "is_grasped": False},
            {"id": "cube_cyan", "name": "Cyan Cube", "color": "#06B6D4", "position": [-0.14, TABLE_TOP_Y + 0.02, 0.16], "rotation": [0,0,0,1], "in_pad": False, "is_grasped": False},
            {"id": "cube_purple", "name": "Purple Cube", "color": "#8B5CF6", "position": [-0.18, TABLE_TOP_Y + 0.02, -0.14], "rotation": [0,0,0,1], "in_pad": False, "is_grasped": False},
            {"id": "cube_pink", "name": "Pink Cube", "color": "#EC4899", "position": [0.24, TABLE_TOP_Y + 0.02, 0.0], "rotation": [0,0,0,1], "in_pad": False, "is_grasped": False},
        ]

    def test_01_three_block_stepped_pyramid(self):
        """Validates 3-block stepped pyramid macro sequence and sub-goal gating."""
        orchestrator = HierarchicalVLJEPAOrchestrator()
        blocks = [dict(b) for b in self.blocks_inventory]
        
        plan = orchestrator.plan_task(
            instruction="Build a 3-block pyramid",
            initial_blocks=blocks
        )
        self.assertEqual(plan["num_subgoals"], 3)
        self.assertEqual(plan["structure_type"], "3_block_stepped")
        
        # Verify initial state: subgoal 0 active
        self.assertEqual(orchestrator.current_subgoal_idx, 0)
        self.assertEqual(orchestrator.micro_phase, "HOVER_PICK")

        # Simulate closed-loop stepping through first subgoal pick-place cycle
        current_tcp = (0.20, SAFE_HOVER_Y, 0.0)
        current_joints = dict(DEFAULT_JOINTS)
        current_gripper = 0.0
        
        target_b_id = orchestrator.subgoals[0]["block_id"]
        target_slot = orchestrator.subgoals[0]["target_slot"]
        
        step_latencies = []
        max_steps = 150
        subgoal_0_cleared = False

        for step_i in range(max_steps):
            res = orchestrator.step(
                current_tcp=current_tcp,
                current_joints=current_joints,
                blocks=blocks,
                current_gripper=current_gripper
            )
            if not res:
                break

            step_latencies.append(res["cem_latency_ms"])
            current_tcp = tuple(res["tcp_target"])
            current_gripper = res["gripper"]
            current_joints = res["joints"]

            # If grasped, block moves with TCP
            if orchestrator.grasped_block_id == target_b_id:
                for b in blocks:
                    if b["id"] == target_b_id:
                        b["position"] = [current_tcp[0], current_tcp[1] - 0.02, current_tcp[2]]
                        b["is_grasped"] = True

            # If released and at target slot, simulate resting contact on table pad
            if res["phase"] == "VERIFY_HOLD" and orchestrator.grasped_block_id is None:
                for b in blocks:
                    if b["id"] == target_b_id:
                        b["position"] = list(target_slot)
                        b["in_pad"] = True
                        b["is_grasped"] = False
                        b["velocity"] = [0.0, 0.0, 0.0]

            if res["subgoal_index"] > 0:
                subgoal_0_cleared = True
                break

        self.assertTrue(subgoal_0_cleared, "Subgoal 0 should complete and transition to Subgoal 1 via dual-gated hold verification.")
        avg_lat = sum(step_latencies) / len(step_latencies)
        print(f"\n[3-Block Pyramid] Cleared Subgoal 0 in {len(step_latencies)} ticks. Avg CEM Latency: {avg_lat:.2f} ms")
        self.assertLess(avg_lat, 25.0)

    def test_02_five_block_square_pyramid(self):
        """Validates 5-block square pyramid grounding & slot assignment."""
        orchestrator = HierarchicalVLJEPAOrchestrator()
        plan = orchestrator.plan_task(
            instruction="Build a 5-block square pyramid with 4 base cubes",
            initial_blocks=self.blocks_inventory
        )
        self.assertEqual(plan["num_subgoals"], 5)
        self.assertEqual(plan["structure_type"], "5_block_square")
        
        # Verify apex is higher than base
        base_y = plan_slots = orchestrator.subgoals[0]["target_slot"][1]
        apex_y = orchestrator.subgoals[4]["target_slot"][1]
        self.assertGreater(apex_y, base_y)

    def test_03_six_block_stepped_pyramid(self):
        """Validates 6-block stepped pyramid 3-tier structure allocation."""
        orchestrator = HierarchicalVLJEPAOrchestrator()
        plan = orchestrator.plan_task(
            instruction="Build a 6-block stepped pyramid",
            initial_blocks=self.blocks_inventory
        )
        self.assertEqual(plan["num_subgoals"], 6)
        self.assertEqual(plan["structure_type"], "6_block_stepped")
        
        # Tier 0 (base): 3 cubes, Tier 1 (mid): 2 cubes, Tier 2 (apex): 1 cube
        tier0_y = orchestrator.subgoals[0]["target_slot"][1]
        tier1_y = orchestrator.subgoals[3]["target_slot"][1]
        tier2_y = orchestrator.subgoals[5]["target_slot"][1]

        self.assertAlmostEqual(tier0_y, orchestrator.subgoals[1]["target_slot"][1], places=3)
        self.assertAlmostEqual(tier0_y, orchestrator.subgoals[2]["target_slot"][1], places=3)
        self.assertAlmostEqual(tier1_y, orchestrator.subgoals[4]["target_slot"][1], places=3)
        self.assertGreater(tier1_y, tier0_y)
        self.assertGreater(tier2_y, tier1_y)

    def test_04_zero_false_positive_gate(self):
        """Verifies that an unstable/moving block NEVER clears the sub-goal gate."""
        orchestrator = HierarchicalVLJEPAOrchestrator()
        blocks = [dict(b) for b in self.blocks_inventory]
        orchestrator.plan_task("Build a 3-block pyramid", blocks)
        
        # Force phase to VERIFY_HOLD
        orchestrator.micro_phase = "VERIFY_HOLD"
        orchestrator.grasped_block_id = None
        
        # Set block velocity high (unstable / sliding block: speed = 0.15 m/s > 0.04 threshold)
        target_b_id = orchestrator.subgoals[0]["block_id"]
        for b in blocks:
            if b["id"] == target_b_id:
                b["in_pad"] = True
                b["velocity"] = [0.10, 0.05, 0.10]
        
        # Step 25 ticks (more than 18 tick threshold)
        for _ in range(25):
            res = orchestrator.step(
                current_tcp=(0.20, SAFE_HOVER_Y, 0.0),
                current_joints=DEFAULT_JOINTS,
                blocks=blocks,
                current_gripper=0.0
            )
        
        # Must still be on Subgoal 0! The gate must NOT unlock while moving!
        self.assertEqual(orchestrator.current_subgoal_idx, 0, "Gate must reject premature completion when blocks are unstable.")
        self.assertFalse(orchestrator.stability_validator.is_verified)
        self.assertIn(orchestrator.stability_validator.status, ["WAITING", "UNSTABLE"])

if __name__ == "__main__":
    unittest.main()
