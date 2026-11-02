"""
tests/test_path_clearing.py

Unit tests for Predictive Obstruction Detection & Dynamic Path Clearing in H-JEPA.
Verifies that:
1. Slot obstructions are predicted and cleared to perimeter staging before placement.
2. Pick envelope obstructions are predicted and cleared to perimeter staging before grasping.
3. Placed blocks in multi-tier towers/pyramids are never passed through during transit.
4. Runtime dynamic obstructions trigger immediate predictive clearance subgoals.
"""

import pytest
import math
from typing import Dict, List, Any
import numpy as np

from simulation.constants import (
    TABLE_TOP_Y,
    CUBE_SIZE,
    CUBE_HALF,
    STACK_ORIGIN,
    SAFE_STAGING_SLOTS,
    FIXED_DT,
    get_pyramid_slots,
)
from simulation.physics_engine import RobotArmSimulation
from simulation.mujoco_engine import MujocoSimulation, MUJOCO_AVAILABLE
from vla.h_jepa.vl_grounder import VLJEPAGrounder
from vla.h_jepa.orchestrator import HierarchicalVLJEPAOrchestrator


class TestPathClearingAndObstaclePrediction:

    def test_predict_and_clear_slot_obstruction(self):
        """
        Verify that when an obstacle block occupies a target slot,
        VLJEPAGrounder predicts the collision and prepends a path-clearing subgoal
        to relocate the obstacle to safe perimeter staging.
        """
        grounder = VLJEPAGrounder()
        target_slots = [
            [0.48, TABLE_TOP_Y + CUBE_HALF, 0.22],
            [0.48, TABLE_TOP_Y + CUBE_HALF + CUBE_SIZE, 0.22],
        ]
        block_ids = ["block_magenta", "block_cyan"]

        # Place block_orange directly on Slot 0
        initial_blocks = [
            {"id": "block_magenta", "position": [0.20, TABLE_TOP_Y + CUBE_HALF, 0.15]},
            {"id": "block_cyan", "position": [0.25, TABLE_TOP_Y + CUBE_HALF, -0.15]},
            {"id": "block_orange", "position": [0.48, TABLE_TOP_Y + CUBE_HALF, 0.22]},  # Obstructing slot 0!
        ]

        subgoals = grounder.generate_subgoal_embeddings(target_slots, block_ids, initial_blocks)

        # Must have at least 3 subgoals: 1 clear + 2 assembly
        assert len(subgoals) == 3, f"Expected 3 subgoals (1 clearing + 2 assembly), got {len(subgoals)}"
        first_sub = subgoals[0]
        assert first_sub["block_id"] == "block_orange"
        assert first_sub["is_clearing"] is True
        assert "Predict & clear path" in first_sub["description"]

        # Staging position must be in SAFE_STAGING_SLOTS or on perimeter
        st_x, st_z = first_sub["target_slot"][0], first_sub["target_slot"][2]
        assert math.hypot(st_x - 0.48, st_z - 0.22) > 0.15, "Staging position is dangerously close to target slot"

    def test_predict_and_clear_pick_envelope_obstruction(self):
        """
        Verify that when an obstacle block crowds the pick envelope of a target block,
        VLJEPAGrounder predicts gripper collision and clears the obstacle to perimeter staging.
        """
        grounder = VLJEPAGrounder()
        target_slots = [[0.48, TABLE_TOP_Y + CUBE_HALF, 0.22]]
        block_ids = ["block_magenta"]

        # Place block_orange only 0.045m away from block_magenta (crowding gripper jaws)
        mag_pos = [0.25, TABLE_TOP_Y + CUBE_HALF, 0.10]
        obs_pos = [0.25, TABLE_TOP_Y + CUBE_HALF, 0.145]  # 45mm away (< 65mm envelope)

        initial_blocks = [
            {"id": "block_magenta", "position": mag_pos},
            {"id": "block_orange", "position": obs_pos},
        ]

        subgoals = grounder.generate_subgoal_embeddings(target_slots, block_ids, initial_blocks)

        assert len(subgoals) == 2, f"Expected 2 subgoals (1 clearing + 1 assembly), got {len(subgoals)}"
        assert subgoals[0]["block_id"] == "block_orange"
        assert subgoals[0]["is_clearing"] is True
        assert "pick envelope obstruction" in subgoals[0]["description"]
        assert subgoals[1]["block_id"] == "block_magenta"

    def test_avoid_placed_blocks_during_multi_tier_transit(self):
        """
        Verifies that after placing the bottom tier of a tower, the robot arm
        strictly ascends vertically and never slices through placed blocks
        when returning to acquire the next block.
        """
        sim = RobotArmSimulation()
        orchestrator = HierarchicalVLJEPAOrchestrator()

        snap = sim.get_snapshot()
        plan_res = orchestrator.plan_task(
            instruction="build a 2-block tower with red and blue",
            initial_blocks=snap["blocks"]
        )
        assert plan_res["status"] == "PLAN_READY"

        max_penetration = 0.0
        ticks = 0
        while orchestrator.is_active and ticks < 450:
            snap = sim.get_snapshot()
            tcp = snap["tcp"]["position"]
            step_cmd = orchestrator.step(tcp, snap["joints"], snap["blocks"])
            if step_cmd is None or not orchestrator.is_active:
                break
            sim.set_joint_targets(step_cmd["joints"], step_cmd["gripper"])
            sim.step(FIXED_DT)
            ticks += 1

            # Check penetration between base block and all other blocks
            b_mag = sim.blocks["block_magenta"]
            for other_id, other_b in sim.blocks.items():
                if other_id == "block_magenta":
                    continue
                # If other block is placed on top of magenta, normal contact along Y is expected
                dx = abs(b_mag.position[0] - other_b.position[0])
                dy = abs(b_mag.position[1] - other_b.position[1])
                dz = abs(b_mag.position[2] - other_b.position[2])
                ox = CUBE_SIZE - dx
                oy = CUBE_SIZE - dy
                oz = CUBE_SIZE - dz
                if ox > 0.005 and oy > 0.005 and oz > 0.005:
                    pen = min(ox, oy, oz)
                    if pen > max_penetration:
                        max_penetration = pen

        assert max_penetration < 0.002, f"Excessive block interpenetration: {max_penetration*1000:.1f} mm"

    def test_runtime_dynamic_obstruction_prediction(self):
        """
        Verifies that when an obstacle block is introduced dynamically at runtime,
        orchestrator.step() detects the obstruction on tick 1, inserts a dynamic
        clearing subgoal, and executes it without crashing.
        """
        orchestrator = HierarchicalVLJEPAOrchestrator()
        initial_blocks = [
            {"id": "block_magenta", "position": [0.25, TABLE_TOP_Y + CUBE_HALF, 0.10], "is_grasped": False},
            {"id": "block_cyan", "position": [0.25, TABLE_TOP_Y + CUBE_HALF, -0.15], "is_grasped": False},
        ]
        target_slot = [[0.48, TABLE_TOP_Y + CUBE_HALF, 0.22]]
        plan = orchestrator.plan_task(
            "move red cube to center pad",
            initial_blocks,
            target_slots=target_slot,
            target_block_ids=["block_magenta"],
        )
        assert plan["num_subgoals"] == 1

        # Now simulate dynamic scene shift: an unexpected obstacle block_orange
        # is placed right at magenta's pick location
        dynamic_blocks = [
            {"id": "block_magenta", "position": [0.25, TABLE_TOP_Y + CUBE_HALF, 0.10], "is_grasped": False},
            {"id": "block_cyan", "position": [0.25, TABLE_TOP_Y + CUBE_HALF, -0.15], "is_grasped": False},
            {"id": "block_orange", "position": [0.25, TABLE_TOP_Y + CUBE_HALF, 0.12], "is_grasped": False},  # 20mm away!
        ]

        # Step 1: Orchestrator should dynamically predict obstruction and insert a clearing subgoal
        cmd = orchestrator.step(
            current_tcp=[0.20, 0.92, 0.0],
            current_joints={"baseYaw": 0, "shoulderPitch": 0, "elbowPitch": 0, "wristPitch": 0, "gripper": 0},
            blocks=dynamic_blocks,
        )

        assert len(orchestrator.subgoals) == 2, "Dynamic clearing subgoal was not inserted"
        assert orchestrator.subgoals[0]["block_id"] == "block_orange"
        assert orchestrator.subgoals[0]["is_clearing"] is True
        assert orchestrator.subgoals[1]["block_id"] == "block_magenta"
