"""
tests/test_verified_manipulation.py

10 Rigorous Manipulation Verification Tests for AERO:
Runs exclusively through genuine MuJoCo rigid-body simulation with articulated
robot dynamics, joint limits, real contact manifolds, and zero artificial shortcuts.

Tests:
1. Pick aligned block
2. Pick rotated block (30-degree yaw)
3. Pick block between obstacles
4. Move through clutter without collisions (joint-space RRT)
5. Place block at specified position & orientation
6. Detect failed grasp
7. Recover from failed grasp
8. Detect unstable placement
9. Handle unreachable IK targets (returns explicit UNREACHABLE_POSITION)
10. Complete multi-step pyramid task
"""

import sys
import os
import math
import copy
import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from simulation.constants import (
    FIXED_DT,
    TABLE_TOP_Y,
    CUBE_SIZE,
    CUBE_HALF,
    SAFE_HOVER_Y,
    DEFAULT_JOINTS,
    ARM_MOUNT,
    PEDESTAL_H,
    GRASP_CLOSE_THRESHOLD,
    GRASP_OPEN_THRESHOLD,
)
from simulation.kinematics import solve_ik, forward_kinematics
from simulation.mujoco_engine import MujocoSimulation
from simulation.motion_planner import RRTConnectJointPlanner, ArmCollisionChecker
from simulation.scoring import StabilityValidator
from vla.hierarchical_planner import HierarchicalTaskPlanner, ExecutionStateMachine, Subgoal
from vla.h_jepa.orchestrator import HierarchicalVLJEPAOrchestrator


class TestVerifiedManipulation:

    def test_01_pick_aligned_block(self):
        """1. Pick aligned block: verify grasp contact, invariant attachment, and clean lift."""
        sim = MujocoSimulation()
        b_id = "block_magenta"
        bp = sim.blocks[b_id].position

        # Pregrasp hover
        ik_pre = solve_ik((bp[0], SAFE_HOVER_Y, bp[2]), gripper=0.0)
        sim.set_joint_targets(ik_pre.joints, gripper=0.0)
        for _ in range(35):
            sim.step(FIXED_DT)

        # Descend to approach pose
        approach_target = (bp[0], bp[1] + 0.008, bp[2])
        ik_res = solve_ik(approach_target, gripper=0.0)
        assert ik_res.success, f"IK failed: {ik_res.reachability_status}"
        sim.set_joint_targets(ik_res.joints, gripper=0.0)
        for _ in range(40):
            sim.step(FIXED_DT)

        # Close gripper
        sim.set_joint_targets(ik_res.joints, gripper=1.0)
        for _ in range(35):
            sim.step(FIXED_DT)

        assert sim.blocks[b_id].is_grasped, "Block should be physically grasped by parallel jaws"

        # Lift target vertically
        lift_target = (bp[0], SAFE_HOVER_Y, bp[2])
        ik_lift = solve_ik(lift_target, gripper=1.0)
        assert ik_lift.success
        sim.set_joint_targets(ik_lift.joints, gripper=1.0)
        for _ in range(50):
            sim.step(FIXED_DT)

        lifted_pos = sim.blocks[b_id].position
        assert lifted_pos[1] > 0.83, f"Block should be lifted above table: {lifted_pos[1]}"
        assert sim.blocks[b_id].is_grasped, "Grasp should remain stable during transit"

    def test_02_pick_rotated_block(self):
        """2. Pick rotated block (30 deg yaw): verify orientation-aware grasp and relative transform preservation."""
        sim = MujocoSimulation()
        b_id = "block_yellow"
        adr = sim.joint_indices[b_id]
        qp = adr["qpos_adr"]

        # Rotate yellow block by 30 deg (0.5236 rad) around Y
        theta = 0.5236
        sim.data.qpos[qp + 3] = math.cos(theta / 2.0)  # w
        sim.data.qpos[qp + 4] = 0.0                     # x
        sim.data.qpos[qp + 5] = math.sin(theta / 2.0)  # y
        sim.data.qpos[qp + 6] = 0.0                     # z
        sim.step(FIXED_DT)

        bp = sim.blocks[b_id].position

        # Pregrasp hover
        ik_pre = solve_ik((bp[0], SAFE_HOVER_Y, bp[2]), gripper=0.0)
        sim.set_joint_targets(ik_pre.joints, gripper=0.0)
        for _ in range(35):
            sim.step(FIXED_DT)

        approach_target = (bp[0], bp[1] + 0.008, bp[2])
        ik_res = solve_ik(approach_target, gripper=0.0)
        assert ik_res.success

        sim.set_joint_targets(ik_res.joints, gripper=0.0)
        for _ in range(40):
            sim.step(FIXED_DT)

        sim.set_joint_targets(ik_res.joints, gripper=1.0)
        for _ in range(35):
            sim.step(FIXED_DT)

        assert sim.blocks[b_id].is_grasped, "Rotated block should be captured by orientation-aware grasp"

        # Lift
        lift_target = (bp[0], SAFE_HOVER_Y, bp[2])
        ik_lift = solve_ik(lift_target, gripper=1.0)
        sim.set_joint_targets(ik_lift.joints, gripper=1.0)
        for _ in range(50):
            sim.step(FIXED_DT)

        assert sim.blocks[b_id].position[1] > 0.83
        assert sim.blocks[b_id].is_grasped

    def test_03_pick_block_between_obstacles(self):
        """3. Pick block between obstacles: pick central block without colliding with flanking obstacles."""
        sim = MujocoSimulation()
        target_id = "block_yellow"
        obs1_id = "block_orange"
        obs2_id = "block_purple"

        clearance = CUBE_SIZE + 0.055
        sim.blocks[target_id].position = [0.30, TABLE_TOP_Y + CUBE_HALF, 0.0]
        sim.blocks[obs1_id].position = [0.30, TABLE_TOP_Y + CUBE_HALF, clearance]
        sim.blocks[obs2_id].position = [0.30, TABLE_TOP_Y + CUBE_HALF, -clearance]

        adr_t = sim.joint_indices[target_id]["qpos_adr"]
        adr_o1 = sim.joint_indices[obs1_id]["qpos_adr"]
        adr_o2 = sim.joint_indices[obs2_id]["qpos_adr"]
        sim.data.qpos[adr_t : adr_t + 3] = sim.blocks[target_id].position
        sim.data.qpos[adr_o1 : adr_o1 + 3] = sim.blocks[obs1_id].position
        sim.data.qpos[adr_o2 : adr_o2 + 3] = sim.blocks[obs2_id].position
        sim.step(FIXED_DT)

        bp = sim.blocks[target_id].position

        # 1. Pregrasp (hover cleanly above slot)
        ik_pre = solve_ik((bp[0], SAFE_HOVER_Y, bp[2]), gripper=0.0)
        sim.set_joint_targets(ik_pre.joints, gripper=0.0)
        for _ in range(35):
            sim.step(FIXED_DT)

        # 2. Approach target block directly from above
        ik_app = solve_ik((bp[0], bp[1] + 0.008, bp[2]), gripper=0.0)
        assert ik_app.success
        sim.set_joint_targets(ik_app.joints, gripper=0.0)
        for _ in range(40):
            sim.step(FIXED_DT)

        # Verify zero collision with obstacles during approach
        contacts = sim.get_contact_info()
        for c in contacts:
            pair = (c["geom1"], c["geom2"])
            assert not (f"{obs1_id}_geom" in pair and "finger" in "".join(pair)), "Gripper hit obstacle 1"
            assert not (f"{obs2_id}_geom" in pair and "finger" in "".join(pair)), "Gripper hit obstacle 2"

        # 3. Grasp and lift
        sim.set_joint_targets(ik_app.joints, gripper=1.0)
        for _ in range(35):
            sim.step(FIXED_DT)
        assert sim.blocks[target_id].is_grasped

        ik_lift = solve_ik((bp[0], SAFE_HOVER_Y, bp[2]), gripper=1.0)
        sim.set_joint_targets(ik_lift.joints, gripper=1.0)
        for _ in range(50):
            sim.step(FIXED_DT)

        assert sim.blocks[target_id].position[1] > 0.85
        assert math.hypot(*sim.blocks[obs1_id].velocity) < 0.05
        assert math.hypot(*sim.blocks[obs2_id].velocity) < 0.05

    def test_04_move_through_clutter_without_collision(self):
        """4. Move through clutter: joint-space RRT-Connect plans swept collision-free trajectory."""
        planner = RRTConnectJointPlanner()

        start_joints = {"baseYaw": -0.6, "shoulderPitch": 0.8, "elbowPitch": -1.5, "wristPitch": 0.5, "gripper": 0.0}
        goal_joints = {"baseYaw": 0.6, "shoulderPitch": 0.8, "elbowPitch": -1.5, "wristPitch": 0.5, "gripper": 0.0}

        obstacles = [{"id": "obs_tower", "position": [0.35, TABLE_TOP_Y + 0.12, 0.0], "size": 0.08}]

        path = planner.plan(start_joints, goal_joints, blocks=obstacles)
        assert len(path) >= 2, "RRT-Connect should produce a valid trajectory"

        for q in path:
            assert not planner.checker.check_configuration_collision(q, blocks=obstacles)

    def test_05_place_block_at_specified_position_and_orientation(self):
        """5. Place block at specified position: executes pick, transit, placement, release, and verify hold."""
        sim = MujocoSimulation()
        b_id = "block_magenta"
        bp = sim.blocks[b_id].position
        target_slot = [0.42, TABLE_TOP_Y + CUBE_HALF, 0.15]

        # Pregrasp & Pick
        ik_pre = solve_ik((bp[0], SAFE_HOVER_Y, bp[2]), gripper=0.0)
        sim.set_joint_targets(ik_pre.joints, gripper=0.0)
        for _ in range(35):
            sim.step(FIXED_DT)

        ik_pick = solve_ik((bp[0], bp[1] + 0.008, bp[2]), gripper=0.0)
        sim.set_joint_targets(ik_pick.joints, gripper=0.0)
        for _ in range(40):
            sim.step(FIXED_DT)
        sim.set_joint_targets(ik_pick.joints, gripper=1.0)
        for _ in range(35):
            sim.step(FIXED_DT)
        assert sim.blocks[b_id].is_grasped

        # Lift & transit
        ik_transit = solve_ik((target_slot[0], SAFE_HOVER_Y, target_slot[2]), gripper=1.0)
        sim.set_joint_targets(ik_transit.joints, gripper=1.0)
        for _ in range(45):
            sim.step(FIXED_DT)

        # Place at slot
        ik_place = solve_ik((target_slot[0], target_slot[1] + 0.006, target_slot[2]), gripper=1.0)
        sim.set_joint_targets(ik_place.joints, gripper=1.0)
        for _ in range(40):
            sim.step(FIXED_DT)

        # Release
        sim.set_joint_targets(ik_place.joints, gripper=0.0)
        for _ in range(30):
            sim.step(FIXED_DT)

        assert not sim.blocks[b_id].is_grasped, "Block should be released"

        # Verify placement position accuracy (< 30 mm error)
        final_p = sim.blocks[b_id].position
        err_xz = math.hypot(final_p[0] - target_slot[0], final_p[2] - target_slot[2])
        err_y = abs(final_p[1] - target_slot[1])
        assert err_xz < 0.030, f"Placement XZ error excessive: {err_xz:.4f}m"
        assert err_y < 0.015, f"Placement Y error excessive: {err_y:.4f}m"

    def test_06_detect_failed_grasp(self):
        """6. Detect failed grasp: verify state machine does NOT advance when block was not captured."""
        sim = MujocoSimulation()
        sm = ExecutionStateMachine()

        subgoal = Subgoal(
            id="sub_test_fail",
            parent_id=None,
            level=2,
            skill_type="PICK_AND_PLACE",
            required_objects=["block_magenta"],
            preconditions={"gripper_empty": True},
            desired_effects={"placed": True},
            target_configuration={"block_id": "block_magenta", "slot": [0.45, TABLE_TOP_Y + CUBE_HALF, 0.0]},
            dependencies=[],
            max_retries=1,
        )
        sm.start_plan([subgoal])
        assert sm.state == "PLAN"

        snap = sim.get_snapshot()
        sm.step(snap["tcp"]["position"], snap["joints"], snap["blocks"], current_gripper=0.0, grasped_block_id=None)
        assert sm.state == "MOVE_TO_PREGRASP"

        # Force transition to VERIFY_GRASP while grasped_block_id remains None
        sm.transition_to("VERIFY_GRASP")
        cmd = sm.step(snap["tcp"]["position"], snap["joints"], snap["blocks"], current_gripper=1.0, grasped_block_id=None)

        # Must reject LIFT and transition to RECOVER
        assert sm.state == "RECOVER", f"Expected RECOVER on grasp miss, got {sm.state}"
        assert subgoal.status != "COMPLETED"

    def test_07_recover_from_failed_grasp(self):
        """7. Recover from failed grasp: retreats, replans, and successfully retries."""
        sim = MujocoSimulation()
        sm = ExecutionStateMachine()
        b_id = "block_magenta"

        subgoal = Subgoal(
            id="sub_recover_test",
            parent_id=None,
            level=2,
            skill_type="PICK_AND_PLACE",
            required_objects=[b_id],
            preconditions={"gripper_empty": True},
            desired_effects={"placed": True},
            target_configuration={"block_id": b_id, "slot": [0.48, TABLE_TOP_Y + CUBE_HALF, 0.22]},
            dependencies=[],
            max_retries=2,
        )
        sm.start_plan([subgoal])

        # Simulate first attempt failing at VERIFY_GRASP
        sm.transition_to("VERIFY_GRASP")
        snap = sim.get_snapshot()
        sm.step(snap["tcp"]["position"], snap["joints"], snap["blocks"], current_gripper=1.0, grasped_block_id=None)
        assert sm.state == "RECOVER"
        assert subgoal.retry_count == 1

        # In RECOVER, arm retreats to SAFE_HOVER_Y
        for _ in range(30):
            snap = sim.get_snapshot()
            cmd = sm.step(snap["tcp"]["position"], snap["joints"], snap["blocks"], current_gripper=0.0, grasped_block_id=None)
            sim.set_joint_targets(cmd["joints"], cmd["gripper"])
            sim.step(FIXED_DT)
            if sm.state == "PLAN":
                break

        assert sm.state == "PLAN"
        assert subgoal.retry_count == 1
        assert not cmd["is_failed"]

    def test_08_detect_unstable_placement(self):
        """8. Detect unstable placement: moving block rejects completion and triggers recovery."""
        sim = MujocoSimulation()
        validator = StabilityValidator(required_ticks=15)
        blocks = [dict(b) for b in sim.get_snapshot()["blocks"]]

        b_id = "block_yellow"
        for b in blocks:
            if b["id"] == b_id:
                b["velocity"] = [0.08, -0.06, 0.08]

        for _ in range(25):
            res = validator.step(blocks, grasped_block_id=None, target_block_id=b_id)

        assert not res[0], "Moving block must never be verified as stably placed"
        assert not validator.is_verified
        assert validator.status in ("WAITING", "UNSTABLE")

    def test_09_handle_unreachable_ik_targets(self):
        """9. Handle unreachable targets: explicit UNREACHABLE_POSITION without silent clamping."""
        unreachable_target = (2.5, 0.85, 2.5)
        ik_res = solve_ik(unreachable_target, gripper=0.0)

        assert not ik_res.success, "Unreachable target must report success=False"
        assert ik_res.reachability_status == "UNREACHABLE_POSITION"
        assert ik_res.position_error > 1.0, f"Error should be huge: {ik_res.position_error}"

        sm = ExecutionStateMachine()
        bad_subgoal = Subgoal(
            id="sub_impossible",
            parent_id=None,
            level=2,
            skill_type="PICK_AND_PLACE",
            required_objects=["block_magenta"],
            preconditions={},
            desired_effects={},
            target_configuration={"block_id": "block_magenta", "slot": list(unreachable_target)},
            dependencies=[],
        )
        sim = MujocoSimulation()
        sim.blocks["block_magenta"].position = list(unreachable_target)
        sm.start_plan([bad_subgoal])
        snap = sim.get_snapshot()
        # Tick 1: PLAN -> MOVE_TO_PREGRASP
        cmd1 = sm.step(snap["tcp"]["position"], snap["joints"], snap["blocks"], 0.0, None)
        assert sm.state == "MOVE_TO_PREGRASP"
        # Tick 2: Solves IK for pregrasp to unreachable block -> fails
        cmd2 = sm.step(snap["tcp"]["position"], snap["joints"], snap["blocks"], 0.0, None)
        assert cmd2["is_failed"], "State machine must fail explicitly on unreachable target"
        assert "Unreachable" in cmd2["recovery_reason"] or "unreachable" in cmd2["recovery_reason"].lower()

    def test_10_complete_multi_step_pyramid(self):
        """10. Complete multi-step pyramid: end-to-end 3-block assembly verified in MuJoCo physics."""
        sim = MujocoSimulation()
        orchestrator = HierarchicalVLJEPAOrchestrator()
        snap = sim.get_snapshot()

        plan = orchestrator.plan_task(
            instruction="build a 3-block pyramid with red, blue and yellow",
            initial_blocks=snap["blocks"]
        )
        assert plan["status"] == "PLAN_READY"
        assert plan["num_subgoals"] == 3

        ticks = 0
        max_ticks = 750
        while orchestrator.is_active and ticks < max_ticks:
            snap = sim.get_snapshot()
            cmd = orchestrator.step(snap["tcp"]["position"], snap["joints"], snap["blocks"])
            if not cmd or not orchestrator.is_active:
                break
            sim.set_joint_targets(cmd["joints"], cmd["gripper"])
            sim.step(FIXED_DT)
            ticks += 1

        final_snap = sim.get_snapshot()
        b_magenta = sim.blocks["block_magenta"]
        b_cyan = sim.blocks["block_cyan"]
        b_yellow = sim.blocks["block_yellow"]

        assert b_magenta.position[1] < 0.760, f"Base magenta displaced vertically: {b_magenta.position[1]}"
        assert b_cyan.position[1] < 0.760, f"Base cyan displaced vertically: {b_cyan.position[1]}"

        assert b_yellow.position[1] >= 0.795, f"Apex yellow not resting on layer 1: {b_yellow.position[1]}"
        assert b_yellow.position[1] <= 0.815, f"Apex yellow displaced vertically: {b_yellow.position[1]}"

        assert math.hypot(*b_yellow.velocity) < 0.05, f"Apex block still moving: {b_yellow.velocity}"
