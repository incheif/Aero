"""
tests/test_block_physics_and_collision.py

Comprehensive verification of rigid block physics, heavy inertia, arm impassability,
and block-block non-penetration in the AERO robotic workcell.
"""

import math
import numpy as np
import pytest

from simulation.constants import (
    FIXED_DT,
    TABLE_TOP_Y,
    CUBE_SIZE,
    CUBE_HALF,
    BLOCK_MASS,
    BLOCK_SPAWNS,
    SAFE_HOVER_Y,
    DEFAULT_JOINTS,
)
from simulation.kinematics import solve_ik, forward_kinematics
from simulation.mujoco_engine import MujocoSimulation, MUJOCO_AVAILABLE


@pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo not installed")
class TestBlockPhysicsAndCollision:
    """Test suite validating heavy rigid block physics and collision impassability."""

    def test_01_heavy_block_equilibrium_and_mass(self):
        """1. Verify blocks have heavy mass (0.35 kg) and settle rigidly under gravity without floating."""
        sim = MujocoSimulation()
        assert BLOCK_MASS == 0.35, f"Expected BLOCK_MASS=0.35 kg, got {BLOCK_MASS}"

        # Verify mass of all block bodies in MuJoCo model
        for spawn in BLOCK_SPAWNS:
            b_id = spawn["id"]
            body_id = sim.model.body(b_id).id
            body_mass = sim.model.body_mass[body_id]
            assert abs(body_mass - BLOCK_MASS) < 1e-4, f"Body {b_id} mass {body_mass} != {BLOCK_MASS}"

        # Step simulation for 30 ticks under gravity (-9.81 m/s^2)
        for _ in range(30):
            sim.step(FIXED_DT)

        snap = sim.get_snapshot()
        expected_table_y = TABLE_TOP_Y + CUBE_HALF  # 0.72 + 0.0275 = 0.7475m

        for b in snap["blocks"]:
            pos = b["position"]
            vel = b["velocity"]
            speed = math.hypot(*vel)

            # Resting altitude on tabletop (penetration/bounce < 1.5 mm)
            assert abs(pos[1] - expected_table_y) < 0.0015, (
                f"Block {b['id']} not resting stably on tabletop: pos_y={pos[1]:.4f}, expected={expected_table_y:.4f}"
            )
            # Velocity must be near zero (zero floating or flying)
            assert speed < 0.05, f"Block {b['id']} drifting with speed {speed:.4f} m/s"

    def test_02_block_block_stacking_non_penetration(self):
        """2. Verify blocks cannot pass through each other: multi-layer stack non-penetration (< 1.0 mm)."""
        sim = MujocoSimulation()

        # Stack block_magenta (base), block_cyan (mid), and block_yellow (top) at STACK_ORIGIN
        stack_x, stack_z = 0.45, 0.15
        base_y = TABLE_TOP_Y + CUBE_HALF
        mid_y = base_y + CUBE_SIZE
        apex_y = mid_y + CUBE_SIZE

        sim.set_block_pose("block_magenta", [stack_x, base_y + 0.002, stack_z], [0, 0, 0, 1])
        sim.set_block_pose("block_cyan", [stack_x, mid_y + 0.004, stack_z], [0, 0, 0, 1])
        sim.set_block_pose("block_yellow", [stack_x, apex_y + 0.006, stack_z], [0, 0, 0, 1])

        # Settle for 60 ticks under physics
        for _ in range(60):
            sim.step(FIXED_DT)

        snap = sim.get_snapshot()
        b_map = {b["id"]: b for b in snap["blocks"]}

        p_base = b_map["block_magenta"]["position"]
        p_mid = b_map["block_cyan"]["position"]
        p_apex = b_map["block_yellow"]["position"]

        # Base rests on table
        assert abs(p_base[1] - base_y) < 0.0015

        # Mid block rests on base block: inter-block distance >= CUBE_SIZE - 0.0010m
        dist_base_mid = p_mid[1] - p_base[1]
        penetration_1 = CUBE_SIZE - dist_base_mid
        assert penetration_1 < 0.0010, f"Mid penetrated base by {penetration_1 * 1000:.2f} mm (target < 1.0 mm)"
        assert dist_base_mid >= CUBE_SIZE - 0.0010, f"Mid block sank into base: dist={dist_base_mid:.4f}"

        # Apex block rests on mid block
        dist_mid_apex = p_apex[1] - p_mid[1]
        penetration_2 = CUBE_SIZE - dist_mid_apex
        assert penetration_2 < 0.0010, f"Apex penetrated mid by {penetration_2 * 1000:.2f} mm (target < 1.0 mm)"
        assert dist_mid_apex >= CUBE_SIZE - 0.0010, f"Apex block sank into mid: dist={dist_mid_apex:.4f}"

    def test_03_arm_impassability_blocks_motion(self):
        """3. Verify arm cannot pass through blocks: arm links physically contact and cannot tunnel."""
        sim = MujocoSimulation()

        # Place an obstacle block directly in front of the arm
        obs_x, obs_y, obs_z = 0.35, TABLE_TOP_Y + CUBE_HALF, 0.0
        sim.set_block_pose("block_orange", [obs_x, obs_y, obs_z], [0, 0, 0, 1])

        # Step to settle block
        for _ in range(15):
            sim.step(FIXED_DT)

        # Command a target that would penetrate right through the center of block_orange
        penetrating_target = (obs_x, obs_y, obs_z)
        ik_res = solve_ik(penetrating_target, gripper=0.0)
        assert ik_res.success

        sim.set_joint_targets(ik_res.joints, gripper=0.0)

        # Step simulation to let the arm attempt to move into the block
        for _ in range(60):
            sim.step(FIXED_DT)

        # Verify arm was blocked by physical impassability
        fk = forward_kinematics(sim.joints)
        tcp_final = fk["tcp"]["position"]

        # The arm should have been stopped before reaching the target inside the block
        dist_to_center = math.hypot(tcp_final[0] - obs_x, tcp_final[1] - obs_y, tcp_final[2] - obs_z)
        assert dist_to_center > 0.020, (
            f"Arm tunneled into block center! Final TCP={tcp_final}, Target={penetrating_target}"
        )

        # Verify block still exists solidly on table without falling through or flying away
        snap = sim.get_snapshot()
        b_orange = next(b for b in snap["blocks"] if b["id"] == "block_orange")
        assert b_orange["position"][1] >= TABLE_TOP_Y + CUBE_HALF - 0.002
        assert b_orange["position"][1] <= TABLE_TOP_Y + CUBE_HALF + 0.015

    def test_04_rotated_block_collision_and_grasp(self):
        """4. Verify orientation-aware grasp and non-destructive transport of rotated block."""
        sim = MujocoSimulation()
        b_id = "block_yellow"

        # Rotate yellow block by 30 deg (0.5236 rad) around Y at its own spawn position
        theta = 0.5236
        qy = math.sin(theta * 0.5)
        qw = math.cos(theta * 0.5)
        rot_30 = [0.0, qy, 0.0, qw]

        target_pos = list(sim.blocks[b_id].position)
        sim.set_block_pose(b_id, target_pos, rot_30)

        for _ in range(15):
            sim.step(FIXED_DT)

        target_pos = list(sim.blocks[b_id].position)

        # 1. Pregrasp hover directly above block
        ik_pre = solve_ik((target_pos[0], SAFE_HOVER_Y, target_pos[2]), gripper=0.0)
        assert ik_pre.success
        sim.set_joint_targets(ik_pre.joints, gripper=0.0)
        for _ in range(35):
            sim.step(FIXED_DT)

        # 2. Vertical descent to grasp approach pose
        approach_target = (target_pos[0], target_pos[1] + 0.008, target_pos[2])
        ik_res = solve_ik(approach_target, gripper=0.0)
        assert ik_res.success
        sim.set_joint_targets(ik_res.joints, gripper=0.0)
        for _ in range(40):
            sim.step(FIXED_DT)

        # 3. Close gripper
        sim.set_joint_targets(ik_res.joints, gripper=1.0)
        for _ in range(35):
            sim.step(FIXED_DT)

        # Confirm grasp attachment
        assert sim.grasped_block_id == "block_yellow", "Failed to grasp 30-deg rotated block"
        assert sim.blocks["block_yellow"].is_grasped

        # 4. Lift block vertically
        lift_target = (target_pos[0], SAFE_HOVER_Y, target_pos[2])
        ik_lift = solve_ik(lift_target, gripper=1.0)
        assert ik_lift.success
        sim.set_joint_targets(ik_lift.joints, gripper=1.0)
        for _ in range(50):
            sim.step(FIXED_DT)

        snap_lifted = sim.get_snapshot()
        b_lifted = next(b for b in snap_lifted["blocks"] if b["id"] == "block_yellow")

        # Verify block ascended with end effector without micro-slipping out
        assert b_lifted["position"][1] > target_pos[1] + 0.08, "Grasped rotated block dropped during lift"
        assert sim.blocks["block_yellow"].is_grasped

    def test_05_cluttered_workspace_jaw_clearance(self):
        """5. Verify 88mm pitch provides clean jaw clearance so neighbor blocks remain undisturbed."""
        sim = MujocoSimulation()

        pitch = 0.088
        ox, oz = 0.45, 0.20
        base_y = TABLE_TOP_Y + CUBE_HALF

        # Spawn 3 blocks in a row along X at 88mm pitch
        sim.set_block_pose("block_magenta", [ox + pitch, base_y, oz], [0, 0, 0, 1])
        sim.set_block_pose("block_cyan", [ox, base_y, oz], [0, 0, 0, 1])
        sim.set_block_pose("block_yellow", [ox - pitch, base_y, oz], [0, 0, 0, 1])

        for _ in range(20):
            sim.step(FIXED_DT)

        # 1. Hover directly above the cyan block
        ik_hover = solve_ik((ox, SAFE_HOVER_Y, oz), gripper=0.70)
        assert ik_hover.success
        sim.set_joint_targets(ik_hover.joints, gripper=0.70)
        for _ in range(35):
            sim.step(FIXED_DT)

        init_snap = sim.get_snapshot()
        init_m = next(b for b in init_snap["blocks"] if b["id"] == "block_magenta")["position"]
        init_y = next(b for b in init_snap["blocks"] if b["id"] == "block_yellow")["position"]

        # 2. Descend vertically between magenta and yellow with gripper at release aperture (0.70)
        center_target = (ox, base_y + 0.008, oz)
        ik_res = solve_ik(center_target, gripper=0.70)
        assert ik_res.success
        sim.set_joint_targets(ik_res.joints, gripper=0.70)
        for _ in range(40):
            sim.step(FIXED_DT)

        # Check positions of flanking blocks (magenta and yellow)
        curr_snap = sim.get_snapshot()
        curr_m = next(b for b in curr_snap["blocks"] if b["id"] == "block_magenta")["position"]
        curr_y = next(b for b in curr_snap["blocks"] if b["id"] == "block_yellow")["position"]

        disp_m = math.hypot(curr_m[0] - init_m[0], curr_m[2] - init_m[2])
        disp_y = math.hypot(curr_y[0] - init_y[0], curr_y[2] - init_y[2])

        # Flanking blocks must not be bumped or displaced by the descending jaws (< 1.5 mm displacement)
        assert disp_m < 0.0015, f"Flanking magenta block bumped: displacement={disp_m * 1000:.2f} mm"
        assert disp_y < 0.0015, f"Flanking yellow block bumped: displacement={disp_y * 1000:.2f} mm"
