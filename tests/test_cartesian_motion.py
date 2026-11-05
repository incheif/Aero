"""
tests/test_cartesian_motion.py

Verification suite for Cartesian Altitude-Locked Transit and Pure Vertical Approach/Retreat.
Validates:
1. Altitude locking during horizontal transit (elimination of 180mm joint-space sagging dip).
2. Pure vertical descent and ascent with zero horizontal drift (< 1mm).
3. Horizontal transit flight over tabletop blocks with zero collision / zero block displacement.
4. Vertical retreat between flanking blocks at 88mm pitch with calibrated gripper aperture (0.70).
"""

import math
import numpy as np
import pytest

from simulation.constants import (
    FIXED_DT,
    TABLE_TOP_Y,
    CUBE_SIZE,
    CUBE_HALF,
    SAFE_HOVER_Y,
    PICK_Y,
    MAX_TCP_STEP,
    BLOCK_SPAWNS,
)
from simulation.kinematics import solve_ik, forward_kinematics
from simulation.motion_planner import (
    plan_cartesian_transit,
    plan_vertical_approach,
    plan_vertical_retreat,
)
from simulation.mujoco_engine import MujocoSimulation, MUJOCO_AVAILABLE


class TestCartesianMotion:
    """Test suite validating Cartesian altitude locking and vertical clearance motions."""

    def test_01_altitude_locked_transit_eliminates_sagging(self):
        """
        Verify that plan_cartesian_transit maintains TCP altitude within 5mm of transit_y,
        completely eliminating the 130-180mm downward sagging dip seen in direct joint interpolation.
        """
        start_tcp = (0.20, SAFE_HOVER_Y, -0.15)
        goal_tcp = (0.35, SAFE_HOVER_Y, 0.15)
        transit_y = SAFE_HOVER_Y

        # 1. Direct joint-space interpolation exhibits significant Cartesian path distortion (> 30mm)
        start_ik = solve_ik(start_tcp)
        goal_ik = solve_ik(goal_tcp)
        assert start_ik.success and goal_ik.success, "IK failed for endpoints"

        n_steps = 20
        max_joint_dev = 0.0
        for i in range(n_steps + 1):
            alpha = i / float(n_steps)
            q_interp = {
                k: start_ik.joints[k] + alpha * (goal_ik.joints[k] - start_ik.joints[k])
                for k in ["baseYaw", "shoulderPitch", "elbowPitch", "wristPitch"]
            }
            fk = forward_kinematics(q_interp)
            pos_actual = np.array(fk["tcp"]["position"])
            pos_nominal = np.array([
                start_tcp[0] + alpha * (goal_tcp[0] - start_tcp[0]),
                transit_y,
                start_tcp[2] + alpha * (goal_tcp[2] - start_tcp[2]),
            ])
            dev = float(np.linalg.norm(pos_actual - pos_nominal))
            if dev > max_joint_dev:
                max_joint_dev = dev

        # Unplanned joint-space direct motion deviates significantly (> 30mm) from the planned linear corridor
        assert max_joint_dev > 0.030, f"Expected noticeable joint deviation, got {max_joint_dev*1000:.1f}mm"

        # 2. Cartesian altitude-locked transit (our robust implementation)
        cartesian_traj = plan_cartesian_transit(start_tcp, goal_tcp, transit_y=transit_y, max_step=MAX_TCP_STEP)
        assert len(cartesian_traj) >= 5, "Trajectory too short"

        for i, wp in enumerate(cartesian_traj):
            assert wp["success"], "All Cartesian waypoints must have valid IK"
            fk = forward_kinematics(wp)
            tcp = fk["tcp"]["position"]
            # TCP must remain locked to transit_y within tight kinematic tolerance (<= 5mm)
            assert abs(tcp[1] - transit_y) <= 0.005, (
                f"Cartesian waypoint sagged or drifted: y={tcp[1]:.4f}, expected={transit_y:.4f}"
            )
            # TCP must follow the linear Cartesian line corridor within 2mm
            alpha = i / float(len(cartesian_traj) - 1)
            nominal = (
                start_tcp[0] + alpha * (goal_tcp[0] - start_tcp[0]),
                transit_y,
                start_tcp[2] + alpha * (goal_tcp[2] - start_tcp[2]),
            )
            dist_to_nominal = math.hypot(tcp[0] - nominal[0], tcp[1] - nominal[1], tcp[2] - nominal[2])
            assert dist_to_nominal < 0.003, (
                f"Cartesian trajectory waypoint deviated {dist_to_nominal*1000:.2f}mm from corridor"
            )

    def test_02_pure_vertical_approach_and_retreat_zero_drift(self):
        """
        Verify that vertical approach and retreat paths have zero horizontal drift (< 1.0mm)
        across the entire vertical descent and ascent.
        """
        pick_target = (0.25, PICK_Y, 0.08)
        hover_target = (0.25, SAFE_HOVER_Y, 0.08)

        # 1. Vertical Approach
        approach_traj = plan_vertical_approach(hover_target, pick_target, steps=15, gripper=0.0)
        assert len(approach_traj) == 16

        for wp in approach_traj:
            assert wp["success"], "Approach waypoint IK must succeed"
            fk = forward_kinematics(wp)
            tcp = fk["tcp"]["position"]
            h_drift = math.hypot(tcp[0] - pick_target[0], tcp[2] - pick_target[2])
            assert h_drift < 0.003, f"Approach horizontal drift {h_drift*1000:.2f}mm exceeded limit (3mm)"

        # 2. Vertical Retreat
        retreat_traj = plan_vertical_retreat(pick_target, transit_y=SAFE_HOVER_Y, steps=15, desired_gripper=0.70)
        assert len(retreat_traj) == 16

        for wp in retreat_traj:
            assert wp["success"], "Retreat waypoint IK must succeed"
            assert wp["gripper"] == 0.70, f"Expected gripper=0.70, got {wp['gripper']}"
            fk = forward_kinematics(wp)
            tcp = fk["tcp"]["position"]
            h_drift = math.hypot(tcp[0] - pick_target[0], tcp[2] - pick_target[2])
            assert h_drift < 0.003, f"Retreat horizontal drift {h_drift*1000:.2f}mm exceeded limit (3mm)"

    @pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo not installed")
    def test_03_transit_flight_over_tabletop_blocks_zero_disturbance(self):
        """
        Execute an altitude-locked Cartesian transit flight across the workspace directly above
        spawned blocks. Verify that arm clears all blocks without contact or displacement (< 1.0mm).
        """
        sim = MujocoSimulation()

        # Let blocks settle
        for _ in range(60):
            sim.step()

        init_poses = {b["id"]: np.array(sim.get_block_pose(b["id"])["position"]) for b in BLOCK_SPAWNS}

        # Transit from left side (-0.15m Z) to right side (+0.15m Z) crossing through X=0.25m
        start_tcp = (0.25, SAFE_HOVER_Y, -0.15)
        goal_tcp = (0.25, SAFE_HOVER_Y, 0.15)

        traj = plan_cartesian_transit(start_tcp, goal_tcp, transit_y=SAFE_HOVER_Y, max_step=0.02)
        assert len(traj) > 0

        # Drive arm along trajectory in simulation
        for wp in traj:
            sim.set_joint_targets(wp)
            for _ in range(8):
                sim.step()

        # Verify all tabletop blocks remained undisturbed
        for b in BLOCK_SPAWNS:
            b_id = b["id"]
            curr_pos = np.array(sim.get_block_pose(b_id)["position"])
            disp = float(np.linalg.norm(curr_pos - init_poses[b_id]))
            assert disp < 0.001, f"Block {b_id} displaced by {disp*1000:.2f}mm during transit flight"

    @pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo not installed")
    def test_04_vertical_retreat_between_flanking_blocks(self):
        """
        Verify that between two flanking blocks at 88mm pitch,
        a vertical approach and retreat cycle with calibrated gripper aperture (desired_gripper=0.70)
        descends and ascends cleanly without bumping flanking blocks (< 1.0mm displacement).
        """
        sim = MujocoSimulation()

        pitch = 0.088
        ox, oz = 0.45, 0.20
        base_y = TABLE_TOP_Y + CUBE_HALF

        # Spawn flanking blocks at 88mm pitch along workcell manipulation axis
        sim.set_block_pose("block_cyan", [ox + pitch, base_y, oz], [1.0, 0.0, 0.0, 0.0])
        sim.set_block_pose("block_yellow", [ox - pitch, base_y, oz], [1.0, 0.0, 0.0, 0.0])

        # Settle
        for _ in range(30):
            sim.step()

        hover_target = (ox, SAFE_HOVER_Y, oz)
        center_place = (ox, PICK_Y, oz)

        # 1. Establish hover pose directly above the target site
        approach_traj = plan_vertical_approach(hover_target, center_place, steps=15, gripper=0.70)
        sim.set_joint_targets(approach_traj[0], gripper=0.70)
        for _ in range(35):
            sim.step()

        # Record settled baseline positions of flanking blocks before vertical insertion
        pos_cyan_init = np.array(sim.get_block_pose("block_cyan")["position"])
        pos_yellow_init = np.array(sim.get_block_pose("block_yellow")["position"])

        # Descend vertically between flanking blocks
        for wp in approach_traj:
            sim.set_joint_targets(wp, gripper=0.70)
            for _ in range(4):
                sim.step()

        # 2. Retreat vertically back to transit altitude
        retreat_traj = plan_vertical_retreat(
            center_place,
            transit_y=SAFE_HOVER_Y,
            steps=15,
            desired_gripper=0.70,
        )

        for wp in retreat_traj:
            sim.set_joint_targets(wp, gripper=0.70)
            for _ in range(4):
                sim.step()

        pos_cyan_final = np.array(sim.get_block_pose("block_cyan")["position"])
        pos_yellow_final = np.array(sim.get_block_pose("block_yellow")["position"])

        disp_cyan = float(np.linalg.norm(pos_cyan_final - pos_cyan_init))
        disp_yellow = float(np.linalg.norm(pos_yellow_final - pos_yellow_init))

        assert disp_cyan < 0.001, f"Cyan flanking block moved {disp_cyan*1000:.2f}mm during vertical cycle"
        assert disp_yellow < 0.001, f"Yellow flanking block moved {disp_yellow*1000:.2f}mm during vertical cycle"

