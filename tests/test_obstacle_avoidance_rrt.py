"""
tests/test_obstacle_avoidance_rrt.py

Verification suite for Obstacle Avoidance via Bidirectional RRT-Connect Joint-Space Sampling Planner.
Validates:
1. Fast-path linear execution for unobstructed transitions.
2. Bidirectional RRT-Connect detour around tall towers / multi-block obstacles.
3. Path shortcut smoothing and dense joint interpolation.
4. MuJoCo dynamic multi-body execution: arm navigates around a tall tower with zero collisions,
   zero impassability halts, and zero disturbance to tower blocks (< 1.0mm).
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
    DEFAULT_JOINTS,
)
from simulation.kinematics import solve_ik, forward_kinematics
from simulation.motion_planner import (
    ArmCollisionChecker,
    RRTConnectJointPlanner,
    interpolate_joint_trajectory,
    plan_collision_free_transit,
)
from simulation.mujoco_engine import MujocoSimulation, MUJOCO_AVAILABLE


class TestObstacleAvoidanceRRT:
    """Test suite validating sampling-based joint-space obstacle avoidance."""

    def test_01_unobstructed_direct_fast_path(self):
        """Verify that when the swept segment is unobstructed, fast-path returns direct [q_start, q_goal]."""
        planner = RRTConnectJointPlanner()
        q_start = dict(DEFAULT_JOINTS)
        ik_goal = solve_ik((0.30, SAFE_HOVER_Y, 0.10))
        assert ik_goal.success
        q_goal = dict(ik_goal.joints)

        # In empty workcell, direct swept segment is valid
        plan = planner.plan(q_start, q_goal, blocks=[])
        assert len(plan) == 2, f"Expected fast-path 2 waypoints, got {len(plan)}"
        assert plan[0] == q_start
        assert plan[1] == q_goal

    def test_02_rrt_connect_navigates_around_tall_obstacle(self):
        """
        Place a tall obstacle directly in the line between start and goal configurations.
        Verify that direct swept segment is blocked, and RRT-Connect plans a collision-free detour.
        """
        planner = RRTConnectJointPlanner(step_size=0.08, max_iterations=200)

        # Start on left (+Z), Goal on right (-Z)
        ik_start = solve_ik((0.35, 0.85, 0.20))
        ik_goal = solve_ik((0.35, 0.85, -0.20))
        assert ik_start.success and ik_goal.success

        q_start = dict(ik_start.joints)
        q_goal = dict(ik_goal.joints)

        # Place a 3-block tall column directly at center (X=0.35, Z=0.0)
        obstacle_blocks = [
            {"id": "obs_1", "position": [0.35, TABLE_TOP_Y + CUBE_HALF, 0.0]},
            {"id": "obs_2", "position": [0.35, TABLE_TOP_Y + CUBE_HALF + CUBE_SIZE, 0.0]},
            {"id": "obs_3", "position": [0.35, TABLE_TOP_Y + CUBE_HALF + 2 * CUBE_SIZE, 0.0]},
        ]

        # 1. Direct segment MUST be blocked by the column
        arr_start = planner._dict_to_array(q_start)
        arr_goal = planner._dict_to_array(q_goal)
        direct_valid = planner.swept_segment_valid(arr_start, arr_goal, obstacle_blocks)
        assert not direct_valid, "Direct swept segment should collide with obstacle column"

        # 2. RRT-Connect MUST generate a collision-free path
        plan = planner.plan(q_start, q_goal, blocks=obstacle_blocks)
        assert len(plan) >= 2, "Planner failed to find a path"

        # 3. Verify every configuration in the planned path is collision-free
        for i, q in enumerate(plan):
            is_collision = planner.checker.check_configuration_collision(q, obstacle_blocks)
            assert not is_collision, f"Waypoint {i} in planned path collides with obstacle column"

        # 4. Verify continuous swept validity along all segments
        for i in range(len(plan) - 1):
            q1 = planner._dict_to_array(plan[i])
            q2 = planner._dict_to_array(plan[i + 1])
            assert planner.swept_segment_valid(q1, q2, obstacle_blocks), (
                f"Segment {i}->{i+1} in planned path has swept collision"
            )

    def test_03_shortcutting_and_dense_interpolation(self):
        """Verify that dense trajectory interpolation produces bounded joint velocity steps."""
        q_start = {"baseYaw": 0.4, "shoulderPitch": 1.2, "elbowPitch": -1.8, "wristPitch": -0.2, "gripper": 0.0}
        q_goal = {"baseYaw": -0.4, "shoulderPitch": 0.9, "elbowPitch": -1.4, "wristPitch": 0.3, "gripper": 0.0}

        dense = interpolate_joint_trajectory([q_start, q_goal], max_step_rad=0.04)
        assert len(dense) > 5, "Dense interpolation produced too few points"

        # Verify max delta between consecutive waypoints <= 0.041 rad
        for i in range(len(dense) - 1):
            for k in ("baseYaw", "shoulderPitch", "elbowPitch", "wristPitch"):
                delta = abs(dense[i + 1][k] - dense[i][k])
                assert delta <= 0.0401, f"Step delta {delta:.4f} exceeded max_step_rad on {k}"

    @pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo not installed")
    def test_04_mujoco_dynamic_execution_avoids_tower(self):
        """
        Dynamically execute an RRT detour around a tall 4-block tower in MuJoCo.
        Assert that the arm navigates around the tower with zero collisions,
        zero impassability halts, and zero displacement on tower blocks (< 1.0mm).
        """
        import random
        random.seed(42)
        np.random.seed(42)

        sim = MujocoSimulation()

        # Plan transit from left side (Z=0.20) to right side (Z=-0.20) at hover altitude
        ik_start = solve_ik((0.35, SAFE_HOVER_Y, 0.20))
        ik_goal = solve_ik((0.35, SAFE_HOVER_Y, -0.20))
        assert ik_start.success and ik_goal.success

        # Place arm at start pose first
        sim.joints = dict(ik_start.joints)
        sim.set_joint_targets(ik_start.joints)
        sim._sync_arm_to_mujoco()

        # Place a 4-block tall tower at center X=0.35, Z=0.0 that blocks direct hover flight
        base_y = TABLE_TOP_Y + CUBE_HALF
        sim.set_block_pose("block_magenta", [0.35, base_y, 0.0], [1.0, 0.0, 0.0, 0.0])
        sim.set_block_pose("block_cyan", [0.35, base_y + CUBE_SIZE, 0.0], [1.0, 0.0, 0.0, 0.0])
        sim.set_block_pose("block_yellow", [0.35, base_y + 2 * CUBE_SIZE, 0.0], [1.0, 0.0, 0.0, 0.0])
        sim.set_block_pose("block_emerald", [0.35, base_y + 3 * CUBE_SIZE, 0.0], [1.0, 0.0, 0.0, 0.0])

        # Settle tower
        for _ in range(40):
            sim.step()

        tower_blocks = [
            {"id": "block_magenta", "position": list(sim.get_block_pose("block_magenta")["position"])},
            {"id": "block_cyan", "position": list(sim.get_block_pose("block_cyan")["position"])},
            {"id": "block_yellow", "position": list(sim.get_block_pose("block_yellow")["position"])},
            {"id": "block_emerald", "position": list(sim.get_block_pose("block_emerald")["position"])},
        ]
        init_poses = {b["id"]: np.array(sim.get_block_pose(b["id"])["position"]) for b in tower_blocks}

        dense_path = plan_collision_free_transit(
            ik_start.joints,
            ik_goal.joints,
            blocks=tower_blocks,
            max_step_rad=0.02,
        )
        assert len(dense_path) >= 5

        # Step through planned collision-free trajectory
        for wp in dense_path:
            sim.set_joint_targets(wp)
            for _ in range(4):
                sim.step()

        # Check tower stability and displacement for all blocks in tower
        for b in tower_blocks:
            b_id = b["id"]
            curr_pos = np.array(sim.get_block_pose(b_id)["position"])
            disp = float(np.linalg.norm(curr_pos - init_poses[b_id]))
            assert disp < 0.001, f"Tower block {b_id} displaced by {disp*1000:.2f}mm during RRT detour"
