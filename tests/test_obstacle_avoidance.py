"""
Unit tests for 3D Obstacle Avoidance & Gantry Motion Profiling in AERO.
Verifies that the robot arm never cuts through or collides with intermediate blocks
during pick-and-place trajectories.
"""

import pytest
import math
from simulation.constants import CUBE_SIZE, TABLE_TOP_Y, FIXED_DT
from simulation.physics_engine import RobotArmSimulation
from simulation.mujoco_engine import MujocoSimulation, MUJOCO_AVAILABLE
from vla.h_jepa.orchestrator import HierarchicalVLJEPAOrchestrator


class TestObstacleAvoidance:

    def test_corridor_detour_geometry(self):
        orchestrator = HierarchicalVLJEPAOrchestrator()
        # Start at Magenta [0.26, 0.12], Goal at Center Pad [0.48, 0.22]
        # Obstacle placed directly on the line of sight at [0.37, 0.75, 0.17]
        blocks = [
            {"id": "block_magenta", "position": [0.26, 0.748, 0.12]},
            {"id": "block_orange", "position": [0.37, 0.748, 0.17]},
        ]
        detour = orchestrator._find_corridor_detour(
            start=(0.26, 0.12),
            goal=(0.48, 0.22),
            blocks=blocks,
            ignore_ids={"block_magenta"},
            clearance=0.075,
        )
        assert detour is not None, "Failed to detect obstacle in direct path"
        # Verify detour maintains safe separation from obstacle
        dist_to_obs = math.hypot(detour[0] - 0.37, detour[1] - 0.17)
        assert dist_to_obs >= 0.070, f"Detour waypoint too close to obstacle: {dist_to_obs*1000:.1f} mm"

    def test_pick_and_place_with_corridor_obstacle_analytical(self):
        sim = RobotArmSimulation()
        orchestrator = HierarchicalVLJEPAOrchestrator()

        # Place block_orange as an obstacle between Magenta and pad
        sim.blocks["block_orange"].position = [0.37, 0.748, 0.17]
        snap = sim.get_snapshot()

        plan = orchestrator.plan_task("move red cube to center pad", snap["blocks"])
        assert plan["status"] == "PLAN_READY"

        max_overlap = 0.0
        ticks = 0
        while orchestrator.is_active and ticks < 300:
            snap = sim.get_snapshot()
            tcp = snap["tcp"]["position"]
            step_cmd = orchestrator.step(tcp, snap["joints"], snap["blocks"])
            if step_cmd is None or not orchestrator.is_active:
                break
            sim.set_joint_targets(step_cmd["joints"], step_cmd["gripper"])
            sim.step(FIXED_DT)
            ticks += 1

            # Check overlap between held block and obstacle
            b_mag = sim.blocks["block_magenta"]
            b_obs = sim.blocks["block_orange"]
            dx = abs(b_mag.position[0] - b_obs.position[0])
            dy = abs(b_mag.position[1] - b_obs.position[1])
            dz = abs(b_mag.position[2] - b_obs.position[2])
            ox = CUBE_SIZE - dx
            oy = CUBE_SIZE - dy
            oz = CUBE_SIZE - dz
            if ox > 0 and oy > 0 and oz > 0:
                overlap = min(ox, oy, oz)
                if overlap > max_overlap:
                    max_overlap = overlap

        assert max_overlap < 0.001, f"Collision detected with obstacle: {max_overlap*1000:.2f} mm"

    @pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo not installed")
    def test_pick_and_place_with_corridor_obstacle_mujoco(self):
        sim = MujocoSimulation()
        orchestrator = HierarchicalVLJEPAOrchestrator()

        # Place block_orange in line of flight in MuJoCo
        obs_pos = [0.37, 0.7475, 0.17]
        sim.blocks["block_orange"].position = list(obs_pos)
        adr = sim.joint_indices["block_orange"]
        sim.data.qpos[adr["qpos_adr"]:adr["qpos_adr"]+3] = obs_pos
        sim.data.qvel[adr["qvel_adr"]:adr["qvel_adr"]+6] = 0.0

        snap = sim.get_snapshot()
        plan = orchestrator.plan_task("move red cube to center pad", snap["blocks"])
        assert plan["status"] == "PLAN_READY"

        max_overlap = 0.0
        ticks = 0
        while orchestrator.is_active and ticks < 300:
            snap = sim.get_snapshot()
            tcp = snap["tcp"]["position"]
            step_cmd = orchestrator.step(tcp, snap["joints"], snap["blocks"])
            if step_cmd is None or not orchestrator.is_active:
                break
            sim.set_joint_targets(step_cmd["joints"], step_cmd["gripper"])
            sim.step(FIXED_DT)
            ticks += 1

            b_mag = sim.blocks["block_magenta"]
            b_obs = sim.blocks["block_orange"]
            dx = abs(b_mag.position[0] - b_obs.position[0])
            dy = abs(b_mag.position[1] - b_obs.position[1])
            dz = abs(b_mag.position[2] - b_obs.position[2])
            ox = CUBE_SIZE - dx
            oy = CUBE_SIZE - dy
            oz = CUBE_SIZE - dz
            if ox > 0 and oy > 0 and oz > 0:
                overlap = min(ox, oy, oz)
                if overlap > max_overlap:
                    max_overlap = overlap

        assert max_overlap < 0.001, f"Collision detected in MuJoCo: {max_overlap*1000:.2f} mm"

