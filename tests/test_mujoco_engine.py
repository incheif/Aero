"""
Unit tests for MuJoCo Physics Backend in AERO.
Verifies contact manifolds, rigid-body equilibrium, grasp dynamics, and snapshot parity.
"""

import pytest
import math
from simulation.mujoco_engine import MujocoSimulation, MUJOCO_AVAILABLE
from simulation.constants import CUBE_SIZE, TABLE_TOP_Y, STACK_ORIGIN, FIXED_DT
from vla.h_jepa.orchestrator import HierarchicalVLJEPAOrchestrator


@pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo not installed")
class TestMujocoEngine:

    def test_mujoco_initialization_and_snapshot(self):
        sim = MujocoSimulation()
        snap = sim.get_snapshot()

        assert snap["tick"] == 0
        assert len(snap["blocks"]) == 7
        assert "tcp" in snap
        assert "joints" in snap
        assert "scores" in snap

        for b in snap["blocks"]:
            assert b["position"][1] >= TABLE_TOP_Y + CUBE_SIZE * 0.5 - 0.005
            assert b["is_grasped"] is False
            assert b["supported_by"] == "table"

    def test_mujoco_table_equilibrium(self):
        sim = MujocoSimulation()
        # Step for 30 ticks to verify gravity settling
        for _ in range(30):
            sim.step()

        snap = sim.get_snapshot()
        for b in snap["blocks"]:
            # Blocks should rest stably on tabletop (Y ~ 0.7475)
            assert 0.740 <= b["position"][1] <= 0.755
            # Velocity should be near zero (settled)
            assert math.hypot(*b["velocity"]) < 0.05

    def test_mujoco_h_jepa_pyramid_assembly(self):
        sim = MujocoSimulation()
        orchestrator = HierarchicalVLJEPAOrchestrator()
        snap = sim.get_snapshot()

        plan_res = orchestrator.plan_task(
            instruction="build a 3-block pyramid with red, blue and yellow",
            initial_blocks=snap["blocks"]
        )
        assert plan_res["status"] == "PLAN_READY"

        ticks = 0
        max_ticks = 600
        while orchestrator.is_active and ticks < max_ticks:
            snap = sim.get_snapshot()
            step_cmd = orchestrator.step(
                current_tcp=snap["tcp"]["position"],
                current_joints=snap["joints"],
                blocks=snap["blocks"],
            )
            if step_cmd is None or not orchestrator.is_active:
                break
            sim.set_joint_targets(step_cmd["joints"], step_cmd["gripper"])
            sim.step(FIXED_DT)
            ticks += 1

        final_snap = sim.get_snapshot()
        b_magenta = sim.blocks["block_magenta"]
        b_cyan = sim.blocks["block_cyan"]
        b_yellow = sim.blocks["block_yellow"]

        # Base blocks are on table (layer 0)
        assert b_magenta.position[1] < 0.760
        assert b_cyan.position[1] < 0.760

        # Apex block (Yellow) is resting on top of base blocks (layer 1)
        assert b_yellow.position[1] >= 0.795
        assert b_yellow.position[1] <= 0.815

        # Check penetration between all blocks
        max_pen = 0.0
        for b1_id, b1 in sim.blocks.items():
            for b2_id, b2 in sim.blocks.items():
                if b1_id >= b2_id:
                    continue
                dx = abs(b1.position[0] - b2.position[0])
                dy = abs(b1.position[1] - b2.position[1])
                dz = abs(b1.position[2] - b2.position[2])
                ox = CUBE_SIZE - dx
                oy = CUBE_SIZE - dy
                oz = CUBE_SIZE - dz
                if ox > 0 and oy > 0 and oz > 0:
                    pen = min(ox, oy, oz)
                    if pen > max_pen:
                        max_pen = pen

        # Native MuJoCo contact compliance is well under 1.0 mm
        assert max_pen < 0.001, f"Excessive penetration in MuJoCo: {max_pen * 1000:.2f} mm"

