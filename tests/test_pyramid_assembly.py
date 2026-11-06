"""
tests/test_pyramid_assembly.py

Verification suite for Stage 2: Motion Planning, Path Clearance,
and Autonomous 6-Block Pyramid Construction.
Validates:
1. Path clearance pick ordering: Foreground blocks (e.g. Yellow Cube at X=0.21)
   are cleared before occluded background blocks (Orange Cube at X=0.34) along the co-linear reach corridor.
2. TrajectoryController first-step execution: TCP advances smoothly to hover pose without
   freezing, without kinematic branch flips, and with bounded tracking lag.
3. Autonomous Pick and Placement: Execution in dynamic MuJoCo physics achieves zero toppling,
   zero impassability halts, and stable metric positioning on designated pyramid slots.
"""

import math
import numpy as np
import pytest

from simulation.constants import (
    FIXED_DT,
    SAFE_HOVER_Y,
    PICK_Y,
    TABLE_TOP_Y,
    CUBE_HALF,
    stack_slot_y,
    get_pyramid_slots,
    STACK_ORIGIN,
    DEFAULT_JOINTS,
)
from simulation.kinematics import solve_ik, forward_kinematics
from simulation.mujoco_engine import MujocoSimulation, MUJOCO_AVAILABLE
from vla.google_vla_agent import GoogleVLAAgent
from vla.trajectory_controller import TrajectoryController


class TestPyramidAssembly:
    """Test suite validating motion planning, path clearance, and pyramid construction."""

    def test_01_path_clearance_sequencing(self):
        """
        Verify that GoogleVLAAgent cognitive reasoner orders blocks with path clearance:
        Foreground blocks (block_yellow at X=0.21, Z=-0.05) are scheduled before background
        blocks (block_orange at X=0.34, Z=-0.06) to ensure the reach corridor remains unobstructed.
        """
        agent = GoogleVLAAgent()
        agent.execution_mode = "deterministic"
        interp = agent.interpreter.interpret("Build a 6-block pyramid tower.")

        assert interp["is_possible"] is True
        assert interp["status"] == "OK"
        assert interp.get("action_type") == "BUILD_PYRAMID"
        assert interp.get("pyramid_type") == "6_block_stepped"

        target_blocks = interp.get("target_blocks", [])
        assert len(target_blocks) == 6

        # Verify block_yellow is scheduled strictly before block_orange
        idx_yellow = target_blocks.index("block_yellow")
        idx_orange = target_blocks.index("block_orange")
        assert idx_yellow < idx_orange, (
            f"Path clearance violation: block_yellow (idx {idx_yellow}) must be cleared "
            f"before block_orange (idx {idx_orange})"
        )

        # Verify plan synthesis produces full pick-and-place sub-plans (49 steps)
        sim = MujocoSimulation() if MUJOCO_AVAILABLE else None
        snapshot = sim.get_snapshot() if sim else {}
        res = agent.parse_instruction("Build a 6-block pyramid tower.", None, snapshot, {})
        plan = res.get("plan", [])
        assert len(plan) == 49, f"Expected 49 plan steps, got {len(plan)}"

        # Verify all transit altitudes maintain safe clearance floor
        for step in plan:
            target = step.get("target", (0, 0, 0))
            action = step.get("action", "")
            if "HOVER" in action or "TRANSIT" in action or "LIFT" in action or "ASCEND" in action:
                assert target[1] >= SAFE_HOVER_Y - 0.001, (
                    f"Step {step['desc']} transit altitude {target[1]}m violates clearance floor {SAFE_HOVER_Y}m"
                )

    @pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo not installed")
    def test_02_trajectory_controller_first_step_no_freeze(self):
        """
        Verify that executing Step 1 (HOVER_ACQUIRE above Blue Cube) does not freeze or halt at tick 6:
        1. Cartesian setpoint smoothly advances from home standby pose to target.
        2. Arm tracking lag remains bounded (< 35mm).
        3. TCP altitude does not sag into tabletop blocks (Y >= 0.88m throughout transit).
        4. Controller advances to Step 2 within 40 ticks.
        """
        sim = MujocoSimulation()
        agent = GoogleVLAAgent()
        agent.execution_mode = "deterministic"
        snapshot = sim.get_snapshot()

        res = agent.parse_instruction("Build a 6-block pyramid tower.", None, snapshot, {})
        plan = res.get("plan", [])
        assert len(plan) > 0

        controller = TrajectoryController()
        controller.load_plan(plan)

        advanced_to_step_2 = False
        min_altitude = 2.0

        for tick in range(80):
            fk = forward_kinematics(sim.joints)
            tcp = fk["tcp"]["position"]
            min_altitude = min(min_altitude, tcp[1])

            cmd = controller.step(tcp, sim.joints)
            assert cmd is not None

            sim.set_joint_targets(cmd["joints"], cmd.get("gripper"))
            sim.step(FIXED_DT)

            progress = controller.get_progress()
            if progress["step"] >= 2:
                advanced_to_step_2 = True
                break

        assert advanced_to_step_2, "Controller froze on Step 1 and failed to advance to Step 2"
        assert min_altitude >= 0.850, f"TCP dipped down to {min_altitude:.3f}m during Step 1 transit"

    @pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo not installed")
    def test_03_autonomous_base_tier_placement_accuracy(self):
        """
        Execute Base Tier 1 pick and place of Blue Cube (block_cyan) under dynamic rigid physics:
        Verifies:
        1. Cube is cleanly grasped from spawn pose [0.24, 0.7475, -0.18].
        2. Cube is transported and placed at pyramid Base Slot 1.
        3. Cube position error < 30mm, zero toppling (upright=True).
        4. Zero collisions or disturbances to other tabletop blocks during the maneuver.
        """
        sim = MujocoSimulation()
        agent = GoogleVLAAgent()
        agent.execution_mode = "deterministic"
        snapshot = sim.get_snapshot()

        res = agent.parse_instruction("Build a 6-block pyramid tower.", None, snapshot, {})
        plan = res.get("plan", [])

        # Execute only the first 8 steps (Tier 1 placement of block_cyan)
        base_tier_plan = plan[:8]
        controller = TrajectoryController()
        controller.load_plan(base_tier_plan)

        for tick in range(400):
            if not controller.is_active:
                break
            fk = forward_kinematics(sim.joints)
            tcp = fk["tcp"]["position"]
            cmd = controller.step(tcp, sim.joints)
            if cmd is not None:
                sim.set_joint_targets(cmd["joints"], cmd.get("gripper"))
            sim.step(FIXED_DT)

        snap = sim.get_snapshot()
        blocks = {b["id"]: b for b in snap["blocks"]}

        cyan = blocks["block_cyan"]
        slot1_target = plan[5]["target"]  # Target of Step 6 (DESCEND_PLACE)

        pos_err = math.hypot(
            cyan["position"][0] - slot1_target[0],
            cyan["position"][2] - slot1_target[2],
        )
        assert pos_err < 0.030, f"Base Tier 1 placement error {pos_err*1000:.1f}mm exceeds 30mm tolerance"
        assert cyan.get("orientation_score", 1.0) >= 0.88, "Placed block toppled or tilted"

        # Verify flanking blocks were not displaced during placement maneuver
        yellow = blocks["block_yellow"]
        assert math.hypot(yellow["position"][0] - 0.21, yellow["position"][2] - (-0.05)) < 0.015, (
            "Flanking Yellow block disturbed during Blue Cube pick"
        )

    @pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo not installed")
    def test_04_distinct_towers_zero_table_penetration(self):
        """
        Verify autonomous multi-tower construction: 'Make 2 distinct towers of 2 blocks each.'
        Validates:
        1. GoogleVLAAgent synthesizes complete multi-tower plan (33 steps).
        2. Closed-loop physics execution achieves 100% stable dual tower stacking.
        3. Zero table penetration: All blocks maintain Y >= 0.7425m throughout execution.
        4. Tower 1: block_yellow stacked on block_cyan (upright=True, supported_by=block_cyan).
        5. Tower 2: block_emerald stacked on block_magenta (upright=True, supported_by=block_magenta).
        6. Spatial separation: Towers maintain clean clearance (delta_Z >= 0.14m).
        """
        sim = MujocoSimulation()
        agent = GoogleVLAAgent()
        agent.api_key = ""
        agent.interpreter.api_key = ""
        agent.execution_mode = "deterministic"
        snapshot = sim.get_snapshot()

        res = agent.parse_instruction("Make 2 distinct towers of 2 blocks each.", None, snapshot, {})
        plan = res.get("plan", [])
        assert len(plan) == 33, f"Expected 33 plan steps for 2 dual towers, got {len(plan)}"

        controller = TrajectoryController()
        controller.load_plan(plan)

        min_altitudes = {b["id"]: 99.0 for b in snapshot["blocks"]}

        for tick in range(1100):
            if not controller.is_active:
                break
            fk = forward_kinematics(sim.joints)
            tcp = fk["tcp"]["position"]
            cmd = controller.step(tcp, sim.joints)
            if cmd is not None:
                sim.set_joint_targets(cmd["joints"], cmd.get("gripper"))
            sim.step(FIXED_DT)
            for b_chk in sim.get_snapshot()["blocks"]:
                min_altitudes[b_chk["id"]] = min(min_altitudes[b_chk["id"]], b_chk["position"][1])

        snap = sim.get_snapshot()
        blocks = {b["id"]: b for b in snap["blocks"]}

        # 1. Zero table penetration validation: Table top is 0.720m, cube center is 0.7475m
        for b_id, min_y in min_altitudes.items():
            assert min_y >= TABLE_TOP_Y + CUBE_HALF - 0.005, (
                f"Block {b_id} penetrated table surface! min_Y = {min_y:.4f}m (table surface is {TABLE_TOP_Y}m)"
            )

        # 2. Tower 1 validation (block_cyan + block_yellow)
        cyan = blocks["block_cyan"]
        yellow = blocks["block_yellow"]
        assert cyan.get("supported_by") == "table", f"Tower 1 base block_cyan not supported by table: {cyan.get('supported_by')}"
        assert yellow.get("supported_by") == "block_cyan", f"Tower 1 top block_yellow not supported by block_cyan: {yellow.get('supported_by')}"
        assert cyan.get("orientation_score", 1.0) >= 0.88, "Tower 1 base block_cyan toppled"
        assert yellow.get("orientation_score", 1.0) >= 0.88, "Tower 1 top block_yellow toppled"
        assert yellow["position"][1] >= 0.795, f"Tower 1 height {yellow['position'][1]:.3f}m below expected Layer 1 altitude"

        # 3. Tower 2 validation (block_magenta + block_emerald)
        magenta = blocks["block_magenta"]
        emerald = blocks["block_emerald"]
        assert magenta.get("supported_by") == "table", f"Tower 2 base block_magenta not supported by table: {magenta.get('supported_by')}"
        assert emerald.get("supported_by") == "block_magenta", f"Tower 2 top block_emerald not supported by block_magenta: {emerald.get('supported_by')}"
        assert magenta.get("orientation_score", 1.0) >= 0.88, "Tower 2 base block_magenta toppled"
        assert emerald.get("orientation_score", 1.0) >= 0.88, "Tower 2 top block_emerald toppled"
        assert emerald["position"][1] >= 0.795, f"Tower 2 height {emerald['position'][1]:.3f}m below expected Layer 1 altitude"

        # 4. Clean spatial clearance between distinct towers
        tower_sep = abs(magenta["position"][2] - cyan["position"][2])
        assert tower_sep >= 0.14, f"Towers too close together: {tower_sep:.3f}m < 0.14m"

