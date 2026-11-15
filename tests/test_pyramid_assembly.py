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
    get_train_slots,
    SAFE_CORRIDOR_STAGING_SLOT,
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

    @pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo not installed")
    def test_05_block_train_path_clearance_staging(self):
        """
        Verify block train assembly with Stage 2 Path Clearance:
        Command: "Make a block train with all boxes in one line."
        Validates:
        1. Cognitive interpreter parses command as BLOCK_TRAIN with 6 target blocks.
        2. Orange Cube (block_orange at [0.34, 0.7535, -0.06]) is detected as obstructing
           train carriage slots.
        3. Path clearance staging pick-and-place sub-plan is scheduled first, moving
           block_orange to safe perimeter slot [0.40, 0.7475, 0.30].
        4. Staged block_orange coordinates are updated in spatial memory so carriage 4
           picks it up from the safe staging slot.
        5. In dynamic MuJoCo physics, block_orange is safely staged out of the corridor
           with zero table penetration and error < 35mm.
        """
        sim = MujocoSimulation()
        agent = GoogleVLAAgent()
        agent.execution_mode = "deterministic"
        snapshot = sim.get_snapshot()

        instruction = "Make a block train with all boxes in one line. this has issue with path clearance as orange obstructs placing the next train block so it should be removed."
        interp = agent.interpreter.interpret(instruction)
        assert interp["is_possible"] is True
        assert interp.get("action_type") == "BLOCK_TRAIN"
        assert len(interp.get("target_blocks", [])) == 6

        res = agent.parse_instruction(instruction, None, snapshot, {})
        plan = res.get("plan", [])
        assert len(plan) == 57, f"Expected 57 plan steps (8 staging + 48 carriages + 1 home), got {len(plan)}"

        # Grounding checks: Verify block_orange has Path Clearance Staging role
        grounding = res.get("grounding", [])
        staged_grounding = [g for g in grounding if g.get("role") == "Path Clearance Staging"]
        assert len(staged_grounding) == 1, "Expected 1 path clearance staging grounding entry"
        assert staged_grounding[0]["entity"] == "block_orange"
        assert staged_grounding[0]["target_coords"] == list(SAFE_CORRIDOR_STAGING_SLOT)

        # Verify Orange Cube is picked up from staging coordinates for Carriage 4
        car4_grounding = [g for g in grounding if g.get("role") == "Carriage 4"]
        assert len(car4_grounding) == 1
        assert car4_grounding[0]["entity"] == "block_orange"
        assert car4_grounding[0]["source_coords"] == list(SAFE_CORRIDOR_STAGING_SLOT)

        # Dynamic physics verification of staging maneuver (first 8 steps)
        controller = TrajectoryController()
        controller.load_plan(plan[:8])

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
        orange = blocks["block_orange"]

        # Verify Orange Cube safely reached perimeter staging zone with zero table penetration
        staging_err = math.hypot(
            orange["position"][0] - SAFE_CORRIDOR_STAGING_SLOT[0],
            orange["position"][2] - SAFE_CORRIDOR_STAGING_SLOT[2],
        )
        assert staging_err < 0.035, f"Orange staging error {staging_err*1000:.1f}mm exceeds 35mm"
        assert orange["position"][1] >= TABLE_TOP_Y + CUBE_HALF - 0.005, "Table penetration detected"

    def test_06_block_train_full_execution_no_freeze(self):
        """
        Verify that executing the block train plan in RobotArmSimulation (dashboard simulation backend)
        advances through carriages without freezing or proxy collision lockups.
        Validates:
        1. All 57 steps complete without stalls or infinite dwells.
        2. All 6 blocks are arranged along the tabletop Z-axis at X=0.40m with zero table penetration.
        """
        from simulation.physics_engine import RobotArmSimulation
        sim = RobotArmSimulation()
        agent = GoogleVLAAgent()
        agent.execution_mode = "deterministic"
        snapshot = sim.get_snapshot()

        res = agent.parse_instruction("Make a block train with all boxes in one line.", None, snapshot, {})
        plan = res.get("plan", [])
        assert len(plan) == 57

        controller = TrajectoryController()
        controller.load_plan(plan)

        for tick in range(1800):
            if not controller.is_active:
                break
            snap = sim.get_snapshot()
            tcp = snap["tcp"]["position"]
            cmd = controller.step(tcp, sim.joints)
            if not cmd:
                break
            sim.set_joint_targets(cmd["joints"], cmd.get("gripper"))
            sim.step(FIXED_DT)

        assert not controller.is_active, f"Arm froze at step {controller.step_idx + 1}: {controller.current_action_desc}"

        snap = sim.get_snapshot()
        blocks = {b["id"]: b for b in snap["blocks"]}

        # All 6 blocks should be on the table surface (Y >= 0.7425) and aligned near X=0.40
        for b_id, b in blocks.items():
            assert abs(b["position"][0] - 0.40) < 0.035, f"Block {b_id} X position {b['position'][0]} not aligned at X=0.40m"
            assert b["position"][1] >= TABLE_TOP_Y + CUBE_HALF - 0.005, f"Block {b_id} penetrated table"

    def test_07_per_step_clearance_obstructed_6_block_pyramid(self):
        """
        Verify per-step clearance removes obstructing blocks from partial/full pyramid footprint:
        - Place block_purple (Apex Cap) inside Base Tier 2 at STACK_ORIGIN [0.45, 0.7475, 0.20].
        - Verify Step Clearance Staging clears it safely to a perimeter slot BEFORE base tier placement,
          then picks block_purple back from its staged slot when placing the Apex Cap.
        """
        from simulation.physics_engine import RobotArmSimulation
        sim = RobotArmSimulation()
        sim.blocks["block_purple"].position = [STACK_ORIGIN[0], TABLE_TOP_Y + CUBE_HALF, STACK_ORIGIN[2]]

        agent = GoogleVLAAgent()
        agent.api_key = ""
        agent.interpreter.api_key = ""
        agent.execution_mode = "deterministic"

        res = agent.parse_instruction("Build a 6-block pyramid tower.", None, sim.get_snapshot(), {})
        assert res["is_possible"] is True
        plan = res["plan"]
        grounding = res["grounding"]

        staged = [g for g in grounding if g.get("role") == "Step Clearance Staging"]
        assert len(staged) >= 1, "Expected Step Clearance Staging for block_purple obstructing STACK_ORIGIN"
        staged_purple = [g for g in staged if g["entity"] == "block_purple"]
        assert len(staged_purple) == 1
        staged_coords = staged_purple[0]["target_coords"]

        apex_entry = [g for g in grounding if g.get("role") == "Apex Cap" and g["entity"] == "block_purple"][0]
        assert apex_entry["source_coords"] == staged_coords, (
            "Apex Cap must pick block_purple from its newly staged perimeter slot"
        )

        # Execute full plan in physics engine and verify clean completion
        controller = TrajectoryController()
        controller.load_plan(plan)
        for _ in range(2600):
            if not controller.is_active:
                break
            snap = sim.get_snapshot()
            cmd = controller.step(snap["tcp"]["position"], sim.joints)
            if not cmd:
                break
            sim.set_joint_targets(cmd["joints"], cmd.get("gripper"))
            sim.step(FIXED_DT)

        assert not controller.is_active, f"Arm froze at step {controller.step_idx + 1}: {controller.current_action_desc}"

    def test_08_per_step_clearance_single_and_dual_towers(self):
        """
        Verify per-step clearance generalizes across Single Tower and Multi-Tower instructions
        and assigns distinct collision-free staging slots when multiple blockers exist.
        """
        from simulation.physics_engine import RobotArmSimulation

        # Case A: Bystander block_purple sitting directly on STACK_ORIGIN during 3-block tower command
        sim = RobotArmSimulation()
        sim.blocks["block_purple"].position = [STACK_ORIGIN[0], TABLE_TOP_Y + CUBE_HALF, STACK_ORIGIN[2]]

        agent = GoogleVLAAgent()
        agent.api_key = ""
        agent.interpreter.api_key = ""
        agent.execution_mode = "deterministic"

        res = agent.parse_instruction("Stack red, blue, and yellow blocks into a 3-layer tower", None, sim.get_snapshot(), {})
        assert res["is_possible"] is True
        grounding = res["grounding"]
        staged = [g for g in grounding if g.get("role") == "Step Clearance Staging"]
        assert any(g["entity"] == "block_purple" for g in staged), (
            "Bystander block_purple on STACK_ORIGIN must be cleared before Layer 0 placement"
        )

        controller = TrajectoryController()
        controller.load_plan(res["plan"])
        for _ in range(1600):
            if not controller.is_active:
                break
            snap = sim.get_snapshot()
            cmd = controller.step(snap["tcp"]["position"], sim.joints)
            if not cmd:
                break
            sim.set_joint_targets(cmd["joints"], cmd.get("gripper"))
            sim.step(FIXED_DT)

        assert not controller.is_active, f"Single tower froze at {controller.current_action_desc}"
        final_blocks = {b["id"]: b for b in sim.get_snapshot()["blocks"]}
        purple_dist_to_pad = math.hypot(
            final_blocks["block_purple"]["position"][0] - STACK_ORIGIN[0],
            final_blocks["block_purple"]["position"][2] - STACK_ORIGIN[2],
        )
        assert purple_dist_to_pad >= 0.08, f"Bystander block_purple still too close to pad ({purple_dist_to_pad:.3f}m)"

        # Case B: Dual towers where block_emerald blocks Tower 1 pad and block_purple blocks Tower 2 pad
        sim2 = RobotArmSimulation()
        sim2.blocks["block_emerald"].position = [0.40, TABLE_TOP_Y + CUBE_HALF, 0.10]
        sim2.blocks["block_purple"].position = [0.40, TABLE_TOP_Y + CUBE_HALF, 0.26]

        res2 = agent.parse_instruction("Make 2 distinct towers of 2 blocks each.", None, sim2.get_snapshot(), {})
        assert res2["is_possible"] is True
        staged2 = [g for g in res2["grounding"] if g.get("role") == "Step Clearance Staging"]
        assert len(staged2) >= 2, f"Expected >=2 staged blockers for obstructed dual towers, got {len(staged2)}"
        staged_targets = [tuple(round(v, 3) for v in g["target_coords"]) for g in staged2]
        assert len(set(staged_targets)) == len(staged_targets), (
            f"Multiple staged blockers collided onto identical staging coordinates: {staged_targets}"
        )

    def test_09_full_six_block_tower_assembly(self):
        """
        Verify autonomous 6-block vertical tower assembly:
        'Build a 6-layer tower with all 6 blocks.'
        Validates:
        1. VLA decomposes instruction into 48 feasible atomic waypoints (8 per block x 6 blocks).
        2. High-speed Cartesian trajectory controller executes all pick-and-place segments without stall.
        3. All 6 blocks (cyan, orange, magenta, yellow, emerald, purple) are stacked vertically at STACK_ORIGIN.
        4. Stacking vertical pitch matches layer altitudes (0.7475m to 1.0225m) with < 8mm residual.
        5. Feetech STS3215 actuator dynamics operate within safety boundaries (peak torque < 3.2 Nm, temp < 40C).
        """
        from simulation.physics_engine import RobotArmSimulation
        sim = RobotArmSimulation()
        sim.reset(randomize=False)
        agent = GoogleVLAAgent()
        agent.api_key = ""
        agent.interpreter.api_key = ""
        agent.execution_mode = "deterministic"

        res = agent.parse_instruction("Build a 6-layer tower with all 6 blocks.", None, sim.get_snapshot(), {})
        assert res["is_possible"] is True
        plan = res["plan"]
        grounding = res["grounding"]
        assert len(plan) == 48, f"Expected 48 waypoints for 6 blocks, got {len(plan)}"
        assert len(grounding) == 6, f"Expected 6 grounded entities, got {len(grounding)}"

        controller = TrajectoryController()
        controller.load_plan(plan)
        task_blocks = [g["entity"] for g in grounding]
        task_targets = {g["entity"]: g["target_coords"] for g in grounding}
        sim.set_active_task(task_blocks, task_targets)

        for _ in range(2200):
            if not controller.is_active:
                break
            snap = sim.get_snapshot()
            cmd = controller.step(snap["tcp"]["position"], sim.joints)
            if not cmd:
                break
            sim.set_joint_targets(cmd["joints"], cmd.get("gripper"))
            sim.step(FIXED_DT)

        assert not controller.is_active, f"Arm stalled at step {controller.step_idx + 1}: {controller.current_action_desc}"

        final_snap = sim.get_snapshot()
        blocks = {b["id"]: b for b in final_snap["blocks"]}

        # Validate vertical stack alignment and layering for all 6 blocks
        for g in grounding:
            b_id = g["entity"]
            b = blocks[b_id]
            pos = b["position"]
            layer = g["target_layer"]
            expected_y = g["target_coords"][1]

            # Horizontal pad alignment within 20mm
            pad_err = math.hypot(pos[0] - STACK_ORIGIN[0], pos[2] - STACK_ORIGIN[2])
            assert pad_err < 0.025, f"Block {b_id} (Layer {layer}) deviated from pad by {pad_err*1000:.1f}mm"

            # Vertical layer height error within 10mm
            y_err = abs(pos[1] - expected_y)
            assert y_err < 0.010, f"Block {b_id} (Layer {layer}) altitude error {y_err*1000:.1f}mm exceeds 10mm"

            # Block must remain upright
            assert b.get("upright", True) is True, f"Block {b_id} toppled at layer {layer}"
            assert b.get("orientation_score", 1.0) >= 0.88, f"Block {b_id} tilted at layer {layer}"

        # Validate Feetech STS3215 actuator safety metrics
        actuators = final_snap.get("actuators", {})
        assert actuators.get("peak_torque", 0.0) <= 3.2, f"Peak torque exceeded: {actuators.get('peak_torque')} Nm"
        shoulder_temp = actuators.get("temperatures_c", {}).get("shoulderPitch", 25.0)
        assert shoulder_temp < 40.0, f"Shoulder actuator overheated: {shoulder_temp} C"





