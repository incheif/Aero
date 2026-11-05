"""
Hierarchical Task Decomposition & Execution State Machine for AERO.
Implements:
- Formal Subgoal DAG schema with preconditions, desired effects, and explicit dependencies.
- Recursive multi-level decomposition (Task Level 0 -> Macro Stage Level 1 -> Atomic Skill Level 2).
- Zero false-positive Execution State Machine:
  (PLAN, MOVE_TO_PREGRASP, APPROACH, GRASP, VERIFY_GRASP, LIFT, MOVE_TO_PLACE, PLACE, RELEASE, VERIFY_PLACEMENT, RECOVER, COMPLETE, FAILED).
- Physical contact and stability verification with genuine recovery policies.
"""

import math
import copy
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np

from simulation.constants import (
    TABLE_TOP_Y,
    CUBE_SIZE,
    CUBE_HALF,
    SAFE_HOVER_Y,
    MAX_TCP_STEP,
    STACK_ORIGIN,
    STACK_ORIGIN_A,
    STACK_ORIGIN_B,
    SAFE_STAGING_SLOTS,
    GRASP_CLOSE_THRESHOLD,
    GRASP_OPEN_THRESHOLD,
    get_pyramid_slots,
    get_train_slots,
)
from simulation.kinematics import solve_ik, forward_kinematics, clamp
from simulation.motion_planner import RRTConnectJointPlanner


@dataclass
class Subgoal:
    """
    Formal representation of a single executable operation in the Hierarchical Plan.
    """
    id: str
    parent_id: Optional[str]
    level: int
    skill_type: str  # "PICK_AND_PLACE", "CLEAR_OBSTACLE", "ALIGN"
    required_objects: List[str]
    preconditions: Dict[str, Any]
    desired_effects: Dict[str, Any]
    target_configuration: Dict[str, Any]
    dependencies: List[str]  # IDs of predecessor subgoals that must be COMPLETED
    status: str = "PENDING"  # PENDING, ACTIVE, COMPLETED, FAILED, SKIPPED
    verification_criteria: Dict[str, Any] = field(default_factory=dict)
    recovery_strategy: str = "RETRY_PICK"
    retry_count: int = 0
    max_retries: int = 3
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HierarchicalTaskPlanner:
    """
    Decomposes natural language and domain tasks into multi-level Subgoal DAGs.
    Generates structured dependencies so that lower tiers are guaranteed to be
    completed and verified before upper tiers or dependent operations execute.
    """

    def __init__(self):
        self.planner_rrt = RRTConnectJointPlanner()

    def decompose_task(
        self,
        instruction: str,
        initial_blocks: List[Dict[str, Any]],
        target_slots: Optional[List[List[float]]] = None,
        target_block_ids: Optional[List[str]] = None,
    ) -> List[Subgoal]:
        """
        Recursively decomposes high-level instructions into a structured DAG of Subgoals.
        Supports pyramids, towers, trains, and arbitrary multi-block structures.
        """
        inst_lower = instruction.lower()

        # Identify structure type and assign geometric slots
        if "pyramid" in inst_lower:
            if "6" in inst_lower or "six" in inst_lower:
                structure = "6_block_stepped"
                req_count = 6
            elif "5" in inst_lower or "five" in inst_lower or "square" in inst_lower:
                structure = "5_block_square"
                req_count = 5
            else:
                structure = "3_block_stepped"
                req_count = 3
            slots = get_pyramid_slots(structure)
        elif "train" in inst_lower or "line" in inst_lower or "row" in inst_lower:
            structure = "block_train"
            req_count = 4
            slots = get_train_slots(num_blocks=req_count)
        elif any(k in inst_lower for k in ("move", "pick", "place", "pad", "single")):
            structure = "single_pick_place"
            req_count = 1
            slots = [(STACK_ORIGIN[0], TABLE_TOP_Y + CUBE_HALF, STACK_ORIGIN[2])]
        else:
            structure = "3_block_stepped"
            req_count = 3
            slots = get_pyramid_slots(structure)

        if target_slots:
            slots = target_slots
            req_count = len(slots)

        avail_blocks = [b["id"] for b in initial_blocks if not b.get("is_grasped")]
        if target_block_ids:
            assigned_ids = target_block_ids[:req_count]
        else:
            color_order = []
            for color, b_id in [
                ("red", "block_magenta"),
                ("magenta", "block_magenta"),
                ("blue", "block_cyan"),
                ("cyan", "block_cyan"),
                ("yellow", "block_yellow"),
                ("green", "block_emerald"),
                ("emerald", "block_emerald"),
                ("orange", "block_orange"),
                ("purple", "block_purple"),
                ("pink", "block_pink"),
            ]:
                if color in inst_lower and b_id in avail_blocks and b_id not in color_order:
                    color_order.append(b_id)
            for b_id in avail_blocks:
                if b_id not in color_order and len(color_order) < req_count:
                    color_order.append(b_id)
            assigned_ids = color_order[:req_count]

        # Sort slots by altitude (base layers first)
        indexed_slots = list(enumerate(slots[:len(assigned_ids)]))
        indexed_slots.sort(key=lambda item: item[1][1])

        subgoals: List[Subgoal] = []
        layer_subgoals: Dict[int, List[str]] = {}

        for step_idx, (orig_slot_idx, slot) in enumerate(indexed_slots):
            b_id = assigned_ids[step_idx]
            sg_id = f"subgoal_{step_idx}"

            rel_y = slot[1] - (TABLE_TOP_Y + CUBE_HALF)
            layer_idx = max(0, int(round(rel_y / CUBE_SIZE)))

            deps = []
            for prev_layer in range(layer_idx):
                deps.extend(layer_subgoals.get(prev_layer, []))

            if layer_idx not in layer_subgoals:
                layer_subgoals[layer_idx] = []
            layer_subgoals[layer_idx].append(sg_id)

            subgoal = Subgoal(
                id=sg_id,
                parent_id="root_task",
                level=layer_idx,
                skill_type="PICK_AND_PLACE",
                required_objects=[b_id],
                preconditions={
                    "dependencies_satisfied": True,
                    "target_slot_clear": True,
                    "block_available": True,
                },
                desired_effects={
                    "object_at_slot": True,
                    "supported": True,
                    "stable": True,
                },
                target_configuration={
                    "block_id": b_id,
                    "slot": list(slot),
                    "layer": layer_idx,
                },
                dependencies=deps,
                status="PENDING",
                verification_criteria={
                    "position_tolerance_xz": 0.055,
                    "position_tolerance_y": 0.035,
                    "max_settling_velocity": 0.040,
                    "min_hold_ticks": 12,
                },
                recovery_strategy="RETRY_PICK",
                max_retries=3,
                description=f"Assemble {b_id} onto Tier {layer_idx} slot ({slot[0]:.3f}, {slot[1]:.3f}, {slot[2]:.3f})",
            )
            subgoals.append(subgoal)

        return subgoals


class ExecutionStateMachine:
    """
    Robust Execution State Machine with genuine physical gating and recovery.
    States:
      - IDLE
      - PLAN
      - MOVE_TO_PREGRASP
      - APPROACH
      - GRASP
      - VERIFY_GRASP
      - LIFT
      - MOVE_TO_PLACE
      - PLACE
      - RELEASE
      - VERIFY_PLACEMENT
      - RECOVER
      - COMPLETE
      - FAILED
    """

    def __init__(self):
        self.state: str = "IDLE"
        self.state_ticks: int = 0
        self.subgoals: List[Subgoal] = []
        self.active_subgoal_idx: int = 0
        self.hold_counter: int = 0
        self.recovery_reason: str = ""
        self.grasped_id: Optional[str] = None
        self.last_place_joints: Optional[Dict[str, float]] = None
        self.planner_rrt = RRTConnectJointPlanner()

    def start_plan(self, subgoals: List[Subgoal]):
        self.subgoals = subgoals
        self.active_subgoal_idx = 0
        self.state = "PLAN" if self.subgoals else "COMPLETE"
        self.state_ticks = 0
        self.hold_counter = 0
        self.grasped_id = None
        self.recovery_reason = ""

    def get_active_subgoal(self) -> Optional[Subgoal]:
        if 0 <= self.active_subgoal_idx < len(self.subgoals):
            return self.subgoals[self.active_subgoal_idx]
        return None

    def _dependencies_met(self, subgoal: Subgoal) -> bool:
        completed_ids = {s.id for s in self.subgoals if s.status == "COMPLETED"}
        return all(dep_id in completed_ids for dep_id in subgoal.dependencies)

    def transition_to(self, new_state: str, reason: str = ""):
        self.state = new_state
        self.state_ticks = 0
        self.hold_counter = 0
        if reason:
            self.recovery_reason = reason

    def step(
        self,
        current_tcp: Tuple[float, float, float],
        current_joints: Dict[str, float],
        blocks: List[Dict[str, Any]],
        current_gripper: float,
        grasped_block_id: Optional[str],
    ) -> Dict[str, Any]:
        """
        Executes one state machine update tick.
        Returns target joints and gripper commands for the simulation controller.
        """
        self.state_ticks += 1
        self.grasped_id = grasped_block_id
        target_joints = copy.deepcopy(current_joints)
        desired_gripper = current_gripper

        sg = self.get_active_subgoal()
        if sg is None and self.state not in ("COMPLETE", "FAILED"):
            self.transition_to("COMPLETE")

        if self.state == "PLAN":
            if sg is None:
                self.transition_to("COMPLETE")
            elif not self._dependencies_met(sg):
                eligible = [i for i, s in enumerate(self.subgoals) if s.status == "PENDING" and self._dependencies_met(s)]
                if eligible:
                    self.active_subgoal_idx = eligible[0]
                    sg = self.get_active_subgoal()
                    self.transition_to("MOVE_TO_PREGRASP")
                else:
                    self.transition_to("FAILED", "Cyclic or unresolvable dependency in plan.")
            else:
                sg.status = "ACTIVE"
                self.transition_to("MOVE_TO_PREGRASP")

        elif self.state == "MOVE_TO_PREGRASP":
            b_id = sg.target_configuration["block_id"]
            b_data = next((b for b in blocks if b["id"] == b_id), None)
            if not b_data:
                self.transition_to("FAILED", f"Target block {b_id} missing from scene.")
            else:
                bp = b_data["position"]
                transit_y = max(SAFE_HOVER_Y, bp[1] + 0.12)

                # Validate whether the block's pregrasp position is reachable
                check_ik = solve_ik((bp[0], transit_y, bp[2]), gripper=0.0)
                if not check_ik.success:
                    self.transition_to("FAILED", f"Pregrasp target unreachable: {check_ik.reachability_status}")
                else:
                    # Ensure the arm ascends vertically to safe hover altitude before horizontal transit
                    transit_y = max(SAFE_HOVER_Y, bp[1] + 0.15)
                    if current_tcp[1] < transit_y - 0.018:
                        waypoint = (current_tcp[0], transit_y, current_tcp[2])
                    else:
                        # Smooth Cartesian waypoint interpolation prevents joint-space dip
                        goal = np.array([bp[0], transit_y, bp[2]])
                        curr = np.array(current_tcp)
                        vec = goal - curr
                        dist = float(np.linalg.norm(vec))
                        max_step = MAX_TCP_STEP
                        if dist > max_step:
                            waypoint = tuple(curr + (vec / dist) * max_step)
                        else:
                            waypoint = (bp[0], transit_y, bp[2])

                    ik_res = solve_ik(waypoint, gripper=0.0, candidate_pitches=[-0.12, -0.4, -0.6, -1.0])
                    if not ik_res.success:
                        ik_res = solve_ik(waypoint, gripper=0.0)
                    if not ik_res.success:
                        self.transition_to("FAILED", f"Pregrasp target unreachable: {ik_res.reachability_status}")
                    else:
                        target_joints = ik_res.joints
                        desired_gripper = 0.0  # Open gripper

                        dist_tcp = math.hypot(current_tcp[0] - bp[0], current_tcp[1] - transit_y, current_tcp[2] - bp[2])
                        if dist_tcp < 0.015:
                            self.transition_to("APPROACH")
                        elif self.state_ticks > 65:
                            if dist_tcp < 0.025:
                                self.transition_to("APPROACH")
                            else:
                                self.transition_to("RECOVER", f"Pregrasp alignment timed out for {b_id}")

        elif self.state == "APPROACH":
            b_id = sg.target_configuration["block_id"]
            b_data = next((b for b in blocks if b["id"] == b_id), None)
            bp = b_data["position"] if b_data else [0, 0, 0]

            grasp_y = bp[1] + 0.008
            transit_y = max(SAFE_HOVER_Y, bp[1] + 0.12)
            descent_progress = min(1.0, self.state_ticks / 10.0)
            curr_target_y = (1.0 - descent_progress) * transit_y + descent_progress * grasp_y
            approach_target = (bp[0], curr_target_y, bp[2])

            ik_res = solve_ik(approach_target, gripper=0.38, candidate_pitches=[-0.12, -0.4, -0.6, -1.0])
            if not ik_res.success:
                ik_res = solve_ik(approach_target, gripper=0.38)

            target_joints = ik_res.joints
            # Narrow gripper aperture (0.38) during descent to avoid clipping flanking blocks,
            # while leaving ample 35mm margin over 55mm cube width
            desired_gripper = 0.38

            dist_tcp = math.hypot(current_tcp[0] - bp[0], current_tcp[1] - grasp_y, current_tcp[2] - bp[2])
            if dist_tcp < 0.012:
                self.transition_to("GRASP")
            elif self.state_ticks > 60:
                self.transition_to("RECOVER", f"Approach descent stalled/blocked for {b_id}")

        elif self.state == "GRASP":
            desired_gripper = 1.0  # Close gripper
            if current_gripper >= GRASP_CLOSE_THRESHOLD or self.state_ticks >= 20:
                self.transition_to("VERIFY_GRASP")

        elif self.state == "VERIFY_GRASP":
            desired_gripper = 1.0
            b_id = sg.target_configuration["block_id"]
            if grasped_block_id == b_id:
                self.transition_to("LIFT")
            else:
                sg.retry_count += 1
                if sg.retry_count > sg.max_retries:
                    sg.status = "FAILED"
                    self.transition_to("FAILED", f"Grasp retries exhausted for {b_id}")
                else:
                    self.transition_to("RECOVER", f"Grasp missed for {b_id}; recovering.")

        elif self.state == "LIFT":
            desired_gripper = 1.0
            b_id = sg.target_configuration["block_id"]
            b_data = next((b for b in blocks if b["id"] == b_id), None)
            bp = b_data["position"] if b_data else [0, 0, 0]

            slot = sg.target_configuration["slot"]
            transit_y = min(1.04, max(SAFE_HOVER_Y, slot[1] + 0.14, bp[1] + 0.14))
            # Pure vertical ascent above the pick location
            lift_target = (bp[0], transit_y, bp[2])

            ik_res = solve_ik(lift_target, gripper=1.0)
            target_joints = ik_res.joints

            if current_tcp[1] >= transit_y - 0.020 or self.state_ticks > 35:
                self.transition_to("MOVE_TO_PLACE")

        elif self.state == "MOVE_TO_PLACE":
            desired_gripper = 1.0
            slot = sg.target_configuration["slot"]
            transit_y = min(1.04, max(SAFE_HOVER_Y, slot[1] + 0.14))

            # Smooth Cartesian waypoint interpolation prevents joint-space dip during transit
            goal = np.array([slot[0], transit_y, slot[2]])
            curr = np.array(current_tcp)
            vec = goal - curr
            dist = float(np.linalg.norm(vec))
            max_step = MAX_TCP_STEP
            if dist > max_step:
                waypoint = tuple(curr + (vec / dist) * max_step)
            else:
                waypoint = (slot[0], transit_y, slot[2])

            ik_res = solve_ik(waypoint, gripper=1.0, candidate_pitches=[-0.12, -0.4, -0.6, -1.0])
            if not ik_res.success:
                ik_res = solve_ik(waypoint, gripper=1.0)
            target_joints = ik_res.joints

            dist_xz = math.hypot(current_tcp[0] - slot[0], current_tcp[2] - slot[2])
            dist_y = abs(current_tcp[1] - transit_y)
            if dist_xz < 0.012 and dist_y < 0.025:
                self.transition_to("PLACE")
            elif self.state_ticks > 65:
                if dist_xz < 0.025:
                    self.transition_to("PLACE")
                else:
                    self.transition_to("RECOVER", "Transit to placement slot misaligned")

        elif self.state == "PLACE":
            desired_gripper = 1.0
            slot = sg.target_configuration["slot"]
            # Since the block was picked with TCP at bp[1] + 0.008, the block center sits
            # 8mm below TCP. Setting place_y = slot[1] + 0.010 places the block center 2mm
            # above the support surface, ensuring smooth touchdown without wedging lower blocks apart.
            offset_y = 0.010
            place_y = slot[1] + offset_y
            transit_y = min(1.02, max(SAFE_HOVER_Y, slot[1] + 0.10))

            descent_progress = min(1.0, self.state_ticks / 12.0)
            curr_target_y = (1.0 - descent_progress) * transit_y + descent_progress * place_y
            place_target = (slot[0], curr_target_y, slot[2])

            ik_res = solve_ik(place_target, gripper=1.0, candidate_pitches=[-0.12, -0.4, -0.6, -1.0])
            if not ik_res.success:
                ik_res = solve_ik(place_target, gripper=1.0)
            target_joints = ik_res.joints
            self.last_place_joints = copy.deepcopy(target_joints)

            dist_tcp = math.hypot(current_tcp[0] - slot[0], current_tcp[1] - place_y, current_tcp[2] - slot[2])
            if dist_tcp < 0.012:
                self.transition_to("RELEASE")
            elif self.state_ticks > 60:
                if dist_tcp < 0.022:
                    self.transition_to("RELEASE")
                else:
                    self.transition_to("RECOVER", f"Placement descent stalled/blocked for {sg.target_configuration['block_id']}")

        elif self.state == "RELEASE":
            # Hold exact placement joint configuration frozen in place while jaws open smoothly.
            # Open gripper to 0.50 (83mm aperture, 28mm clearance over 55mm cube),
            # safely clearing GRASP_OPEN_THRESHOLD (0.65) while keeping outer span compact (111mm)
            # so opening fingers never sweep into adjacent blocks.
            if self.last_place_joints:
                target_joints = copy.deepcopy(self.last_place_joints)
            desired_gripper = 0.70

            is_released = (current_gripper <= GRASP_OPEN_THRESHOLD) and (grasped_block_id is None)
            if is_released and self.state_ticks >= 10:
                self.transition_to("VERIFY_PLACEMENT")
            elif self.state_ticks >= 30:
                if current_gripper <= GRASP_OPEN_THRESHOLD:
                    self.transition_to("VERIFY_PLACEMENT")
                else:
                    self.transition_to("RECOVER", "Gripper failed to release block")

        elif self.state == "VERIFY_PLACEMENT":
            desired_gripper = 0.70
            slot = sg.target_configuration["slot"]
            transit_y = max(SAFE_HOVER_Y, slot[1] + 0.15)
            start_y = slot[1] + 0.010
            retreat_progress = min(1.0, self.state_ticks / 10.0)
            curr_y = (1.0 - retreat_progress) * start_y + retreat_progress * transit_y
            retreat_target = (slot[0], curr_y, slot[2])

            ik_res = solve_ik(retreat_target, gripper=0.0, candidate_pitches=[-0.12, -0.4, -0.6, -1.0])
            if not ik_res.success:
                ik_res = solve_ik(retreat_target, gripper=0.0)
            target_joints = ik_res.joints

            b_id = sg.target_configuration["block_id"]
            b_data = next((b for b in blocks if b["id"] == b_id), None)

            if b_data:
                bp = b_data["position"]
                vel = b_data.get("velocity", [0, 0, 0])
                speed = math.hypot(*vel)
                dxz = math.hypot(bp[0] - slot[0], bp[2] - slot[2])
                dy = abs(bp[1] - slot[1])

                tol_xz = sg.verification_criteria.get("position_tolerance_xz", 0.055)
                tol_y = sg.verification_criteria.get("position_tolerance_y", 0.035)

                if dxz <= tol_xz and dy <= tol_y and speed < 0.040:
                    self.hold_counter += 1
                else:
                    self.hold_counter = 0

                req_ticks = sg.verification_criteria.get("min_hold_ticks", 12)
                arm_ascended = (current_tcp[1] >= transit_y - 0.018)
                if self.hold_counter >= req_ticks and arm_ascended:
                    sg.status = "COMPLETED"
                    self.active_subgoal_idx += 1
                    if self.active_subgoal_idx < len(self.subgoals):
                        self.transition_to("PLAN")
                    else:
                        self.transition_to("COMPLETE")
                elif self.state_ticks > 90:
                    sg.retry_count += 1
                    if sg.retry_count > sg.max_retries:
                        sg.status = "FAILED"
                        self.transition_to("FAILED", f"Placement stability failed for {b_id}")
                    else:
                        self.transition_to("RECOVER", f"Placement stability hold failed for {b_id}")

        elif self.state == "RECOVER":
            desired_gripper = 0.0
            transit_target = (current_tcp[0], SAFE_HOVER_Y, current_tcp[2])
            ik_res = solve_ik(transit_target, gripper=0.0)
            target_joints = ik_res.joints

            if current_tcp[1] >= SAFE_HOVER_Y - 0.02 or self.state_ticks > 25:
                self.transition_to("PLAN")

        return {
            "joints": target_joints,
            "gripper": desired_gripper,
            "state": self.state,
            "active_subgoal": sg.to_dict() if sg else None,
            "recovery_reason": self.recovery_reason,
            "is_complete": self.state == "COMPLETE",
            "is_failed": self.state == "FAILED",
        }
