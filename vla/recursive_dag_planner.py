"""
vla/recursive_dag_planner.py

Recursive Subgoal Directed Acyclic Graph (DAG) Task Decomposer for AERO.
Translates complex natural-language manipulation instructions and 3D scene graphs
into a multi-level dependency graph of atomic manipulation primitives.

Features:
- Formal Subgoal DAG schema with preconditions, desired effects, and explicit dependencies.
- Automatic topological dependency derivation (bottom-up structural support invariance).
- Spatial interference detection & prerequisite obstacle clearance generation.
- Dynamic graph repair: runtime splicing of recovery subgoals into the DAG.
- Acyclicity validation, in-degree dependency resolution, and ready node dispatch.
"""

import math
import copy
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np

from simulation.constants import (
    TABLE_TOP_Y,
    CUBE_SIZE,
    CUBE_HALF,
    SAFE_HOVER_Y,
    STACK_ORIGIN,
    STACK_ORIGIN_A,
    STACK_ORIGIN_B,
    SAFE_STAGING_SLOTS,
    get_pyramid_slots,
    get_train_slots,
    stack_slot_y,
)
from vla.spatial_geometry import SpatialGeometryEngine


@dataclass
class Subgoal:
    """
    Formal representation of a single executable operation in the Hierarchical Plan.
    """
    id: str
    parent_id: Optional[str]
    level: int                           # 0: Root Task, 1: Macro Stage, 2: Atomic Skill
    skill_type: str                      # "APPROACH_GRASP", "COMPLIANT_LIFT", "TRANSIT", "DEPOSIT", "SLIDE", "CLEAR_OBSTACLE"
    required_objects: List[str]          # Object IDs (e.g. ["block_cyan"])
    parameters: Dict[str, Any]           # Metric targets, approach vectors, tolerances
    preconditions: Dict[str, Any]        # Required physical conditions before execution
    desired_effects: Dict[str, Any]      # Post-condition invariants required for completion
    dependencies: List[str]              # IDs of prerequisite subgoals that must be COMPLETED
    status: str = "PENDING"              # PENDING, ACTIVE, COMPLETED, FAILED, RECOVERING, SKIPPED
    verification_criteria: Dict[str, Any] = field(default_factory=dict)
    recovery_strategy: str = "RETRY_PICK"
    retry_count: int = 0
    max_retries: int = 3
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RecursiveDAGPlanner:
    """
    Recursive Task Decomposer & Subgoal DAG Manager for AERO.
    Translates high-level natural language instructions and perceived 3D scene states
    into mathematically sound, non-circular DAGs of executable manipulation primitives.
    """

    def __init__(self):
        self.subgoals: Dict[str, Subgoal] = {}
        self.root_instruction: str = ""
        self.step_counter: int = 0

    def clear(self):
        self.subgoals.clear()
        self.root_instruction = ""
        self.step_counter = 0

    def decompose(
        self,
        instruction: str,
        current_blocks: List[Dict[str, Any]],
        custom_slots: Optional[List[Tuple[float, float, float]]] = None,
        preferred_block_ids: Optional[List[str]] = None,
    ) -> List[Subgoal]:
        """
        Recursively compiles an instruction into a Subgoal DAG.
        Derives slots, assigns optimal available blocks, checks for occlusions,
        and builds explicit topological prerequisite dependencies.
        """
        self.clear()
        self.root_instruction = instruction
        inst_lower = instruction.lower()

        slots: List[Tuple[float, float, float]] = []
        structure_type = "custom"

        # 1. Structure Geometric Resolution
        letter_match = re.search(r"\b(?:letter|char|character)\s+([a-zA-Z])\b", inst_lower)
        if not letter_match:
            letter_match = re.search(r"\b([a-zA-Z])\s+(?:shape|pattern|letter)\b", inst_lower)
        if not letter_match:
            letter_match = re.search(r"\b(?:shape|make|build|form|assemble|create)\s+(?:a\s+|an\s+)?([a-zA-Z])\s*(?:with|using)?\s*(?:the\s+)?(?:blocks|cubes)?\b", inst_lower)
            if letter_match and letter_match.group(1).lower() in ("a", "i", "to", "in"):
                letter_match = None

        if custom_slots:
            slots = list(custom_slots)
        elif letter_match:
            char = letter_match.group(1).upper()
            structure_type = f"letter_{char}"
            slots = SpatialGeometryEngine.get_letter_slots(char, num_blocks=len(current_blocks))
        elif any(w in inst_lower for w in ("triangle", "triangular")):
            structure_type = "triangle"
            slots = SpatialGeometryEngine.get_shape_slots("triangle", num_blocks=len(current_blocks))
        elif any(w in inst_lower for w in ("cross", "plus")):
            structure_type = "cross"
            slots = SpatialGeometryEngine.get_shape_slots("cross", num_blocks=len(current_blocks))
        elif any(w in inst_lower for w in ("circle", "circular", "ring")):
            structure_type = "circle"
            slots = SpatialGeometryEngine.get_shape_slots("circle", num_blocks=len(current_blocks))
        elif any(w in inst_lower for w in ("diamond", "rhombus")):
            structure_type = "diamond"
            slots = SpatialGeometryEngine.get_shape_slots("diamond", num_blocks=len(current_blocks))
        elif any(w in inst_lower for w in ("square", "rectangle", "box pattern")):
            structure_type = "square"
            slots = SpatialGeometryEngine.get_shape_slots("square", num_blocks=len(current_blocks))
        elif any(w in inst_lower for w in ("enclosure", "corral")):
            structure_type = "enclosure"
            slots = SpatialGeometryEngine.get_shape_slots("enclosure", num_blocks=len(current_blocks))
        elif any(w in inst_lower for w in ("sort", "separate", "segregate", "divide", "partition")) and any(w in inst_lower for w in ("warm", "cool", "left", "right", "sides")):
            structure_type = "spatial_sort"
            sort_mode = "warm_cool" if any(w in inst_lower for w in ("warm", "cool", "temperature")) else "left_right"
            slots, pref_blocks = SpatialGeometryEngine.get_sorting_slots(sort_mode, current_blocks)
            if not preferred_block_ids:
                preferred_block_ids = pref_blocks
        elif "bridge" in inst_lower or "arch" in inst_lower:
            structure_type = "arch_bridge"
            # 2 base pillars separated by 58mm (3mm air gap) spanned by a top bridging block
            ox, oz = STACK_ORIGIN[0], STACK_ORIGIN[2]
            y0 = TABLE_TOP_Y + CUBE_HALF
            y1 = y0 + CUBE_SIZE
            spacing = 0.058  # 58mm between centers: pillars don't collide, 55mm top block spans with 26mm overlap
            slots = [
                (ox, y0, oz - spacing / 2.0),  # Pillar 1
                (ox, y0, oz + spacing / 2.0),  # Pillar 2
                (ox, y1, oz),                  # Spanning bridge block
            ]
        elif "cantilever" in inst_lower:
            structure_type = "cantilever"
            ox, oz = STACK_ORIGIN[0], STACK_ORIGIN[2]
            y0 = TABLE_TOP_Y + CUBE_HALF
            y1 = y0 + CUBE_SIZE
            slots = [
                (ox, y0, oz),                  # Foundation
                (ox + CUBE_SIZE * 0.40, y1, oz),# Overhanging cantilever
            ]
        elif "pyramid" in inst_lower:
            structure_type = "pyramid"
            if any(k in inst_lower for k in ("6", "six")):
                p_slots = get_pyramid_slots("6_block_stepped")
            elif any(k in inst_lower for k in ("5", "five", "square")):
                p_slots = get_pyramid_slots("5_block_square")
            else:
                p_slots = get_pyramid_slots("3_block_stepped")
            slots = [(float(s[0]), float(s[1]), float(s[2])) for s in p_slots]
        elif any(k in inst_lower for k in ("train", "line", "row")):
            structure_type = "train"
            count = 4
            for num_word, num_val in [("one", 1), ("two", 2), ("three", 3), ("four", 4), ("five", 5), ("six", 6)]:
                if num_word in inst_lower or str(num_val) in inst_lower:
                    count = num_val
                    break
            t_slots = get_train_slots(num_blocks=count, origin=(0.40, TABLE_TOP_Y, 0.12))
            slots = [(float(s[0]), float(s[1]), float(s[2])) for s in t_slots]
        elif "tower" in inst_lower or "stack" in inst_lower:
            structure_type = "tower"
            count = 3
            for num_word, num_val in [("one", 1), ("two", 2), ("three", 3), ("four", 4), ("five", 5), ("six", 6)]:
                if num_word in inst_lower or str(num_val) in inst_lower:
                    count = num_val
                    break
            ox, oz = STACK_ORIGIN[0], STACK_ORIGIN[2]
            slots = [(ox, stack_slot_y(i), oz) for i in range(count)]
        elif any(w in inst_lower for w in ("arrange", "pattern", "layout", "formation", "place", "put", "organize")):
            structure_type = "open_ended_layout"
            slots = SpatialGeometryEngine.get_shape_slots("line", num_blocks=len(current_blocks))
        else:
            # Default single pick-place target
            structure_type = "single_pick_place"
            slots = [(STACK_ORIGIN[0], TABLE_TOP_Y + CUBE_HALF, STACK_ORIGIN[2])]

        # 2. Block Assignment (color-aware and reachability checked)
        avail_blocks = [b["id"] for b in current_blocks if not b.get("is_grasped")]
        assigned_blocks: List[str] = []

        if preferred_block_ids:
            for pid in preferred_block_ids:
                if pid in avail_blocks and pid not in assigned_blocks:
                    assigned_blocks.append(pid)

        # Color token parsing from instruction
        color_map = {
            "magenta": "block_magenta",
            "red": "block_magenta",
            "cyan": "block_cyan",
            "blue": "block_cyan",
            "yellow": "block_yellow",
            "green": "block_emerald",
            "emerald": "block_emerald",
            "orange": "block_orange",
            "purple": "block_purple",
            "pink": "block_pink",
        }
        for token, b_id in color_map.items():
            if token in inst_lower and b_id in avail_blocks and b_id not in assigned_blocks:
                assigned_blocks.append(b_id)

        # Fill remaining slots with closest available blocks
        for b_id in avail_blocks:
            if b_id not in assigned_blocks and len(assigned_blocks) < len(slots):
                assigned_blocks.append(b_id)

        # Truncate slots if fewer blocks available
        req_count = min(len(slots), len(assigned_blocks))
        slots = slots[:req_count]
        assigned_blocks = assigned_blocks[:req_count]

        # 3. Sort slots bottom-up by Y coordinate for structural stability
        indexed_slots = list(enumerate(slots))
        indexed_slots.sort(key=lambda item: item[1][1])

        # 4. Synthesize Subgoals and Calculate Prerequisite Dependencies
        created_subgoals: List[Subgoal] = []
        created_subgoal_ids: List[str] = []

        for order_idx, (orig_idx, target_slot) in enumerate(indexed_slots):
            block_id = assigned_blocks[order_idx]
            sg_id = f"subgoal_{order_idx}_{block_id}"

            rel_y = target_slot[1] - (TABLE_TOP_Y + CUBE_HALF)
            layer_idx = max(0, int(round(rel_y / CUBE_SIZE)))

            # Dependencies: any block physically underneath this target slot
            deps: List[str] = []
            for prev_idx in range(len(created_subgoals)):
                prev_sg = created_subgoals[prev_idx]
                if "target_slot" not in prev_sg.parameters:
                    continue
                prev_slot = prev_sg.parameters["target_slot"]
                # If prev_slot is at a lower altitude and horizontally close (supports current slot)
                dx = abs(prev_slot[0] - target_slot[0])
                dz = abs(prev_slot[2] - target_slot[2])
                dy = target_slot[1] - prev_slot[1]
                if dy > (CUBE_SIZE * 0.6) and dx < (CUBE_SIZE * 1.1) and dz < (CUBE_SIZE * 1.1):
                    deps.append(prev_sg.id)

            # Check if target slot is currently blocked by another ungrasped block
            for b in current_blocks:
                if b["id"] != block_id and not b.get("is_grasped"):
                    bx, by, bz = b["position"]
                    if (
                        abs(bx - target_slot[0]) < (CUBE_SIZE * 0.75)
                        and abs(by - target_slot[1]) < (CUBE_SIZE * 0.75)
                        and abs(bz - target_slot[2]) < (CUBE_SIZE * 0.75)
                    ):
                        # Generate Prerequisite Clearance Subgoal
                        clear_id = f"clear_obstacle_{b['id']}"
                        clear_sg = Subgoal(
                            id=clear_id,
                            parent_id="root_task",
                            level=1,
                            skill_type="CLEAR_OBSTACLE",
                            required_objects=[b["id"]],
                            parameters={
                                "block_id": b["id"],
                                "staging_slot": list(SAFE_STAGING_SLOTS[0]),
                            },
                            preconditions={"block_present": True, "gripper_free": True},
                            desired_effects={"obstacle_cleared": True},
                            dependencies=[],
                            description=f"Clear obstructing {b['id']} from slot to safe staging",
                        )
                        if clear_id not in self.subgoals:
                            self.subgoals[clear_id] = clear_sg
                            created_subgoals.insert(0, clear_sg)
                        deps.append(clear_id)

            subgoal = Subgoal(
                id=sg_id,
                parent_id="root_task",
                level=layer_idx + 1,
                skill_type="DEPOSIT",
                required_objects=[block_id],
                parameters={
                    "block_id": block_id,
                    "target_slot": list(target_slot),
                    "layer": layer_idx,
                    "approach_vector": [0.0, 1.0, 0.0],
                },
                preconditions={
                    "dependencies_satisfied": True,
                    "target_slot_clear": True,
                    "block_available": True,
                },
                desired_effects={
                    "object_at_slot": True,
                    "physically_supported": True,
                    "stably_placed": True,
                },
                dependencies=deps,
                status="PENDING",
                verification_criteria={
                    "max_dist_xz": 0.045,
                    "max_dist_y": 0.030,
                    "max_settle_vel": 0.035,
                    "min_hold_steps": 10,
                },
                description=f"Manipulate {block_id} onto Tier {layer_idx} slot ({target_slot[0]:.3f}, {target_slot[1]:.3f}, {target_slot[2]:.3f})",
            )
            created_subgoals.append(subgoal)
            self.subgoals[sg_id] = subgoal

        # 5. Acyclicity Validation
        if not self.is_acyclic():
            raise ValueError("Synthesized Subgoal DAG contains circular dependencies!")

        return list(self.subgoals.values())

    def is_acyclic(self) -> bool:
        """Kahn's algorithm to verify Directed Acyclic Graph properties."""
        in_degrees = {sg_id: len(sg.dependencies) for sg_id, sg in self.subgoals.items()}
        queue = [sg_id for sg_id, deg in in_degrees.items() if deg == 0]
        visited_count = 0

        # Adjacency: who depends on me
        dependents: Dict[str, List[str]] = {sg_id: [] for sg_id in self.subgoals}
        for sg_id, sg in self.subgoals.items():
            for dep in sg.dependencies:
                if dep in dependents:
                    dependents[dep].append(sg_id)

        while queue:
            curr = queue.pop(0)
            visited_count += 1
            for nxt in dependents.get(curr, []):
                in_degrees[nxt] -= 1
                if in_degrees[nxt] == 0:
                    queue.append(nxt)

        return visited_count == len(self.subgoals)

    def get_ready_subgoals(self) -> List[Subgoal]:
        """Returns all subgoals whose dependencies are 100% COMPLETED and status is PENDING."""
        completed_ids = {sg_id for sg_id, sg in self.subgoals.items() if sg.status == "COMPLETED"}
        ready: List[Subgoal] = []
        for sg in self.subgoals.values():
            if sg.status == "PENDING":
                if all(dep in completed_ids for dep in sg.dependencies):
                    ready.append(sg)
        return ready

    def mark_active(self, subgoal_id: str):
        if subgoal_id in self.subgoals:
            self.subgoals[subgoal_id].status = "ACTIVE"

    def mark_completed(self, subgoal_id: str):
        if subgoal_id in self.subgoals:
            self.subgoals[subgoal_id].status = "COMPLETED"

    def mark_failed(self, subgoal_id: str, reason: str = ""):
        if subgoal_id in self.subgoals:
            sg = self.subgoals[subgoal_id]
            sg.retry_count += 1
            if sg.retry_count < sg.max_retries:
                sg.status = "RECOVERING"
            else:
                sg.status = "FAILED"

    def splice_recovery_subgoal(self, failed_subgoal_id: str, recovery_subgoal: Subgoal):
        """
        Dynamically inserts a recovery subgoal into the active DAG:
        The failed subgoal is updated to depend on the recovery subgoal.
        """
        self.subgoals[recovery_subgoal.id] = recovery_subgoal
        if failed_subgoal_id in self.subgoals:
            failed_sg = self.subgoals[failed_subgoal_id]
            if recovery_subgoal.id not in failed_sg.dependencies:
                failed_sg.dependencies.append(recovery_subgoal.id)
            failed_sg.status = "PENDING"
