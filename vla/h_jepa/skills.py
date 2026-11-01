"""
vla/h_jepa/skills.py

Primitive Robotic Manipulation Skills for Hierarchical VLA-JEPA.
Defines composable, closed-loop behavioral primitives:
- PICK(block_id)
- PLACE(block_id, target_slot)
- STACK(block_id, support_ids)
- CLEAR(obstructing_block_id, staging_slot)
- MOVE(block_id, from_slot, to_slot)

Each skill handles explicit pre-conditions, physical micro-phases,
and verifiable post-conditions.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import math

from simulation.constants import (
    CUBE_SIZE,
    CUBE_HALF,
    TABLE_TOP_Y,
    SAFE_HOVER_Y,
    PICK_Y,
    GRASP_CLOSE_THRESHOLD,
    GRASP_OPEN_THRESHOLD,
)


class BaseSkill(ABC):
    """Abstract base class for all manipulation primitives."""
    def __init__(self, name: str, timeout_ticks: int = 180):
        self.name = name
        self.timeout_ticks = timeout_ticks
        self.elapsed_ticks: int = 0
        self.is_completed: bool = False
        self.is_failed: bool = False
        self.failure_reason: str = ""

    def reset(self):
        self.elapsed_ticks = 0
        self.is_completed = False
        self.is_failed = False
        self.failure_reason = ""

    @abstractmethod
    def check_preconditions(self, world_state: Dict[str, Any]) -> Tuple[bool, str]:
        """Verifies physical requirements before starting the skill."""
        pass

    @abstractmethod
    def step(
        self,
        current_tcp: List[float],
        current_gripper: float,
        world_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Executes one 60Hz tick of the skill.
        Returns:
            {
                "target_pos": [x, y, z],
                "desired_gripper": float,
                "phase": str,
                "is_done": bool,
            }
        """
        pass

    @abstractmethod
    def check_postconditions(self, world_state: Dict[str, Any]) -> Tuple[bool, str]:
        """Verifies physical criteria for skill completion."""
        pass


class PickSkill(BaseSkill):
    """
    Primitive skill to acquire and grasp a block cleanly from tabletop or stack.
    Micro-phases: HOVER -> DESCEND -> GRASP -> LIFT.
    """
    def __init__(self, block_id: str):
        super().__init__(name=f"PICK({block_id})", timeout_ticks=150)
        self.block_id = block_id
        self.phase: str = "HOVER"

    def check_preconditions(self, world_state: Dict[str, Any]) -> Tuple[bool, str]:
        blocks = {b["id"]: b for b in world_state.get("blocks", [])}
        if self.block_id not in blocks:
            return False, f"Target block {self.block_id} does not exist in workspace."
        block = blocks[self.block_id]
        if block.get("is_grasped"):
            return False, f"Block {self.block_id} is already grasped."
        return True, "Ready to pick."

    def step(
        self,
        current_tcp: List[float],
        current_gripper: float,
        world_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        self.elapsed_ticks += 1
        blocks = {b["id"]: b for b in world_state.get("blocks", [])}
        block = blocks.get(self.block_id)
        if not block:
            self.is_failed = True
            self.failure_reason = "Block disappeared"
            return {"target_pos": current_tcp, "desired_gripper": current_gripper, "phase": "FAILED", "is_done": True}

        b_pos = block["position"]
        max_scene_y = max((b.get("position", [0, 0, 0])[1] for b in blocks.values() if not b.get("is_grasped")), default=TABLE_TOP_Y)
        transit_y = max(SAFE_HOVER_Y, b_pos[1] + 0.080, max_scene_y + CUBE_SIZE * 1.5 + 0.035)
        transit_y = min(transit_y, 1.08)
        pick_y = max(0.756, b_pos[1] + 0.010)

        target_pos = list(current_tcp)
        desired_gripper = current_gripper

        if self.phase == "HOVER":
            desired_gripper = 0.0
            dist = math.hypot(current_tcp[0] - b_pos[0], current_tcp[2] - b_pos[2])
            # Ascend-First Guard: if low, ascend vertically before translating horizontally
            if current_tcp[1] < transit_y - 0.025 and dist > 0.025:
                target_pos = [current_tcp[0], transit_y, current_tcp[2]]
            else:
                target_pos = [b_pos[0], transit_y, b_pos[2]]
            if (dist < 0.015 and abs(current_tcp[1] - transit_y) < 0.025) or self.elapsed_ticks > 60:
                self.phase = "DESCEND"

        elif self.phase == "DESCEND":
            target_pos = [b_pos[0], pick_y, b_pos[2]]
            desired_gripper = 0.0
            h_err = math.hypot(current_tcp[0] - b_pos[0], current_tcp[2] - b_pos[2])
            v_err = abs(current_tcp[1] - pick_y)
            if (h_err < 0.010 and v_err < 0.005) or self.elapsed_ticks > 90:
                self.phase = "GRASP"

        elif self.phase == "GRASP":
            target_pos = [b_pos[0], pick_y, b_pos[2]]
            desired_gripper = 1.0
            if current_gripper >= GRASP_CLOSE_THRESHOLD + 0.02 and block.get("is_grasped"):
                self.phase = "LIFT"
            elif self.elapsed_ticks > 120:
                self.is_failed = True
                self.failure_reason = "Grasp timeout"

        elif self.phase == "LIFT":
            target_pos = [current_tcp[0], transit_y, current_tcp[2]]
            desired_gripper = 1.0
            if abs(current_tcp[1] - transit_y) < 0.025:
                self.is_completed = True

        return {
            "target_pos": target_pos,
            "desired_gripper": desired_gripper,
            "phase": self.phase,
            "is_done": self.is_completed or self.is_failed,
        }

    def check_postconditions(self, world_state: Dict[str, Any]) -> Tuple[bool, str]:
        blocks = {b["id"]: b for b in world_state.get("blocks", [])}
        block = blocks.get(self.block_id)
        if block and block.get("is_grasped"):
            return True, f"Block {self.block_id} successfully grasped and lifted."
        return False, f"Block {self.block_id} not grasped."


class PlaceSkill(BaseSkill):
    """
    Primitive skill to deliver a held block to an exact target slot and release it.
    Micro-phases: TRANSIT -> DESCEND -> RELEASE -> RETREAT.
    """
    def __init__(self, block_id: str, target_slot: List[float]):
        super().__init__(name=f"PLACE({block_id} -> {target_slot})", timeout_ticks=180)
        self.block_id = block_id
        self.target_slot = list(target_slot)
        self.phase: str = "TRANSIT"

    def check_preconditions(self, world_state: Dict[str, Any]) -> Tuple[bool, str]:
        blocks = {b["id"]: b for b in world_state.get("blocks", [])}
        block = blocks.get(self.block_id)
        if not block or not block.get("is_grasped"):
            return False, f"Block {self.block_id} is not currently grasped."
        return True, "Ready to place."

    def step(
        self,
        current_tcp: List[float],
        current_gripper: float,
        world_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        self.elapsed_ticks += 1
        max_scene_y = max((b.get("position", [0, 0, 0])[1] for b in blocks.values() if not b.get("is_grasped")), default=TABLE_TOP_Y)
        transit_y = max(SAFE_HOVER_Y, self.target_slot[1] + 0.080, max_scene_y + CUBE_SIZE * 1.5 + 0.035)
        transit_y = min(transit_y, 1.08)
        place_y = self.target_slot[1] + 0.008

        target_pos = list(current_tcp)
        desired_gripper = current_gripper

        if self.phase == "TRANSIT":
            desired_gripper = 1.0
            h_dist = math.hypot(current_tcp[0] - self.target_slot[0], current_tcp[2] - self.target_slot[2])
            # Ascend-First Guard: ensure safe transit altitude is reached before horizontal motion
            if current_tcp[1] < transit_y - 0.020 and h_dist > 0.025:
                target_pos = [current_tcp[0], transit_y, current_tcp[2]]
            else:
                target_pos = [self.target_slot[0], transit_y, self.target_slot[2]]
            if (h_dist < 0.018 and abs(current_tcp[1] - transit_y) < 0.025) or self.elapsed_ticks > 70:
                self.phase = "DESCEND"

        elif self.phase == "DESCEND":
            target_pos = [self.target_slot[0], place_y, self.target_slot[2]]
            desired_gripper = 1.0
            h_err = math.hypot(current_tcp[0] - self.target_slot[0], current_tcp[2] - self.target_slot[2])
            v_err = abs(current_tcp[1] - place_y)
            if (h_err < 0.015 and v_err < 0.010) or self.elapsed_ticks > 100:
                self.phase = "RELEASE"

        elif self.phase == "RELEASE":
            target_pos = [self.target_slot[0], place_y, self.target_slot[2]]
            desired_gripper = 0.0
            blocks = {b["id"]: b for b in world_state.get("blocks", [])}
            block = blocks.get(self.block_id)
            if current_gripper <= GRASP_OPEN_THRESHOLD - 0.05 and (block and not block.get("is_grasped")):
                self.phase = "RETREAT"
            elif self.elapsed_ticks > 140:
                self.phase = "RETREAT"

        elif self.phase == "RETREAT":
            target_pos = [self.target_slot[0], transit_y, self.target_slot[2]]
            desired_gripper = 0.0
            if abs(current_tcp[1] - transit_y) < 0.030:
                self.is_completed = True

        return {
            "target_pos": target_pos,
            "desired_gripper": desired_gripper,
            "phase": self.phase,
            "is_done": self.is_completed or self.is_failed,
        }

    def check_postconditions(self, world_state: Dict[str, Any]) -> Tuple[bool, str]:
        blocks = {b["id"]: b for b in world_state.get("blocks", [])}
        block = blocks.get(self.block_id)
        if not block:
            return False, f"Block {self.block_id} not found."
        b_pos = block["position"]
        h_err = math.hypot(b_pos[0] - self.target_slot[0], b_pos[2] - self.target_slot[2])
        v_err = abs(b_pos[1] - self.target_slot[1])
        if h_err <= 0.045 and v_err <= 0.035 and not block.get("is_grasped"):
            return True, f"Block {self.block_id} placed accurately within tolerance."
        return False, f"Block {self.block_id} placement error: h={h_err*1000:.1f}mm, v={v_err*1000:.1f}mm."


class StackSkill(PlaceSkill):
    """
    Specialized Placement Skill that stacks on top of underlying support block(s).
    Aligns placement position dynamically with the real observed center of mass
    of the supporting blocks to prevent toppling or slanting.
    """
    def __init__(self, block_id: str, support_ids: List[str], nominal_slot: List[float]):
        super().__init__(block_id=block_id, target_slot=nominal_slot)
        self.support_ids = list(support_ids)
        self.name = f"STACK({block_id} on {support_ids})"

    def check_preconditions(self, world_state: Dict[str, Any]) -> Tuple[bool, str]:
        blocks = {b["id"]: b for b in world_state.get("blocks", [])}
        for s_id in self.support_ids:
            if s_id not in blocks:
                return False, f"Supporting block {s_id} is missing."
            s_block = blocks[s_id]
            if s_block.get("is_grasped"):
                return False, f"Supporting block {s_id} is currently grasped."
        return True, "Supports valid and stable."

    def step(
        self,
        current_tcp: List[float],
        current_gripper: float,
        world_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Dynamic Center of Mass Alignment: adjust target slot X and Z to match supporting block(s)
        blocks = {b["id"]: b for b in world_state.get("blocks", [])}
        valid_supports = [blocks[s_id] for s_id in self.support_ids if s_id in blocks]
        if valid_supports:
            avg_x = sum(b["position"][0] for b in valid_supports) / len(valid_supports)
            avg_z = sum(b["position"][2] for b in valid_supports) / len(valid_supports)
            max_y = max(b["position"][1] for b in valid_supports)
            # Align with support center of mass
            self.target_slot[0] = avg_x
            self.target_slot[1] = max_y + CUBE_SIZE
            self.target_slot[2] = avg_z

        return super().step(current_tcp, current_gripper, world_state)
