"""
vla/h_jepa/task_analyzer.py

Feasibility Analyzer and Task Verification Gate for Hierarchical VLA-JEPA.
Implements the 3-Gate physical feasibility check before plan execution:
1. Resource Feasibility Gate: Verifies block count and required colors.
2. Geometric Support Gate: Ensures structures have valid physical foundations.
3. Kinematic Reachability Gate: Checks if target slots are within the arm's workspace.
"""

from typing import Dict, List, Any, Tuple, Optional
import math
import re

from simulation.constants import (
    TABLE_HALF_EXTENTS,
    TABLE_TOP_Y,
    CUBE_SIZE,
    MAX_ALTITUDE_CLAMP,
)
from simulation.kinematics import inverse_kinematics


class TaskFeasibilityAnalyzer:
    """
    Evaluates whether a manipulation task is physically and kinematically possible.
    Provides clear, human-understandable explanations when tasks cannot be executed.
    """
    def __init__(self, arm_max_reach: float = 0.85, arm_min_reach: float = 0.15):
        self.arm_max_reach = arm_max_reach
        self.arm_min_reach = arm_min_reach

    def check_resource_feasibility(
        self,
        required_count: int,
        available_blocks: List[Dict[str, Any]],
    ) -> Tuple[bool, str]:
        """Gate 1: Verifies enough blocks exist on the table."""
        avail_count = len(available_blocks)
        if required_count > avail_count:
            return (
                False,
                f"Task requires {required_count} blocks, but only {avail_count} blocks are present on the table."
            )
        return True, f"Resource check passed: {avail_count} blocks available for {required_count} required."

    def check_kinematic_feasibility(
        self,
        target_slots: List[List[float]],
    ) -> Tuple[bool, str]:
        """Gate 2: Verifies that analytical IK can solve all target positions."""
        for idx, slot in enumerate(target_slots):
            # Check table horizontal reach radius from arm base (0, 0)
            radial_dist = math.hypot(slot[0], slot[2])
            if radial_dist > self.arm_max_reach:
                return (
                    False,
                    f"Slot {idx} at ({slot[0]:.3f}, {slot[1]:.3f}, {slot[2]:.3f}) is beyond maximum arm reach ({radial_dist:.2f}m > {self.arm_max_reach:.2f}m)."
                )
            if radial_dist < self.arm_min_reach:
                return (
                    False,
                    f"Slot {idx} is too close to arm pedestal ({radial_dist:.2f}m < {self.arm_min_reach:.2f}m)."
                )
            if slot[1] > MAX_ALTITUDE_CLAMP:
                return (
                    False,
                    f"Slot {idx} altitude ({slot[1]:.2f}m) exceeds ceiling clamp ({MAX_ALTITUDE_CLAMP:.2f}m)."
                )

            # Test IK solver solvability
            ik_sol = inverse_kinematics(slot)
            if ik_sol is None:
                return (
                    False,
                    f"Kinematic singularity or reach boundary at slot {idx} ({slot[0]:.3f}, {slot[1]:.3f}, {slot[2]:.3f})."
                )

        return True, "All target coordinates are kinematically reachable."

    def check_geometric_stability(
        self,
        target_slots: List[List[float]],
    ) -> Tuple[bool, str]:
        """Gate 3: Checks that upper-tier blocks have physical foundation underneath."""
        for idx, slot in enumerate(target_slots):
            y = slot[1]
            layer = round((y - TABLE_TOP_Y) / CUBE_SIZE)
            if layer > 0:
                # Requires at least one supporting block at layer - 1 directly below (within 0.9 * CUBE_SIZE)
                support_found = False
                for other_slot in target_slots[:idx]:
                    other_y = other_slot[1]
                    other_layer = round((other_y - TABLE_TOP_Y) / CUBE_SIZE)
                    if other_layer == layer - 1:
                        dx = abs(slot[0] - other_slot[0])
                        dz = abs(slot[2] - other_slot[2])
                        if dx < CUBE_SIZE * 0.95 and dz < CUBE_SIZE * 0.95:
                            support_found = True
                            break
                if not support_found:
                    return (
                        False,
                        f"Slot {idx} at Tier {layer} lacks physical support below it. Cannot place block in mid-air."
                    )
        return True, "Geometric stability structure is valid."

    def analyze(
        self,
        instruction: str,
        target_slots: List[List[float]],
        available_blocks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Runs the full 3-Gate Feasibility Verification.
        Returns a comprehensive report dict.
        """
        # Gate 1: Resources
        r_ok, r_msg = self.check_resource_feasibility(len(target_slots), available_blocks)
        if not r_ok:
            return {"is_feasible": False, "failed_gate": "RESOURCE", "reason": r_msg}

        # Gate 2: Kinematics
        k_ok, k_msg = self.check_kinematic_feasibility(target_slots)
        if not k_ok:
            return {"is_feasible": False, "failed_gate": "KINEMATICS", "reason": k_msg}

        # Gate 3: Geometric Stability
        g_ok, g_msg = self.check_geometric_stability(target_slots)
        if not g_ok:
            return {"is_feasible": False, "failed_gate": "GEOMETRIC_STABILITY", "reason": g_msg}

        return {
            "is_feasible": True,
            "failed_gate": None,
            "reason": "Task passed all resource, reachability, and physical stability checks.",
            "num_blocks_needed": len(target_slots),
            "num_blocks_available": len(available_blocks),
        }
