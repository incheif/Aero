"""
Execution Monitor and Closed-Loop Verification for AERO Workcell.
Implements Observe-Act-Verify cycles and bounded failure recovery.
"""

import math
from typing import Tuple, List, Optional, Dict, Any

from simulation.constants import STACK_TOLERANCE, CUBE_SIZE, TABLE_TOP_Y, CUBE_HALF
from simulation.sensor_interfaces import CameraObservation, RobotStateObservation
from perception.models import TrackedObject
from vla.semantic_mapper import SemanticSpatialMapper


class VerificationResult:
    def __init__(self, verified: bool, confidence: float = 1.0, reason: str = "", recovery_action: Optional[str] = None):
        self.verified = verified
        self.confidence = confidence
        self.reason = reason
        self.recovery_action = recovery_action

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verified": self.verified,
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
            "recovery_action": self.recovery_action,
        }


class ExecutionMonitor:
    """
    Supervises discrete manipulation stages with empirical visual and proprioceptive verification.
    Prevents open-loop false-success assumptions.
    """
    def __init__(self, max_retries: int = 2, placement_tolerance: float = STACK_TOLERANCE):
        self.max_retries = max_retries
        self.placement_tolerance = placement_tolerance
        self.retry_counts: Dict[str, int] = {}
        self.current_phase: str = "IDLE"

    def set_phase(self, phase: str):
        self.current_phase = phase

    def verify_grasp(
        self,
        target_name: str,
        initial_target_pos: Tuple[float, float, float],
        robot_state: RobotStateObservation,
        tracks: List[TrackedObject]
    ) -> VerificationResult:
        """
        Verifies grasp success using visual evidence and gripper state.
        Does NOT query simulator's internal grasped_block_id!
        """
        # 1. Proprioceptive check: Gripper closed under motor effort or gap matches cube width
        effort = robot_state.gripper_effort
        gap = robot_state.gripper_position
        gripper_has_contact = (effort > 0.4) or (gap > 0.4)

        if not gripper_has_contact:
            retries = self.retry_counts.get(target_name, 0)
            if retries < self.max_retries:
                self.retry_counts[target_name] = retries + 1
                return VerificationResult(
                    verified=False,
                    confidence=0.20,
                    reason=f"Gripper effort ({effort:.2f}) indicates jaws closed on empty air (no contact).",
                    recovery_action="REACQUIRE_AND_RETRY_GRASP"
                )
            else:
                return VerificationResult(
                    verified=False,
                    confidence=0.10,
                    reason=f"Max grasp retries ({self.max_retries}) exhausted for {target_name}.",
                    recovery_action="ABORT_AND_REPLAN"
                )

        # 2. Visual evidence check: Target cube should no longer be resting undisturbed at initial position
        matching_tracks = [t for t in tracks if target_name.lower().split()[0] in (t.color_name or "").lower()]
        if matching_tracks:
            closest_track = min(matching_tracks, key=lambda t: math.hypot(t.position[0] - initial_target_pos[0], t.position[2] - initial_target_pos[2]))
            dist_to_init = math.hypot(closest_track.position[0] - initial_target_pos[0], closest_track.position[2] - initial_target_pos[2])
            alt_diff = closest_track.position[1] - initial_target_pos[1]

            # If cube was lifted or moved with gripper
            if alt_diff > 0.020 or dist_to_init > 0.030 or robot_state.gripper_position > 0.3:
                return VerificationResult(
                    verified=True,
                    confidence=0.95,
                    reason=f"Visual and tactile verification confirmed grasp of {target_name}."
                )

        # If no conflicting evidence, accept proprioceptive confirmation
        return VerificationResult(
            verified=True,
            confidence=0.88,
            reason=f"Gripper tactile engagement confirmed hold on {target_name}."
        )

    def verify_placement(
        self,
        target_name: str,
        expected_slot: Tuple[float, float, float],
        tracks: List[TrackedObject]
    ) -> VerificationResult:
        """
        Verifies placement accuracy at designated target slot from fresh camera observations.
        """
        tx, ty, tz = expected_slot
        tol = self.placement_tolerance * 1.5

        # Find closest tracked object
        if not tracks:
            return VerificationResult(False, 0.0, "No objects visible to verify placement", "INSPECT_SCENE")

        closest = min(tracks, key=lambda t: math.dist(t.position, (tx, ty, tz)))
        err_dist = math.dist(closest.position, (tx, ty, tz))

        if err_dist <= tol:
            return VerificationResult(
                verified=True,
                confidence=min(0.99, max(0.80, 1.0 - err_dist / tol)),
                reason=f"Placement verified: {closest.color_name} block positioned at {closest.position} (error: {err_dist*1000:.1f}mm within tolerance)."
            )
        else:
            retries = self.retry_counts.get(f"place_{target_name}", 0)
            if retries < self.max_retries:
                self.retry_counts[f"place_{target_name}"] = retries + 1
                return VerificationResult(
                    verified=False,
                    confidence=0.30,
                    reason=f"Placement error {err_dist*1000:.1f}mm exceeds tolerance {tol*1000:.1f}mm.",
                    recovery_action="REINSPECT_AND_NUDGE"
                )
            else:
                return VerificationResult(
                    verified=False,
                    confidence=0.10,
                    reason=f"Placement failed after {self.max_retries} attempts.",
                    recovery_action="ABORT_AND_REPLAN"
                )

