"""
simulation/scoring.py

Rigorous Multi-Metric Evaluation & Physical Stability Suite.
Directly implements the telemetry & verification standards from vsarena:
  1. Spatial Accuracy: 70% 3D Cartesian position precision + 30% upright orientation alignment.
  2. Task Completion Score: Layer-wise stack seating verification.
  3. Physical Stability Hold Verification: Enforces at least 18 consecutive physics ticks (0.3s)
     of stable resting contact after gripper release before declaring completion.
  4. Joint Effort & Torque Telemetry: Measures kinematic work proxy Σ|Δq| * gain with peak & avg telemetry.
"""

import math
from typing import Dict, List, Tuple, Optional, Any
from .constants import STACK_ORIGIN, STACK_TOLERANCE, CUBE_SIZE, TABLE_TOP_Y, stack_slot_y

POS_SCALE_M = 0.35
TORQUE_GAIN = 4.2
STABILITY_HOLD_THRESHOLD = 18  # 18 ticks at 60 Hz = 0.30 seconds

def clamp(val: float, low: float, high: float) -> float:
    return max(low, min(high, val))

def vec_dist(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a[:3], b[:3])))

def quat_rotate_vec(q: List[float], v: List[float]) -> List[float]:
    """Rotates a 3D vector v by quaternion q = [x, y, z, w]."""
    x, y, z, w = q[0], q[1], q[2], q[3]
    vx, vy, vz = v[0], v[1], v[2]
    
    # Calculate q * v * q^-1
    ix = w * vx + y * vz - z * vy
    iy = w * vy + z * vx - x * vz
    iz = w * vz + x * vy - y * vx
    iw = -x * vx - y * vy - z * vz
    
    rx = ix * w + iw * -x + iy * -z - iz * -y
    ry = iy * w + iw * -y + iz * -x - ix * -z
    rz = iz * w + iw * -z + ix * -y - iy * -x
    return [rx, ry, rz]

def position_score(actual: List[float], target: List[float]) -> float:
    """Position accuracy in [0, 1]: 1 at exact target, 0 at or beyond POS_SCALE_M (0.35m)."""
    dist = vec_dist(actual, target)
    return clamp(1.0 - dist / POS_SCALE_M, 0.0, 1.0)

def orientation_score(rotation: List[float]) -> float:
    """Upright score in [0, 1]: 1 when local +Y perfectly aligns with world up [0, 1, 0]."""
    if len(rotation) < 4:
        return 1.0
    up = quat_rotate_vec(rotation, [0.0, 1.0, 0.0])
    return clamp(abs(up[1]), 0.0, 1.0)

def spatial_accuracy(
    blocks: List[Dict[str, Any]],
    target_origin: List[float] = None,
    task_block_ids: Optional[List[str]] = None,
    task_targets: Optional[Dict[str, List[float]]] = None,
) -> float:
    """
    Task-Adaptive Spatial Accuracy:
    70% position alignment + 30% upright orientation.

    - If task_block_ids are specified (e.g. from VLA plan, ColorSeek, or user command):
      Evaluates strictly the blocks assigned to this task against their intended target slots.
      Completing a 1-block pick, 2-block or 3-block stack scores up to 100%.
    - If task_block_ids is None (idle baseline before task, or free exploration):
      Identifies blocks that have been moved / placed in target zone. If none are placed yet,
      returns 0.0 (idle state, no false baseline penalty).
    """
    origin = target_origin or STACK_ORIGIN
    if not blocks:
        return 0.0

    blocks_by_id = {b.get("id"): b for b in blocks}

    # 1. Explicit task specified
    if task_block_ids is not None and len(task_block_ids) > 0:
        eval_blocks = []
        for idx, b_id in enumerate(task_block_ids):
            b = blocks_by_id.get(b_id)
            if not b:
                continue
            if task_targets and b_id in task_targets:
                target_pos = task_targets[b_id]
            else:
                target_pos = [origin[0], stack_slot_y(idx), origin[2]]
            eval_blocks.append((b, target_pos))

        if not eval_blocks:
            return 0.0

        total = 0.0
        for b, target_pos in eval_blocks:
            pos = b.get("position", [0.0, 0.0, 0.0])
            rot = b.get("rotation", [0.0, 0.0, 0.0, 1.0])
            p_score = position_score(pos, target_pos)
            o_score = orientation_score(rot)
            total += 0.7 * p_score + 0.3 * o_score

        return round(total / len(eval_blocks), 4)

    # 2. No explicit task (idle baseline or unconstrained manual play)
    active_in_pad = [b for b in blocks if b.get("in_pad")]
    
    if not active_in_pad:
        return 0.0

    sorted_active = sorted(active_in_pad, key=lambda b: b.get("position", [0, 0, 0])[1])
    total = 0.0
    for idx, b in enumerate(sorted_active):
        target_pos = [origin[0], stack_slot_y(idx), origin[2]]
        pos = b.get("position", [0.0, 0.0, 0.0])
        rot = b.get("rotation", [0.0, 0.0, 0.0, 1.0])
        p_score = position_score(pos, target_pos)
        o_score = orientation_score(rot)
        total += 0.7 * p_score + 0.3 * o_score

    return round(total / len(sorted_active), 4)

def task_completion_score(
    blocks: List[Dict[str, Any]],
    target_origin: List[float] = None,
    grasped_block_id: Optional[str] = None,
    task_block_ids: Optional[List[str]] = None,
    task_targets: Optional[Dict[str, List[float]]] = None,
) -> float:
    """
    Task-Adaptive Task Completion Score in [0, 1]:
    - Returns 1.0 when all required task blocks are stably seated in their target slots.
    - A block currently held in gripper jaws scores 0.0.
    - If no task is active and no blocks are on pad, returns 0.0.
    """
    origin = target_origin or STACK_ORIGIN
    if not blocks:
        return 0.0

    blocks_by_id = {b.get("id"): b for b in blocks}

    # 1. Explicit task specified
    if task_block_ids is not None and len(task_block_ids) > 0:
        scores = []
        for idx, b_id in enumerate(task_block_ids):
            b = blocks_by_id.get(b_id)
            if not b or b.get("id") == grasped_block_id or b.get("is_grasped"):
                scores.append(0.0)
                continue

            pos = b.get("position", [0.0, 0.0, 0.0])
            if task_targets and b_id in task_targets:
                target_pos = task_targets[b_id]
            else:
                target_pos = [origin[0], stack_slot_y(idx), origin[2]]

            dist = vec_dist(pos, target_pos)
            if dist <= STACK_TOLERANCE:
                scores.append(1.0)
            else:
                scores.append(0.5 * clamp(1.0 - dist / 0.55, 0.0, 1.0))

        if not scores:
            return 0.0
        if all(s >= 1.0 for s in scores):
            return 1.0
        return round(sum(scores) / len(scores), 4)

    # 2. No explicit task specified (manual or idle)
    active_in_pad = [b for b in blocks if b.get("in_pad")]
    if not active_in_pad:
        return 0.0

    sorted_active = sorted(active_in_pad, key=lambda b: b.get("position", [0, 0, 0])[1])
    scores = []
    for idx, b in enumerate(sorted_active):
        if b.get("id") == grasped_block_id or b.get("is_grasped"):
            scores.append(0.0)
            continue
        pos = b.get("position", [0.0, 0.0, 0.0])
        target_pos = [origin[0], stack_slot_y(idx), origin[2]]
        dist = vec_dist(pos, target_pos)
        if dist <= STACK_TOLERANCE:
            scores.append(1.0)
        else:
            scores.append(0.5 * clamp(1.0 - dist / 0.55, 0.0, 1.0))

    if all(s >= 1.0 for s in scores):
        return 1.0
    return round(sum(scores) / len(scores), 4)

class TorqueTracker:
    """
    Tracks kinematic effort (proxy for joint torque) across 60 Hz simulation steps.
    Records Σ|Δq| * gain with running average and peak telemetry.
    """
    def __init__(self, max_samples: int = 600):
        self.samples: List[float] = []
        self.max_samples = max_samples
        self.running_sum = 0.0
        self.peak = 0.0

    def reset(self):
        self.samples.clear()
        self.running_sum = 0.0
        self.peak = 0.0

    def sample(self, prev_joints: Dict[str, float], next_joints: Dict[str, float]) -> float:
        effort = (
            abs(next_joints.get("baseYaw", 0) - prev_joints.get("baseYaw", 0)) +
            abs(next_joints.get("shoulderPitch", 0) - prev_joints.get("shoulderPitch", 0)) +
            abs(next_joints.get("elbowPitch", 0) - prev_joints.get("elbowPitch", 0)) +
            abs(next_joints.get("wristPitch", 0) - prev_joints.get("wristPitch", 0)) +
            abs(next_joints.get("gripper", 0) - prev_joints.get("gripper", 0))
        ) * TORQUE_GAIN

        self.samples.append(effort)
        self.running_sum += effort
        if effort > self.peak:
            self.peak = effort
            
        if len(self.samples) > self.max_samples:
            removed = self.samples.pop(0)
            self.running_sum -= removed
            
        return effort

    def summarize(self) -> Dict[str, float]:
        avg = self.running_sum / len(self.samples) if self.samples else 0.0
        return {
            "peak": round(self.peak, 3),
            "avg": round(avg, 3)
        }

class StabilityValidator:
    """
    Enforces the vsarena 18-tick physical stability hold requirement.
    Ensures blocks settle under gravity and remain stationary on the pad
    without toppling or sliding off before declaring completion.
    """
    def __init__(self, required_ticks: int = STABILITY_HOLD_THRESHOLD):
        self.required_ticks = required_ticks
        self.hold_ticks: int = 0
        self.is_verified: bool = False
        self.status: str = "IDLE"  # "IDLE", "HOLDING", "VERIFIED", "UNSTABLE"

    def reset(self):
        self.hold_ticks = 0
        self.is_verified = False
        self.status = "IDLE"

    def step(
        self,
        blocks: List[Dict[str, Any]],
        grasped_block_id: Optional[str] = None,
        target_block_id: Optional[str] = None,
    ) -> Tuple[bool, int, str]:
        """
        Updates stability hold counter.
        Returns: (is_verified, hold_ticks, status_string)
        """
        if grasped_block_id is not None:
            self.hold_ticks = 0
            self.is_verified = False
            self.status = "GRASPING"
            return False, 0, self.status

        # If a specific target_block_id is provided, verify that block directly
        if target_block_id is not None:
            target_b = next((b for b in blocks if b.get("id") == target_block_id), None)
            if not target_b or target_b.get("is_grasped"):
                self.hold_ticks = 0
                self.status = "WAITING"
                return False, 0, self.status
            vel = target_b.get("velocity", [0.0, 0.0, 0.0])
            speed = math.hypot(vel[0], vel[1], vel[2])
            settled = speed <= 0.04
            valid_blocks = [target_b]
        else:
            # Check if stacked blocks are in target pad
            valid_blocks = [b for b in blocks if b.get("in_pad")]
            settled = True
            for b in valid_blocks:
                vel = b.get("velocity", [0.0, 0.0, 0.0])
                speed = math.hypot(vel[0], vel[1], vel[2])
                if speed > 0.04:
                    settled = False
                    break

        if len(valid_blocks) >= 1 and settled:
            self.hold_ticks = min(self.hold_ticks + 1, self.required_ticks)
            if self.hold_ticks >= self.required_ticks:
                self.is_verified = True
                self.status = "VERIFIED"
            else:
                self.status = f"HOLDING ({self.hold_ticks}/{self.required_ticks})"
        else:
            if self.hold_ticks > 0 and not self.is_verified:
                self.status = "UNSTABLE"
            elif not self.is_verified:
                self.status = "IDLE" if len(valid_blocks) == 0 else "WAITING"
            self.hold_ticks = 0

        return self.is_verified, self.hold_ticks, self.status
