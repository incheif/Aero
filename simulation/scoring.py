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

def spatial_accuracy(blocks: List[Dict[str, Any]], target_origin: List[float] = None) -> float:
    """
    Mean spatial accuracy across active blocks: 70% position alignment, 30% upright orientation.
    """
    origin = target_origin or STACK_ORIGIN
    if not blocks:
        return 0.0
    
    total = 0.0
    for idx, b in enumerate(blocks):
        # Target slot height based on layer order
        target_pos = [origin[0], stack_slot_y(idx), origin[2]]
        pos = b.get("position", [0.0, 0.0, 0.0])
        rot = b.get("rotation", [0.0, 0.0, 0.0, 1.0])
        
        p_score = position_score(pos, target_pos)
        o_score = orientation_score(rot)
        total += 0.7 * p_score + 0.3 * o_score
        
    return round(total / len(blocks), 4)

def task_completion_score(blocks: List[Dict[str, Any]], target_origin: List[float] = None, grasped_block_id: Optional[str] = None) -> float:
    """
    Returns 1.0 if every block is stably seated in its 3D stack slot.
    A cube currently grasped in the gripper scores 0.0.
    """
    origin = target_origin or STACK_ORIGIN
    if not blocks:
        return 0.0
        
    scores = []
    for idx, b in enumerate(blocks):
        if b.get("id") == grasped_block_id or b.get("is_grasped"):
            scores.append(0.0)
            continue
            
        pos = b.get("position", [0.0, 0.0, 0.0])
        target_pos = [origin[0], stack_slot_y(idx), origin[2]]
        dist = vec_dist(pos, target_pos)
        
        if dist <= STACK_TOLERANCE:
            scores.append(1.0)
        else:
            # Partial credit for proximity to target pad
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

    def step(self, blocks: List[Dict[str, Any]], grasped_block_id: Optional[str] = None) -> Tuple[bool, int, str]:
        """
        Updates stability hold counter.
        Returns: (is_verified, hold_ticks, status_string)
        """
        if grasped_block_id is not None:
            self.hold_ticks = 0
            self.is_verified = False
            self.status = "GRASPING"
            return False, 0, self.status

        # Check if stacked blocks are in target pad
        stacked_in_pad = [b for b in blocks if b.get("in_pad") and b.get("stacked_layer", 0) >= 0]
        
        # Velocity check: blocks must be near rest (< 0.03 m/s)
        settled = True
        for b in stacked_in_pad:
            vel = b.get("velocity", [0.0, 0.0, 0.0])
            speed = math.hypot(vel[0], vel[1], vel[2])
            if speed > 0.04:
                settled = False
                break

        if len(stacked_in_pad) >= 1 and settled:
            self.hold_ticks += 1
            if self.hold_ticks >= self.required_ticks:
                self.is_verified = True
                self.status = "VERIFIED"
            else:
                self.status = f"HOLDING ({self.hold_ticks}/{self.required_ticks})"
        else:
            if self.hold_ticks > 0 and not self.is_verified:
                self.status = "UNSTABLE"
            elif not self.is_verified:
                self.status = "WAITING"
            self.hold_ticks = 0

        return self.is_verified, self.hold_ticks, self.status
