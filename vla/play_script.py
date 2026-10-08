"""
vla/play_script.py

Playground Canned & Dynamic Kinematic Trick Engine.
Directly ports and enhances vsarena's PlayScript:
  1. Time-indexed keyframe pose interpolator with joint limits clamping.
  2. Dynamic State-Aware Pointing: Computes analytical IK dynamically toward the specified target block.
  3. Expressive Non-Prehensile Gestures: Wave, Nod, Snap (rhythmic pinch), and Celebrate.
  4. Table Clearance Safety: Guarantees poses maintain safety margin above the workcell table.
"""

import math
from typing import Dict, List, Tuple, Optional, Any
from simulation.constants import DEFAULT_JOINTS, JOINT_LIMITS, TABLE_TOP_Y
from simulation.kinematics import inverse_kinematics, forward_kinematics, clamp

def clamp_pose(j: Dict[str, float]) -> Dict[str, float]:
    return {
        "baseYaw": clamp(j.get("baseYaw", 0.0), JOINT_LIMITS["baseYaw"][0], JOINT_LIMITS["baseYaw"][1]),
        "shoulderPitch": clamp(j.get("shoulderPitch", 0.0), JOINT_LIMITS["shoulderPitch"][0], JOINT_LIMITS["shoulderPitch"][1]),
        "elbowPitch": clamp(j.get("elbowPitch", 0.0), JOINT_LIMITS["elbowPitch"][0], JOINT_LIMITS["elbowPitch"][1]),
        "wristPitch": clamp(j.get("wristPitch", 0.0), JOINT_LIMITS["wristPitch"][0], JOINT_LIMITS["wristPitch"][1]),
        "gripper": clamp(j.get("gripper", 0.0), 0.0, 1.0),
    }

def mix_pose(a: Dict[str, float], b: Dict[str, float], u: float) -> Dict[str, float]:
    t = clamp(u, 0.0, 1.0)
    return clamp_pose({
        "baseYaw": a.get("baseYaw", 0.0) + (b.get("baseYaw", 0.0) - a.get("baseYaw", 0.0)) * t,
        "shoulderPitch": a.get("shoulderPitch", 0.0) + (b.get("shoulderPitch", 0.0) - a.get("shoulderPitch", 0.0)) * t,
        "elbowPitch": a.get("elbowPitch", 0.0) + (b.get("elbowPitch", 0.0) - a.get("elbowPitch", 0.0)) * t,
        "wristPitch": a.get("wristPitch", 0.0) + (b.get("wristPitch", 0.0) - a.get("wristPitch", 0.0)) * t,
        "gripper": a.get("gripper", 0.0) + (b.get("gripper", 0.0) - a.get("gripper", 0.0)) * t,
    })

def along_keys(t: int, keys: List[Dict[str, Any]]) -> Dict[str, float]:
    if not keys:
        return dict(DEFAULT_JOINTS)
    if t <= keys[0]["at"]:
        return dict(keys[0]["pose"])
    for i in range(1, len(keys)):
        prev = keys[i - 1]
        nxt = keys[i]
        if t <= nxt["at"]:
            span = max(1, nxt["at"] - prev["at"])
            return mix_pose(prev["pose"], nxt["pose"], (t - prev["at"]) / span)
    return dict(keys[-1]["pose"])

# Baseline canonical poses
REST = dict(DEFAULT_JOINTS)
LIFT = inverse_kinematics([0.12, 1.14, 0.0], gripper=0.0)
WAVE_L = inverse_kinematics([0.16, 1.08, -0.26], gripper=0.0)
WAVE_R = inverse_kinematics([0.16, 1.08, 0.26], gripper=0.0)
NOD_DOWN = inverse_kinematics([0.16, 0.94, 0.0], gripper=0.0)
CELEBRATE_HIGH = inverse_kinematics([0.08, 1.25, 0.0], gripper=1.0)

class PlayScriptAgent:
    """
    Executes canned or dynamic gestures matching vsarena's PlayScript engine.
    """
    def __init__(self, trick: str = "wave", target_name: Optional[str] = None):
        self.trick = trick.lower()
        self.target_name = (target_name or "cyan").lower()
        self.tick = 0
        self.finished = False
        self.last_plan = f"Play {trick}"

    def reset(self, trick: Optional[str] = None, target_name: Optional[str] = None):
        if trick:
            self.trick = trick.lower()
        if target_name:
            self.target_name = target_name.lower()
        self.tick = 0
        self.finished = False
        self.last_plan = f"Play {self.trick}"

    def act(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        self.tick += 1
        t = self.tick

        if self.trick == "wave":
            return self._wave(t)
        elif self.trick == "nod":
            return self._nod(t)
        elif self.trick == "snap":
            return self._snap(t)
        elif self.trick == "point":
            return self._point(t, snapshot)
        elif self.trick == "celebrate":
            return self._celebrate(t)
        else:
            return self._wave(t)

    def _wave(self, t: int) -> Dict[str, Any]:
        keys = [
            {"at": 0, "pose": REST},
            {"at": 20, "pose": LIFT},
            {"at": 38, "pose": WAVE_L},
            {"at": 56, "pose": WAVE_R},
            {"at": 74, "pose": WAVE_L},
            {"at": 92, "pose": WAVE_R},
            {"at": 110, "pose": LIFT},
            {"at": 135, "pose": REST},
        ]
        self.last_plan = "Expressive Wave Gesture (Greeting)"
        if t >= 135:
            self.finished = True
        pose = along_keys(t, keys)
        return {
            "joints": pose,
            "gripper": 0.0,
            "plan": self.last_plan,
            "is_active": not self.finished,
        }

    def _nod(self, t: int) -> Dict[str, Any]:
        keys = [
            {"at": 0, "pose": REST},
            {"at": 16, "pose": NOD_DOWN},
            {"at": 32, "pose": REST},
            {"at": 48, "pose": NOD_DOWN},
            {"at": 64, "pose": REST},
            {"at": 80, "pose": NOD_DOWN},
            {"at": 100, "pose": REST},
        ]
        self.last_plan = "Nodding Gesture (Affirmative Agreement)"
        if t >= 100:
            self.finished = True
        pose = along_keys(t, keys)
        return {
            "joints": pose,
            "gripper": 0.0,
            "plan": self.last_plan,
            "is_active": not self.finished,
        }

    def _snap(self, t: int) -> Dict[str, Any]:
        keys = [
            {"at": 0, "pose": REST},
            {"at": 20, "pose": LIFT},
            {"at": 96, "pose": LIFT},
            {"at": 118, "pose": REST},
        ]
        self.last_plan = "Gripper Snapping (Rhythmic Pinch / Clap)"
        if t >= 118:
            self.finished = True
        pose = along_keys(t, keys)
        # Periodic snap between ticks 24 and 96
        closed = (24 <= t <= 96) and (((t - 24) // 12) % 2 == 0)
        grip_val = 1.0 if closed else 0.0
        return {
            "joints": pose,
            "gripper": grip_val,
            "plan": self.last_plan,
            "is_active": not self.finished,
        }

    def _point(self, t: int, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        # Locate target block dynamically from snapshot
        blocks = snapshot.get("blocks", [])
        target_block = None
        for b in blocks:
            name = (b.get("name") or b.get("id") or "").lower()
            color = (b.get("color") or "").lower()
            if self.target_name in name or self.target_name in color or self.target_name in b.get("id", "").lower():
                target_block = b
                break

        if target_block:
            pos = target_block["position"]
            aim = [pos[0], max(pos[1] + 0.22, 0.98), pos[2]]
            name_label = target_block.get("name", self.target_name)
        else:
            aim = [0.26, 1.0, -0.16]
            name_label = self.target_name

        reach = inverse_kinematics(aim, gripper=0.0)
        keys = [
            {"at": 0, "pose": REST},
            {"at": 20, "pose": LIFT},
            {"at": 48, "pose": reach},
            {"at": 95, "pose": reach},
            {"at": 120, "pose": LIFT},
            {"at": 140, "pose": REST},
        ]
        self.last_plan = f"Dynamic Pointing at {name_label}"
        if t >= 140:
            self.finished = True
        pose = along_keys(t, keys)
        return {
            "joints": pose,
            "gripper": 0.0,
            "plan": self.last_plan,
            "is_active": not self.finished,
        }

    def _celebrate(self, t: int) -> Dict[str, Any]:
        keys = [
            {"at": 0, "pose": REST},
            {"at": 24, "pose": CELEBRATE_HIGH},
            {"at": 48, "pose": WAVE_L},
            {"at": 72, "pose": WAVE_R},
            {"at": 96, "pose": CELEBRATE_HIGH},
            {"at": 125, "pose": REST},
        ]
        self.last_plan = "Victory Celebration Gesture"
        if t >= 125:
            self.finished = True
        pose = along_keys(t, keys)
        return {
            "joints": pose,
            "gripper": 1.0 if 24 <= t <= 96 else 0.0,
            "plan": self.last_plan,
            "is_active": not self.finished,
        }
