"""
vla/color_seek_policy.py

In-Tab Closed-Loop Visual Servoing Policy (ColorSeek VLA).
Directly adopts the vsarena ColorSeek & findBlobs architecture:
  1. Pure Vision Contract: Operates exclusively on top-down 128x128 RGB camera frames.
     No privileged 3D object GPS coordinates are accessed.
  2. Color-Space Blob Detection: Segment visual centroids of cubes, TCP crosshair, and target pad.
  3. Visual Servoing: Maps pixel errors (du, dv) directly to Cartesian tool deltas (dx, dz).
  4. Multi-Phase Stacking State Machine: Closed-loop hover, down, pinch, lift, carry, drop, release.
"""

import math
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image
import numpy as np

from simulation.constants import (
    VLA_IMAGE_SIZE,
    VLA_X_MIN,
    VLA_X_MAX,
    VLA_Z_MIN,
    VLA_Z_MAX,
    STACK_ORIGIN,
    TABLE_TOP_Y,
    CUBE_HALF,
    stack_slot_y,
)
from simulation.kinematics import inverse_kinematics

STEP_LIMIT = 0.03
XY_CLOSE_PX = 6
XY_NEAR_PX = 14
HELD_PX = 10
STACKED_PAD_PX = 8

HOVER_Y = TABLE_TOP_Y + CUBE_HALF + 0.12
PICK_Y = TABLE_TOP_Y + CUBE_HALF + 0.012

DOWN_TICKS = 24
PINCH_TICKS = 16
LIFT_TICKS = 12
CARRY_MAX_TICKS = 80
DROP_TICKS = 18
OPEN_TICKS = 12
PLACE_XY_TOL = 0.03
CARRY_CLEARANCE = 0.11

ORDER = ["cyan", "orange", "magenta"]

def classify_pixel(r: int, g: int, b: int) -> Optional[str]:
    """Color classification matching vsarena blob thresholds."""
    if r > 220 and g > 220 and b > 220:
        return "tcp"
    if r < 80 and g > 120 and b > 180:
        return "cyan"
    if r > 200 and 90 < g < 210 and b < 90:
        return "orange"
    if r > 160 and g < 90 and b > 90:
        return "magenta"
    if r > 180 and g > 180 and b < 80:
        return "yellow"
    if r < 60 and g > 140 and b < 100:
        return "emerald"
    if r < 65 and 50 < g < 110 and 65 < b < 125:
        return "pad"
    return None

def find_blobs(img: Image.Image) -> Dict[str, Optional[Tuple[float, float]]]:
    """
    Extracts 2D centroid (u, v) for cubes, robot TCP, and pad from raw RGB image.
    """
    if img.size != (VLA_IMAGE_SIZE, VLA_IMAGE_SIZE):
        img = img.resize((VLA_IMAGE_SIZE, VLA_IMAGE_SIZE))
        
    arr = np.array(img.convert("RGB"))
    height, width, _ = arr.shape
    
    sums: Dict[str, List[float]] = {
        "cyan": [0.0, 0.0, 0.0],
        "orange": [0.0, 0.0, 0.0],
        "magenta": [0.0, 0.0, 0.0],
        "yellow": [0.0, 0.0, 0.0],
        "emerald": [0.0, 0.0, 0.0],
        "tcp": [0.0, 0.0, 0.0],
        "pad": [0.0, 0.0, 0.0],
    }
    
    for v in range(height):
        for u in range(width):
            r, g, b = int(arr[v, u, 0]), int(arr[v, u, 1]), int(arr[v, u, 2])
            tag = classify_pixel(r, g, b)
            if tag and tag in sums:
                sums[tag][0] += u
                sums[tag][1] += v
                sums[tag][2] += 1
                
    blobs: Dict[str, Optional[Tuple[float, float]]] = {}
    min_counts = {"tcp": 3, "pad": 6, "cyan": 5, "orange": 5, "magenta": 5, "yellow": 5, "emerald": 5}
    for tag, (su, sv, n) in sums.items():
        min_n = min_counts.get(tag, 4)
        if n >= min_n:
            blobs[tag] = (su / n, sv / n)
        else:
            blobs[tag] = None
            
    return blobs

def pixel_to_ee_delta(du: float, dv: float, size: int = VLA_IMAGE_SIZE) -> Tuple[float, float]:
    """Converts image pixel error (du, dv) to table-plane metric end-effector delta (dx, dz)."""
    span = max(1.0, float(size - 1))
    dx = (du / span) * (VLA_X_MAX - VLA_X_MIN)
    dz = -(dv / span) * (VLA_Z_MAX - VLA_Z_MIN)
    return dx, dz

def clamp_step(dx: float, dy: float, dz: float, max_step: float = STEP_LIMIT) -> Tuple[float, float, float]:
    m = math.hypot(dx, dy, dz)
    if m <= max_step or m < 1e-9:
        return dx, dy, dz
    s = max_step / m
    return dx * s, dy * s, dz * s

class ColorSeekPolicy:
    """
    Closed-loop vision pick-and-place policy running exclusively on RGB frames.
    """
    def __init__(self, target_order: List[str] = None):
        self.order = list(target_order or ORDER)
        self.phase: str = "hover"
        self.hold: int = 0
        self.target_idx: int = 0
        self.last_goal: Optional[Tuple[float, float]] = None
        self.latest_plan: str = "ColorSeek Idle"
        self.is_done: bool = False

    def reset(self, target_order: List[str] = None):
        if target_order:
            self.order = list(target_order)
        self.phase = "hover"
        self.hold = 0
        self.target_idx = 0
        self.last_goal = None
        self.latest_plan = "ColorSeek Reset"
        self.is_done = False

    @property
    def current_target(self) -> Optional[str]:
        if self.target_idx < len(self.order):
            return self.order[self.target_idx]
        return None

    def step(self, img: Image.Image, current_joints: Dict[str, float], tcp_world: Tuple[float, float, float]) -> Dict[str, Any]:
        """
        Processes one vision step. Returns action command dictionary with target joints and gripper.
        """
        if self.is_done or self.current_target is None:
            self.latest_plan = "ColorSeek Stacking Completed"
            return {
                "joints": inverse_kinematics(tcp_world, gripper=0.0),
                "gripper": 0.0,
                "plan": self.latest_plan,
                "is_active": False,
            }

        blobs = find_blobs(img)
        tcp_blob = blobs.get("tcp")
        pad_blob = blobs.get("pad")
        target_name = self.current_target

        # If TCP marker is occluded or not detected, use fallback world projection
        if tcp_blob is None:
            from simulation.camera import world_to_pixel
            tcp_u, tcp_v = world_to_pixel(tcp_world[0], tcp_world[2])
            tcp_blob = (float(tcp_u), float(tcp_v))

        target_blob = blobs.get(target_name)
        if target_blob is not None:
            self.last_goal = target_blob

        goal_blob = target_blob or self.last_goal

        # Check if already stacked
        if target_blob and pad_blob:
            dist_to_pad_px = math.hypot(target_blob[0] - pad_blob[0], target_blob[1] - pad_blob[1])
            if dist_to_pad_px <= STACKED_PAD_PX and self.phase == "hover":
                # Advance to next target
                self.target_idx += 1
                self.hold = 0
                self.last_goal = None
                if self.current_target is None:
                    self.is_done = True
                    self.latest_plan = "All targets stacked!"
                    return {
                        "joints": inverse_kinematics(tcp_world, gripper=0.0),
                        "gripper": 0.0,
                        "plan": self.latest_plan,
                        "is_active": False,
                    }
                return self.step(img, current_joints, tcp_world)

        aim_blob = goal_blob or tcp_blob
        du = aim_blob[0] - tcp_blob[0]
        dv = aim_blob[1] - tcp_blob[1]
        dx, dz = pixel_to_ee_delta(du, dv)
        xy_dist_px = math.hypot(du, dv)
        xy_close = xy_dist_px <= XY_CLOSE_PX

        curr_x, curr_y, curr_z = tcp_world
        slot_y = stack_slot_y(self.target_idx)
        pad_world = [STACK_ORIGIN[0], slot_y, STACK_ORIGIN[2]]

        # State Machine Transitions
        grip_state = 0.0  # 0.0 = open, 1.0 = closed
        target_pos = [curr_x, curr_y, curr_z]

        if self.phase == "hover":
            self.latest_plan = f"Seek {target_name.capitalize()} (Pixel Delta: {round(xy_dist_px, 1)}px)"
            self.hold += 1
            if (xy_close and self.hold > 3) or (xy_dist_px <= XY_NEAR_PX and self.hold > 35):
                self.phase = "down"
                self.hold = 0
            step_dx, step_dy, step_dz = clamp_step(dx, HOVER_Y - curr_y, dz)
            target_pos = [curr_x + step_dx, curr_y + step_dy, curr_z + step_dz]
            grip_state = 0.0

        elif self.phase == "down":
            self.latest_plan = f"Descend to {target_name.capitalize()} (Y: {round(curr_y, 3)}m)"
            self.hold += 1
            if curr_y <= PICK_Y + 0.025 or self.hold > DOWN_TICKS:
                self.phase = "pinch"
                self.hold = 0
            step_dx, step_dy, step_dz = clamp_step(dx * 0.35, PICK_Y - curr_y, dz * 0.35)
            target_pos = [curr_x + step_dx, curr_y + step_dy, curr_z + step_dz]
            grip_state = 0.0

        elif self.phase == "pinch":
            self.latest_plan = f"Pinch Grip on {target_name.capitalize()}"
            self.hold += 1
            if self.hold > PINCH_TICKS:
                self.phase = "lift"
                self.hold = 0
            step_dx, step_dy, step_dz = clamp_step(dx * 0.15, PICK_Y - curr_y, dz * 0.15)
            target_pos = [curr_x + step_dx, curr_y + step_dy, curr_z + step_dz]
            grip_state = 1.0

        elif self.phase == "lift":
            self.latest_plan = f"Lift {target_name.capitalize()} clear of table"
            self.hold += 1
            if self.hold > LIFT_TICKS:
                self.phase = "carry"
                self.hold = 0
            target_pos = [curr_x, curr_y + 0.024, curr_z]
            grip_state = 1.0

        elif self.phase == "carry":
            self.latest_plan = f"Carry {target_name.capitalize()} to Target Pad"
            self.hold += 1
            hover_slot_y = pad_world[1] + CARRY_CLEARANCE
            to_pad_x = pad_world[0] - curr_x
            to_pad_y = hover_slot_y - curr_y
            to_pad_z = pad_world[2] - curr_z
            dist_to_pad = math.hypot(to_pad_x, to_pad_z)

            if (dist_to_pad <= PLACE_XY_TOL and abs(to_pad_y) < 0.04) or self.hold > CARRY_MAX_TICKS:
                self.phase = "drop"
                self.hold = 0
            step_dx, step_dy, step_dz = clamp_step(to_pad_x, to_pad_y, to_pad_z)
            target_pos = [curr_x + step_dx, curr_y + step_dy, curr_z + step_dz]
            grip_state = 1.0

        elif self.phase == "drop":
            self.latest_plan = f"Lower {target_name.capitalize()} onto Stack Slot #{self.target_idx + 1}"
            self.hold += 1
            place_y = pad_world[1] + 0.012
            to_pad_x = pad_world[0] - curr_x
            to_pad_y = place_y - curr_y
            to_pad_z = pad_world[2] - curr_z
            dist_to_pad = math.hypot(to_pad_x, to_pad_z)

            if (dist_to_pad <= PLACE_XY_TOL + 0.01 and curr_y <= place_y + 0.025 and self.hold > 4) or self.hold > DROP_TICKS:
                self.phase = "open"
                self.hold = 0
            step_dx, step_dy, step_dz = clamp_step(to_pad_x, to_pad_y, to_pad_z)
            target_pos = [curr_x + step_dx, curr_y + step_dy, curr_z + step_dz]
            grip_state = 1.0

        elif self.phase == "open":
            self.latest_plan = f"Release {target_name.capitalize()} and Retract"
            self.hold += 1
            if self.hold > OPEN_TICKS:
                self.target_idx += 1
                self.phase = "hover"
                self.last_goal = None
                self.hold = 0
                if self.target_idx >= len(self.order):
                    self.is_done = True
                    self.latest_plan = "ColorSeek: Multi-layer stacking sequence completed!"
            target_pos = [curr_x, curr_y + 0.015, curr_z]
            grip_state = 0.0

        target_joints = inverse_kinematics(target_pos, gripper=grip_state)
        return {
            "joints": target_joints,
            "gripper": grip_state,
            "plan": self.latest_plan,
            "is_active": not self.is_done,
        }
