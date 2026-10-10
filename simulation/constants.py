"""
Physical and geometric constants for the Robot Arm VLA Workcell.
Directly aligns with the VSArena Rapier simulation specifications from new_project_1.
"""

from typing import Dict, Tuple, List

# Timing & Physics
FIXED_DT: float = 1.0 / 60.0
PHYS_HZ: int = 60
VLA_POLICY_HZ: int = 5
MAX_SUBSTEPS: int = 5
GRAVITY_Y: float = -9.81

# Table Workspace Geometry
TABLE_TOP_Y: float = 0.72
TABLE_HALF_EXTENTS: Tuple[float, float, float] = (0.70, 0.04, 0.45)
TABLE_CENTER_Y: float = TABLE_TOP_Y - TABLE_HALF_EXTENTS[1]
TABLE_WALL_HALF_THICKNESS: float = 0.018
TABLE_WALL_HALF_HEIGHT: float = 0.08
FLOOR_HALF_EXTENTS: Tuple[float, float, float] = (8.0, 0.05, 8.0)

# 5-DOF Articulated Robot Arm Kinematic Dimensions
ARM_MOUNT: Tuple[float, float, float] = (-0.28, TABLE_TOP_Y, 0.0)
PEDESTAL_H: float = 0.11
PEDESTAL_R: float = 0.07
L_UPPER: float = 0.38
L_FOREARM: float = 0.32
L_WRIST: float = 0.13
LINK_SIZE: float = 0.05
JAW_LENGTH: float = 0.095
JAW_HEIGHT: float = 0.048
JAW_THICKNESS: float = 0.014
JAW_MIN_SEP: float = 0.069   # Pad-to-pad distance when fully closed on cube
JAW_MAX_SEP: float = 0.125   # Pad-to-pad distance when fully open
GRASP_DEPTH: float = 0.05    # Mid-finger TCP pinch point forward of palm
WRIST_WORLD_PITCH: float = -0.12 # Slight downward tilt for tabletop manipulation

# Manipulated Objects (Cubes)
CUBE_SIZE: float = 0.055
CUBE_HALF: float = CUBE_SIZE / 2.0

# Grasping Thresholds
GRASP_RADIUS: float = 0.04
GRASP_BOX = {"x": 0.028, "y": 0.02, "z": 0.03}
GRASP_CLOSE_THRESHOLD: float = 0.82
GRASP_OPEN_THRESHOLD: float = 0.38

# Motion & Velocities
JOINT_SPEED: float = 3.4
GRIPPER_SPEED: float = 3.6
MAX_TCP_STEP: float = 0.036  # Max EE displacement per tick (m)
SAFE_HOVER_Y: float = 0.920  # Safe search, acquisition, and transit altitude (generous clearance above table & stack)
PICK_Y: float = 0.7575       # TCP pick altitude clears table surface (0.7555m)
MAX_ALTITUDE_CLAMP: float = 1.12 # Enforces maximum vertical reach limit (permits 5-layer tower reach up to 1.05m)

# Joint Limits (Radians for arm joints, 0..1 for gripper)
JOINT_LIMITS: Dict[str, Tuple[float, float]] = {
    "baseYaw": (-3.14159, 3.14159),
    "shoulderPitch": (-0.35, 1.75),
    "elbowPitch": (-2.40, 0.15),
    "wristPitch": (-1.60, 1.60),
    "gripper": (0.0, 1.0),
}

# Default Home Pose
DEFAULT_JOINTS: Dict[str, float] = {
    "baseYaw": 0.0,
    "shoulderPitch": 0.97,
    "elbowPitch": -1.92,
    "wristPitch": 0.83,
    "gripper": 0.0,
}

# Initial Spawns for 5 Tabletop Cubes (Ultra-vivid, high-contrast, distinct primary & secondary colors)
BLOCK_SPAWNS = [
    {
        "id": "block_magenta",
        "name": "Red Cube",
        "color": "#FF1E27",
        "color_rgb": (255, 30, 39),
        "position": [0.26, TABLE_TOP_Y + CUBE_HALF + 0.006, 0.12],
    },
    {
        "id": "block_cyan",
        "name": "Blue Cube",
        "color": "#1E60FF",
        "color_rgb": (30, 96, 255),
        "position": [0.24, TABLE_TOP_Y + CUBE_HALF + 0.006, -0.18],
    },
    {
        "id": "block_yellow",
        "name": "Yellow Cube",
        "color": "#FFC300",
        "color_rgb": (255, 195, 0),
        "position": [0.15, TABLE_TOP_Y + CUBE_HALF + 0.006, -0.06],
    },
    {
        "id": "block_emerald",
        "name": "Green Cube",
        "color": "#00B341",
        "color_rgb": (0, 179, 65),
        "position": [0.18, TABLE_TOP_Y + CUBE_HALF + 0.006, 0.24],
    },
    {
        "id": "block_orange",
        "name": "Orange Cube",
        "color": "#FF5500",
        "color_rgb": (255, 85, 0),
        "position": [0.34, TABLE_TOP_Y + CUBE_HALF + 0.006, -0.06],
    },
]

# Target Landing Zones
STACK_ORIGIN: Tuple[float, float, float] = (0.48, TABLE_TOP_Y, 0.22)
# Distinct Dual Tower Landing Zones (separated along Z to eliminate workspace collision)
STACK_ORIGIN_A: Tuple[float, float, float] = (0.46, TABLE_TOP_Y, 0.10)
STACK_ORIGIN_B: Tuple[float, float, float] = (0.46, TABLE_TOP_Y, 0.32)

TARGET_ZONE_RADIUS: float = 0.048
STACK_TOLERANCE: float = 0.038  # Tolerance in meters to verify stacked placement

def stack_slot_y(layer: int) -> float:
    """Center Y altitude for a cube placed at vertical stack layer (0=base, 1=mid, 2=top, etc.)."""
    return TABLE_TOP_Y + CUBE_HALF + layer * CUBE_SIZE

def get_pyramid_slots(style: str = "3_block_stepped", origin: Tuple[float, float, float] = STACK_ORIGIN) -> List[Tuple[float, float, float]]:
    """
    Computes metric target coordinates for pyramid assembly.
    - 3_block_stepped: 2 base cubes touching along Z + 1 apex cube centered on top.
    - 5_block_square: 4 base cubes touching in a 2x2 square + 1 apex cube centered on top.
    """
    ox, oy, oz = origin
    if style == "5_block_square":
        half = CUBE_HALF
        base_y = stack_slot_y(0)
        apex_y = stack_slot_y(1)
        return [
            (ox - half, base_y, oz - half),
            (ox + half, base_y, oz - half),
            (ox - half, base_y, oz + half),
            (ox + half, base_y, oz + half),
            (ox, apex_y, oz),
        ]
    else:
        # Default: 3-block 2D stepped pyramid
        half = CUBE_HALF
        base_y = stack_slot_y(0)
        apex_y = stack_slot_y(1)
        return [
            (ox, base_y, oz - half),
            (ox, base_y, oz + half),
            (ox, apex_y, oz),
        ]

# Target coordinates for multi-layer tower (up to 5 layers)
BLOCK_TARGETS = {
    "block_cyan": (STACK_ORIGIN[0], stack_slot_y(0), STACK_ORIGIN[2]),
    "block_orange": (STACK_ORIGIN[0], stack_slot_y(1), STACK_ORIGIN[2]),
    "block_magenta": (STACK_ORIGIN[0], stack_slot_y(2), STACK_ORIGIN[2]),
    "block_yellow": (STACK_ORIGIN[0], stack_slot_y(3), STACK_ORIGIN[2]),
    "block_emerald": (STACK_ORIGIN[0], stack_slot_y(4), STACK_ORIGIN[2]),
}

# 128x128 Orthographic Work-Cell Camera Coordinates
VLA_IMAGE_SIZE: int = 128
VLA_X_MIN: float = -0.55
VLA_X_MAX: float = 0.75
VLA_Z_MIN: float = -0.50
VLA_Z_MAX: float = 0.50
