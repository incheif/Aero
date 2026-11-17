"""
vla/spatial_geometry.py

Open-Ended 3D Spatial Geometry Engine for AERO.
Synthesizes metric 3D tabletop placement coordinates [X, Y, Z] for novel,
unhurd-of complex tasks (letters, geometric shapes, spatial sorting, custom patterns)
using pretrained basic manipulation skills without hardcoded category restrictions.
"""

import math
import re
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from simulation.constants import (
    TABLE_TOP_Y,
    CUBE_SIZE,
    CUBE_HALF,
    STACK_ORIGIN,
    stack_slot_y,
)

# Metric parameters for tabletop spatial layouts
SLOT_SPACING = 0.052  # 52mm center-to-center: 45mm cube + 7mm inter-cube clearance for gripper fingers
DEFAULT_CENTER_X = 0.40  # Ideal 5-DOF SO-100 kinematic reach sweet spot
DEFAULT_CENTER_Z = 0.00
BASE_SURFACE_Y = TABLE_TOP_Y + CUBE_HALF  # ~0.7425m center of cube on tabletop


def is_within_reach(x: float, y: float, z: float) -> bool:
    """Verifies that a 3D coordinate is safely within the robot's reachable kinematic envelope."""
    planar_radius = math.sqrt(x ** 2 + z ** 2)
    return (
        0.26 <= planar_radius <= 0.52 and
        -0.35 <= z <= 0.35 and
        0.20 <= x <= 0.55 and
        TABLE_TOP_Y <= y <= 1.15
    )


def clamp_to_reach(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Clamps coordinates to the nearest valid point in the physical reachable workspace."""
    planar_radius = math.sqrt(x ** 2 + z ** 2)
    if planar_radius < 0.28:
        scale = 0.28 / max(1e-4, planar_radius)
        x *= scale
        z *= scale
    elif planar_radius > 0.49:
        scale = 0.49 / planar_radius
        x *= scale
        z *= scale
    x = max(0.24, min(0.50, x))
    z = max(-0.30, min(0.30, z))
    y = max(BASE_SURFACE_Y, min(1.10, y))
    return (float(x), float(y), float(z))


class SpatialGeometryEngine:
    """
    Synthesizes kinematically verified 3D target coordinates for arbitrary complex tasks:
    - Letter formations ('D', 'L', 'T', 'C', 'U', 'H', 'O', 'V', 'X', 'Z', etc.)
    - Geometric shapes (triangle, cross, square, circle, diamond, star, enclosure)
    - Spatial sorting & grouping (warm vs cool colors, left vs right, quadrants)
    - Multi-level and custom vertical topologies
    """

    @classmethod
    def get_letter_slots(
        cls,
        letter: str,
        num_blocks: int = 6,
        origin: Tuple[float, float, float] = (DEFAULT_CENTER_X, BASE_SURFACE_Y, DEFAULT_CENTER_Z),
        scale: float = SLOT_SPACING,
    ) -> List[Tuple[float, float, float]]:
        """
        Generates 3D metric coordinates forming a capital letter on the tabletop.
        Default orientation: readable from the front overhead perspective (+X towards user, +Z right).
        """
        ox, oy, oz = origin
        char = letter.upper().strip()
        slots: List[Tuple[float, float, float]] = []

        if char == "D":
            # 6 blocks forming letter D:
            # 3 vertical spine blocks along Z-axis (or X-axis)
            # 3 curved loop blocks to the right
            slots = [
                (ox - scale, oy, oz - scale * 0.7),  # Spine Top
                (ox, oy, oz - scale * 0.7),          # Spine Center
                (ox + scale, oy, oz - scale * 0.7),  # Spine Bottom
                (ox - scale, oy, oz + scale * 0.3),  # Loop Top Right
                (ox, oy, oz + scale * 0.9),          # Loop Apex Outer Right
                (ox + scale, oy, oz + scale * 0.3),  # Loop Bottom Right
            ]

        elif char == "L":
            # 5 or 6 blocks forming letter L
            slots = [
                (ox - scale * 1.5, oy, oz - scale),  # Vertical stem top
                (ox - scale * 0.5, oy, oz - scale),  # Vertical stem mid
                (ox + scale * 0.5, oy, oz - scale),  # Vertical stem lower
                (ox + scale * 1.5, oy, oz - scale),  # Corner base
                (ox + scale * 1.5, oy, oz),          # Base foot 1
                (ox + scale * 1.5, oy, oz + scale),  # Base foot 2
            ]

        elif char == "T":
            # 5 or 6 blocks forming letter T
            slots = [
                (ox - scale * 1.2, oy, oz - scale),    # Top bar left
                (ox - scale * 1.2, oy, oz),            # Top bar center
                (ox - scale * 1.2, oy, oz + scale),    # Top bar right
                (ox - scale * 0.2, oy, oz),            # Stem upper
                (ox + scale * 0.8, oy, oz),            # Stem mid
                (ox + scale * 1.8, oy, oz),            # Stem bottom
            ]

        elif char in ("C", "U"):
            # Open curved enclosure
            if char == "C":
                slots = [
                    (ox - scale, oy, oz + scale * 0.6),  # Top tip
                    (ox - scale, oy, oz - scale * 0.5),  # Top left corner
                    (ox, oy, oz - scale * 0.8),          # Mid back
                    (ox + scale, oy, oz - scale * 0.5),  # Bottom left corner
                    (ox + scale, oy, oz + scale * 0.6),  # Bottom tip
                    (ox, oy, oz - scale * 0.2),          # Infill
                ]
            else:  # U
                slots = [
                    (ox - scale * 1.2, oy, oz - scale * 0.8), # Left tip
                    (ox, oy, oz - scale * 0.8),               # Left spine
                    (ox + scale * 1.2, oy, oz - scale * 0.5), # Bottom left
                    (ox + scale * 1.2, oy, oz + scale * 0.5), # Bottom right
                    (ox, oy, oz + scale * 0.8),               # Right spine
                    (ox - scale * 1.2, oy, oz + scale * 0.8), # Right tip
                ]

        elif char == "H":
            # 2 vertical pillars spanned by horizontal crossbar
            slots = [
                (ox - scale, oy, oz - scale * 0.7),  # Left pillar top
                (ox + scale, oy, oz - scale * 0.7),  # Left pillar bottom
                (ox, oy, oz - scale * 0.7),          # Left pillar mid
                (ox, oy, oz),                        # Crossbar center
                (ox - scale, oy, oz + scale * 0.7),  # Right pillar top
                (ox + scale, oy, oz + scale * 0.7),  # Right pillar bottom
            ]

        elif char in ("O", "0"):
            # Enclosed ring / loop
            slots = cls.get_shape_slots("circle", num_blocks=6, origin=origin, scale=scale)

        elif char == "V":
            # Diagonal converging lines
            slots = [
                (ox - scale * 1.2, oy, oz - scale * 1.0),
                (ox - scale * 0.2, oy, oz - scale * 0.5),
                (ox + scale * 1.0, oy, oz),
                (ox - scale * 0.2, oy, oz + scale * 0.5),
                (ox - scale * 1.2, oy, oz + scale * 1.0),
                (ox + scale * 0.4, oy, oz),
            ]

        elif char == "X":
            # Cross intersection
            slots = [
                (ox - scale, oy, oz - scale),
                (ox + scale, oy, oz + scale),
                (ox, oy, oz),
                (ox - scale, oy, oz + scale),
                (ox + scale, oy, oz - scale),
                (ox + scale * 1.6, oy, oz),
            ]

        else:
            # Fallback for any unmapped letter: clean linear line with character offset
            slots = cls.get_shape_slots("line", num_blocks=num_blocks, origin=origin, scale=scale)

        # Truncate or pad to num_blocks and clamp to reachable workspace
        valid_slots = [clamp_to_reach(s[0], s[1], s[2]) for s in slots[:num_blocks]]
        return valid_slots

    @classmethod
    def get_shape_slots(
        cls,
        shape: str,
        num_blocks: int = 6,
        origin: Tuple[float, float, float] = (DEFAULT_CENTER_X, BASE_SURFACE_Y, DEFAULT_CENTER_Z),
        scale: float = SLOT_SPACING,
    ) -> List[Tuple[float, float, float]]:
        """
        Generates 3D metric coordinates forming geometric 2D shapes on the tabletop.
        Supported: 'triangle', 'cross', 'plus', 'square', 'rectangle', 'circle', 'ring',
                   'diamond', 'enclosure', 'line', 'arc'.
        """
        ox, oy, oz = origin
        shape_lower = shape.lower().strip()
        slots: List[Tuple[float, float, float]] = []

        if any(w in shape_lower for w in ("triangle", "triangular")):
            if num_blocks == 3:
                # 3-block flat triangle
                r = scale * 1.1
                for i in range(3):
                    angle = i * (2.0 * math.pi / 3.0) - math.pi / 2.0
                    slots.append((ox + r * math.cos(angle), oy, oz + r * math.sin(angle)))
            else:
                # 6-block flat triangle (3 base, 2 mid, 1 apex on tabletop)
                slots = [
                    (ox + scale, oy, oz - scale),      # Base left
                    (ox + scale, oy, oz),              # Base center
                    (ox + scale, oy, oz + scale),      # Base right
                    (ox, oy, oz - scale * 0.5),        # Mid left
                    (ox, oy, oz + scale * 0.5),        # Mid right
                    (ox - scale, oy, oz),              # Apex
                ]

        elif any(w in shape_lower for w in ("cross", "plus")):
            # 5-block cross / plus sign with 6th block as extra support
            slots = [
                (ox, oy, oz),                        # Center
                (ox - scale, oy, oz),                # Top
                (ox + scale, oy, oz),                # Bottom
                (ox, oy, oz - scale),                # Left
                (ox, oy, oz + scale),                # Right
                (ox + scale * 2.0, oy, oz),          # Extended bottom stem
            ]

        elif any(w in shape_lower for w in ("circle", "ring", "round", "oval")):
            # Radial circular distribution
            radius = scale * 1.15
            for i in range(num_blocks):
                angle = i * (2.0 * math.pi / float(num_blocks))
                slots.append((
                    ox + radius * math.cos(angle),
                    oy,
                    oz + radius * math.sin(angle)
                ))

        elif any(w in shape_lower for w in ("square", "box", "rectangle")):
            # 4 corners + midpoints
            slots = [
                (ox - scale * 0.7, oy, oz - scale * 0.7),  # Top left
                (ox - scale * 0.7, oy, oz + scale * 0.7),  # Top right
                (ox + scale * 0.7, oy, oz + scale * 0.7),  # Bottom right
                (ox + scale * 0.7, oy, oz - scale * 0.7),  # Bottom left
                (ox - scale * 0.7, oy, oz),                # Top mid
                (ox + scale * 0.7, oy, oz),                # Bottom mid
            ]

        elif any(w in shape_lower for w in ("diamond", "rhombus")):
            # Diamond layout
            slots = [
                (ox - scale * 1.3, oy, oz),                # Top vertex
                (ox, oy, oz - scale * 1.3),                # Left vertex
                (ox + scale * 1.3, oy, oz),                # Bottom vertex
                (ox, oy, oz + scale * 1.3),                # Right vertex
                (ox - scale * 0.65, oy, oz - scale * 0.65),# Upper left edge
                (ox + scale * 0.65, oy, oz + scale * 0.65),# Lower right edge
            ]

        elif any(w in shape_lower for w in ("enclosure", "corral", "boundary")):
            # Enclosure with a central clear area
            r = scale * 1.35
            for i in range(num_blocks):
                angle = i * (2.0 * math.pi / float(num_blocks))
                slots.append((
                    ox + r * math.cos(angle),
                    oy,
                    oz + r * math.sin(angle)
                ))

        else:
            # Default straight line
            half = (num_blocks - 1) / 2.0
            for i in range(num_blocks):
                slots.append((ox, oy, oz + (i - half) * scale))

        return [clamp_to_reach(s[0], s[1], s[2]) for s in slots[:num_blocks]]

    @classmethod
    def get_layered_structure_slots(
        cls,
        layer_counts: List[int],
        origin: Tuple[float, float, float] = (DEFAULT_CENTER_X, BASE_SURFACE_Y, DEFAULT_CENTER_Z),
        scale: float = SLOT_SPACING,
    ) -> List[Tuple[float, float, float]]:
        """
        Synthesizes 3D slot coordinates for a multi-layered structure with specified blocks per tier.
        Bottom tier (layer 0) is placed on tabletop, subsequent tiers are stacked directly above.
        """
        ox, oy, oz = origin
        slots: List[Tuple[float, float, float]] = []
        for l_idx, count in enumerate(layer_counts):
            y = oy + l_idx * CUBE_SIZE
            half = (count - 1) / 2.0
            for i in range(count):
                z_offset = (i - half) * scale
                slot = (ox, y, oz + z_offset)
                slots.append(clamp_to_reach(slot[0], slot[1], slot[2]))
        return slots

    @classmethod
    def get_sorting_slots(
        cls,
        sort_mode: str,
        current_blocks: List[Dict[str, Any]],
        origin: Tuple[float, float, float] = (DEFAULT_CENTER_X, BASE_SURFACE_Y, DEFAULT_CENTER_Z),
        scale: float = SLOT_SPACING,
    ) -> Tuple[List[Tuple[float, float, float]], List[str]]:
        """
        Synthesizes spatial partition slots for semantic sorting tasks:
        - "warm_cool": Warm colors (red, orange, yellow) to the right; cool colors (blue, green, purple) to the left.
        - "left_right": Even split into two opposing lateral columns.
        - "front_back": Split into near and far rows.
        - "center_perimeter": 1 central VIP block, 5 perimeter sentinels.
        Returns (target_slots, ordered_block_ids).
        """
        ox, oy, oz = origin
        avail_ids = [b["id"] for b in current_blocks if not b.get("is_grasped")]
        all_blocks = ["block_magenta", "block_orange", "block_yellow", "block_cyan", "block_emerald", "block_purple"]
        pool = [b for b in all_blocks if b in avail_ids] + [b for b in avail_ids if b not in all_blocks]

        slots: List[Tuple[float, float, float]] = []
        ordered_blocks: List[str] = []

        if "warm" in sort_mode or "cool" in sort_mode or "temperature" in sort_mode:
            # Warm colors on the right (Z > 0), Cool colors on the left (Z < 0)
            warm_blocks = [b for b in pool if b in ("block_magenta", "block_orange", "block_yellow")]
            cool_blocks = [b for b in pool if b in ("block_cyan", "block_emerald", "block_purple")]

            # Warm slots: Z = +0.08 to +0.18
            warm_x = ox
            for i, b_id in enumerate(warm_blocks):
                slot = (warm_x + (i - 1) * scale * 0.9, oy, oz + 0.12)
                slots.append(clamp_to_reach(slot[0], slot[1], slot[2]))
                ordered_blocks.append(b_id)

            # Cool slots: Z = -0.08 to -0.18
            cool_x = ox
            for i, b_id in enumerate(cool_blocks):
                slot = (cool_x + (i - 1) * scale * 0.9, oy, oz - 0.12)
                slots.append(clamp_to_reach(slot[0], slot[1], slot[2]))
                ordered_blocks.append(b_id)

        elif "left" in sort_mode or "right" in sort_mode or "sides" in sort_mode or "split" in sort_mode:
            # Split pool into two lateral columns
            half_len = len(pool) // 2
            col_a = pool[:half_len]
            col_b = pool[half_len:]

            for i, b_id in enumerate(col_a):
                slot = (ox + (i - 1) * scale, oy, oz - 0.13)
                slots.append(clamp_to_reach(slot[0], slot[1], slot[2]))
                ordered_blocks.append(b_id)

            for i, b_id in enumerate(col_b):
                slot = (ox + (i - 1) * scale, oy, oz + 0.13)
                slots.append(clamp_to_reach(slot[0], slot[1], slot[2]))
                ordered_blocks.append(b_id)

        else:
            # Front vs Back split
            half_len = len(pool) // 2
            row_front = pool[:half_len]
            row_back = pool[half_len:]

            for i, b_id in enumerate(row_front):
                slot = (ox + 0.08, oy, oz + (i - 1) * scale)
                slots.append(clamp_to_reach(slot[0], slot[1], slot[2]))
                ordered_blocks.append(b_id)

            for i, b_id in enumerate(row_back):
                slot = (ox - 0.08, oy, oz + (i - 1) * scale)
                slots.append(clamp_to_reach(slot[0], slot[1], slot[2]))
                ordered_blocks.append(b_id)

        return slots, ordered_blocks


def parse_multi_layer_spec(instruction: str) -> Optional[List[int]]:
    """
    Parses multi-layer structure commands like:
    - 'make a structure using 4 blocks, 1st layer with 2 blocks and 2nd layer with 2 blocks'
    - 'build a structure with layer 1 having 2 blocks and layer 2 having 2 blocks'
    - 'first layer with 2 blocks and second layer with 2 blocks'
    - 'build 2 layers of 2 blocks each'
    - 'make a 2x2 wall of blocks'
    - 'structure with bottom layer of 2 blocks and top layer of 2 blocks'
    Returns a list of block counts per layer, e.g. [2, 2] (base to top).
    """
    cmd = (instruction or "").lower().strip()
    if not cmd:
        return None

    # Pattern 1: Explicit per-layer declarations
    layer_pat = r'(?:(?:(\d+(?:st|nd|rd|th)?|first|second|third|fourth|bottom|top|base|ground|upper|lower)\s+layer|layer\s*(\d+|one|two|three|four))[a-z\s]*?(\d+|one|two|three|four)\s+(?:blocks?|cubes?))'
    matches = re.findall(layer_pat, cmd)
    if len(matches) >= 2:
        num_map = {
            '1': 1, '1st': 1, 'first': 1, 'bottom': 1, 'base': 1, 'ground': 1, 'lower': 1,
            '2': 2, '2nd': 2, 'second': 2, 'top': 2, 'upper': 2,
            '3': 3, '3rd': 3, 'third': 3,
            '4': 4, '4th': 4, 'fourth': 4,
            'one': 1, 'two': 2, 'three': 3, 'four': 4
        }
        layers = []
        for g1, g2, cnt_term in matches:
            l_term = g1 or g2
            l_num = num_map.get(l_term, len(layers) + 1)
            cnt = int(cnt_term) if cnt_term.isdigit() else num_map.get(cnt_term, 2)
            layers.append((l_num, cnt))
        layers.sort(key=lambda x: x[0])
        return [cnt for _, cnt in layers]

    # Pattern 2: '2 layers of 2 blocks' / '2 layers with 2 blocks each'
    m2 = re.search(r'(\d+|one|two|three)\s+layers?\s+(?:of|with)\s+(\d+|one|two|three)\s+(?:blocks?|cubes?)(?:\s+each)?', cmd)
    if m2:
        num_map = {'one': 1, 'two': 2, 'three': 3, '1': 1, '2': 2, '3': 3}
        nl = num_map.get(m2.group(1), 2)
        nc = num_map.get(m2.group(2), 2)
        return [nc] * nl

    # Pattern 3: '2x2 wall' / '2 by 2 structure' / '2x2 block structure'
    m3 = re.search(r'(\d+)\s*(?:x|by)\s*(\d+)\s*(?:wall|structure|grid|formation|block)', cmd)
    if m3:
        w = int(m3.group(1))
        h = int(m3.group(2))
        return [w] * h

    return None


