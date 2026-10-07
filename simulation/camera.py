"""
Work-Cell Camera Simulation & Rasterizer.
Produces 128x128 top-down orthographic RGB frames matching new_project_1's rasterScene,
as well as base64 images for the Google Gemini VLA multimodal pipeline.
"""

import io
import base64
from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image, ImageDraw

from .constants import (
    VLA_IMAGE_SIZE,
    VLA_X_MIN,
    VLA_X_MAX,
    VLA_Z_MIN,
    VLA_Z_MAX,
    TABLE_HALF_EXTENTS,
    TABLE_TOP_Y,
    CUBE_SIZE,
    STACK_ORIGIN,
)

def clamp(val: float, low: float, high: float) -> float:
    return max(low, min(high, val))

def world_to_pixel(x: float, z: float, size: int = VLA_IMAGE_SIZE) -> Tuple[int, int]:
    """Projects world (x, z) coordinates into camera pixel (u, v) coordinates."""
    u = ((x - VLA_X_MIN) / (VLA_X_MAX - VLA_X_MIN)) * (size - 1)
    v = ((VLA_Z_MAX - z) / (VLA_Z_MAX - VLA_Z_MIN)) * (size - 1)
    return int(clamp(round(u), 0, size - 1)), int(clamp(round(v), 0, size - 1))

def pixel_to_world(u: float, v: float, size: int = VLA_IMAGE_SIZE) -> Tuple[float, float]:
    """Converts camera pixel (u, v) into world metric table coordinates (x, z)."""
    denom = max(1.0, float(size - 1))
    x = VLA_X_MIN + (u / denom) * (VLA_X_MAX - VLA_X_MIN)
    z = VLA_Z_MAX - (v / denom) * (VLA_Z_MAX - VLA_Z_MIN)
    return x, z

def render_camera_frame(snapshot: Dict[str, Any], size: int = VLA_IMAGE_SIZE) -> Image.Image:
    """
    Renders top-down orthographic RGB view matching new_project_1 evaluation raster:
    - Dark gray studio background
    - Charcoal table surface
    - Blue/green target pad
    - Colored cubes (Cyan, Orange, Magenta) with height displacement
    - White crosshair for robot TCP
    """
    img = Image.new("RGB", (size, size), color=(14, 16, 22))
    draw = ImageDraw.Draw(img)

    # 1. Draw Table Surface
    t_min_u, t_min_v = world_to_pixel(-TABLE_HALF_EXTENTS[0], TABLE_HALF_EXTENTS[2], size)
    t_max_u, t_max_v = world_to_pixel(TABLE_HALF_EXTENTS[0], -TABLE_HALF_EXTENTS[2], size)
    draw.rectangle([t_min_u, t_min_v, t_max_u, t_max_v], fill=(28, 32, 40), outline=(45, 52, 65))

    # 2. Draw Target Zone Pad
    pad_u, pad_v = world_to_pixel(STACK_ORIGIN[0], STACK_ORIGIN[2], size)
    pad_r = 5
    draw.ellipse([pad_u - pad_r, pad_v - pad_r, pad_u + pad_r, pad_v + pad_r], fill=(35, 75, 95), outline=(50, 160, 200))

    # 3. Draw Blocks (ordered by altitude so lower blocks are drawn first)
    sorted_blocks = sorted(snapshot.get("blocks", []), key=lambda b: b["position"][1])
    cube_half_px = max(2, int(round((CUBE_SIZE / (VLA_X_MAX - VLA_X_MIN)) * size * 0.5)))

    for b in sorted_blocks:
        pos = b["position"]
        bu, bv = world_to_pixel(pos[0], pos[2], size)
        # Visual elevation cue for stacked or lifted cubes
        lift = int(clamp(round((pos[1] - TABLE_TOP_Y) * 35), 0, 6))

        # Parse hex color
        hex_c = b.get("color", "#FFFFFF").lstrip("#")
        color_rgb = tuple(int(hex_c[i:i+2], 16) for i in (0, 2, 4)) if len(hex_c) == 6 else (200, 200, 200)

        # Draw cube shadow if lifted
        if lift > 1:
            draw.rectangle([bu - cube_half_px + 2, bv - cube_half_px + 2, bu + cube_half_px + 2, bv + cube_half_px + 2], fill=(18, 20, 26))

        # Draw cube top face
        draw.rectangle([bu - cube_half_px, bv - cube_half_px - lift, bu + cube_half_px, bv + cube_half_px - lift], fill=color_rgb, outline=(255, 255, 255, 120))

    # 4. Draw Robot TCP indicator (white crosshair)
    tcp = snapshot.get("tcp", {}).get("position", [0.0, 0.8, 0.0])
    tu, tv = world_to_pixel(tcp[0], tcp[2], size)
    draw.line([tu - 3, tv, tu + 3, tv], fill=(245, 245, 245), width=1)
    draw.line([tu, tv - 3, tu, tv + 3], fill=(245, 245, 245), width=1)

    return img

def encode_image_base64(img: Image.Image) -> str:
    """Encodes PIL image to Base64 PNG string for Gemini multimodal API."""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
