"""
Work-Cell Camera Simulation, Sensor Abstraction & Rasterizer.
Produces 128x128 top-down orthographic RGB and synthetic depth frames matching new_project_1's rasterScene,
as well as base64 images for the Google Gemini VLA multimodal pipeline.
Implements CameraInterface and Sensor contracts.
"""

import io
import time
import base64
from typing import Dict, Any, Tuple, Optional, List
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
    CUBE_HALF,
    STACK_ORIGIN,
)
from .sensor_interfaces import (
    CameraCalibration,
    CameraObservation,
    CameraInterface,
    RobotStateObservation,
    RobotStateInterface,
)
from .coordinate_frames import OVERHEAD_CAMERA_HEIGHT, CoordinateFrameManager
from .kinematics import forward_kinematics

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

def render_camera_frame(
    snapshot: Dict[str, Any],
    size: int = VLA_IMAGE_SIZE,
    annotated: bool = False,
    target_block_id: Optional[str] = None
) -> Image.Image:
    """
    Renders top-down orthographic RGB view matching new_project_1 evaluation raster:
    - Dark gray studio background
    - Charcoal table surface
    - Blue/green target pad
    - Colored cubes with height displacement
    - White crosshair for robot TCP
    - Optional AI Perception Overlay (bounding brackets, tracking vector line, pixel HUD)
    """
    img = Image.new("RGB", (size, size), color=(14, 16, 22))
    draw = ImageDraw.Draw(img)

    # 1. Draw Table Surface
    t_min_u, t_min_v = world_to_pixel(-TABLE_HALF_EXTENTS[0], TABLE_HALF_EXTENTS[2], size)
    t_max_u, t_max_v = world_to_pixel(TABLE_HALF_EXTENTS[0], -TABLE_HALF_EXTENTS[2], size)
    draw.rectangle([t_min_u, t_min_v, t_max_u, t_max_v], fill=(28, 32, 40), outline=(45, 52, 65))

    # 2. Draw Blocks (ordered by altitude so lower blocks are drawn first)
    sorted_blocks = sorted(snapshot.get("blocks", []), key=lambda b: b["position"][1])
    cube_half_px = max(2, int(round((CUBE_SIZE / (VLA_X_MAX - VLA_X_MIN)) * size * 0.5)))

    target_center_px: Optional[Tuple[int, int]] = None

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

        # Check if this block is the active target
        target_str = str(target_block_id).lower() if target_block_id else ""
        b_id_str = str(b.get("id", "")).lower()
        b_name_str = str(b.get("name", "")).lower()
        is_target = bool(target_str and (
            b_id_str == target_str
            or b_name_str == target_str
            or target_str in b_name_str
            or target_str in b_id_str
            or b_id_str in target_str
        ))
        if is_target:
            target_center_px = (bu, bv - lift)

        # AI Perception HUD: Draw computer vision bounding brackets
        if annotated:
            box_pad = cube_half_px + 3
            bx1, by1 = bu - box_pad, bv - lift - box_pad
            bx2, by2 = bu + box_pad, bv - lift + box_pad
            bracket_len = 3

            bracket_col = (16, 185, 129) if is_target else (100, 116, 139)
            draw.line([(bx1, by1), (bx1 + bracket_len, by1)], fill=bracket_col, width=1)
            draw.line([(bx1, by1), (bx1, by1 + bracket_len)], fill=bracket_col, width=1)

            draw.line([(bx2, by1), (bx2 - bracket_len, by1)], fill=bracket_col, width=1)
            draw.line([(bx2, by1), (bx2, by1 + bracket_len)], fill=bracket_col, width=1)

            draw.line([(bx1, by2), (bx1 + bracket_len, by2)], fill=bracket_col, width=1)
            draw.line([(bx1, by2), (bx1, by2 - bracket_len)], fill=bracket_col, width=1)

            draw.line([(bx2, by2), (bx2 - bracket_len, by2)], fill=bracket_col, width=1)
            draw.line([(bx2, by2), (bx2, by2 - bracket_len)], fill=bracket_col, width=1)

    # 3. Draw Robot TCP indicator (white crosshair)
    tcp = snapshot.get("tcp", {}).get("position", [0.0, 0.8, 0.0])
    tu, tv = world_to_pixel(tcp[0], tcp[2], size)
    draw.line([tu - 3, tv, tu + 3, tv], fill=(245, 245, 245), width=1)
    draw.line([tu, tv - 3, tu, tv + 3], fill=(245, 245, 245), width=1)

    # 4. AI Perception HUD tracking vector line from TCP to Target Cube
    if annotated and target_center_px:
        tx_u, tx_v = target_center_px
        draw.line([(tu, tv), (tx_u, tx_v)], fill=(56, 189, 248), width=1)
        draw.ellipse([tx_u - 1, tx_v - 1, tx_u + 1, tx_v + 1], fill=(56, 189, 248))

    return img


def render_depth_frame(
    snapshot: Dict[str, Any],
    size: int = VLA_IMAGE_SIZE,
    camera_height: float = OVERHEAD_CAMERA_HEIGHT
) -> np.ndarray:
    """
    Renders top-down orthographic metric depth map (H, W) in float32 meters from camera plane.
    Depth values:
    - Background: 2.0m (clipped/out-of-bounds)
    - Tabletop: camera_height - TABLE_TOP_Y (~0.85m)
    - Cubes on table: camera_height - (block_y + CUBE_HALF) (~0.823m)
    - Stacked cubes: depth decreases proportionally to altitude.
    """
    # Background plane at 2.0 meters
    depth = np.full((size, size), 2.0, dtype=np.float32)

    # 1. Fill Table surface depth
    t_min_u, t_min_v = world_to_pixel(-TABLE_HALF_EXTENTS[0], TABLE_HALF_EXTENTS[2], size)
    t_max_u, t_max_v = world_to_pixel(TABLE_HALF_EXTENTS[0], -TABLE_HALF_EXTENTS[2], size)
    table_depth = float(camera_height - TABLE_TOP_Y)
    depth[t_min_v:t_max_v + 1, t_min_u:t_max_u + 1] = table_depth

    # 2. Render Blocks (lowest to highest)
    sorted_blocks = sorted(snapshot.get("blocks", []), key=lambda b: b["position"][1])
    cube_half_px = max(2, int(round((CUBE_SIZE / (VLA_X_MAX - VLA_X_MIN)) * size * 0.5)))

    for b in sorted_blocks:
        pos = b["position"]
        bu, bv = world_to_pixel(pos[0], pos[2], size)
        lift = int(clamp(round((pos[1] - TABLE_TOP_Y) * 35), 0, 6))

        top_y = pos[1] + CUBE_HALF
        block_depth = float(max(0.1, camera_height - top_y))

        v1 = max(0, bv - cube_half_px - lift)
        v2 = min(size - 1, bv + cube_half_px - lift)
        u1 = max(0, bu - cube_half_px)
        u2 = min(size - 1, bu + cube_half_px)

        depth[v1:v2 + 1, u1:u2 + 1] = block_depth

    return depth


def render_depth_visual(
    snapshot: Dict[str, Any],
    size: int = VLA_IMAGE_SIZE,
    camera_height: float = OVERHEAD_CAMERA_HEIGHT
) -> Image.Image:
    """
    Renders metric depth buffer into an intuitive false-color image for frontend display:
    - Bright amber/cyan for elevated blocks
    - Deep blue for tabletop surface
    - Dark studio charcoal for background
    """
    depth = render_depth_frame(snapshot, size, camera_height)
    table_depth = camera_height - TABLE_TOP_Y
    vis = np.zeros((size, size, 3), dtype=np.uint8)

    # 1. Background pixels
    vis[depth >= 1.8] = [12, 14, 18]

    # 2. Table pixels
    table_mask = (depth < 1.8) & (depth >= (table_depth - 0.008))
    vis[table_mask] = [28, 48, 85]

    # 3. Block pixels (elevated above table)
    block_mask = depth < (table_depth - 0.008)
    if np.any(block_mask):
        elev = np.clip((table_depth - depth[block_mask]) / 0.22, 0.0, 1.0)
        r = np.clip(elev * 240 + 20, 0, 255).astype(np.uint8)
        g = np.clip(160 + elev * 85, 0, 255).astype(np.uint8)
        b = np.clip((1.0 - elev) * 220, 0, 255).astype(np.uint8)
        vis[block_mask, 0] = r
        vis[block_mask, 1] = g
        vis[block_mask, 2] = b

    return Image.fromarray(vis)


def render_wrist_camera_frame(
    snapshot: Dict[str, Any],
    size: int = VLA_IMAGE_SIZE,
    target_block_id: Optional[str] = None
) -> Image.Image:
    """
    Renders eye-in-hand / wrist camera feed mounted on the robot TCP:
    - End-effector centered perspective view looking downwards
    - Dynamic scale/zoom based on arm altitude above the tabletop
    - Visible gripper jaw outlines responding to opening/closing
    - Laser alignment reticle and distance-to-table readout
    """
    img = Image.new("RGB", (size, size), color=(18, 22, 28))
    draw = ImageDraw.Draw(img)

    tcp = snapshot.get("tcp", {})
    tcp_pos = tcp.get("position", [0.08, 0.85, 0.0])
    tx, ty, tz = float(tcp_pos[0]), float(tcp_pos[1]), float(tcp_pos[2])
    gripper_val = float(tcp.get("gripper", 0.04))

    # Altitude above table and dynamic FOV span in meters
    alt = max(0.015, ty - TABLE_TOP_Y)
    fov_span = max(0.10, min(0.42, 0.08 + alt * 2.0))
    scale = size / fov_span

    # Draw tabletop surface background
    draw.rectangle([0, 0, size - 1, size - 1], fill=(24, 28, 36))

    # Draw table fine reference grid (every 4cm)
    grid_spacing = 0.04
    grid_color = (36, 44, 56)
    for gx in np.arange(-0.4, 0.45, grid_spacing):
        px = int(size * 0.5 + (gx - tx) * scale)
        if 0 <= px < size:
            draw.line([(px, 0), (px, size)], fill=grid_color, width=1)
    for gz in np.arange(-0.4, 0.45, grid_spacing):
        py = int(size * 0.5 + (gz - tz) * scale)
        if 0 <= py < size:
            draw.line([(0, py), (size, py)], fill=grid_color, width=1)

    # Draw blocks within wrist camera FOV
    sorted_blocks = sorted(snapshot.get("blocks", []), key=lambda b: b["position"][1])
    cube_half_px = max(3, int(round((CUBE_SIZE * 0.5) * scale)))

    for b in sorted_blocks:
        bx, by, bz = b["position"]
        dx = bx - tx
        dz = bz - tz
        # If within horizontal sensor aperture
        if abs(dx) < (fov_span * 0.6) and abs(dz) < (fov_span * 0.6):
            cx = int(size * 0.5 + dx * scale)
            cy = int(size * 0.5 + dz * scale)

            hex_c = b.get("color", "#FFFFFF").lstrip("#")
            color_rgb = tuple(int(hex_c[i:i+2], 16) for i in (0, 2, 4)) if len(hex_c) == 6 else (200, 200, 200)

            # Draw block shadow and top face
            draw.rectangle(
                [cx - cube_half_px, cy - cube_half_px, cx + cube_half_px, cy + cube_half_px],
                fill=color_rgb,
                outline=(255, 255, 255, 140)
            )

            # Target indicator if matching
            if target_block_id and b.get("id") == target_block_id:
                draw.rectangle(
                    [cx - cube_half_px - 2, cy - cube_half_px - 2, cx + cube_half_px + 2, cy + cube_half_px + 2],
                    outline=(56, 189, 248),
                    width=2
                )

    # Draw gripper jaws on left and right flanks
    jaw_offset = int(clamp((gripper_val / 0.08) * (size * 0.32) + (size * 0.12), size * 0.14, size * 0.46))
    left_jaw_x = int(size * 0.5 - jaw_offset)
    right_jaw_x = int(size * 0.5 + jaw_offset)

    # Left gripper jaw silhouette
    draw.rectangle([0, int(size * 0.25), left_jaw_x, int(size * 0.75)], fill=(12, 14, 18), outline=(60, 68, 80))
    draw.rectangle([max(0, left_jaw_x - 3), int(size * 0.35), left_jaw_x, int(size * 0.65)], fill=(56, 189, 248))

    # Right gripper jaw silhouette
    draw.rectangle([right_jaw_x, int(size * 0.25), size - 1, int(size * 0.75)], fill=(12, 14, 18), outline=(60, 68, 80))
    draw.rectangle([right_jaw_x, int(size * 0.35), min(size - 1, right_jaw_x + 3), int(size * 0.65)], fill=(56, 189, 248))

    # Central Laser Sight / Alignment Crosshair
    mid = size // 2
    laser_color = (16, 185, 129) if alt < 0.06 else (56, 189, 248)
    draw.line([(mid - 10, mid), (mid - 3, mid)], fill=laser_color, width=1)
    draw.line([(mid + 3, mid), (mid + 10, mid)], fill=laser_color, width=1)
    draw.line([(mid, mid - 10), (mid, mid - 3)], fill=laser_color, width=1)
    draw.line([(mid, mid + 3), (mid, mid + 10)], fill=laser_color, width=1)
    draw.ellipse([mid - 14, mid - 14, mid + 14, mid + 14], outline=laser_color, width=1)

    # HUD readout (Altitude and Camera tag)
    draw.text((4, 3), "WRIST (EYE-IN-HAND)", fill=(160, 175, 200))
    draw.text((4, size - 13), f"ALT: {alt*100:.1f}cm", fill=(200, 215, 230))

    return img


def render_occupancy_map_visual(
    explorer_data: Dict[str, Any],
    tracks: List[Any],
    tcp_pos: Tuple[float, float, float] = (0.08, 0.85, 0.0),
    size: int = VLA_IMAGE_SIZE
) -> Image.Image:
    """
    Renders 2D tabletop occupancy & frontier navigation grid:
    - Explored free-space cells (cool teal/cyan)
    - Unexplored unknown cells (dark charcoal/slate)
    - Occupied object cells (amber/orange)
    - Active frontier boundary & robot TCP tracking
    """
    img = Image.new("RGB", (size, size), color=(10, 13, 18))
    draw = ImageDraw.Draw(img)

    # Grid specifications (10x10)
    grid_n = 10
    cell_w = (size - 16) / grid_n
    margin = 8

    # Tabletop boundary box
    draw.rectangle([margin, margin, size - margin, size - margin], outline=(40, 50, 68), width=1)

    pct = explorer_data.get("exploration_pct", 25.0)
    # Generate cells based on exploration coverage
    for r in range(grid_n):
        for c in range(grid_n):
            x1 = int(margin + c * cell_w)
            y1 = int(margin + r * cell_w)
            x2 = int(x1 + cell_w - 1)
            y2 = int(y1 + cell_w - 1)

            # Determine cell exploration status
            diag = (r + c) / (grid_n * 2.0)
            is_explored = (diag * 100.0) <= (pct + 15.0)

            if is_explored:
                fill_c = (18, 38, 48)
                out_c = (30, 64, 82)
            else:
                fill_c = (14, 18, 24)
                out_c = (24, 30, 40)

            draw.rectangle([x1, y1, x2, y2], fill=fill_c, outline=out_c)

    # Draw occupied cells from perceived tracks
    for t in tracks:
        pos = getattr(t, "position", None) or (t.get("position") if isinstance(t, dict) else None)
        if pos:
            u = int(margin + clamp((pos[0] - VLA_X_MIN) / (VLA_X_MAX - VLA_X_MIN), 0.05, 0.95) * (size - 2 * margin))
            v = int(margin + clamp((VLA_Z_MAX - pos[2]) / (VLA_Z_MAX - VLA_Z_MIN), 0.05, 0.95) * (size - 2 * margin))

            c_rgb = getattr(t, "color_rgb", None) or (245, 158, 11)
            draw.rectangle([u - 4, v - 4, u + 4, v + 4], fill=tuple(c_rgb[:3]), outline=(255, 255, 255))

    # Draw Robot TCP projection
    tu = int(margin + clamp((tcp_pos[0] - VLA_X_MIN) / (VLA_X_MAX - VLA_X_MIN), 0.05, 0.95) * (size - 2 * margin))
    tv = int(margin + clamp((VLA_Z_MAX - tcp_pos[2]) / (VLA_Z_MAX - VLA_Z_MIN), 0.05, 0.95) * (size - 2 * margin))
    draw.ellipse([tu - 5, tv - 5, tu + 5, tv + 5], fill=(56, 189, 248), outline=(255, 255, 255))

    return img


def encode_image_base64(img: Image.Image) -> str:
    """Encodes PIL image to Base64 PNG string for Gemini multimodal API."""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


class SimulationCamera:
    """
    Adapter implementing CameraInterface for analytical and MuJoCo simulation backends.
    Produces clean CameraObservation instances with no hidden physics metadata.
    """
    def __init__(
        self,
        sim: Any,
        size: int = VLA_IMAGE_SIZE,
        camera_height: float = OVERHEAD_CAMERA_HEIGHT
    ):
        self.sim = sim
        self.size = size
        self.camera_height = camera_height
        self._frame_mgr = CoordinateFrameManager(camera_height=camera_height)

        self.calibration = CameraCalibration(
            model_type="orthographic",
            width=size,
            height=size,
            bounds_x=(VLA_X_MIN, VLA_X_MAX),
            bounds_z=(VLA_Z_MIN, VLA_Z_MAX),
            camera_height=camera_height,
            cam_to_base_matrix=self._frame_mgr.T_cam_to_base,
            near=0.1,
            far=2.0
        )

    def capture(self) -> CameraObservation:
        """Capture a fresh RGB-D camera observation from the simulator."""
        # Query snapshot only for visual rendering
        snapshot = self.sim.get_snapshot()
        rgb_img = render_camera_frame(snapshot, size=self.size)
        depth_map = render_depth_frame(snapshot, size=self.size, camera_height=self.camera_height)

        t_sim = getattr(self.sim, "sim_time", time.time())
        tick = getattr(self.sim, "tick_count", 0)

        return CameraObservation(
            timestamp=float(t_sim),
            tick=int(tick),
            rgb=rgb_img,
            depth=depth_map,
            calibration=self.calibration,
            width=self.size,
            height=self.size,
            is_valid=True
        )

    def get_calibration(self) -> CameraCalibration:
        return self.calibration

    def is_available(self) -> bool:
        return self.sim is not None


class SimulationRobotStateSensor:
    """
    Adapter implementing RobotStateInterface.
    Reads joint angles, velocities, and gripper status.
    Calculates estimated TCP pose via forward kinematics.
    Does NOT leak object ground truth.
    """
    def __init__(self, sim: Any):
        self.sim = sim

    def get_state(self) -> RobotStateObservation:
        joint_names = ["baseYaw", "shoulderPitch", "elbowPitch", "wristPitch", "wristRoll", "gripper"]
        sim_joints = getattr(self.sim, "joints", {})
        if not sim_joints and hasattr(self.sim, "joint_angles"):
            sim_joints = self.sim.joint_angles
        gripper_val = sim_joints.get("gripper", getattr(self.sim, "gripper_target", 0.04))

        joint_angles = {}
        for j in ["baseYaw", "shoulderPitch", "elbowPitch", "wristPitch", "wristRoll"]:
            joint_angles[j] = float(sim_joints.get(j, 0.0))
        joint_angles["gripper"] = float(gripper_val)

        joint_pos_list = [joint_angles[j] for j in joint_names]
        joint_vel_list = [0.0] * len(joint_names)

        grip_pos = float(joint_angles.get("gripper", 0.04))
        grip_closed = bool(getattr(self.sim, "grasped_block_id", False) or getattr(self.sim, "gripper_closed", False) or grip_pos > 0.6)
        grip_effort = 1.0 if grip_closed else 0.0

        # Compute estimated TCP purely from FK
        fk_data = forward_kinematics(joint_angles)
        tcp_pos = fk_data["tcp"]["position"]
        tcp_est = (float(tcp_pos[0]), float(tcp_pos[1]), float(tcp_pos[2]), 0.0, float(joint_angles.get("wristPitch", -1.0)), float(joint_angles.get("baseYaw", 0.0)))

        t_sim = getattr(self.sim, "sim_time", time.time())
        is_e_stopped = bool(getattr(self.sim, "e_stopped", False))

        return RobotStateObservation(
            timestamp=float(t_sim),
            joint_positions=joint_pos_list,
            joint_velocities=joint_vel_list,
            gripper_position=grip_pos,
            gripper_effort=grip_effort,
            tcp_pose_estimated=tcp_est,
            is_e_stopped=is_e_stopped
        )
