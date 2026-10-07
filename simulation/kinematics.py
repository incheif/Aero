"""
Forward and Inverse Kinematics for the 5-DOF Articulated Robot Arm.
Implements the exact chain and analytical geometric IK from new_project_1.
"""

import math
from typing import Dict, Tuple, List, Any
from .constants import (
    ARM_MOUNT,
    PEDESTAL_H,
    L_UPPER,
    L_FOREARM,
    L_WRIST,
    GRASP_DEPTH,
    WRIST_WORLD_PITCH,
    JAW_MIN_SEP,
    JAW_MAX_SEP,
    JAW_LENGTH,
    TABLE_TOP_Y,
    CUBE_HALF,
    JOINT_LIMITS,
    MAX_ALTITUDE_CLAMP,
    TABLE_HALF_EXTENTS,
)

def clamp(val: float, low: float, high: float) -> float:
    return max(low, min(high, val))

def quat_from_axis_angle(x: float, y: float, z: float, rad: float) -> Tuple[float, float, float, float]:
    half = rad * 0.5
    s = math.sin(half)
    return (x * s, y * s, z * s, math.cos(half))

def quat_mul(q1: Tuple[float, float, float, float], q2: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )

def quat_conjugate(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    return (-q[0], -q[1], -q[2], q[3])

def quat_rotate_vec(q: Tuple[float, float, float, float], v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    vx, vy, vz = v
    qx, qy, qz, qw = q
    ix = qw * vx + qy * vz - qz * vy
    iy = qw * vy + qz * vx - qx * vz
    iz = qw * vz + qx * vy - qy * vx
    iw = -qx * vx - qy * vy - qz * vz
    return (
        ix * qw + iw * -qx + iy * -qz - iz * -qy,
        iy * qw + iw * -qy + iz * -qx - ix * -qz,
        iz * qw + iw * -qz + ix * -qy - iy * -qx,
    )

def vec_add(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

def is_over_table(x: float, z: float) -> bool:
    """Checks if the (x, z) coordinates lie within the physical table surface bounds."""
    half_x, _, half_z = TABLE_HALF_EXTENTS
    return (-half_x <= x <= half_x) and (-half_z <= z <= half_z)

def forward_kinematics(joints: Dict[str, float], mount: Tuple[float, float, float] = ARM_MOUNT) -> Dict[str, Any]:
    """
    Computes forward kinematics for all links and end-effector TCP.
    Chain: baseYaw (Y) -> shoulderPitch (Z) -> elbowPitch (Z) -> wristPitch (Z) -> gripper separation.
    """
    pedestal_pos = (mount[0], mount[1] + PEDESTAL_H * 0.5, mount[2])
    pedestal_rot = (0.0, 0.0, 0.0, 1.0)

    origin = (mount[0], mount[1] + PEDESTAL_H, mount[2])
    rot = quat_from_axis_angle(0.0, 1.0, 0.0, joints["baseYaw"])
    shoulder_pos = tuple(origin)
    shoulder_rot = rot

    rot = quat_mul(rot, quat_from_axis_angle(0.0, 0.0, 1.0, joints["shoulderPitch"]))
    upper_arm_center = vec_add(origin, quat_rotate_vec(rot, (L_UPPER * 0.5, 0.0, 0.0)))
    origin = vec_add(origin, quat_rotate_vec(rot, (L_UPPER, 0.0, 0.0)))
    upper_arm_rot = rot

    rot = quat_mul(rot, quat_from_axis_angle(0.0, 0.0, 1.0, joints["elbowPitch"]))
    elbow_pos = tuple(origin)
    elbow_rot = rot
    forearm_center = vec_add(origin, quat_rotate_vec(rot, (L_FOREARM * 0.5, 0.0, 0.0)))
    origin = vec_add(origin, quat_rotate_vec(rot, (L_FOREARM, 0.0, 0.0)))
    forearm_rot = rot

    rot = quat_mul(rot, quat_from_axis_angle(0.0, 0.0, 1.0, joints["wristPitch"]))
    wrist_pos = tuple(origin)
    wrist_rot = rot
    origin = vec_add(origin, quat_rotate_vec(rot, (L_WRIST, 0.0, 0.0)))
    palm_pos = tuple(origin)
    palm_rot = rot

    tcp_pos = vec_add(origin, quat_rotate_vec(rot, (GRASP_DEPTH, 0.0, 0.0)))
    tcp_rot = rot

    gripper_val = clamp(joints.get("gripper", 0.0), 0.0, 1.0)
    sep = JAW_MIN_SEP * 0.5 + ((JAW_MAX_SEP - JAW_MIN_SEP) * 0.5) * (1.0 - gripper_val)

    jaw_left_pos = vec_add(origin, quat_rotate_vec(rot, (JAW_LENGTH * 0.5, 0.0, sep)))
    jaw_right_pos = vec_add(origin, quat_rotate_vec(rot, (JAW_LENGTH * 0.5, 0.0, -sep)))

    return {
        "pedestal": {"position": pedestal_pos, "rotation": pedestal_rot},
        "shoulder": {"position": shoulder_pos, "rotation": shoulder_rot},
        "upperArm": {"position": upper_arm_center, "rotation": upper_arm_rot},
        "elbow": {"position": elbow_pos, "rotation": elbow_rot},
        "forearm": {"position": forearm_center, "rotation": forearm_rot},
        "wrist": {"position": wrist_pos, "rotation": wrist_rot},
        "palm": {"position": palm_pos, "rotation": palm_rot},
        "jawLeft": {"position": jaw_left_pos, "rotation": rot},
        "jawRight": {"position": jaw_right_pos, "rotation": rot},
        "tcp": {"position": tcp_pos, "rotation": tcp_rot},
    }

def inverse_kinematics(target: Tuple[float, float, float], gripper: float = 0.0) -> Dict[str, float]:
    """
    Analytical geometric IK for aiming the mid-finger TCP at world target coordinate (tx, ty, tz).
    Enforces tabletop altitude safety, workspace boundaries, and reach clamping.
    """
    tx, ty, tz = target

    # Clamping transit altitude to prevent 2-link boundary elbow reach clamping
    clamped_y = min(ty, MAX_ALTITUDE_CLAMP)

    # Minimum Y clearance above table or floor
    min_y = (TABLE_TOP_Y + CUBE_HALF + 0.008) if is_over_table(tx, tz) else (CUBE_HALF + 0.008)
    aim_y = max(clamped_y, min_y)

    shoulder_origin = (ARM_MOUNT[0], ARM_MOUNT[1] + PEDESTAL_H, ARM_MOUNT[2])
    dx = tx - shoulder_origin[0]
    dy = aim_y - shoulder_origin[1]
    dz = tz - shoulder_origin[2]

    # Base yaw to aim towards target
    yaw_limit = JOINT_LIMITS["baseYaw"]
    base_yaw = clamp(math.atan2(-dz, dx), yaw_limit[0], yaw_limit[1])

    # Rotate vector into arm's 2D pitch plane
    yaw_q = quat_from_axis_angle(0.0, 1.0, 0.0, base_yaw)
    local = quat_rotate_vec(quat_conjugate(yaw_q), (dx, dy, dz))

    # Offset for wrist reach to TCP
    wrist_reach = L_WRIST + GRASP_DEPTH
    px = local[0] - wrist_reach * math.cos(WRIST_WORLD_PITCH)
    py = local[1] - wrist_reach * math.sin(WRIST_WORLD_PITCH)

    # 2-link planar kinematics for upper arm (L_UPPER) and forearm (L_FOREARM)
    l1 = L_UPPER
    l2 = L_FOREARM
    max_reach = l1 + l2 - 0.02
    min_reach = abs(l1 - l2) + 0.02

    d = math.hypot(px, py)
    if d < 1e-6:
        px, py, d = min_reach, 0.0, min_reach
    elif d > max_reach:
        scale = max_reach / d
        px *= scale
        py *= scale
        d = max_reach
    elif d < min_reach:
        scale = min_reach / d
        px *= scale
        py *= scale
        d = min_reach

    # Law of cosines for elbow interior angle
    cos_elbow = clamp((l1 * l1 + l2 * l2 - d * d) / (2.0 * l1 * l2), -1.0, 1.0)
    elbow_interior = math.acos(cos_elbow)
    elbow_pitch_lim = JOINT_LIMITS["elbowPitch"]
    elbow_pitch = clamp(elbow_interior - math.pi, elbow_pitch_lim[0], elbow_pitch_lim[1])

    # Shoulder pitch
    to_target = math.atan2(py, px)
    cos_shoulder = clamp((l1 * l1 + d * d - l2 * l2) / (2.0 * l1 * d), -1.0, 1.0)
    shoulder_offset = math.acos(cos_shoulder)
    shoulder_pitch_lim = JOINT_LIMITS["shoulderPitch"]
    shoulder_pitch = clamp(to_target + shoulder_offset, shoulder_pitch_lim[0], shoulder_pitch_lim[1])

    # Wrist pitch keeping world orientation consistent
    wrist_pitch_lim = JOINT_LIMITS["wristPitch"]
    wrist_pitch = clamp(WRIST_WORLD_PITCH - shoulder_pitch - elbow_pitch, wrist_pitch_lim[0], wrist_pitch_lim[1])

    return {
        "baseYaw": base_yaw,
        "shoulderPitch": shoulder_pitch,
        "elbowPitch": elbow_pitch,
        "wristPitch": wrist_pitch,
        "gripper": clamp(gripper, 0.0, 1.0),
    }
