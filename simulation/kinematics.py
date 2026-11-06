"""
Forward and Inverse Kinematics for the 5-DOF Articulated Robot Arm.
Implements the exact chain and analytical geometric IK from new_project_1.
"""

import math
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List, Any, Optional
import numpy as np

@dataclass
class IKResult:
    """
    Structured outcome of an Inverse Kinematics solve.
    Contains full feasibility, residuals, joint validity, and manipulability.
    """
    success: bool
    joints: Dict[str, float]
    position_error: float
    orientation_error: float
    fk_residual: float
    joint_limits_valid: bool
    reachability_status: str  # "FEASIBLE", "UNREACHABLE_POSITION", "JOINT_LIMIT_VIOLATION", "RESIDUAL_EXCEEDED"
    manipulability: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class JointDictionary(dict):
    """
    A dictionary of joint values that also provides IKResult metadata attributes
    and dict-key access for seamless backwards compatibility.
    """
    def __init__(self, mapping: Dict[str, float], result: Optional[IKResult] = None):
        super().__init__(mapping)
        self.result: Optional[IKResult] = result
        if result:
            self["success"] = result.success
            self["status"] = result.reachability_status
            self["position_error"] = result.position_error
            self["fk_residual"] = result.fk_residual
            self["manipulability"] = result.manipulability
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

def compute_adaptive_pitch(local_x: float) -> float:
    """
    Computes optimal world pitch angle for the end-effector.
    - Close targets (local_x <= 0.72m from shoulder / world_x <= 0.44m): pitches downward (-1.25 rad / -71.6 deg)
      to execute clean, vertical top-down grasps without folding elbow into singularity.
    - Distant targets (local_x >= 0.84m from shoulder / world_x >= 0.56m): pitches forward (-0.12 rad / -6.9 deg)
      to maximize forward horizontal reach.
    - Smooth linear transition between 0.72m and 0.84m.
    """
    if local_x <= 0.72:
        return -1.25
    elif local_x >= 0.84:
        return WRIST_WORLD_PITCH
    else:
        alpha = (local_x - 0.72) / (0.84 - 0.72)
        return -1.25 + alpha * (WRIST_WORLD_PITCH - (-1.25))

def compute_manipulability(joints: Dict[str, float]) -> float:
    """
    Computes the Yoshikawa manipulability index w = sqrt(det(J * J^T))
    for the 4-DOF planar/spatial arm chain using finite difference Jacobian.
    """
    q_keys = ["baseYaw", "shoulderPitch", "elbowPitch", "wristPitch"]
    eps = 1e-4
    fk_nominal = forward_kinematics(joints)
    p_nom = np.array(fk_nominal["tcp"]["position"])

    J = np.zeros((3, 4))
    for i, k in enumerate(q_keys):
        j_pert = dict(joints)
        j_pert[k] += eps
        fk_pert = forward_kinematics(j_pert)
        p_pert = np.array(fk_pert["tcp"]["position"])
        J[:, i] = (p_pert - p_nom) / eps

    JJt = J @ J.T
    det = np.linalg.det(JJt)
    return float(math.sqrt(max(0.0, det)))

def solve_ik(
    target: Tuple[float, float, float],
    gripper: float = 0.0,
    world_pitch: Optional[float] = None,
    candidate_pitches: Optional[List[float]] = None,
    tolerance: float = 0.008,
) -> IKResult:
    """
    Rigorous Inverse Kinematics solver with explicit feasibility validation.
    Checks reachability, joint limits, forward kinematics residuals, and manipulability.
    Never silently reports an unreachable or clamped configuration as successful.
    """
    tx, ty, tz = target
    shoulder_origin = (ARM_MOUNT[0], ARM_MOUNT[1] + PEDESTAL_H, ARM_MOUNT[2])
    dx = tx - shoulder_origin[0]
    dy = ty - shoulder_origin[1]
    dz = tz - shoulder_origin[2]

    # Base yaw to aim towards target
    yaw_limit = JOINT_LIMITS["baseYaw"]
    raw_yaw = math.atan2(-dz, dx)
    yaw_in_limit = yaw_limit[0] <= raw_yaw <= yaw_limit[1]
    base_yaw = clamp(raw_yaw, yaw_limit[0], yaw_limit[1])

    yaw_q = quat_from_axis_angle(0.0, 1.0, 0.0, base_yaw)
    local = quat_rotate_vec(quat_conjugate(yaw_q), (dx, dy, dz))

    l1 = L_UPPER
    l2 = L_FOREARM
    wrist_reach = L_WRIST + GRASP_DEPTH
    max_reach = l1 + l2 - 0.01
    min_reach = abs(l1 - l2) + 0.02

    # Determine candidate pitches to try
    if world_pitch is not None:
        pitches_to_try = [world_pitch]
    elif candidate_pitches is not None:
        pitches_to_try = candidate_pitches
    else:
        adaptive = compute_adaptive_pitch(local[0])
        if ty <= TABLE_TOP_Y + 0.15:
            pitches_to_try = [-1.45, -1.25, -1.0, adaptive, WRIST_WORLD_PITCH, -0.60]
        else:
            pitches_to_try = [adaptive, -1.25, WRIST_WORLD_PITCH, -0.60]

    best_result: Optional[IKResult] = None
    min_err = float("inf")

    for pitch in pitches_to_try:
        px = local[0] - wrist_reach * math.cos(pitch)
        py = local[1] - wrist_reach * math.sin(pitch)
        d = math.hypot(px, py)

        is_reachable = (min_reach <= d <= max_reach)
        if not is_reachable:
            status = "UNREACHABLE_POSITION"
            # Scale for approximate solution evaluation
            scale = (max_reach / d) if d > max_reach else (min_reach / max(1e-6, d))
            px_eff = px * scale
            py_eff = py * scale
            d_eff = max_reach if d > max_reach else min_reach
        else:
            px_eff, py_eff, d_eff = px, py, d
            status = "FEASIBLE"

        cos_elbow = clamp((l1 * l1 + l2 * l2 - d_eff * d_eff) / (2.0 * l1 * l2), -1.0, 1.0)
        elbow_interior = math.acos(cos_elbow)
        raw_elbow = elbow_interior - math.pi
        elbow_lim = JOINT_LIMITS["elbowPitch"]
        elbow_in_limit = elbow_lim[0] <= raw_elbow <= elbow_lim[1]
        elbow_pitch = clamp(raw_elbow, elbow_lim[0], elbow_lim[1])

        to_target = math.atan2(py_eff, px_eff)
        cos_shoulder = clamp((l1 * l1 + d_eff * d_eff - l2 * l2) / (2.0 * l1 * d_eff), -1.0, 1.0)
        shoulder_offset = math.acos(cos_shoulder)
        raw_shoulder = to_target + shoulder_offset
        shoulder_lim = JOINT_LIMITS["shoulderPitch"]
        shoulder_in_limit = shoulder_lim[0] <= raw_shoulder <= shoulder_lim[1]
        shoulder_pitch = clamp(raw_shoulder, shoulder_lim[0], shoulder_lim[1])

        raw_wrist = pitch - shoulder_pitch - elbow_pitch
        wrist_lim = JOINT_LIMITS["wristPitch"]
        wrist_in_limit = wrist_lim[0] <= raw_wrist <= wrist_lim[1]
        wrist_pitch = clamp(raw_wrist, wrist_lim[0], wrist_lim[1])

        joint_limits_valid = yaw_in_limit and elbow_in_limit and shoulder_in_limit and wrist_in_limit
        if not joint_limits_valid and is_reachable:
            status = "JOINT_LIMIT_VIOLATION"

        joints = {
            "baseYaw": base_yaw,
            "shoulderPitch": shoulder_pitch,
            "elbowPitch": elbow_pitch,
            "wristPitch": wrist_pitch,
            "gripper": clamp(gripper, 0.0, 1.0),
        }

        fk = forward_kinematics(joints)
        tcp_pos = fk["tcp"]["position"]
        pos_err = math.hypot(tx - tcp_pos[0], ty - tcp_pos[1], tz - tcp_pos[2])
        ori_err = abs(pitch - (shoulder_pitch + elbow_pitch + wrist_pitch))
        residual = pos_err

        success = is_reachable and joint_limits_valid and (pos_err <= tolerance)
        if not success and status == "FEASIBLE" and pos_err > tolerance:
            status = "RESIDUAL_EXCEEDED"

        manip = compute_manipulability(joints)

        candidate_res = IKResult(
            success=success,
            joints=joints,
            position_error=pos_err,
            orientation_error=ori_err,
            fk_residual=residual,
            joint_limits_valid=joint_limits_valid,
            reachability_status=status,
            manipulability=manip,
        )

        if success:
            return candidate_res

        if pos_err < min_err:
            min_err = pos_err
            best_result = candidate_res

    return best_result


def inverse_kinematics(
    target: Tuple[float, float, float],
    gripper: float = 0.0,
    world_pitch: Optional[float] = None,
    candidate_pitches: Optional[List[float]] = None,
) -> JointDictionary:
    """
    Inverse Kinematics interface.
    Returns JointDictionary that behaves as Dict[str, float] with full IKResult metadata.
    """
    res = solve_ik(target=target, gripper=gripper, world_pitch=world_pitch, candidate_pitches=candidate_pitches)
    return JointDictionary(res.joints, result=res)
