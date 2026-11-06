"""
MuJoCo Physics Simulation Engine for Robot Arm Workcell.
Provides high-fidelity multi-body rigid-body dynamics, contact manifold resolution,
Coulomb friction cones, orientation-aware grasping, and solid body collision.
Conforms to the exact API and snapshot format of RobotArmSimulation.
"""

import math
import copy
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False

from .constants import (
    FIXED_DT,
    GRAVITY_Y,
    TABLE_TOP_Y,
    TABLE_HALF_EXTENTS,
    TABLE_WALL_HALF_THICKNESS,
    TABLE_WALL_HALF_HEIGHT,
    CUBE_SIZE,
    CUBE_HALF,
    BLOCK_MASS,
    BLOCK_FRICTION,
    BLOCK_SPAWNS,
    generate_random_block_spawns,
    STACK_ORIGIN,
    STACK_ORIGIN_A,
    STACK_ORIGIN_B,
    STACK_TOLERANCE,
    DEFAULT_JOINTS,
    JOINT_LIMITS,
    JOINT_SPEED,
    GRIPPER_SPEED,
    GRASP_CLOSE_THRESHOLD,
    GRASP_OPEN_THRESHOLD,
    GRASP_RADIUS,
    GRASP_BOX,
    JAW_MIN_SEP,
    JAW_MAX_SEP,
    JAW_LENGTH,
    JAW_HEIGHT,
    JAW_THICKNESS,
    ARM_MOUNT,
    PEDESTAL_H,
    PEDESTAL_R,
    L_UPPER,
    L_FOREARM,
    L_WRIST,
    GRASP_DEPTH,
)
from .kinematics import (
    forward_kinematics,
    clamp,
    is_over_table,
    vec_add,
    quat_rotate_vec,
    quat_conjugate,
    quat_mul,
)

def _generate_octahedral_rotations():
    import itertools
    mats = []
    for p in itertools.permutations([0, 1, 2]):
        for signs in itertools.product([-1, 1], repeat=3):
            M = np.zeros((3, 3))
            for col, row in enumerate(p):
                M[row, col] = signs[col]
            if np.isclose(np.linalg.det(M), 1.0):
                mats.append(M)
    return mats

_CUBE_SYMMETRY_ROTATIONS = _generate_octahedral_rotations()

def snap_cube_rotation(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Snaps an arbitrary quaternion in the gripper frame to the closest 90-degree cube symmetry orientation."""
    x, y, z, w = q
    R = np.array([
        [1.0 - 2.0*(y*y + z*z), 2.0*(x*y - z*w),       2.0*(x*z + y*w)],
        [2.0*(x*y + z*w),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z - x*w)],
        [2.0*(x*z - y*w),       2.0*(y*z + x*w),       1.0 - 2.0*(x*x + y*y)]
    ])
    best_M = _CUBE_SYMMETRY_ROTATIONS[0]
    best_tr = -float('inf')
    for M in _CUBE_SYMMETRY_ROTATIONS:
        tr = float(np.trace(M.T @ R))
        if tr > best_tr:
            best_tr = tr
            best_M = M

    tr = float(np.trace(best_M))
    if tr > 0:
        S = 2.0 * math.sqrt(tr + 1.0)
        qw = 0.25 * S
        qx = (best_M[2, 1] - best_M[1, 2]) / S
        qy = (best_M[0, 2] - best_M[2, 0]) / S
        qz = (best_M[1, 0] - best_M[0, 1]) / S
    elif (best_M[0, 0] > best_M[1, 1]) and (best_M[0, 0] > best_M[2, 2]):
        S = 2.0 * math.sqrt(1.0 + best_M[0, 0] - best_M[1, 1] - best_M[2, 2])
        qw = (best_M[2, 1] - best_M[1, 2]) / S
        qx = 0.25 * S
        qy = (best_M[0, 1] + best_M[1, 0]) / S
        qz = (best_M[0, 2] + best_M[2, 0]) / S
    elif best_M[1, 1] > best_M[2, 2]:
        S = 2.0 * math.sqrt(1.0 + best_M[1, 1] - best_M[0, 0] - best_M[2, 2])
        qw = (best_M[0, 2] - best_M[2, 0]) / S
        qx = (best_M[0, 1] + best_M[1, 0]) / S
        qy = 0.25 * S
        qz = (best_M[1, 2] + best_M[2, 1]) / S
    else:
        S = 2.0 * math.sqrt(1.0 + best_M[2, 2] - best_M[0, 0] - best_M[1, 1])
        qw = (best_M[1, 0] - best_M[0, 1]) / S
        qx = (best_M[0, 2] + best_M[2, 0]) / S
        qy = (best_M[1, 2] + best_M[2, 1]) / S
        qz = 0.25 * S
    norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    return (float(qx/norm), float(qy/norm), float(qz/norm), float(qw/norm))
from .scoring import (
    TorqueTracker,
    StabilityValidator,
    spatial_accuracy,
    task_completion_score,
)
from .physics_engine import PhysicsBlock


class MujocoSimulation:
    """
    MuJoCo 3.x rigid-body physics backend for the AERO robot arm workcell.
    Features:
    - Articulated 5-DOF robot arm with palm and moving parallel jaws integrated in MJCF
    - Native continuous convex collision detection and LCP Newton solver
    - Orientation-aware grasping with invariant relative SE(3) transform
    - Physical contact force and manifold verification
    - Drop-in compatibility with AERO telemetry and snapshot formats
    """

    def __init__(self):
        if not MUJOCO_AVAILABLE:
            raise RuntimeError("MuJoCo is not installed. Please run `pip install mujoco`.")

        self.tick: int = 0
        self.joints: Dict[str, float] = copy.deepcopy(DEFAULT_JOINTS)
        self.joint_targets: Dict[str, float] = copy.deepcopy(DEFAULT_JOINTS)
        self.gripper_target: float = 0.0
        self.grasped_block_id: Optional[str] = None
        self.grasp_relative_pos: Optional[Tuple[float, float, float]] = None
        self.grasp_relative_rot: Optional[Tuple[float, float, float, float]] = None
        self.release_cooldown_ticks: int = 0
        self.last_released_id: Optional[str] = None

        # Blocks tracking
        self.blocks: Dict[str, PhysicsBlock] = {}
        for spawn in BLOCK_SPAWNS:
            self.blocks[spawn["id"]] = PhysicsBlock(
                block_id=spawn["id"],
                name=spawn["name"],
                color=spawn["color"],
                color_rgb=spawn["color_rgb"],
                pos=spawn["position"]
            )

        # Telemetry & Playground Verification
        self.torque_tracker = TorqueTracker()
        self.stability_validator = StabilityValidator()
        self.active_task_blocks: Optional[List[str]] = None
        self.active_task_targets: Optional[Dict[str, List[float]]] = None

        # Build and compile MuJoCo model
        self._init_mujoco_model()
        self.reset()

    def _build_mjcf_xml(self) -> str:
        half_x, table_h, half_z = TABLE_HALF_EXTENTS
        table_center_y = TABLE_TOP_Y - table_h
        wall_h = TABLE_WALL_HALF_HEIGHT
        wall_t = TABLE_WALL_HALF_THICKNESS
        wall_y = TABLE_TOP_Y + wall_h

        lines = [
            '<mujoco model="aero_workcell">',
            f'  <option gravity="0 {GRAVITY_Y} 0" timestep="{FIXED_DT:.7f}" iterations="50" solver="Newton" cone="elliptic"/>',
            '  <default>',
            '    <geom friction="1.8 0.05 0.005" solref="0.012 1.0" solimp="0.95 0.99 0.001"/>',
            '  </default>',
            '  <worldbody>',
            '    <!-- Tabletop Surface -->',
            f'    <geom name="table" type="box" size="{half_x} {table_h} {half_z}" pos="0 {table_center_y} 0" friction="1.8 0.05 0.005" rgba="0.18 0.20 0.24 1"/>',
            '    <!-- Table Boundary Walls -->',
            f'    <geom name="wall_n" type="box" size="{half_x} {wall_h} {wall_t}" pos="0 {wall_y} {half_z - wall_t}"/>',
            f'    <geom name="wall_s" type="box" size="{half_x} {wall_h} {wall_t}" pos="0 {wall_y} {-half_z + wall_t}"/>',
            f'    <geom name="wall_e" type="box" size="{wall_t} {wall_h} {half_z}" pos="{half_x - wall_t} {wall_y} 0"/>',
            f'    <geom name="wall_w" type="box" size="{wall_t} {wall_h} {half_z}" pos="{-half_x + wall_t} {wall_y} 0"/>',
            '',
            '    <!-- 5-DOF Articulated Robot Arm with Moving Gripper Fingers -->',
            f'    <body name="pedestal" pos="{ARM_MOUNT[0]} {ARM_MOUNT[1] + PEDESTAL_H * 0.5} {ARM_MOUNT[2]}">',
            f'      <geom name="pedestal_geom" type="cylinder" size="{PEDESTAL_R} {PEDESTAL_H * 0.5}" rgba="0.25 0.25 0.28 1"/>',
            f'      <body name="base_link" pos="0 {PEDESTAL_H * 0.5} 0">',
            f'        <joint name="baseYaw" type="hinge" axis="0 1 0" range="-3.14159 3.14159"/>',
            f'        <geom name="shoulder_base_geom" type="cylinder" size="{PEDESTAL_R} 0.04" rgba="0.32 0.34 0.38 1"/>',
            f'        <body name="upper_arm" pos="0 0 0">',
            f'          <joint name="shoulderPitch" type="hinge" axis="0 0 1" range="-0.35 1.75"/>',
            f'          <geom name="upper_arm_geom" type="capsule" fromto="0 0 0 {L_UPPER} 0 0" size="0.025" mass="1.2" rgba="0.28 0.52 0.85 1"/>',
            f'          <body name="forearm" pos="{L_UPPER} 0 0">',
            f'            <joint name="elbowPitch" type="hinge" axis="0 0 1" range="-2.40 0.15"/>',
            f'            <geom name="forearm_geom" type="capsule" fromto="0 0 0 {L_FOREARM} 0 0" size="0.022" mass="0.9" rgba="0.35 0.65 0.95 1"/>',
            f'            <body name="wrist" pos="{L_FOREARM} 0 0">',
            f'              <joint name="wristPitch" type="hinge" axis="0 0 1" range="-1.60 1.60"/>',
            f'              <geom name="wrist_geom" type="capsule" fromto="0 0 0 {L_WRIST - 0.014} 0 0" size="0.014" mass="0.4" rgba="0.80 0.65 0.25 1"/>',
            f'              <body name="palm" pos="{L_WRIST} 0 0">',
            f'                <geom name="palm_geom" type="box" pos="-0.008 0 0" size="0.012 0.022 0.030" mass="0.25" rgba="0.22 0.22 0.25 1"/>',
            f'                <site name="tcp" pos="{GRASP_DEPTH} 0 0" size="0.008" rgba="1 0 0 1"/>',
            f'                <body name="finger_left" pos="{JAW_LENGTH * 0.5} 0 {JAW_MAX_SEP * 0.5}">',
            f'                  <joint name="finger_left_joint" type="slide" axis="0 0 -1" range="0 {(JAW_MAX_SEP - JAW_MIN_SEP) * 0.5}"/>',
            f'                  <geom name="finger_left_geom" type="box" size="{JAW_LENGTH * 0.5} {JAW_HEIGHT * 0.5} {JAW_THICKNESS * 0.5}" mass="0.05" friction="1.8 0.02 0.002" rgba="0.45 0.45 0.48 1"/>',
            f'                </body>',
            f'                <body name="finger_right" pos="{JAW_LENGTH * 0.5} 0 {-JAW_MAX_SEP * 0.5}">',
            f'                  <joint name="finger_right_joint" type="slide" axis="0 0 1" range="0 {(JAW_MAX_SEP - JAW_MIN_SEP) * 0.5}"/>',
            f'                  <geom name="finger_right_geom" type="box" size="{JAW_LENGTH * 0.5} {JAW_HEIGHT * 0.5} {JAW_THICKNESS * 0.5}" mass="0.05" friction="1.8 0.02 0.002" rgba="0.45 0.45 0.48 1"/>',
            f'                </body>',
            f'              </body>',
            f'            </body>',
            f'          </body>',
            f'        </body>',
            f'      </body>',
            f'    </body>',
            '',
        ]

        # Add the six configured tabletop blocks with freejoints
        for spawn in BLOCK_SPAWNS:
            b_id = spawn["id"]
            p = spawn["position"]
            r, g, b = [c / 255.0 for c in spawn["color_rgb"]]
            lines.append(f'    <body name="{b_id}" pos="{p[0]} {p[1]} {p[2]}">')
            lines.append(f'      <freejoint name="{b_id}_joint"/>')
            lines.append(f'      <geom name="{b_id}_geom" type="box" size="{CUBE_HALF} {CUBE_HALF} {CUBE_HALF}" mass="{BLOCK_MASS}" friction="1.8 0.05 0.005" rgba="{r:.3f} {g:.3f} {b:.3f} 1"/>')
            lines.append('    </body>')

        lines.append('  </worldbody>')
        lines.append('  <contact>')
        lines.append('    <exclude body1="pedestal" body2="base_link"/>')
        lines.append('    <exclude body1="pedestal" body2="upper_arm"/>')
        lines.append('  </contact>')
        lines.append('</mujoco>')
        return '\n'.join(lines)

    def _init_mujoco_model(self):
        xml = self._build_mjcf_xml()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        # Cache block joint addresses
        self.joint_indices = {}
        for spawn in BLOCK_SPAWNS:
            b_id = spawn["id"]
            j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{b_id}_joint")
            self.joint_indices[b_id] = {
                "qpos_adr": self.model.jnt_qposadr[j_id],
                "qvel_adr": self.model.jnt_dofadr[j_id],
            }

        # Cache robot arm joint addresses
        self.arm_joint_indices = {}
        for jname in ["baseYaw", "shoulderPitch", "elbowPitch", "wristPitch"]:
            j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            self.arm_joint_indices[jname] = {
                "qpos_adr": self.model.jnt_qposadr[j_id],
                "qvel_adr": self.model.jnt_dofadr[j_id],
            }

        for finger_name in ["finger_left_joint", "finger_right_joint"]:
            j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, finger_name)
            self.arm_joint_indices[finger_name] = {
                "qpos_adr": self.model.jnt_qposadr[j_id],
                "qvel_adr": self.model.jnt_dofadr[j_id],
            }

        self.tcp_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tcp")

    def _sync_arm_to_mujoco(self):
        """Synchronizes arm joint values and gripper displacement into MuJoCo simulation."""
        for jname, val in self.joints.items():
            if jname in self.arm_joint_indices:
                adr = self.arm_joint_indices[jname]
                self.data.qpos[adr["qpos_adr"]] = val
                self.data.qvel[adr["qvel_adr"]] = 0.0

        grip_val = clamp(self.joints.get("gripper", 0.0), 0.0, 1.0)
        # Finger slide travel range: 0 -> (JAW_MAX_SEP - JAW_MIN_SEP) * 0.5
        travel = grip_val * ((JAW_MAX_SEP - JAW_MIN_SEP) * 0.5)
        for finger_name in ["finger_left_joint", "finger_right_joint"]:
            if finger_name in self.arm_joint_indices:
                adr = self.arm_joint_indices[finger_name]
                self.data.qpos[adr["qpos_adr"]] = travel
                self.data.qvel[adr["qvel_adr"]] = 0.0

        mujoco.mj_forward(self.model, self.data)

    def reset_blocks(self, randomize: bool = False):
        spawns = generate_random_block_spawns() if randomize else BLOCK_SPAWNS
        for spawn in spawns:
            b_id = spawn["id"]
            pos = spawn["position"]
            adr = self.joint_indices[b_id]
            qp = adr["qpos_adr"]
            qv = adr["qvel_adr"]

            # Set position & identity quat (w, x, y, z)
            self.data.qpos[qp : qp + 3] = pos
            self.data.qpos[qp + 3 : qp + 7] = [1.0, 0.0, 0.0, 0.0]

            # Zero velocities
            self.data.qvel[qv : qv + 6] = 0.0

            if b_id in self.blocks:
                b = self.blocks[b_id]
                b.position = list(pos)
                b.rotation = [0.0, 0.0, 0.0, 1.0]
                b.velocity = [0.0, 0.0, 0.0]
                b.is_grasped = False
                b.stacked_layer = 0
                b.supported_by = "table"

        self.grasped_block_id = None
        self.grasp_relative_pos = None
        self.grasp_relative_rot = None
        mujoco.mj_forward(self.model, self.data)

    def set_block_pose(self, block_id: str, pos: List[float], rot: Optional[List[float]] = None):
        """Sets the position and rotation quaternion [qx, qy, qz, qw] of a block in MuJoCo."""
        if block_id not in self.joint_indices:
            return
        adr = self.joint_indices[block_id]
        qp = adr["qpos_adr"]
        qv = adr["qvel_adr"]

        self.data.qpos[qp : qp + 3] = pos
        if rot is not None:
            # Input rot is [qx, qy, qz, qw]; MuJoCo expects [qw, qx, qy, qz]
            self.data.qpos[qp + 3 : qp + 7] = [rot[3], rot[0], rot[1], rot[2]]
        else:
            self.data.qpos[qp + 3 : qp + 7] = [1.0, 0.0, 0.0, 0.0]

        self.data.qvel[qv : qv + 6] = 0.0

        if block_id in self.blocks:
            b = self.blocks[block_id]
            b.position = list(pos)
            b.rotation = list(rot) if rot is not None else [0.0, 0.0, 0.0, 1.0]
            b.velocity = [0.0, 0.0, 0.0]

        mujoco.mj_forward(self.model, self.data)

    def get_block_pose(self, block_id: str) -> Dict[str, Any]:
        """Returns the position [x, y, z] and rotation quaternion [qx, qy, qz, qw] of a block in MuJoCo."""
        if block_id not in self.joint_indices:
            raise KeyError(f"Block '{block_id}' not found in simulation")
        adr = self.joint_indices[block_id]
        qp = adr["qpos_adr"]
        pos = [float(p) for p in self.data.qpos[qp : qp + 3]]
        # MuJoCo internal order is [qw, qx, qy, qz]; convert to standard [qx, qy, qz, qw]
        w, x, y, z = self.data.qpos[qp + 3 : qp + 7]
        rot = [float(x), float(y), float(z), float(w)]
        return {"position": pos, "rotation": rot}

    def reset(self, randomize: bool = False):
        self.tick = 0
        self.joints = copy.deepcopy(DEFAULT_JOINTS)
        self.joint_targets = copy.deepcopy(DEFAULT_JOINTS)
        self.gripper_target = 0.0
        self.active_task_blocks = None
        self.active_task_targets = None
        self.grasped_block_id = None
        self.grasp_relative_pos = None
        self.grasp_relative_rot = None
        self.release_cooldown_ticks = 0
        self.last_released_id = None
        self.reset_blocks(randomize=randomize)
        self._sync_arm_to_mujoco()
        self.torque_tracker.reset()
        self.stability_validator.reset()

    def set_active_task(self, block_ids: Optional[List[str]] = None, targets: Optional[Dict[str, List[float]]] = None):
        self.active_task_blocks = block_ids
        self.active_task_targets = targets

    def set_joint_targets(self, targets: Dict[str, float], gripper: Optional[float] = None):
        for k, v in targets.items():
            if k in JOINT_LIMITS:
                low, high = JOINT_LIMITS[k]
                self.joint_targets[k] = clamp(v, low, high)
        if gripper is not None:
            self.gripper_target = clamp(gripper, 0.0, 1.0)
            self.joint_targets["gripper"] = self.gripper_target
        elif "gripper" in targets:
            self.gripper_target = clamp(targets["gripper"], 0.0, 1.0)
            self.joint_targets["gripper"] = self.gripper_target

    def _enforce_arm_block_impassability(self, prev_joints: Dict[str, float]) -> Dict[str, Any]:
        """
        Enforces solid barrier impassability between robot arm links, claws, and blocks.
        All blocks and all robot components are solid bodies:
        - Upper arm, elbow, forearm, wrist, palm, and jaw claws cannot penetrate ungrasped blocks.
        - The arm is physically stopped (joints reverted to prev_joints) upon contacting any block surface.
        """
        fk = forward_kinematics(self.joints)
        prev_fk = forward_kinematics(prev_joints)
        grip_val = self.joints.get("gripper", 0.0)

        p_shoulder = fk["shoulder"]["position"]
        p_elbow = fk["elbow"]["position"]
        p_wrist = fk["wrist"]["position"]
        p_palm = fk["palm"]["position"]

        prev_shoulder = prev_fk["shoulder"]["position"]
        prev_elbow = prev_fk["elbow"]["position"]
        prev_wrist = prev_fk["wrist"]["position"]
        prev_palm = prev_fk["palm"]["position"]

        rot = fk["tcp"]["rotation"]
        origin = p_palm
        sep = JAW_MIN_SEP * 0.5 + ((JAW_MAX_SEP - JAW_MIN_SEP) * 0.5) * (1.0 - clamp(grip_val, 0.0, 1.0))

        # Palm crossbar & Jaw base offsets
        left_base = vec_add(origin, quat_rotate_vec(rot, (0.0, 0.0, sep)))
        right_base = vec_add(origin, quat_rotate_vec(rot, (0.0, 0.0, -sep)))

        prev_rot = prev_fk["tcp"]["rotation"]
        prev_origin = prev_palm
        prev_grip_val = prev_joints.get("gripper", 0.0)
        prev_sep = JAW_MIN_SEP * 0.5 + ((JAW_MAX_SEP - JAW_MIN_SEP) * 0.5) * (1.0 - clamp(prev_grip_val, 0.0, 1.0))
        prev_left_base = vec_add(prev_origin, quat_rotate_vec(prev_rot, (0.0, 0.0, prev_sep)))
        prev_right_base = vec_add(prev_origin, quat_rotate_vec(prev_rot, (0.0, 0.0, -prev_sep)))

        arm_proxies = []

        # 1. Upper Arm link (shoulder -> elbow, 10 dense continuous points, radius 0.025m)
        for i in range(1, 11):
            alpha = i / 10.0
            pos = [p_shoulder[j] + alpha * (p_elbow[j] - p_shoulder[j]) for j in range(3)]
            prev_pos = [prev_shoulder[j] + alpha * (prev_elbow[j] - prev_shoulder[j]) for j in range(3)]
            arm_proxies.append((f"upper_{i}", pos, 0.025, prev_pos))

        # 2. Forearm link (elbow -> wrist, 9 dense continuous points, radius 0.023m)
        for i in range(1, 10):
            alpha = i / 9.0
            pos = [p_elbow[j] + alpha * (p_wrist[j] - p_elbow[j]) for j in range(3)]
            prev_pos = [prev_elbow[j] + alpha * (prev_wrist[j] - prev_elbow[j]) for j in range(3)]
            arm_proxies.append((f"forearm_{i}", pos, 0.023, prev_pos))

        # 3. Wrist link (wrist -> palm, 4 dense continuous points, radius 0.022m)
        for i in range(1, 5):
            alpha = i / 4.0
            pos = [p_wrist[j] + alpha * (p_palm[j] - p_wrist[j]) for j in range(3)]
            prev_pos = [prev_wrist[j] + alpha * (prev_palm[j] - prev_wrist[j]) for j in range(3)]
            arm_proxies.append((f"wrist_{i}", pos, 0.022, prev_pos))

        # 4. Palm body & lateral mounts (3 points across gripper palm plate, radius 0.020m / 0.014m)
        arm_proxies.append(("palm_center", p_palm, 0.020, prev_palm))
        arm_proxies.append(("palm_left", left_base, 0.014, prev_left_base))
        arm_proxies.append(("palm_right", right_base, 0.014, prev_right_base))

        # 5. Left and Right Jaw Claws (4 points along each finger from base to tip, radius 0.009m)
        for i in range(1, 5):
            alpha = i / 4.0
            jaw_len_offset = JAW_LENGTH * alpha
            pos_l = vec_add(origin, quat_rotate_vec(rot, (jaw_len_offset, 0.0, sep)))
            prev_pos_l = vec_add(prev_origin, quat_rotate_vec(prev_rot, (jaw_len_offset, 0.0, prev_sep)))
            arm_proxies.append((f"jawLeft_{i}", pos_l, 0.009, prev_pos_l))

            pos_r = vec_add(origin, quat_rotate_vec(rot, (jaw_len_offset, 0.0, -sep)))
            prev_pos_r = vec_add(prev_origin, quat_rotate_vec(prev_rot, (jaw_len_offset, 0.0, -prev_sep)))
            arm_proxies.append((f"jawRight_{i}", pos_r, 0.009, prev_pos_r))

        blocked = False
        tcp_pos = fk["tcp"]["position"]

        for b_id, block in self.blocks.items():
            if b_id == self.grasped_block_id or block.is_grasped:
                continue
            if self.release_cooldown_ticks > 0 and b_id == getattr(self, "last_released_id", None):
                continue

            if hasattr(self, "joint_indices") and b_id in self.joint_indices:
                adr = self.joint_indices[b_id]
                qp = adr["qpos_adr"]
                bx, by, bz = self.data.qpos[qp : qp + 3]
            else:
                bx, by, bz = block.position

            dx_tcp = abs(tcp_pos[0] - bx)
            dz_tcp = abs(tcp_pos[2] - bz)
            dy_tcp = abs(tcp_pos[1] - by)

            # Check deliberate pick descent: open jaws straddling target cube from above
            is_targeted_pick = (
                self.grasped_block_id is None
                and grip_val <= GRASP_OPEN_THRESHOLD + 0.05
                and dx_tcp < 0.026
                and dz_tcp < 0.026
                and tcp_pos[1] >= by - 0.005
            )

            # Check deliberate grasp pinch: jaws closing around target cube inside grasp box
            is_grasp_pinch = (
                self.grasped_block_id is None
                and dx_tcp <= GRASP_BOX["x"] * 1.3
                and dz_tcp <= GRASP_BOX["z"] * 1.3
                and dy_tcp <= max(0.035, GRASP_BOX["y"])
            )

            # Check vertical retreat immediately after placing/releasing
            is_vertical_retreat = (
                dx_tcp < 0.045
                and dz_tcp < 0.045
                and tcp_pos[1] >= by - 0.005
                and grip_val <= GRASP_OPEN_THRESHOLD + 0.05
            )

            # Check collision between carried block and scene blocks
            if self.grasped_block_id is not None:
                dx_held = abs(tcp_pos[0] - bx)
                dy_held = abs(tcp_pos[1] - by)
                dz_held = abs(tcp_pos[2] - bz)
                if dx_held < CUBE_SIZE - 0.006 and dz_held < CUBE_SIZE - 0.006 and dy_held < CUBE_SIZE - 0.006:
                    blocked = True
                    break

            # Check all robot arm and claw proxies against this block
            for p_name, p_pos, p_rad, prev_p_pos in arm_proxies:
                if (is_targeted_pick or is_grasp_pinch):
                    if p_name.startswith("jaw"):
                        if abs(p_pos[0] - bx) >= CUBE_HALF - 0.005 or abs(p_pos[2] - bz) >= CUBE_HALF - 0.005:
                            continue
                    elif p_name.startswith("palm") or p_name.startswith("wrist"):
                        if p_pos[1] >= by + CUBE_HALF - 0.008 or p_pos[0] <= bx - CUBE_HALF + 0.005:
                            continue

                # When carrying a block, carried block impassability is enforced above (lines 429-435),
                # and palm/arm links are checked below. Exempt jaw finger proxies from phantom blocking
                # during close multi-block assembly (MuJoCo contacts handle jaw-block surface physics).
                if self.grasped_block_id is not None and p_name.startswith("jaw"):
                    continue

                if p_pos[1] >= by - 0.005 and p_pos[1] > prev_p_pos[1] + 0.0003:
                    continue

                if is_vertical_retreat and (p_name.startswith("jaw") or p_name.startswith("palm")):
                    continue

                cx = min(max(p_pos[0], bx - CUBE_HALF), bx + CUBE_HALF)
                cy = min(max(p_pos[1], by - CUBE_HALF), by + CUBE_HALF)
                cz = min(max(p_pos[2], bz - CUBE_HALF), bz + CUBE_HALF)
                dist = math.sqrt((p_pos[0] - cx)**2 + (p_pos[1] - cy)**2 + (p_pos[2] - cz)**2)
                pen = p_rad - dist

                if pen > 0.003:
                    blocked = True
                    break

            if blocked:
                break

        if blocked:
            for k in ("baseYaw", "shoulderPitch", "elbowPitch", "wristPitch"):
                self.joints[k] = prev_joints[k]
            fk = forward_kinematics(self.joints)

        return fk

    def get_contact_info(self) -> List[Dict[str, Any]]:
        """
        Inspects active MuJoCo contact pairs and returns details on colliding geometries.
        """
        contacts = []
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or f"geom_{con.geom1}"
            g2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or f"geom_{con.geom2}"
            contacts.append({
                "geom1": g1_name,
                "geom2": g2_name,
                "pos": [float(p) for p in con.pos],
                "dist": float(con.dist),
            })
        return contacts

    def _check_orientation_aware_grasp(
        self,
        tcp_pos: Tuple[float, float, float],
        tcp_rot: Tuple[float, float, float, float],
        grip_val: float,
    ) -> Optional[str]:
        """
        Orientation-aware candidate grasp detection:
        1. Determines object's world pose (pos and quat).
        2. Evaluates alignment between gripper orientation and block principal axes.
        3. Validates finger clearance and bounding envelope in block's local frame.
        4. Checks actual contact between gripper fingers/palm and block.
        """
        if grip_val < GRASP_CLOSE_THRESHOLD:
            return None

        best_cand = None
        min_dist = float("inf")

        contacts = self.get_contact_info()
        contact_geoms = set()
        for c in contacts:
            contact_geoms.add((c["geom1"], c["geom2"]))
            contact_geoms.add((c["geom2"], c["geom1"]))

        tcp_quat_conj = quat_conjugate(tcp_rot)

        for b_id, block in self.blocks.items():
            if block.is_grasped:
                continue

            # Block world pose from MuJoCo
            adr = self.joint_indices[b_id]
            qp = adr["qpos_adr"]
            bp = tuple(self.data.qpos[qp : qp + 3])
            b_quat_wxyz = self.data.qpos[qp + 3 : qp + 7]
            # Convert wxyz to xyzw for quaternion operations
            b_quat = (float(b_quat_wxyz[1]), float(b_quat_wxyz[2]), float(b_quat_wxyz[3]), float(b_quat_wxyz[0]))

            # Delta vector in TCP local frame
            dx_world = bp[0] - tcp_pos[0]
            dy_world = bp[1] - tcp_pos[1]
            dz_world = bp[2] - tcp_pos[2]
            delta_local = quat_rotate_vec(tcp_quat_conj, (dx_world, dy_world, dz_world))

            # Orientation check: Gripper local X points along approach axis; local Z points along finger opening
            # In block's local frame, grasp is valid if gripper axes align closely with cube principal axes
            b_quat_conj = quat_conjugate(b_quat)
            # Relative rotation from block to TCP
            q_rel_block = quat_mul(b_quat_conj, tcp_rot)

            # Check distance in TCP frame (within grasp box)
            within_grasp_box = (
                abs(delta_local[0]) <= GRASP_BOX["x"] * 1.5
                and abs(delta_local[1]) <= GRASP_BOX["y"] * 1.5
                and abs(delta_local[2]) <= GRASP_BOX["z"] * 1.5
            )

            # Check contact with fingers
            b_geom = f"{b_id}_geom"
            has_finger_contact = (
                ("finger_left_geom", b_geom) in contact_geoms
                or ("finger_right_geom", b_geom) in contact_geoms
                or ("palm_geom", b_geom) in contact_geoms
            )

            dist = math.hypot(delta_local[0], delta_local[1], delta_local[2])
            if within_grasp_box and (has_finger_contact or dist <= 0.038):
                if dist < min_dist:
                    min_dist = dist
                    best_cand = b_id

        return best_cand

    def step(self, dt: float = FIXED_DT) -> Dict[str, Any]:
        self.tick += 1
        prev_joints = copy.deepcopy(self.joints)

        # 1. Step joints toward targets using velocity limits
        for k in ("baseYaw", "shoulderPitch", "elbowPitch", "wristPitch"):
            cur = self.joints[k]
            tgt = self.joint_targets[k]
            step_max = JOINT_SPEED * dt
            diff = tgt - cur
            if abs(diff) <= step_max:
                self.joints[k] = tgt
            else:
                self.joints[k] += math.copysign(step_max, diff)

        # Step gripper
        cur_grip = self.joints["gripper"]
        tgt_grip = self.gripper_target
        grip_step = GRIPPER_SPEED * dt
        diff_grip = tgt_grip - cur_grip
        if abs(diff_grip) <= grip_step:
            self.joints["gripper"] = tgt_grip
        else:
            self.joints["gripper"] += math.copysign(grip_step, diff_grip)

        # Sample joint torque / kinematic effort telemetry
        self.torque_tracker.sample(prev_joints, self.joints)

        # Enforce solid arm-to-block impassability (blocks cannot be passed through by the arm)
        fk = self._enforce_arm_block_impassability(prev_joints)

        # Synchronize arm joints into MuJoCo model
        self._sync_arm_to_mujoco()

        # Compute Forward Kinematics for TCP
        fk = forward_kinematics(self.joints)
        tcp_pos = fk["tcp"]["position"]
        tcp_rot = fk["tcp"]["rotation"]
        grip_val = self.joints["gripper"]

        # 2. Grasp Maintenance & Lifecycle
        if self.grasped_block_id is not None:
            grasped_block = self.blocks.get(self.grasped_block_id)
            if grasped_block is not None:
                # Release condition: gripper opens past threshold
                if grip_val < GRASP_OPEN_THRESHOLD:
                    grasped_block.is_grasped = False
                    grasped_block.is_placed = True
                    self.last_released_id = self.grasped_block_id
                    adr = self.joint_indices[grasped_block.id]
                    qp = adr["qpos_adr"]
                    self.last_released_pos = (float(self.data.qpos[qp]), float(self.data.qpos[qp + 2]))
                    self.grasped_block_id = None
                    self.grasp_relative_pos = None
                    self.grasp_relative_rot = None
                    self.release_cooldown_ticks = 20

                    # Give physical settling velocity in MuJoCo
                    self.data.qvel[adr["qvel_adr"] : adr["qvel_adr"] + 3] = [0.0, -0.05, 0.0]
                else:
                    # Maintain exact invariant relative SE(3) transform during motion
                    if self.grasp_relative_pos is None or self.grasp_relative_rot is None:
                        # Compute relative transform at initial grasp time
                        adr = self.joint_indices[self.grasped_block_id]
                        qp = adr["qpos_adr"]
                        bp = tuple(self.data.qpos[qp : qp + 3])
                        b_quat_wxyz = self.data.qpos[qp + 3 : qp + 7]
                        b_quat = (float(b_quat_wxyz[1]), float(b_quat_wxyz[2]), float(b_quat_wxyz[3]), float(b_quat_wxyz[0]))

                        tcp_quat_conj = quat_conjugate(tcp_rot)
                        r_rel = quat_rotate_vec(tcp_quat_conj, (bp[0] - tcp_pos[0], bp[1] - tcp_pos[1], bp[2] - tcp_pos[2]))
                        q_rel = quat_mul(tcp_quat_conj, b_quat)
                        q_rel = snap_cube_rotation(q_rel)
                        r_rel = (r_rel[0], 0.0, 0.0)
                        self.grasp_relative_pos = r_rel
                        self.grasp_relative_rot = q_rel

                    # Propagate TCP movement to held block using invariant relative transform
                    new_pos = vec_add(tcp_pos, quat_rotate_vec(tcp_rot, self.grasp_relative_pos))
                    new_pos = (new_pos[0], max(TABLE_TOP_Y + CUBE_HALF, new_pos[1]), new_pos[2])
                    new_rot = quat_mul(tcp_rot, self.grasp_relative_rot)

                    # Update MuJoCo rigid-body qpos directly
                    adr = self.joint_indices[self.grasped_block_id]
                    qp = adr["qpos_adr"]
                    qv = adr["qvel_adr"]

                    self.data.qpos[qp : qp + 3] = new_pos
                    # MuJoCo quat is (w, x, y, z)
                    self.data.qpos[qp + 3 : qp + 7] = [new_rot[3], new_rot[0], new_rot[1], new_rot[2]]
                    self.data.qvel[qv : qv + 6] = 0.0

                    grasped_block.position = list(new_pos)
                    grasped_block.rotation = list(new_rot)
                    grasped_block.velocity = [0.0, 0.0, 0.0]

        elif self.release_cooldown_ticks > 0:
            self.release_cooldown_ticks -= 1
        elif grip_val >= GRASP_CLOSE_THRESHOLD:
            # Orientation-aware candidate grasp detection
            candidate = self._check_orientation_aware_grasp(tcp_pos, tcp_rot, grip_val)
            if candidate is not None:
                self.grasped_block_id = candidate
                block = self.blocks[candidate]
                block.is_grasped = True

                # Compute invariant relative transform
                adr = self.joint_indices[candidate]
                qp = adr["qpos_adr"]
                bp = tuple(self.data.qpos[qp : qp + 3])
                b_quat_wxyz = self.data.qpos[qp + 3 : qp + 7]
                b_quat = (float(b_quat_wxyz[1]), float(b_quat_wxyz[2]), float(b_quat_wxyz[3]), float(b_quat_wxyz[0]))

                tcp_quat_conj = quat_conjugate(tcp_rot)
                r_rel = quat_rotate_vec(tcp_quat_conj, (bp[0] - tcp_pos[0], bp[1] - tcp_pos[1], bp[2] - tcp_pos[2]))
                q_rel = quat_mul(tcp_quat_conj, b_quat)
                q_rel = snap_cube_rotation(q_rel)
                r_rel = (r_rel[0], 0.0, 0.0)

                self.grasp_relative_pos = r_rel
                self.grasp_relative_rot = q_rel

        # 3. Step MuJoCo Dynamics Solver
        mujoco.mj_step(self.model, self.data)

        # Ensure held block maintains exact kinematic transform without internal solver stress
        if self.grasped_block_id is not None and self.grasp_relative_pos is not None:
            new_pos = vec_add(tcp_pos, quat_rotate_vec(tcp_rot, self.grasp_relative_pos))
            new_rot = quat_mul(tcp_rot, self.grasp_relative_rot)
            adr = self.joint_indices[self.grasped_block_id]
            qp = adr["qpos_adr"]
            qv = adr["qvel_adr"]
            self.data.qpos[qp : qp + 3] = new_pos
            self.data.qpos[qp + 3 : qp + 7] = [new_rot[3], new_rot[0], new_rot[1], new_rot[2]]
            self.data.qvel[qv : qv + 6] = 0.0

        # Grounding & velocity regulation for ungrasped tabletop blocks (heavy 350g, no flying)
        half_x_bound, _, half_z_bound = TABLE_HALF_EXTENTS
        min_table_y_check = TABLE_TOP_Y + CUBE_HALF
        for b_id_chk, block_chk in self.blocks.items():
            if block_chk.is_grasped:
                continue
            adr_chk = self.joint_indices[b_id_chk]
            qp_chk = adr_chk["qpos_adr"]
            qv_chk = adr_chk["qvel_adr"]

            # During release cooldown, ensure opening fingers do not kick or drift the placed block
            if self.release_cooldown_ticks > 0 and b_id_chk == getattr(self, "last_released_id", None):
                if getattr(self, "last_released_pos", None) is not None:
                    self.data.qpos[qp_chk] = self.last_released_pos[0]
                    self.data.qpos[qp_chk + 2] = self.last_released_pos[1]
                self.data.qvel[qv_chk] = 0.0
                self.data.qvel[qv_chk + 2] = 0.0
                self.data.qvel[qv_chk + 3 : qv_chk + 6] = 0.0

            # Grounded stability: ungrasped blocks resting on the table cannot acquire upward ballistic pop
            if self.data.qpos[qp_chk + 1] <= min_table_y_check + 0.025:
                if self.data.qvel[qv_chk + 1] > 0.0:
                    self.data.qvel[qv_chk + 1] = 0.0

            # Heavy tabletop sliding friction damping (blocks don't slide forever or float)
            self.data.qvel[qv_chk] *= 0.88
            self.data.qvel[qv_chk + 2] *= 0.88
            self.data.qvel[qv_chk + 3 : qv_chk + 6] *= 0.85

            # Clamp maximum linear velocity (suppresses explosive impulses)
            spd = math.hypot(self.data.qvel[qv_chk], self.data.qvel[qv_chk + 1], self.data.qvel[qv_chk + 2])
            if spd > 0.35:
                scale = 0.35 / spd
                self.data.qvel[qv_chk : qv_chk + 3] *= scale

        # 4. Read back block states from MuJoCo and update PhysicsBlock records
        half_x, _, half_z = TABLE_HALF_EXTENTS
        margin = TABLE_WALL_HALF_THICKNESS + CUBE_HALF
        min_table_y = TABLE_TOP_Y + CUBE_HALF

        contacts = self.get_contact_info()
        contact_geoms = set()
        for c in contacts:
            contact_geoms.add((c["geom1"], c["geom2"]))
            contact_geoms.add((c["geom2"], c["geom1"]))

        for b_id, block in self.blocks.items():
            adr = self.joint_indices[b_id]
            qp = adr["qpos_adr"]
            qv = adr["qvel_adr"]

            if block.is_grasped:
                continue

            raw_pos = self.data.qpos[qp : qp + 3].tolist()
            raw_quat_wxyz = self.data.qpos[qp + 3 : qp + 7].tolist()
            raw_vel = self.data.qvel[qv : qv + 3].tolist()

            # Convert quaternion (w, x, y, z) -> (x, y, z, w) for Three.js
            block.position = [
                clamp(raw_pos[0], -half_x + margin, half_x - margin),
                max(min_table_y, raw_pos[1]),
                clamp(raw_pos[2], -half_z + margin, half_z - margin),
            ]
            block.rotation = [raw_quat_wxyz[1], raw_quat_wxyz[2], raw_quat_wxyz[3], raw_quat_wxyz[0]]
            block.velocity = raw_vel

            # Determine support relationship from physical contacts
            b_geom = f"{b_id}_geom"
            block.supported_by = "table"
            block.stacked_layer = 0

            for other_id, other in self.blocks.items():
                if other_id == b_id or other.is_grasped:
                    continue
                other_geom = f"{other_id}_geom"
                # If contact exists and other block is physically underneath
                if (b_geom, other_geom) in contact_geoms:
                    if block.position[1] > other.position[1] + 0.02:
                        block.supported_by = other_id
                        block.stacked_layer = (other.stacked_layer or 0) + 1

        return self.get_snapshot()

    def get_snapshot(self) -> Dict[str, Any]:
        fk = forward_kinematics(self.joints)
        blocks_data = []
        for b_id, b in self.blocks.items():
            pad_dist = min(
                math.hypot(b.position[0] - STACK_ORIGIN[0], b.position[2] - STACK_ORIGIN[2]),
                math.hypot(b.position[0] - STACK_ORIGIN_A[0], b.position[2] - STACK_ORIGIN_A[2]),
                math.hypot(b.position[0] - STACK_ORIGIN_B[0], b.position[2] - STACK_ORIGIN_B[2]),
            )
            in_pad = pad_dist <= (STACK_TOLERANCE + 0.015)

            blocks_data.append({
                "id": b.id,
                "name": b.name,
                "color": b.color,
                "position": [round(p, 4) for p in b.position],
                "rotation": [round(r, 4) for r in b.rotation],
                "velocity": [round(v, 4) for v in b.velocity],
                "is_grasped": b.is_grasped,
                "stacked_layer": b.stacked_layer,
                "supported_by": b.supported_by,
                "in_pad": in_pad,
                "pad_dist": round(pad_dist, 4),
            })

        # Update stability hold validator
        is_verified, hold_ticks, stability_status = self.stability_validator.step(
            blocks_data, self.grasped_block_id
        )

        # Compute multi-metric scores
        spat_acc = spatial_accuracy(
            blocks_data,
            STACK_ORIGIN,
            task_block_ids=self.active_task_blocks,
            task_targets=self.active_task_targets,
        )
        comp_score = task_completion_score(
            blocks_data,
            STACK_ORIGIN,
            self.grasped_block_id,
            task_block_ids=self.active_task_blocks,
            task_targets=self.active_task_targets,
        )
        torque_summary = self.torque_tracker.summarize()

        return {
            "tick": self.tick,
            "joints": {k: round(v, 4) for k, v in self.joints.items()},
            "joint_targets": {k: round(v, 4) for k, v in self.joint_targets.items()},
            "arm": fk,
            "tcp": {
                "position": [round(p, 4) for p in fk["tcp"]["position"]],
                "rotation": [round(r, 4) for r in fk["tcp"]["rotation"]],
            },
            "blocks": blocks_data,
            "grasped_block_id": self.grasped_block_id,
            "target_zone": {
                "position": STACK_ORIGIN,
                "radius": STACK_TOLERANCE,
                "zones": [
                    {"id": "center", "position": STACK_ORIGIN, "label": "Center Pad"},
                    {"id": "zone_a", "position": STACK_ORIGIN_A, "label": "Pad A (North)"},
                    {"id": "zone_b", "position": STACK_ORIGIN_B, "label": "Pad B (South)"},
                ],
            },
            "scores": {
                "spatial_accuracy": spat_acc,
                "task_completion": comp_score,
                "torque": torque_summary,
                "stability": {
                    "is_verified": is_verified,
                    "hold_ticks": hold_ticks,
                    "required_ticks": self.stability_validator.required_ticks,
                    "status": stability_status,
                }
            }
        }
