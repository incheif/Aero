"""
MuJoCo Physics Simulation Engine for Robot Arm Workcell.
Provides high-fidelity multi-body rigid-body dynamics, contact manifold resolution,
friction cones, and zero-penetration solid body stacking.
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
)
from .kinematics import forward_kinematics, clamp, is_over_table
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
    - Native continuous convex collision detection and LCP/Newton solver
    - Coulomb friction cones and contact restitution
    - Kinematic precision arm trajectory tracking with dynamic multi-block physics
    - Complete drop-in compatibility with RobotArmSimulation snapshot telemetry
    """

    def __init__(self):
        if not MUJOCO_AVAILABLE:
            raise RuntimeError("MuJoCo is not installed. Please run `pip install mujoco`.")

        self.tick: int = 0
        self.joints: Dict[str, float] = copy.deepcopy(DEFAULT_JOINTS)
        self.joint_targets: Dict[str, float] = copy.deepcopy(DEFAULT_JOINTS)
        self.gripper_target: float = 0.0
        self.grasped_block_id: Optional[str] = None
        self.release_cooldown_ticks: int = 0

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
            '    <geom friction="1.2 0.01 0.001" solref="0.008 1" solimp="0.95 0.99 0.001"/>',
            '  </default>',
            '  <worldbody>',
            '    <!-- Tabletop Surface -->',
            f'    <geom name="table" type="box" size="{half_x} {table_h} {half_z}" pos="0 {table_center_y} 0" rgba="0.18 0.20 0.24 1"/>',
            '    <!-- Table Boundary Walls -->',
            f'    <geom name="wall_n" type="box" size="{half_x} {wall_h} {wall_t}" pos="0 {wall_y} {half_z - wall_t}"/>',
            f'    <geom name="wall_s" type="box" size="{half_x} {wall_h} {wall_t}" pos="0 {wall_y} {-half_z + wall_t}"/>',
            f'    <geom name="wall_e" type="box" size="{wall_t} {wall_h} {half_z}" pos="{half_x - wall_t} {wall_y} 0"/>',
            f'    <geom name="wall_w" type="box" size="{wall_t} {wall_h} {half_z}" pos="{-half_x + wall_t} {wall_y} 0"/>',
        ]

        # Add 7 Tabletop Blocks with Freejoints
        for spawn in BLOCK_SPAWNS:
            b_id = spawn["id"]
            p = spawn["position"]
            r, g, b = [c / 255.0 for c in spawn["color_rgb"]]
            lines.append(f'    <body name="{b_id}" pos="{p[0]} {p[1]} {p[2]}">')
            lines.append(f'      <freejoint name="{b_id}_joint"/>')
            lines.append(f'      <geom name="{b_id}_geom" type="box" size="{CUBE_HALF} {CUBE_HALF} {CUBE_HALF}" mass="0.05" rgba="{r:.3f} {g:.3f} {b:.3f} 1"/>')
            lines.append('    </body>')

        lines.append('  </worldbody>')
        lines.append('</mujoco>')
        return '\n'.join(lines)

    def _init_mujoco_model(self):
        xml = self._build_mjcf_xml()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        # Cache joint addresses for fast indexing
        self.joint_indices = {}
        for spawn in BLOCK_SPAWNS:
            b_id = spawn["id"]
            j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{b_id}_joint")
            self.joint_indices[b_id] = {
                "qpos_adr": self.model.jnt_qposadr[j_id],
                "qvel_adr": self.model.jnt_dofadr[j_id],
            }

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
        mujoco.mj_forward(self.model, self.data)

    def reset(self, randomize: bool = False):
        self.tick = 0
        self.joints = copy.deepcopy(DEFAULT_JOINTS)
        self.joint_targets = copy.deepcopy(DEFAULT_JOINTS)
        self.gripper_target = 0.0
        self.active_task_blocks = None
        self.active_task_targets = None
        self.reset_blocks(randomize=randomize)
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
        Enforces physical solid impassability between robot arm links/gripper and blocks.
        Blocks cannot be passed through by the arm:
        - Placed/stacked blocks are immovable solid anchors: arm is physically blocked at contact face.
        - Arm links (upper arm, elbow, forearm, wrist) cannot penetrate ANY block.
        - Free tabletop blocks are pushed along contact normal when contacted by end-effector.
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

        arm_proxies = []
        # Upper Arm link (shoulder -> elbow, 4 points)
        for alpha in (0.25, 0.5, 0.75, 1.0):
            pos = [p_shoulder[i] + alpha * (p_elbow[i] - p_shoulder[i]) for i in range(3)]
            prev_pos = [prev_shoulder[i] + alpha * (prev_elbow[i] - prev_shoulder[i]) for i in range(3)]
            arm_proxies.append((f"upper_{alpha}", pos, 0.026, prev_pos))

        # Forearm link (elbow -> wrist, 4 points)
        for alpha in (0.25, 0.5, 0.75, 1.0):
            pos = [p_elbow[i] + alpha * (p_wrist[i] - p_elbow[i]) for i in range(3)]
            prev_pos = [prev_elbow[i] + alpha * (prev_wrist[i] - prev_elbow[i]) for i in range(3)]
            arm_proxies.append((f"forearm_{alpha}", pos, 0.024, prev_pos))

        # Wrist link (wrist -> palm, 2 points)
        for alpha in (0.5, 1.0):
            pos = [p_wrist[i] + alpha * (p_palm[i] - p_wrist[i]) for i in range(3)]
            prev_pos = [prev_wrist[i] + alpha * (prev_palm[i] - prev_wrist[i]) for i in range(3)]
            arm_proxies.append((f"wrist_{alpha}", pos, 0.024, prev_pos))

        if self.grasped_block_id is None:
            arm_proxies.extend([
                ("tcp", fk["tcp"]["position"], 0.020, prev_fk["tcp"]["position"]),
                ("jawLeft", fk["jawLeft"]["position"], 0.012, prev_fk["jawLeft"]["position"]),
                ("jawRight", fk["jawRight"]["position"], 0.012, prev_fk["jawRight"]["position"]),
            ])

        blocked_by_placed = False
        for b_id, block in self.blocks.items():
            if b_id == self.grasped_block_id or block.is_grasped:
                continue
            if self.release_cooldown_ticks > 0 and b_id == getattr(self, "last_released_id", None):
                continue

            bx, by, bz = block.position
            dx_tcp = abs(fk["tcp"]["position"][0] - bx)
            dz_tcp = abs(fk["tcp"]["position"][2] - bz)
            dy_tcp = abs(fk["tcp"]["position"][1] - by)

            is_placed = (
                getattr(block, "is_placed", False)
                or block.supported_by != "table"
                or (block.stacked_layer or 0) > 0
                or (self.active_task_targets and b_id in self.active_task_targets and
                    math.hypot(bx - self.active_task_targets[b_id][0], bz - self.active_task_targets[b_id][2]) < 0.038)
            )

            # Allow unobstructed pick: target block within grasp envelope or vertical approach
            is_grasp_candidate = (dx_tcp <= GRASP_BOX["x"] * 1.5 and dz_tcp <= GRASP_BOX["z"] * 1.5 and dy_tcp <= 0.040)
            is_pick_approach = (dx_tcp < 0.048 and dz_tcp < 0.048 and fk["tcp"]["position"][1] >= by - 0.008)
            if not is_placed and (is_grasp_candidate or is_pick_approach):
                continue

            # Allow unobstructed vertical ascent when releasing and retreating upward from placed block
            is_vertical_retreat = (dx_tcp < 0.045 and dz_tcp < 0.045 and fk["tcp"]["position"][1] >= by and grip_val <= GRASP_OPEN_THRESHOLD + 0.05)
            if is_vertical_retreat:
                continue

            for p_name, p_pos, p_rad, prev_p_pos in arm_proxies:
                # Closest point on block box to proxy center
                cx = min(max(p_pos[0], bx - CUBE_HALF), bx + CUBE_HALF)
                cy = min(max(p_pos[1], by - CUBE_HALF), by + CUBE_HALF)
                cz = min(max(p_pos[2], bz - CUBE_HALF), bz + CUBE_HALF)
                dist = math.sqrt((p_pos[0] - cx)**2 + (p_pos[1] - cy)**2 + (p_pos[2] - cz)**2)
                pen = p_rad - dist

                # True penetration beyond margin (> 4mm)
                if pen > 0.004:
                    # If this proxy is at/above block and strictly moving upward, it is ascending away
                    is_ascending_away = (p_pos[1] >= by - 0.005 and p_pos[1] > prev_p_pos[1] + 0.0003)
                    if is_ascending_away:
                        continue

                    is_arm_link = not p_name.startswith("jaw") and p_name != "tcp"
                    # Placed blocks are immovable solid anchors.
                    # Arm links (upper arm, elbow, forearm, wrist) cannot penetrate ANY block!
                    if is_placed or not is_over_table(bx, bz) or is_arm_link:
                        blocked_by_placed = True
                        break
                    else:
                        sign_x = 1.0 if bx >= p_pos[0] else -1.0
                        sign_z = 1.0 if bz >= p_pos[2] else -1.0
                        block.position[0] += pen * sign_x * 0.5
                        block.velocity[0] = sign_x * 0.15
                        block.position[2] += pen * sign_z * 0.5
                        block.velocity[2] = sign_z * 0.15

            if blocked_by_placed:
                break

        if blocked_by_placed:
            # Revert positioning arm joints to prevent slicing through block; allow gripper to actuate
            for k in ("baseYaw", "shoulderPitch", "elbowPitch", "wristPitch"):
                self.joints[k] = prev_joints[k]
            fk = forward_kinematics(self.joints)

        return fk

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
        # 2. Forward Kinematics to find current link and TCP positions
        tcp_pos = fk["tcp"]["position"]
        tcp_rot = fk["tcp"]["rotation"]

        # 3. Handle Grasp Dynamics
        grip_val = self.joints["gripper"]

        if self.grasped_block_id is not None:
            grasped_block = self.blocks.get(self.grasped_block_id)
            if grasped_block is not None:
                # Check for release
                if grip_val < GRASP_OPEN_THRESHOLD:
                    grasped_block.is_grasped = False
                    grasped_block.is_placed = True
                    self.last_released_id = self.grasped_block_id
                    self.grasped_block_id = None
                    self.release_cooldown_ticks = 30
                    # Give slight downward settling velocity in MuJoCo
                    adr = self.joint_indices[grasped_block.id]
                    self.data.qvel[adr["qvel_adr"] : adr["qvel_adr"] + 3] = [0.0, -0.05, 0.0]
                else:
                    # Move block to lock onto TCP mid-finger pinch point
                    # Physical vertical clamp: held block cannot sink into underlying blocks or table
                    min_support_y = TABLE_TOP_Y + CUBE_HALF + 0.002
                    for other_id, other in self.blocks.items():
                        if other_id == self.grasped_block_id or other.is_grasped:
                            continue
                        dx = abs(tcp_pos[0] - other.position[0])
                        dz = abs(tcp_pos[2] - other.position[2])
                        if dx < CUBE_SIZE - 0.024 and dz < CUBE_SIZE - 0.024:
                            if tcp_pos[1] >= other.position[1] - 0.015:
                                cand_y = other.position[1] + CUBE_SIZE
                                if cand_y > min_support_y:
                                    min_support_y = cand_y

                    grasped_pos = [tcp_pos[0], max(min_support_y, tcp_pos[1]), tcp_pos[2]]
                    grasped_block.position = list(grasped_pos)
                    grasped_block.rotation = list(tcp_rot)
                    grasped_block.velocity = [0.0, 0.0, 0.0]

                    # Synchronize grasped block state into MuJoCo
                    adr = self.joint_indices[self.grasped_block_id]
                    qp = adr["qpos_adr"]
                    qv = adr["qvel_adr"]
                    self.data.qpos[qp : qp + 3] = grasped_pos
                    # MuJoCo quat is (w, x, y, z)
                    self.data.qpos[qp + 3 : qp + 7] = [tcp_rot[3], tcp_rot[0], tcp_rot[1], tcp_rot[2]]
                    self.data.qvel[qv : qv + 6] = 0.0

        elif self.release_cooldown_ticks > 0:
            self.release_cooldown_ticks -= 1
        elif grip_val >= GRASP_CLOSE_THRESHOLD:
            # Look for graspable block within grasp bounding envelope
            best_candidate = None
            min_dist = float("inf")
            for b_id, block in self.blocks.items():
                if block.is_grasped:
                    continue
                dx = abs(block.position[0] - tcp_pos[0])
                dy = abs(block.position[1] - tcp_pos[1])
                dz = abs(block.position[2] - tcp_pos[2])
                if dx <= GRASP_BOX["x"] and dy <= GRASP_BOX["y"] and dz <= GRASP_BOX["z"]:
                    dist = math.hypot(dx, dy, dz)
                    if dist < min_dist:
                        min_dist = dist
                        best_candidate = b_id

            if best_candidate is not None:
                self.grasped_block_id = best_candidate
                self.blocks[best_candidate].is_grasped = True
                self.blocks[best_candidate].position = list(tcp_pos)
                self.blocks[best_candidate].rotation = list(tcp_rot)
                adr = self.joint_indices[best_candidate]
                self.data.qpos[adr["qpos_adr"] : adr["qpos_adr"] + 3] = tcp_pos
                self.data.qpos[adr["qpos_adr"] + 3 : adr["qpos_adr"] + 7] = [tcp_rot[3], tcp_rot[0], tcp_rot[1], tcp_rot[2]]
                self.data.qvel[adr["qvel_adr"] : adr["qvel_adr"] + 6] = 0.0

        # 4. Step MuJoCo Dynamics
        mujoco.mj_step(self.model, self.data)

        # 5. Read back block states from MuJoCo and update PhysicsBlock records
        half_x, _, half_z = TABLE_HALF_EXTENTS
        margin = TABLE_WALL_HALF_THICKNESS + CUBE_HALF
        min_table_y = TABLE_TOP_Y + CUBE_HALF

        for b_id, block in self.blocks.items():
            adr = self.joint_indices[b_id]
            qp = adr["qpos_adr"]
            qv = adr["qvel_adr"]

            if block.is_grasped:
                # Keep held block anchored to TCP
                continue

            # Read pos, quat, vel from MuJoCo
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

            # Compute stacking support relationship
            support_y = min_table_y
            block.supported_by = "table"
            block.stacked_layer = 0

            for other_id, other in self.blocks.items():
                if other_id == b_id or other.is_grasped:
                    continue
                if block.position[1] >= other.position[1] - 0.012:
                    dx = abs(block.position[0] - other.position[0])
                    dz = abs(block.position[2] - other.position[2])
                    overlap_x = CUBE_SIZE - dx
                    overlap_z = CUBE_SIZE - dz
                    if overlap_x > 0.001 and overlap_z > 0.001:
                        candidate_support_y = other.position[1] + CUBE_SIZE
                        if candidate_support_y > support_y:
                            support_y = candidate_support_y
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

