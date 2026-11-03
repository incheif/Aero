"""
Robot Arm Simulation Physics Engine.
Implements the 60 Hz rigid-body dynamics, kinematic grasping, joint actuation,
table collision bounds, and stacking tolerances matching new_project_1.
"""

import math
import copy
from typing import Dict, List, Tuple, Optional, Any
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
    JAW_MIN_SEP,
    JAW_MAX_SEP,
    JAW_LENGTH,
    stack_slot_y,
)
from .kinematics import forward_kinematics, clamp, is_over_table, vec_add, quat_rotate_vec
from .scoring import (
    TorqueTracker,
    StabilityValidator,
    spatial_accuracy,
    task_completion_score,
)

class PhysicsBlock:
    def __init__(self, block_id: str, name: str, color: str, color_rgb: Tuple[int, int, int], pos: List[float]):
        self.id = block_id
        self.name = name
        self.color = color
        self.color_rgb = color_rgb
        self.position = list(pos)
        self.rotation = [0.0, 0.0, 0.0, 1.0]
        self.is_placed: bool = False
        self.velocity = [0.0, 0.0, 0.0]
        self.is_grasped = False
        self.stacked_layer: Optional[int] = None
        self.supported_by: Optional[str] = None

    def clone(self) -> 'PhysicsBlock':
        b = PhysicsBlock(self.id, self.name, self.color, self.color_rgb, list(self.position))
        b.rotation = list(self.rotation)
        b.velocity = list(self.velocity)
        b.is_grasped = self.is_grasped
        b.stacked_layer = self.stacked_layer
        b.supported_by = self.supported_by
        return b

class RobotArmSimulation:
    """
    Complete physical simulation matching new_project_1 (Rapier/Kinematic hybrid).
    Steps arm joints, checks collisions, updates block dynamics, and maintains grasp contacts.
    """
    def __init__(self):
        self.tick: int = 0
        self.joints: Dict[str, float] = copy.deepcopy(DEFAULT_JOINTS)
        self.joint_targets: Dict[str, float] = copy.deepcopy(DEFAULT_JOINTS)
        self.gripper_target: float = 0.0
        self.grasped_block_id: Optional[str] = None
        self.release_cooldown_ticks: int = 0

        # Blocks
        self.blocks: Dict[str, PhysicsBlock] = {}
        self.reset_blocks()

        # Telemetry & Playground Verification
        self.torque_tracker = TorqueTracker()
        self.stability_validator = StabilityValidator()
        self.active_task_blocks: Optional[List[str]] = None
        self.active_task_targets: Optional[Dict[str, List[float]]] = None

    def reset_blocks(self, randomize: bool = False):
        self.blocks.clear()
        spawns = generate_random_block_spawns() if randomize else BLOCK_SPAWNS
        for spawn in spawns:
            self.blocks[spawn["id"]] = PhysicsBlock(
                block_id=spawn["id"],
                name=spawn["name"],
                color=spawn["color"],
                color_rgb=spawn["color_rgb"],
                pos=spawn["position"]
            )
        self.grasped_block_id = None

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
        Enforces physical solid impassability between robot arm links, claws, carried block, and blocks.
        All blocks and all robot components are solid bodies:
        - Upper arm, elbow, forearm, wrist, palm, and jaw claws cannot pass through or penetrate ANY block.
        - The arm is physically stopped (joints reverted) upon contacting any block surface.
        - Tabletop blocks are pushed along contact normal without allowing the arm to penetrate.
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

            bx, by, bz = block.position
            dx_tcp = abs(tcp_pos[0] - bx)
            dz_tcp = abs(tcp_pos[2] - bz)
            dy_tcp = abs(tcp_pos[1] - by)

            # Placed or stacked status
            is_placed = (
                getattr(block, "is_placed", False)
                or block.supported_by != "table"
                or (block.stacked_layer or 0) > 0
                or (self.active_task_targets and b_id in self.active_task_targets and
                    math.hypot(bx - self.active_task_targets[b_id][0], bz - self.active_task_targets[b_id][2]) < 0.038)
            )

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
                and dy_tcp <= 0.032
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
                # If target pick or grasp pinch, jaws pinch or straddle outside faces of cube
                if (is_targeted_pick or is_grasp_pinch) and p_name.startswith("jaw"):
                    if abs(p_pos[0] - bx) >= CUBE_HALF - 0.005 or abs(p_pos[2] - bz) >= CUBE_HALF - 0.005:
                        continue

                # Upward ascent away from contact is always allowed
                if p_pos[1] >= by - 0.005 and p_pos[1] > prev_p_pos[1] + 0.0003:
                    continue

                if is_vertical_retreat and (p_name.startswith("jaw") or p_name.startswith("palm")):
                    continue

                # Closest point on block cube to proxy center
                cx = min(max(p_pos[0], bx - CUBE_HALF), bx + CUBE_HALF)
                cy = min(max(p_pos[1], by - CUBE_HALF), by + CUBE_HALF)
                cz = min(max(p_pos[2], bz - CUBE_HALF), bz + CUBE_HALF)
                dist = math.sqrt((p_pos[0] - cx)**2 + (p_pos[1] - cy)**2 + (p_pos[2] - cz)**2)
                pen = p_rad - dist

                # Impassability threshold: 3mm
                if pen > 0.003:
                    blocked = True
                    # If free tabletop block, repel away along normal without allowing arm to penetrate
                    if not is_placed and is_over_table(bx, bz):
                        sign_x = 1.0 if bx >= p_pos[0] else -1.0
                        sign_z = 1.0 if bz >= p_pos[2] else -1.0
                        block.position[0] += pen * sign_x * 0.4
                        block.velocity[0] = sign_x * 0.12
                        block.position[2] += pen * sign_z * 0.4
                        block.velocity[2] = sign_z * 0.12
                        if hasattr(self, "joint_indices") and b_id in self.joint_indices:
                            adr = self.joint_indices[b_id]
                            self.data.qpos[adr["qpos_adr"]] = block.position[0]
                            self.data.qpos[adr["qpos_adr"] + 2] = block.position[2]
                            self.data.qvel[adr["qvel_adr"]] = block.velocity[0]
                            self.data.qvel[adr["qvel_adr"] + 2] = block.velocity[2]
                    break

            if blocked:
                break

        if blocked:
            # Physically stop positioning joints at contact face to enforce 100% solid barrier
            for k in ("baseYaw", "shoulderPitch", "elbowPitch", "wristPitch"):
                self.joints[k] = prev_joints[k]
            fk = forward_kinematics(self.joints)

        return fk

    def step(self, dt: float = FIXED_DT):
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
                    grasped_block.velocity = [0.0, -0.05, 0.0]
                    self.last_released_id = self.grasped_block_id
                    self.grasped_block_id = None
                    self.release_cooldown_ticks = 30
                else:
                    # Move block to lock onto TCP mid-finger pinch point
                    # Physical clamp: carried block can never penetrate table surface OR any underlying blocks
                    min_support_y = TABLE_TOP_Y + CUBE_HALF + 0.002
                    for other_id, other in self.blocks.items():
                        if other_id == self.grasped_block_id or other.is_grasped:
                            continue
                        dx = abs(tcp_pos[0] - other.position[0])
                        dz = abs(tcp_pos[2] - other.position[2])
                        # Check horizontal AABB overlap with any other block
                        if dx < CUBE_SIZE - 0.024 and dz < CUBE_SIZE - 0.024:
                            if tcp_pos[1] >= other.position[1] - 0.015:
                                cand_y = other.position[1] + CUBE_SIZE
                                if cand_y > min_support_y:
                                    min_support_y = cand_y

                    grasped_block.position = [tcp_pos[0], max(min_support_y, tcp_pos[1]), tcp_pos[2]]
                    grasped_block.rotation = list(tcp_rot)
                    grasped_block.velocity = [0.0, 0.0, 0.0]

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

        # 4. Step Un-grasped Block Dynamics & Stacking Physics
        half_x, _, half_z = TABLE_HALF_EXTENTS
        min_table_y = TABLE_TOP_Y + CUBE_HALF

        for b_id, block in self.blocks.items():
            if block.is_grasped:
                continue

            # Gravity
            block.velocity[1] += GRAVITY_Y * dt
            # Velocity damping
            block.velocity[0] *= 0.92
            block.velocity[1] *= 0.98
            block.velocity[2] *= 0.92

            # Integrate position
            block.position[0] += block.velocity[0] * dt
            block.position[1] += block.velocity[1] * dt
            block.position[2] += block.velocity[2] * dt

            # Keep block within table boundary rim
            margin = TABLE_WALL_HALF_THICKNESS + CUBE_HALF
            block.position[0] = clamp(block.position[0], -half_x + margin, half_x - margin)
            block.position[2] = clamp(block.position[2], -half_z + margin, half_z - margin)

            # Table surface collision
            support_y = min_table_y
            block.supported_by = "table"
            block.stacked_layer = 0

            # Check collision with other blocks (stacking support)
            for other_id, other in self.blocks.items():
                if other_id == b_id:
                    continue
                # Grasped blocks being carried by gripper cannot act as support platforms
                if other.is_grasped:
                    continue
                # For other block to support this block, this block must be above other (or slightly resting on top)
                if block.position[1] >= other.position[1] - 0.012:
                    dx = abs(block.position[0] - other.position[0])
                    dz = abs(block.position[2] - other.position[2])
                    overlap_x = CUBE_SIZE - dx
                    overlap_z = CUBE_SIZE - dz
                    # Horizontal overlap exists when AABBs share contact area (supports 1-on-1 stacking and pyramid bridging)
                    if overlap_x > 0.001 and overlap_z > 0.001:
                        candidate_support_y = other.position[1] + CUBE_SIZE
                        if candidate_support_y > support_y:
                            support_y = candidate_support_y
                            block.supported_by = other_id
                            block.stacked_layer = (other.stacked_layer or 0) + 1

            if block.position[1] <= support_y:
                block.position[1] = support_y
                block.velocity[1] = 0.0

        # 5. Full 3D Solid-Body Collision Resolution between Blocks (Multi-Pass Constraint Solver)
        block_keys = list(self.blocks.keys())
        margin = TABLE_WALL_HALF_THICKNESS + CUBE_HALF
        # Run 4 solver relaxation iterations to ensure transitive multi-contact stability
        for _solver_iter in range(4):
            for i in range(len(block_keys)):
                b1_id = block_keys[i]
                b1 = self.blocks[b1_id]
                for j in range(i + 1, len(block_keys)):
                    b2_id = block_keys[j]
                    b2 = self.blocks[b2_id]

                    if b1.is_grasped and b2.is_grasped:
                        continue

                    dx = b1.position[0] - b2.position[0]
                    dy = b1.position[1] - b2.position[1]
                    dz = b1.position[2] - b2.position[2]
                    overlap_x = CUBE_SIZE - abs(dx)
                    overlap_y = CUBE_SIZE - abs(dy)
                    overlap_z = CUBE_SIZE - abs(dz)

                    if overlap_x > 0.0005 and overlap_y > 0.0005 and overlap_z > 0.0005:
                        # Cubes interpenetrate in 3D!
                        # Case 1: Vertical stacking penetration (one is above the other with horizontal overlap)
                        if overlap_y <= min(overlap_x, overlap_z) or (overlap_x > 0.008 and overlap_z > 0.008 and abs(dy) > CUBE_HALF * 0.8):
                            if dy >= 0.0:
                                # b1 is above b2: enforce b1 rests flush on top of b2
                                if b1.is_grasped:
                                    # Held block b1 cannot sink into underlying support block b2
                                    b1.position[1] = max(b1.position[1], b2.position[1] + CUBE_SIZE)
                                elif b2.is_grasped:
                                    b1.position[1] = b2.position[1] + CUBE_SIZE
                                    b1.velocity[1] = 0.0
                                else:
                                    b1.position[1] = b2.position[1] + CUBE_SIZE
                                    b1.velocity[1] = 0.0
                            else:
                                # b2 is above b1: enforce b2 rests flush on top of b1
                                if b2.is_grasped:
                                    # Held block b2 cannot sink into underlying support block b1
                                    b2.position[1] = max(b2.position[1], b1.position[1] + CUBE_SIZE)
                                elif b1.is_grasped:
                                    b2.position[1] = b1.position[1] + CUBE_SIZE
                                    b2.velocity[1] = 0.0
                                else:
                                    b2.position[1] = b1.position[1] + CUBE_SIZE
                                    b2.velocity[1] = 0.0

                        # Case 2: Horizontal side-by-side penetration
                        else:
                            if overlap_x < overlap_z:
                                sign = 1.0 if dx >= 0.0 else -1.0
                                if b1.is_grasped:
                                    b2.position[0] -= overlap_x * sign
                                    b2.velocity[0] = -0.04 * sign
                                elif b2.is_grasped:
                                    b1.position[0] += overlap_x * sign
                                    b1.velocity[0] = 0.04 * sign
                                else:
                                    b1.position[0] += (overlap_x * 0.5) * sign
                                    b2.position[0] -= (overlap_x * 0.5) * sign
                                    b1.velocity[0] *= 0.5
                                    b2.velocity[0] *= 0.5
                            else:
                                sign = 1.0 if dz >= 0.0 else -1.0
                                if b1.is_grasped:
                                    b2.position[2] -= overlap_z * sign
                                    b2.velocity[2] = -0.04 * sign
                                elif b2.is_grasped:
                                    b1.position[2] += overlap_z * sign
                                    b1.velocity[2] = 0.04 * sign
                                else:
                                    b1.position[2] += (overlap_z * 0.5) * sign
                                    b2.position[2] -= (overlap_z * 0.5) * sign
                                    b1.velocity[2] *= 0.5
                                    b2.velocity[2] *= 0.5

                        if not b1.is_grasped:
                            b1.position[0] = clamp(b1.position[0], -half_x + margin, half_x - margin)
                            b1.position[2] = clamp(b1.position[2], -half_z + margin, half_z - margin)
                        if not b2.is_grasped:
                            b2.position[0] = clamp(b2.position[0], -half_x + margin, half_x - margin)
                            b2.position[2] = clamp(b2.position[2], -half_z + margin, half_z - margin)

        # Return snapshot
        return self.get_snapshot()

    def get_snapshot(self) -> Dict[str, Any]:
        fk = forward_kinematics(self.joints)
        blocks_data = []
        for b_id, b in self.blocks.items():
            # Check proximity to target pad (Center, Zone A, or Zone B)
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

        # Compute multi-metric scores (task-adaptive)
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
