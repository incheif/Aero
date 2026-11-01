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
    stack_slot_y,
)
from .kinematics import forward_kinematics, clamp
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

    def reset_blocks(self):
        self.blocks.clear()
        for spawn in BLOCK_SPAWNS:
            self.blocks[spawn["id"]] = PhysicsBlock(
                block_id=spawn["id"],
                name=spawn["name"],
                color=spawn["color"],
                color_rgb=spawn["color_rgb"],
                pos=spawn["position"]
            )
        self.grasped_block_id = None

    def reset(self):
        self.tick = 0
        self.joints = copy.deepcopy(DEFAULT_JOINTS)
        self.joint_targets = copy.deepcopy(DEFAULT_JOINTS)
        self.gripper_target = 0.0
        self.active_task_blocks = None
        self.active_task_targets = None
        self.reset_blocks()
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

        # 2. Forward Kinematics to find current link and TCP positions
        fk = forward_kinematics(self.joints)
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
                    grasped_block.velocity = [0.0, -0.05, 0.0]
                    self.grasped_block_id = None
                    self.release_cooldown_ticks = 15
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
                        if dx < CUBE_SIZE - 0.0005 and dz < CUBE_SIZE - 0.0005:
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
