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
        self.reset_blocks()
        self.torque_tracker.reset()
        self.stability_validator.reset()

    def set_joint_targets(self, targets: Dict[str, float], gripper: Optional[float] = None):
        for k, v in targets.items():
            if k in JOINT_LIMITS:
                low, high = JOINT_LIMITS[k]
                self.joint_targets[k] = clamp(v, low, high)
        if gripper is not None:
            self.gripper_target = clamp(gripper, 0.0, 1.0)
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
                    grasped_block.position = list(tcp_pos)
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
                # If other block is below this block
                if other.position[1] < block.position[1]:
                    horiz_dist = math.hypot(block.position[0] - other.position[0], block.position[2] - other.position[2])
                    if horiz_dist <= CUBE_SIZE * 0.95:
                        candidate_support_y = other.position[1] + CUBE_SIZE
                        if candidate_support_y > support_y:
                            support_y = candidate_support_y
                            block.supported_by = other_id
                            block.stacked_layer = (other.stacked_layer or 0) + 1

            if block.position[1] <= support_y:
                block.position[1] = support_y
                block.velocity[1] = 0.0

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

        # Compute multi-metric scores
        spat_acc = spatial_accuracy(blocks_data, STACK_ORIGIN)
        comp_score = task_completion_score(blocks_data, STACK_ORIGIN, self.grasped_block_id)
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
