"""
Joint-Space Motion Planner for 5-DOF Articulated Robot Arm.
Implements:
- Continuous swept-volume collision checking for all robot links, jaws, and carried blocks.
- Orientation-aware obstacle collision checking against rotated tabletop blocks.
- Bidirectional RRT-Connect joint-space sampling planner with trajectory shortcutting.
- Parameterized Approach, Transit, and Retreat trajectory generators.
"""

import math
import random
from typing import Dict, List, Tuple, Optional, Set, Any
import numpy as np

from .constants import (
    TABLE_TOP_Y,
    TABLE_HALF_EXTENTS,
    TABLE_WALL_HALF_HEIGHT,
    TABLE_WALL_HALF_THICKNESS,
    CUBE_SIZE,
    CUBE_HALF,
    JOINT_LIMITS,
    JAW_LENGTH,
    JAW_HEIGHT,
    JAW_THICKNESS,
    JAW_MIN_SEP,
    JAW_MAX_SEP,
    GRASP_DEPTH,
    L_UPPER,
    L_FOREARM,
    L_WRIST,
    ARM_MOUNT,
    PEDESTAL_H,
    SAFE_HOVER_Y,
    MAX_TCP_STEP,
    WRIST_WORLD_PITCH,
)
from .kinematics import (
    forward_kinematics,
    solve_ik,
    clamp,
    quat_rotate_vec,
    quat_conjugate,
    vec_add,
)


class ArmCollisionChecker:
    """
    Validates collision safety for articulated robot arm links and held objects
    against table surface, boundary walls, and tabletop blocks.
    """

    def __init__(self, table_margin: float = 0.004):
        self.table_margin = table_margin
        half_x, _, half_z = TABLE_HALF_EXTENTS
        self.bounds_x = (-half_x + 0.03, half_x - 0.03)
        self.bounds_z = (-half_z + 0.03, half_z - 0.03)

    def dist_point_to_segment(
        self,
        p: Tuple[float, float, float],
        a: Tuple[float, float, float],
        b: Tuple[float, float, float],
    ) -> float:
        """Computes shortest distance from point p to segment ab."""
        ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        ap = (p[0] - a[0], p[1] - a[1], p[2] - a[2])
        len_sq = ab[0]**2 + ab[1]**2 + ab[2]**2
        if len_sq < 1e-8:
            return math.hypot(ap[0], ap[1], ap[2])
        t = max(0.0, min(1.0, (ap[0]*ab[0] + ap[1]*ab[1] + ap[2]*ab[2]) / len_sq))
        proj = (a[0] + t * ab[0], a[1] + t * ab[1], a[2] + t * ab[2])
        return math.hypot(p[0] - proj[0], p[1] - proj[1], p[2] - proj[2])

    def check_configuration_collision(
        self,
        joints: Dict[str, float],
        blocks: List[Dict[str, Any]],
        grasped_block_id: Optional[str] = None,
        ignore_block_ids: Optional[Set[str]] = None,
        allow_table_contact: bool = False,
    ) -> bool:
        """
        Evaluates collision for a single joint configuration.
        Returns True if in collision, False if safe.
        """
        fk = forward_kinematics(joints)
        p_shoulder = fk["shoulder"]["position"]
        p_elbow = fk["elbow"]["position"]
        p_wrist = fk["wrist"]["position"]
        p_palm = fk["palm"]["position"]
        p_tcp = fk["tcp"]["position"]
        rot = fk["tcp"]["rotation"]

        ignore = ignore_block_ids or set()
        if grasped_block_id:
            ignore.add(grasped_block_id)

        # 1. Floor / Tabletop clearance for arm elbows and wrist
        min_arm_y = TABLE_TOP_Y + (0.005 if allow_table_contact else 0.020)
        if p_elbow[1] < min_arm_y or p_wrist[1] < min_arm_y:
            return True

        # TCP floor clearance unless deliberately allowed during touchdown
        if not allow_table_contact and p_tcp[1] < TABLE_TOP_Y + 0.015:
            return True

        # 2. Check each link against obstacle blocks
        link_segments = [
            (p_shoulder, p_elbow, 0.028),  # Upper arm capsule
            (p_elbow, p_wrist, 0.025),      # Forearm capsule
            (p_wrist, p_palm, 0.022),       # Wrist capsule
            (p_palm, p_tcp, 0.020),         # Palm-to-TCP gripper core
        ]

        grip_val = joints.get("gripper", 0.0)
        sep = JAW_MIN_SEP * 0.5 + ((JAW_MAX_SEP - JAW_MIN_SEP) * 0.5) * (1.0 - clamp(grip_val, 0.0, 1.0))
        jaw_l = vec_add(p_palm, quat_rotate_vec(rot, (JAW_LENGTH * 0.5, 0.0, sep)))
        jaw_r = vec_add(p_palm, quat_rotate_vec(rot, (JAW_LENGTH * 0.5, 0.0, -sep)))
        jaw_l_tip = vec_add(p_palm, quat_rotate_vec(rot, (JAW_LENGTH, 0.0, sep)))
        jaw_r_tip = vec_add(p_palm, quat_rotate_vec(rot, (JAW_LENGTH, 0.0, -sep)))

        finger_points = [
            (jaw_l, 0.024),
            (jaw_r, 0.024),
            (jaw_l_tip, 0.024),
            (jaw_r_tip, 0.024),
            (p_palm, 0.024),
            (p_tcp, 0.020),
        ]

        safety_margin = 0.008

        for b in blocks:
            b_id = b.get("id")
            if b_id in ignore or b.get("is_grasped"):
                continue

            bp = b.get("position")
            if not bp or len(bp) < 3:
                continue
            bx, by, bz = bp[0], bp[1], bp[2]

            # 1. Link capsules vs actual clamped box volume (samples along segment)
            for a, c, rad in link_segments:
                for t in (0.0, 0.25, 0.5, 0.75, 1.0):
                    px = a[0] + t * (c[0] - a[0])
                    py = a[1] + t * (c[1] - a[1])
                    pz = a[2] + t * (c[2] - a[2])
                    cx = min(max(px, bx - CUBE_HALF), bx + CUBE_HALF)
                    cy = min(max(py, by - CUBE_HALF), by + CUBE_HALF)
                    cz = min(max(pz, bz - CUBE_HALF), bz + CUBE_HALF)
                    dist = math.sqrt((px - cx)**2 + (py - cy)**2 + (pz - cz)**2)
                    if dist < (rad + safety_margin):
                        return True

            # 2. Fingers / Palm / TCP vs actual clamped box volume
            for pt, rad in finger_points:
                cx = min(max(pt[0], bx - CUBE_HALF), bx + CUBE_HALF)
                cy = min(max(pt[1], by - CUBE_HALF), by + CUBE_HALF)
                cz = min(max(pt[2], bz - CUBE_HALF), bz + CUBE_HALF)
                dist = math.sqrt((pt[0] - cx)**2 + (pt[1] - cy)**2 + (pt[2] - cz)**2)
                if dist < (rad + safety_margin):
                    return True

            # 3. Carried block vs obstacle blocks
            if grasped_block_id is not None:
                cx = min(max(p_tcp[0], bx - CUBE_HALF), bx + CUBE_HALF)
                cy = min(max(p_tcp[1], by - CUBE_HALF), by + CUBE_HALF)
                cz = min(max(p_tcp[2], bz - CUBE_HALF), bz + CUBE_HALF)
                dist_held = math.sqrt((p_tcp[0] - cx)**2 + (p_tcp[1] - cy)**2 + (p_tcp[2] - cz)**2)
                if dist_held < (CUBE_HALF + safety_margin):
                    return True

        return False


class RRTConnectJointPlanner:
    """
    Sampling-based Bidirectional RRT-Connect planner in 4-DOF Joint Space
    (baseYaw, shoulderPitch, elbowPitch, wristPitch).
    Generates collision-free trajectories avoiding obstacle blocks and workcell limits.
    """

    def __init__(self, step_size: float = 0.10, max_iterations: int = 250):
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.checker = ArmCollisionChecker()
        self.joint_names = ["baseYaw", "shoulderPitch", "elbowPitch", "wristPitch"]

    def _dict_to_array(self, q: Dict[str, float]) -> np.ndarray:
        return np.array([q[k] for k in self.joint_names], dtype=np.float32)

    def _array_to_dict(self, arr: np.ndarray, gripper: float = 0.0) -> Dict[str, float]:
        res = {k: float(arr[i]) for i, k in enumerate(self.joint_names)}
        res["gripper"] = float(gripper)
        return res

    def _sample_random_config(self) -> np.ndarray:
        sample = []
        for k in self.joint_names:
            low, high = JOINT_LIMITS[k]
            sample.append(random.uniform(low, high))
        return np.array(sample, dtype=np.float32)

    def swept_segment_valid(
        self,
        q1: np.ndarray,
        q2: np.ndarray,
        blocks: List[Dict[str, Any]],
        grasped_block_id: Optional[str] = None,
        ignore_block_ids: Optional[Set[str]] = None,
        step_rad: float = 0.03,
    ) -> bool:
        """Validates swept path between two joint configurations in fine increments."""
        dist = float(np.linalg.norm(q2 - q1))
        steps = max(2, int(math.ceil(dist / step_rad)))
        for i in range(steps + 1):
            alpha = i / float(steps)
            q_interp = q1 + alpha * (q2 - q1)
            j_dict = self._array_to_dict(q_interp)
            if self.checker.check_configuration_collision(
                joints=j_dict,
                blocks=blocks,
                grasped_block_id=grasped_block_id,
                ignore_block_ids=ignore_block_ids,
            ):
                return False
        return True

    def plan(
        self,
        q_start: Dict[str, float],
        q_goal: Dict[str, float],
        blocks: List[Dict[str, Any]],
        grasped_block_id: Optional[str] = None,
        ignore_block_ids: Optional[Set[str]] = None,
    ) -> List[Dict[str, float]]:
        """
        Plans a collision-free path from q_start to q_goal.
        Fast-path: direct joint linear interpolation if collision-free.
        Fallback: Bidirectional RRT-Connect with path shortcutting.
        """
        arr_start = self._dict_to_array(q_start)
        arr_goal = self._dict_to_array(q_goal)
        grip_val = q_start.get("gripper", 0.0)

        # 1. Fast-Path: Direct swept segment check
        if self.swept_segment_valid(arr_start, arr_goal, blocks, grasped_block_id, ignore_block_ids):
            return [q_start, q_goal]

        # 2. Bidirectional RRT-Connect with explicit start and goal trees
        tree_start = [arr_start]
        parent_start = {0: -1}
        tree_goal = [arr_goal]
        parent_goal = {0: -1}

        # Reference pointers for alternating tree expansion
        t_a, p_a = tree_start, parent_start
        t_b, p_b = tree_goal, parent_goal
        swapped = False

        for _ in range(self.max_iterations):
            # Goal biasing: 15% probability of sampling the target configuration directly
            if random.random() < 0.15:
                q_rand = arr_goal if not swapped else arr_start
            else:
                q_rand = self._sample_random_config()

            # Extend tree A toward q_rand
            dists = [np.linalg.norm(q - q_rand) for q in t_a]
            idx_near_a = int(np.argmin(dists))
            q_near_a = t_a[idx_near_a]
            dir_a = q_rand - q_near_a
            dist_a = np.linalg.norm(dir_a)
            if dist_a < 1e-6:
                continue

            q_new_a = q_near_a + min(self.step_size, dist_a) * (dir_a / dist_a)

            if not self.swept_segment_valid(q_near_a, q_new_a, blocks, grasped_block_id, ignore_block_ids):
                continue

            idx_new_a = len(t_a)
            t_a.append(q_new_a)
            p_a[idx_new_a] = idx_near_a

            # Greedily connect tree B toward q_new_a until reached or collision
            dists_b = [np.linalg.norm(q - q_new_a) for q in t_b]
            idx_curr_b = int(np.argmin(dists_b))
            q_curr_b = t_b[idx_curr_b]
            connected = False

            while True:
                dir_b = q_new_a - q_curr_b
                dist_b = np.linalg.norm(dir_b)
                if dist_b < 1e-5:
                    break
                step_b = min(self.step_size, dist_b)
                q_step_b = q_curr_b + step_b * (dir_b / dist_b)

                if not self.swept_segment_valid(q_curr_b, q_step_b, blocks, grasped_block_id, ignore_block_ids):
                    break

                idx_new_b = len(t_b)
                t_b.append(q_step_b)
                p_b[idx_new_b] = idx_curr_b
                q_curr_b = q_step_b
                idx_curr_b = idx_new_b

                if np.linalg.norm(q_new_a - q_step_b) < 0.04 or self.swept_segment_valid(
                    q_new_a, q_step_b, blocks, grasped_block_id, ignore_block_ids
                ):
                    connected = True
                    break

            if connected:
                # Connection successful! Build path from start to goal
                if not swapped:
                    idx_start_junction = idx_new_a
                    idx_goal_junction = idx_curr_b
                else:
                    idx_start_junction = idx_curr_b
                    idx_goal_junction = idx_new_a

                    # Path from start to junction
                    path_from_start = []
                    curr = idx_start_junction
                    while curr != -1:
                        path_from_start.append(tree_start[curr])
                        curr = parent_start[curr]
                    path_from_start.reverse()  # now [arr_start, ..., junction]

                    # Path from junction to goal
                    path_to_goal = []
                    curr = idx_goal_junction
                    while curr != -1:
                        path_to_goal.append(tree_goal[curr])
                        curr = parent_goal[curr]
                    # path_to_goal is already [junction, ..., arr_goal]

                    full_path = path_from_start + path_to_goal

                    # Short-cut smoothing
                    smoothed = self._smooth_path(full_path, blocks, grasped_block_id, ignore_block_ids)
                    return [self._array_to_dict(q, grip_val) for q in smoothed]

            # Alternate tree roles for symmetric bidirectional growth
            t_a, t_b = t_b, t_a
            p_a, p_b = p_b, p_a
            swapped = not swapped

        # Fallback to direct path if RRT reached max iterations
        return [q_start, q_goal]

    def _smooth_path(
        self,
        path: List[np.ndarray],
        blocks: List[Dict[str, Any]],
        grasped_block_id: Optional[str],
        ignore_block_ids: Optional[Set[str]],
        max_shortcut_passes: int = 15,
    ) -> List[np.ndarray]:
        """Shortens and smooths the piecewise linear joint path."""
        if len(path) <= 2:
            return path
        res = list(path)
        for _ in range(max_shortcut_passes):
            if len(res) <= 2:
                break
            i = random.randint(0, len(res) - 2)
            j = random.randint(i + 1, len(res) - 1)
            if j - i <= 1:
                continue
            if self.swept_segment_valid(res[i], res[j], blocks, grasped_block_id, ignore_block_ids):
                res = res[: i + 1] + res[j:]
        return res

    def interpolate_trajectory(
        self,
        path: List[Dict[str, float]],
        max_step_rad: float = 0.04,
    ) -> List[Dict[str, float]]:
        """Densely interpolates a joint waypoint path into fine-grained motion steps."""
        if len(path) <= 1:
            return path
        dense: List[Dict[str, float]] = [path[0]]
        for i in range(len(path) - 1):
            q_a = path[i]
            q_b = path[i + 1]
            grip = q_b.get("gripper", q_a.get("gripper", 0.0))
            max_d = max(abs(q_b[k] - q_a[k]) for k in self.joint_names)
            steps = max(1, int(math.ceil(max_d / max_step_rad)))
            for s in range(1, steps + 1):
                alpha = s / float(steps)
                interp = {k: q_a[k] + alpha * (q_b[k] - q_a[k]) for k in self.joint_names}
                interp["gripper"] = float(grip)
                dense.append(interp)
        return dense


def interpolate_joint_trajectory(
    path: List[Dict[str, float]],
    max_step_rad: float = 0.04,
) -> List[Dict[str, float]]:
    """Top-level helper to densely interpolate a joint waypoint trajectory."""
    planner = RRTConnectJointPlanner()
    return planner.interpolate_trajectory(path, max_step_rad=max_step_rad)


def plan_collision_free_transit(
    start_joints: Dict[str, float],
    goal_joints: Dict[str, float],
    blocks: List[Dict[str, Any]],
    grasped_block_id: Optional[str] = None,
    ignore_block_ids: Optional[Set[str]] = None,
    max_step_rad: float = 0.04,
) -> List[Dict[str, float]]:
    """
    Plans a collision-free transit trajectory between two joint configurations.
    Uses fast-path linear swept segment if unobstructed, or RRT-Connect obstacle detour if blocked.
    Densely interpolates the resulting trajectory for continuous kinematic tracking.
    """
    planner = RRTConnectJointPlanner()
    raw_path = planner.plan(
        start_joints,
        goal_joints,
        blocks=blocks,
        grasped_block_id=grasped_block_id,
        ignore_block_ids=ignore_block_ids,
    )
    return interpolate_joint_trajectory(raw_path, max_step_rad=max_step_rad)



def plan_cartesian_transit(
    start_tcp: Tuple[float, float, float] | List[float],
    goal_tcp: Tuple[float, float, float] | List[float],
    transit_y: float = SAFE_HOVER_Y,
    max_step: float = MAX_TCP_STEP,
    gripper: float = 0.0,
    candidate_pitches: Optional[List[float]] = None,
) -> List[Dict[str, float]]:
    """
    Plans an altitude-locked linear Cartesian transit between start_tcp and goal_tcp.
    Enforces that altitude remains locked at transit_y for the entire flight, preventing
    joint-space sagging (which can dip 130-180mm into tabletop blocks).
    Solves IK for each intermediate waypoint.
    """
    sx, _, sz = start_tcp[0], start_tcp[1], start_tcp[2]
    gx, _, gz = goal_tcp[0], goal_tcp[1], goal_tcp[2]

    dist = math.hypot(gx - sx, gz - sz)
    n_steps = max(2, int(math.ceil(dist / max(1e-4, max_step))))

    # Prioritize pitch continuity from start configuration
    if candidate_pitches is None:
        start_ik = solve_ik((sx, transit_y, sz), gripper=gripper)
        if start_ik.success:
            s_pitch = round(start_ik.joints["shoulderPitch"] + start_ik.joints["elbowPitch"] + start_ik.joints["wristPitch"], 4)
            candidate_pitches = [s_pitch, -1.25, -1.0, -0.60, WRIST_WORLD_PITCH]

    trajectory: List[Dict[str, float]] = []
    for i in range(n_steps + 1):
        alpha = i / float(n_steps)
        x = sx + alpha * (gx - sx)
        z = sz + alpha * (gz - sz)
        target = (x, transit_y, z)

        ik_res = solve_ik(target, gripper=gripper, candidate_pitches=candidate_pitches)
        j_dict = dict(ik_res.joints)
        j_dict["gripper"] = float(gripper)
        j_dict["success"] = ik_res.success
        trajectory.append(j_dict)

    return trajectory


def plan_vertical_approach(
    hover_target: Tuple[float, float, float] | List[float],
    pick_target: Tuple[float, float, float] | List[float],
    steps: int = 12,
    gripper: float = 0.0,
    candidate_pitches: Optional[List[float]] = None,
) -> List[Dict[str, float]]:
    """
    Plans a pure vertical descent trajectory from hover altitude down to pick/place target.
    Locks X and Z coordinates to pick_target to eliminate horizontal drift and prevent
    clipping neighboring blocks during jaw insertion.
    Locks end-effector pitch consistently across the descent to prevent mid-flight branch flips.
    """
    px, py_goal, pz = pick_target[0], pick_target[1], pick_target[2]
    py_start = hover_target[1]
    n_steps = max(2, steps)

    if candidate_pitches is None:
        target_ik = solve_ik(pick_target, gripper=gripper)
        if target_ik.success:
            locked_world_pitch = round(
                target_ik.joints["shoulderPitch"] + target_ik.joints["elbowPitch"] + target_ik.joints["wristPitch"], 4
            )
            candidate_pitches = [locked_world_pitch]

    trajectory: List[Dict[str, float]] = []
    for i in range(n_steps + 1):
        alpha = i / float(n_steps)
        y = py_start + alpha * (py_goal - py_start)
        target = (px, y, pz)

        ik_res = solve_ik(target, gripper=gripper, candidate_pitches=candidate_pitches)
        j_dict = dict(ik_res.joints)
        j_dict["gripper"] = float(gripper)
        j_dict["success"] = ik_res.success
        trajectory.append(j_dict)

    return trajectory


def plan_vertical_retreat(
    place_target: Tuple[float, float, float] | List[float],
    transit_y: float = SAFE_HOVER_Y,
    steps: int = 12,
    desired_gripper: float = 0.70,
    candidate_pitches: Optional[List[float]] = None,
) -> List[Dict[str, float]]:
    """
    Plans a pure vertical ascent trajectory from place target up to transit altitude.
    Locks X and Z coordinates and maintains calibrated gripper aperture (desired_gripper=0.70)
    to safely disengage from placed blocks without sweeping flanking blocks.
    Locks end-effector pitch consistently across the ascent to prevent mid-flight branch flips.
    """
    px, py_start, pz = place_target[0], place_target[1], place_target[2]
    n_steps = max(2, steps)

    if candidate_pitches is None:
        target_ik = solve_ik(place_target, gripper=desired_gripper)
        if target_ik.success:
            locked_world_pitch = round(
                target_ik.joints["shoulderPitch"] + target_ik.joints["elbowPitch"] + target_ik.joints["wristPitch"], 4
            )
            candidate_pitches = [locked_world_pitch]

    trajectory: List[Dict[str, float]] = []
    for i in range(n_steps + 1):
        alpha = i / float(n_steps)
        y = py_start + alpha * (transit_y - py_start)
        target = (px, y, pz)

        ik_res = solve_ik(target, gripper=desired_gripper, candidate_pitches=candidate_pitches)
        j_dict = dict(ik_res.joints)
        j_dict["gripper"] = float(desired_gripper)
        j_dict["success"] = ik_res.success
        trajectory.append(j_dict)

    return trajectory


