"""
tests/test_so100_kinematics.py

Unit tests for 6-DOF Kinematics, Yaw-Oriented Grasping, and SO-ARM100 Geometry.
Validates forward kinematics, inverse kinematics, reachability, and joint limits.
"""

import math
import pytest
from simulation.constants import DEFAULT_JOINTS, JOINT_LIMITS, ARM_MOUNT, TABLE_TOP_Y
from simulation.kinematics import forward_kinematics, inverse_kinematics, solve_ik, compute_manipulability


class TestSO100Kinematics:
    def test_01_default_joints_has_wrist_roll(self):
        """Validates that 6-DOF joint dict includes wristRoll and matches home configuration."""
        assert "wristRoll" in DEFAULT_JOINTS
        assert DEFAULT_JOINTS["wristRoll"] == 0.0
        assert "wristRoll" in JOINT_LIMITS
        assert JOINT_LIMITS["wristRoll"][0] <= 0.0 <= JOINT_LIMITS["wristRoll"][1]

    def test_02_forward_kinematics_tcp_invariance_under_roll(self):
        """
        Rotating wristRoll around the local X approach axis must rotate the gripper jaws
        while leaving the TCP position along the approach axis invariant (zero translation error).
        """
        j_zero = dict(DEFAULT_JOINTS)
        j_zero["wristRoll"] = 0.0
        fk_zero = forward_kinematics(j_zero)
        tcp_zero = fk_zero["tcp"]["position"]

        for test_angle in [math.pi / 4, math.pi / 2, -math.pi / 3, math.pi]:
            j_rolled = dict(DEFAULT_JOINTS)
            j_rolled["wristRoll"] = test_angle
            fk_rolled = forward_kinematics(j_rolled)
            tcp_rolled = fk_rolled["tcp"]["position"]

            # TCP position must remain identical to sub-millimeter precision (< 1e-5m)
            dist = math.hypot(
                tcp_zero[0] - tcp_rolled[0],
                tcp_zero[1] - tcp_rolled[1],
                tcp_zero[2] - tcp_rolled[2]
            )
            assert dist < 1e-4, f"TCP shifted by {dist}m under wristRoll {test_angle}"

            # Jaw positions must rotate around the pinch axis
            jaw_l_zero = fk_zero["jawLeft"]["position"]
            jaw_l_rolled = fk_rolled["jawLeft"]["position"]
            jaw_shift = math.hypot(
                jaw_l_zero[0] - jaw_l_rolled[0],
                jaw_l_zero[1] - jaw_l_rolled[1],
                jaw_l_zero[2] - jaw_l_rolled[2]
            )
            assert jaw_shift > 0.01, "Jaw position did not rotate under wristRoll"

    def test_03_inverse_kinematics_reaches_tabletop_target(self):
        """Validates that solve_ik reaches standard manipulation coordinates within tolerance."""
        target = (0.25, TABLE_TOP_Y + 0.05, 0.0)
        res = solve_ik(target=target)
        assert res.success, f"Failed to reach target {target}: status={res.reachability_status}"
        assert res.position_error < 0.008

        # Check FK of solved joints matches target
        fk = forward_kinematics(res.joints)
        pos = fk["tcp"]["position"]
        err = math.hypot(pos[0] - target[0], pos[1] - target[1], pos[2] - target[2])
        assert err < 0.008

    def test_04_inverse_kinematics_solves_yaw_aligned_grasp(self):
        """
        When target_yaw is specified, wristRoll must orient the end-effector
        such that the gripper orientation matches target_yaw - baseYaw.
        """
        target = (0.30, TABLE_TOP_Y + 0.04, 0.15)
        for target_yaw in [0.0, math.pi / 4, -math.pi / 6, math.pi / 3]:
            res = solve_ik(target=target, target_yaw=target_yaw)
            assert res.success
            assert "wristRoll" in res.joints

            base_yaw = res.joints["baseYaw"]
            wrist_roll = res.joints["wristRoll"]
            expected_diff = math.atan2(math.sin(target_yaw - base_yaw), math.cos(target_yaw - base_yaw))
            assert abs(wrist_roll - expected_diff) < 1e-4

    def test_05_manipulability_nonzero_in_workspace(self):
        """Validates that Yoshikawa manipulability index is positive in typical reachable space."""
        j = dict(DEFAULT_JOINTS)
        w = compute_manipulability(j)
        assert w > 0.001, f"Manipulability too low or zero: {w}"

    def test_06_joint_limits_respected(self):
        """Validates that no IK solution violates joint limits for reachable table coordinates."""
        test_points = [
            (0.18, TABLE_TOP_Y + 0.03, -0.15),
            (0.35, TABLE_TOP_Y + 0.03, 0.0),
            (0.28, TABLE_TOP_Y + 0.12, 0.20),
            (0.42, TABLE_TOP_Y + 0.05, -0.10),
        ]
        for pt in test_points:
            res = solve_ik(target=pt)
            assert res.success, f"Point {pt} should be reachable"
            for k, val in res.joints.items():
                if k in JOINT_LIMITS:
                    low, high = JOINT_LIMITS[k]
                    assert low - 1e-4 <= val <= high + 1e-4, f"Joint {k}={val} exceeded limits [{low}, {high}]"

    def test_07_unreachable_target_handled_gracefully(self):
        """Validates that unreachable/extreme points return success=False without raising exceptions or NaNs."""
        extreme_targets = [
            (1.5, 0.85, 0.0),       # Far beyond forward reach (distance > 1.4m vs max reach 0.56m)
            (0.08, 2.5, 0.0),       # Far above maximum reach ceiling
            (0.08, 0.85, 1.8),      # Far beyond lateral side reach
        ]
        for pt in extreme_targets:
            res = solve_ik(target=pt)
            assert not res.success
            assert res.reachability_status in ("UNREACHABLE_POSITION", "JOINT_LIMIT_VIOLATION", "RESIDUAL_EXCEEDED")
            for k, val in res.joints.items():
                assert not math.isnan(val), f"Joint {k} is NaN for target {pt}"

    def test_08_forward_kinematics_handles_missing_wrist_roll(self):
        """Validates that legacy 5-DOF joint dictionary without wristRoll defaults to 0.0 roll cleanly."""
        legacy_joints = {
            "baseYaw": 0.2,
            "shoulderPitch": 0.5,
            "elbowPitch": -0.4,
            "wristPitch": 0.1,
            "gripper": 0.5,
        }
        fk = forward_kinematics(legacy_joints)
        assert "tcp" in fk
        assert "palm" in fk
        assert len(fk["tcp"]["position"]) == 3


