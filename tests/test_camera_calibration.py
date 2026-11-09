"""
Unit tests for Camera Calibration, Coordinate Frames & Sensor Interfaces.
Validates Stage 1 requirements of AERO perception foundation.
"""

import math
import numpy as np
import pytest

from simulation.constants import TABLE_TOP_Y, STACK_ORIGIN, VLA_IMAGE_SIZE, VLA_X_MIN, VLA_X_MAX, VLA_Z_MIN, VLA_Z_MAX
from simulation.coordinate_frames import (
    CoordinateFrameManager,
    OVERHEAD_CAMERA_HEIGHT,
    make_transform,
    inverse_transform,
    transform_point,
    transform_vector,
    euler_to_rot_matrix,
)
from simulation.sensor_interfaces import CameraCalibration, CameraObservation, RobotStateObservation
from simulation.camera import SimulationCamera, SimulationRobotStateSensor
from simulation.physics_engine import RobotArmSimulation


class TestCoordinateFramesAndCalibration:
    def test_01_rigid_transform_and_inverse(self):
        """Validates that T @ T_inv = Identity for arbitrary translations and rotations."""
        trans = [0.15, -0.42, 0.88]
        rot = (0.3, -0.5, 0.7)  # roll, pitch, yaw
        T = make_transform(trans, rot)
        T_inv = inverse_transform(T)

        ident = T @ T_inv
        np.testing.assert_allclose(ident, np.eye(4), atol=1e-6)

        # Roundtrip point transformation
        pt = np.array([0.25, 0.75, -0.10])
        pt_trans = transform_point(pt, T)
        pt_restored = transform_point(pt_trans, T_inv)
        np.testing.assert_allclose(pt, pt_restored, atol=1e-6)

    def test_02_coordinate_frame_manager_landmarks(self):
        """Verifies landmark coordinate mapping between base, table, and camera frames."""
        mgr = CoordinateFrameManager(camera_height=OVERHEAD_CAMERA_HEIGHT)

        # Table surface point in table frame [0, 0, 0] must map to [0, TABLE_TOP_Y, 0] in base
        pt_table = (0.0, 0.0, 0.0)
        pt_base = mgr.table_to_base(pt_table)
        assert abs(pt_base[0] - 0.0) < 1e-5
        assert abs(pt_base[1] - TABLE_TOP_Y) < 1e-5
        assert abs(pt_base[2] - 0.0) < 1e-5

        # Restoring from base to table
        pt_table_restored = mgr.base_to_table(pt_base)
        np.testing.assert_allclose(pt_table_restored, pt_table, atol=1e-5)

        # Camera frame origin [0, 0, 0] in camera frame maps to [0, OVERHEAD_CAMERA_HEIGHT, 0] in base
        cam_origin_base = mgr.camera_to_base((0.0, 0.0, 0.0))
        assert abs(cam_origin_base[1] - OVERHEAD_CAMERA_HEIGHT) < 1e-5

    def test_03_camera_calibration_projection_roundtrip(self):
        """Tests that pixel_to_world and world_to_pixel invert accurately across workspace."""
        calib = CameraCalibration(
            model_type="orthographic",
            width=VLA_IMAGE_SIZE,
            height=VLA_IMAGE_SIZE,
            bounds_x=(VLA_X_MIN, VLA_X_MAX),
            bounds_z=(VLA_Z_MIN, VLA_Z_MAX),
            camera_height=OVERHEAD_CAMERA_HEIGHT
        )

        test_points = [
            (0.0, 0.0),
            (STACK_ORIGIN[0], STACK_ORIGIN[2]),
            (0.20, -0.15),
            (-0.18, 0.22),
        ]

        for x_gt, z_gt in test_points:
            u, v = calib.world_to_pixel(x_gt, z_gt)
            x_est, z_est = calib.pixel_to_world(u, v)
            # Grid resolution quantization tolerance: ~ (0.70m / 128) ≈ 0.0055m
            assert abs(x_est - x_gt) < 0.01
            assert abs(z_est - z_gt) < 0.01

    def test_04_camera_depth_to_height(self):
        """Verifies depth to table-height conversion."""
        calib = CameraCalibration(
            model_type="orthographic",
            width=128,
            height=128,
            bounds_x=(-0.35, 0.35),
            bounds_z=(-0.35, 0.35),
            camera_height=1.60
        )
        # Depth for tabletop
        expected_depth = 1.60 - TABLE_TOP_Y
        assert abs(calib.y_to_depth(TABLE_TOP_Y) - expected_depth) < 1e-5
        assert abs(calib.depth_to_y(expected_depth) - TABLE_TOP_Y) < 1e-5

    def test_05_simulation_camera_and_robot_state_sensor(self):
        """Verifies SimulationCamera and SimulationRobotStateSensor enforce observational boundaries."""
        sim = RobotArmSimulation()
        sim.reset()

        cam = SimulationCamera(sim, size=128)
        assert cam.is_available()
        obs = cam.capture()

        assert isinstance(obs, CameraObservation)
        assert obs.rgb.size == (128, 128)
        assert obs.depth is not None
        assert obs.depth.shape == (128, 128)
        assert obs.is_valid is True

        # Check depth on table center (u=64, v=64)
        center_depth = obs.depth[64, 64]
        assert abs(center_depth - (OVERHEAD_CAMERA_HEIGHT - TABLE_TOP_Y)) < 0.05

        # Check RobotStateSensor
        sensor = SimulationRobotStateSensor(sim)
        state = sensor.get_state()
        assert isinstance(state, RobotStateObservation)
        assert len(state.joint_positions) == 5
        assert state.tcp_pose_estimated is not None
        assert abs(state.tcp_pose_estimated[1] - 0.80) < 0.10  # default TCP altitude near 0.8m

        # CRITICAL: Verify observation does NOT expose hidden simulator block dictionaries
        assert not hasattr(obs, "blocks")
        assert not hasattr(state, "blocks")
        assert not hasattr(state, "grasped_block_id")
