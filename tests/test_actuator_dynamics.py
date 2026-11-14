"""
tests/test_actuator_dynamics.py

Unit and integration tests for Feetech STS3215 Actuator & Physics Model in AERO.
Verifies torque limits, electrical current estimation, load %, Joule heating thermal dissipation,
power consumption, and integration with the RobotArmSimulation engine.
"""

import math
import pytest
from simulation.constants import DEFAULT_JOINTS, FIXED_DT
from simulation.actuator_model import ActuatorModel, STS3215_SPECS
from simulation.physics_engine import RobotArmSimulation


def test_actuator_model_initial_state():
    """Verify actuator model initializes with default ambient temperatures and idle currents."""
    model = ActuatorModel()
    telemetry = model.get_telemetry()

    assert telemetry["voltage_v"] == 7.4
    assert telemetry["total_power_watts"] > 0.0
    for joint in model.joint_names:
        assert telemetry["temperatures_c"][joint] == pytest.approx(STS3215_SPECS["ambient_temp_c"], rel=1e-2)
        assert telemetry["currents_ma"][joint] == 80  # Quiescent idle current: 80 mA
        assert telemetry["torques"][joint] == 0.0
        assert telemetry["loads_pct"][joint] == 0.0


def test_torque_clamping_limits():
    """Verify raw calculated torques are strictly clamped to STS3215 stall limits."""
    model = ActuatorModel()

    # Extreme instant acceleration to generate high inertial torque
    prev_j = {"baseYaw": 0.0, "shoulderPitch": 0.0, "elbowPitch": 0.0, "wristPitch": 0.0, "wristRoll": 0.0, "gripper": 0.0}
    next_j = {"baseYaw": 2.0, "shoulderPitch": 2.0, "elbowPitch": 2.0, "wristPitch": 2.0, "wristRoll": 2.0, "gripper": 1.0}

    # Step with very small dt to produce massive acceleration
    telemetry = model.sample(prev_j, next_j, dt=0.001, has_grasped_block=True)

    # Major joints must not exceed 2.80 Nm
    assert telemetry["torques"]["baseYaw"] <= STS3215_SPECS["torque_stall_shoulder"]
    assert telemetry["torques"]["shoulderPitch"] <= STS3215_SPECS["torque_stall_shoulder"]
    assert telemetry["torques"]["elbowPitch"] <= STS3215_SPECS["torque_stall_shoulder"]

    # Wrist & Gripper must not exceed 1.90 Nm
    assert telemetry["torques"]["wristPitch"] <= STS3215_SPECS["torque_stall_wrist"]
    assert telemetry["torques"]["wristRoll"] <= STS3215_SPECS["torque_stall_wrist"]
    assert telemetry["torques"]["gripper"] <= STS3215_SPECS["torque_stall_wrist"]


def test_payload_increases_shoulder_torque():
    """Verify that carrying a grasped block increases gravitational torque on shoulder & elbow."""
    model_empty = ActuatorModel()
    model_loaded = ActuatorModel()

    angled_pose = {
        "baseYaw": 0.0,
        "shoulderPitch": 0.8, # Angled arm (non-saturated torque)
        "elbowPitch": -0.4,
        "wristPitch": 0.2,
        "wristRoll": 0.0,
        "gripper": 0.9,
    }

    t_empty = model_empty.sample(angled_pose, angled_pose, dt=FIXED_DT, has_grasped_block=False)
    t_loaded = model_loaded.sample(angled_pose, angled_pose, dt=FIXED_DT, has_grasped_block=True)

    # Loaded arm must experience higher gravity moment on shoulder and elbow
    assert t_loaded["torques"]["shoulderPitch"] > t_empty["torques"]["shoulderPitch"]
    assert t_loaded["torques"]["elbowPitch"] > t_empty["torques"]["elbowPitch"]
    assert t_loaded["currents_ma"]["shoulderPitch"] > t_empty["currents_ma"]["shoulderPitch"]
    assert t_loaded["total_power_watts"] > t_empty["total_power_watts"]


def test_thermal_dissipation_and_heating():
    """Verify Joule heating warms actuators under sustained load, and cooling occurs toward ambient."""
    model = ActuatorModel()
    initial_temp = model.temperatures["shoulderPitch"]

    pose = {
        "baseYaw": 0.0,
        "shoulderPitch": 0.0,  # Max gravity torque
        "elbowPitch": 0.0,
        "wristPitch": 0.0,
        "wristRoll": 0.0,
        "gripper": 0.9,
    }

    # Simulate 500 steps of sustained holding
    for _ in range(500):
        telemetry = model.sample(pose, pose, dt=FIXED_DT, has_grasped_block=True)

    heated_temp = telemetry["temperatures_c"]["shoulderPitch"]
    assert heated_temp > initial_temp

    # Now simulate resting vertical pose to observe cooling toward ambient
    rest_pose = {
        "baseYaw": 0.0,
        "shoulderPitch": 1.5708,  # Vertical straight (zero gravity cantilever)
        "elbowPitch": 0.0,
        "wristPitch": 0.0,
        "wristRoll": 0.0,
        "gripper": 0.0,
    }
    for _ in range(1200):
        telemetry = model.sample(rest_pose, rest_pose, dt=FIXED_DT, has_grasped_block=False)

    cooled_temp = telemetry["temperatures_c"]["shoulderPitch"]
    assert cooled_temp < heated_temp


def test_simulation_integration_actuators_telemetry():
    """Verify RobotArmSimulation integrates ActuatorModel and outputs telemetry in snapshots."""
    sim = RobotArmSimulation()
    snapshot = sim.get_snapshot()

    assert "actuators" in snapshot
    act = snapshot["actuators"]
    assert "torques" in act
    assert "loads_pct" in act
    assert "currents_ma" in act
    assert "temperatures_c" in act
    assert "total_power_watts" in act
    assert "voltage_v" in act

    # Step simulation with arm motion
    sim.set_joint_targets({"shoulderPitch": 0.5, "elbowPitch": -0.4, "wristRoll": 0.3})
    for _ in range(15):
        snapshot = sim.step(FIXED_DT)

    act_stepped = snapshot["actuators"]
    assert act_stepped["peak_torque"] > 0.0
    assert act_stepped["total_power_watts"] > 0.5


def test_simulation_reset_restores_actuator_state():
    """Verify sim.reset() resets actuator temperatures and peak tracking."""
    sim = RobotArmSimulation()
    sim.set_joint_targets({"shoulderPitch": 0.8, "elbowPitch": -0.8})
    for _ in range(30):
        sim.step(FIXED_DT)

    sim.reset()
    snap = sim.get_snapshot()
    act = snap["actuators"]
    assert act["peak_torque"] == 0.0
    for joint in sim.actuator_model.joint_names:
        assert act["torques"][joint] == 0.0
        assert act["temperatures_c"][joint] == pytest.approx(STS3215_SPECS["ambient_temp_c"], rel=1e-2)
