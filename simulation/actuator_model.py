"""
simulation/actuator_model.py

Feetech STS3215 Smart Serial Bus Servo Physics & Actuator Model for AERO.
Simulates real-world motor torque limits, electrical current (mA), load %,
thermal dissipation, and power consumption for the SO-ARM100 digital twin.
"""

import math
from typing import Dict, Any, Optional
from .constants import (
    L_UPPER,
    L_FOREARM,
    L_WRIST,
    BLOCK_MASS,
)
from .kinematics import clamp

# Technical specifications for Feetech STS3215 smart serial bus servo
STS3215_SPECS = {
    "voltage_nominal": 7.4,         # Volts (2S LiPo / Bench DC Power Supply)
    "torque_stall_shoulder": 2.80,  # Nm (approx 28.5 kg·cm @ 7.4V)
    "torque_stall_wrist": 1.90,     # Nm (approx 19.4 kg·cm @ 7.4V)
    "current_stall": 2.50,          # Amperes (2500 mA)
    "current_idle": 0.08,           # Amperes (80 mA quiescent draw)
    "torque_constant_kt": 1.25,     # Nm / A
    "winding_resistance": 3.8,      # Ohms
    "max_velocity_rad_s": 6.28,     # rad/s (~60 RPM)
    "thermal_dissipation": 0.05,    # Cooling coefficient per second
    "ambient_temp_c": 24.5,         # Ambient room temperature
}

# SO-ARM100 3D printed PETG link masses (kg) & center of mass fractions
LINK_MASSES = {
    "shoulder": 0.165,
    "upper_arm": 0.285,
    "forearm": 0.190,
    "wrist": 0.125,
    "gripper": 0.145,
}


class ActuatorModel:
    """
    Simulates the 6-DOF daisy-chained Feetech STS3215 bus servos on the SO-ARM100.
    Tracks torque, current, load %, temperature, and power for each individual joint.
    """

    def __init__(self):
        self.joint_names = ["baseYaw", "shoulderPitch", "elbowPitch", "wristPitch", "wristRoll", "gripper"]
        self.prev_vel: Dict[str, float] = {k: 0.0 for k in self.joint_names}
        self.temperatures: Dict[str, float] = {k: STS3215_SPECS["ambient_temp_c"] for k in self.joint_names}
        self.torques: Dict[str, float] = {k: 0.0 for k in self.joint_names}
        self.raw_torques: Dict[str, float] = {k: 0.0 for k in self.joint_names}
        self.currents_ma: Dict[str, int] = {k: 80 for k in self.joint_names}
        self.loads_pct: Dict[str, float] = {k: 0.0 for k in self.joint_names}
        self.peak_torque: float = 0.0
        self.total_power_watts: float = 0.6
        self.sample_count: int = 0
        self.torque_sum: float = 0.0

    def reset(self):
        self.prev_vel = {k: 0.0 for k in self.joint_names}
        self.temperatures = {k: STS3215_SPECS["ambient_temp_c"] for k in self.joint_names}
        self.torques = {k: 0.0 for k in self.joint_names}
        self.raw_torques = {k: 0.0 for k in self.joint_names}
        self.currents_ma = {k: 80 for k in self.joint_names}
        self.loads_pct = {k: 0.0 for k in self.joint_names}
        self.peak_torque = 0.0
        self.total_power_watts = 0.6
        self.sample_count = 0
        self.torque_sum = 0.0

    def sample(
        self,
        prev_joints: Dict[str, float],
        next_joints: Dict[str, float],
        dt: float = 1.0 / 60.0,
        has_grasped_block: bool = False,
    ) -> Dict[str, Any]:
        """
        Samples joint movements, updates dynamic torques, currents, loads, and heat dissipation.
        """
        dt = max(1e-4, dt)
        g = 9.80665

        # 1. Joint Velocities and Accelerations
        vel: Dict[str, float] = {}
        acc: Dict[str, float] = {}
        for k in self.joint_names:
            v = (next_joints.get(k, 0.0) - prev_joints.get(k, 0.0)) / dt
            a = (v - self.prev_vel.get(k, 0.0)) / dt
            vel[k] = v
            acc[k] = a
            self.prev_vel[k] = v

        # 2. Gravity Moments & Effective Mass Leverage
        # Link positions from shoulder joint:
        q_sh = next_joints.get("shoulderPitch", 0.0)
        q_el = next_joints.get("elbowPitch", 0.0)
        q_wr = next_joints.get("wristPitch", 0.0)

        arm_pitch_world = q_sh
        forearm_pitch_world = q_sh + q_el
        wrist_pitch_world = q_sh + q_el + q_wr

        m_block = BLOCK_MASS if has_grasped_block else 0.0
        m_hand = LINK_MASSES["wrist"] + LINK_MASSES["gripper"] + m_block

        # Static gravity torques (tau = r x F):
        tau_wrist = m_hand * g * (L_WRIST * 0.5) * math.cos(wrist_pitch_world)

        r_forearm_cg = L_UPPER + L_FOREARM * 0.5
        r_hand_cg = L_UPPER + L_FOREARM + L_WRIST * 0.5
        tau_elbow = (
            LINK_MASSES["forearm"] * g * (L_FOREARM * 0.5) * math.cos(forearm_pitch_world)
            + m_hand * g * (L_FOREARM + L_WRIST * 0.5) * math.cos(wrist_pitch_world)
        )

        tau_shoulder = (
            LINK_MASSES["upper_arm"] * g * (L_UPPER * 0.5) * math.cos(arm_pitch_world)
            + LINK_MASSES["forearm"] * g * (L_UPPER * math.cos(arm_pitch_world) + L_FOREARM * 0.5 * math.cos(forearm_pitch_world))
            + m_hand * g * (L_UPPER * math.cos(arm_pitch_world) + L_FOREARM * math.cos(forearm_pitch_world) + L_WRIST * 0.5 * math.cos(wrist_pitch_world))
        )

        # Dynamic Inertial Torques: tau = I * alpha
        # Approximate rotational inertia around joint pivots (kg·m²):
        i_base = 0.018
        i_shoulder = 0.045
        i_elbow = 0.024
        i_wrist = 0.008
        i_roll = 0.003
        i_grip = 0.004

        raw_torques = {
            "baseYaw": abs(i_base * acc["baseYaw"]) + 0.12 * abs(vel["baseYaw"]),
            "shoulderPitch": abs(tau_shoulder) + abs(i_shoulder * acc["shoulderPitch"]) + 0.25 * abs(vel["shoulderPitch"]),
            "elbowPitch": abs(tau_elbow) + abs(i_elbow * acc["elbowPitch"]) + 0.18 * abs(vel["elbowPitch"]),
            "wristPitch": abs(tau_wrist) + abs(i_wrist * acc["wristPitch"]) + 0.10 * abs(vel["wristPitch"]),
            "wristRoll": abs(i_roll * acc["wristRoll"]) + 0.04 * abs(vel["wristRoll"]),
            "gripper": (0.65 if has_grasped_block else 0.08) + abs(i_grip * acc["gripper"]),
        }

        # 3. Apply Feetech STS3215 Torque Limits & Electrical Modeling
        total_current_a = 0.0
        for k in self.joint_names:
            limit = (
                STS3215_SPECS["torque_stall_shoulder"]
                if k in ("baseYaw", "shoulderPitch", "elbowPitch")
                else STS3215_SPECS["torque_stall_wrist"]
            )
            self.raw_torques[k] = raw_torques[k]
            clamped_tau = clamp(raw_torques[k], 0.0, limit)
            self.torques[k] = clamped_tau

            # Current calculation: I = |tau| / kt + I_idle
            c_amps = (clamped_tau / STS3215_SPECS["torque_constant_kt"]) + STS3215_SPECS["current_idle"]
            c_amps = clamp(c_amps, 0.08, STS3215_SPECS["current_stall"])
            self.currents_ma[k] = int(round(c_amps * 1000.0))
            total_current_a += c_amps

            # Motor load percentage
            self.loads_pct[k] = round(min(100.0, (clamped_tau / limit) * 100.0), 1)

            # Thermal model (Joule heating I²R - cooling to ambient)
            p_heat = (c_amps ** 2) * STS3215_SPECS["winding_resistance"] * 0.015
            t_diff = self.temperatures[k] - STS3215_SPECS["ambient_temp_c"]
            cooling_rate = t_diff * STS3215_SPECS["thermal_dissipation"]
            self.temperatures[k] = clamp(self.temperatures[k] + (p_heat - cooling_rate) * dt, 24.5, 75.0)

        # 4. Total Power Consumption (Watts = V * I)
        self.total_power_watts = round(STS3215_SPECS["voltage_nominal"] * total_current_a, 2)

        # Update peak and average tracking
        curr_peak = max(self.torques.values())
        if curr_peak > self.peak_torque:
            self.peak_torque = round(curr_peak, 3)

        self.sample_count += 1
        self.torque_sum += curr_peak

        return self.get_telemetry()

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns structured actuator telemetry conforming to dashboard WebSocket payload."""
        avg_torque = round(self.torque_sum / max(1, self.sample_count), 3)
        return {
            "torques": {k: round(v, 3) for k, v in self.torques.items()},
            "raw_torques": {k: round(v, 3) for k, v in self.raw_torques.items()},
            "loads_pct": dict(self.loads_pct),
            "currents_ma": dict(self.currents_ma),
            "temperatures_c": {k: round(v, 1) for k, v in self.temperatures.items()},
            "total_power_watts": self.total_power_watts,
            "peak_torque": self.peak_torque,
            "avg_torque": avg_torque,
            "voltage_v": STS3215_SPECS["voltage_nominal"],
        }
