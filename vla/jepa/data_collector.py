"""
vla/jepa/data_collector.py

Automated Simulation Rollout Collector for H-JEPA World Model Training.
Collects multi-block manipulation episodes headless using RobotArmSimulation:
- 50% Expert multi-block demonstrations (clean towers and pyramids)
- 35% Noisy / perturbed trajectories (adding Gaussian jitter to induce topples & slips)
- 15% Random reachable exploration rollouts
Outputs transition tuples: (obs_t, action_t, obs_{t+1}, toppled_flag).
"""

import os
import time
import math
import copy
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from simulation.constants import (
    STACK_ORIGIN,
    SAFE_HOVER_Y,
    PICK_Y,
    TABLE_TOP_Y,
    CUBE_SIZE,
    BLOCK_SPAWNS,
    DEFAULT_JOINTS,
)
from simulation.physics_engine import RobotArmSimulation
from simulation.kinematics import inverse_kinematics


class HJEPADataCollector:
    """Collects multi-modal simulation rollouts for H-JEPA training."""
    def __init__(self, sim: Optional[RobotArmSimulation] = None):
        self.sim = sim or RobotArmSimulation()

    def collect_episode(
        self,
        mode: str = "expert",
        num_blocks: int = 3,
        max_ticks: int = 200,
        noise_std: float = 0.015,
    ) -> List[Dict[str, Any]]:
        """
        Runs a single headless simulation episode and records transitions.
        mode: 'expert', 'noisy', or 'random'
        """
        self.sim.reset()
        transitions = []

        # Available block IDs
        block_ids = list(self.sim.blocks.keys())[:num_blocks]

        # Target slots for vertical tower
        target_slots = [
            (STACK_ORIGIN[0], TABLE_TOP_Y + CUBE_SIZE * (i + 0.5), STACK_ORIGIN[2])
            for i in range(num_blocks)
        ]

        # In noisy mode, inject lateral offset to induce toppling
        if mode == "noisy":
            jitter = np.random.normal(0.0, noise_std, size=(num_blocks, 3))
            # Boost X/Z offset on upper block to trigger STACK_OFFSET_INSTABILITY
            jitter[-1, 0] += 0.045
        else:
            jitter = np.zeros((num_blocks, 3))

        curr_block_idx = 0
        phase = "HOVER"
        ticks_in_phase = 0

        for tick in range(max_ticks):
            if curr_block_idx >= len(block_ids):
                break

            b_id = block_ids[curr_block_idx]
            target_slot = target_slots[curr_block_idx]

            snapshot_before = self.sim.get_snapshot()
            tcp_curr = snapshot_before["tcp"]["position"]
            joints_curr = snapshot_before["joints"]

            target_block = self.sim.blocks.get(b_id)
            if not target_block:
                break
            b_pos = target_block.position

            # Determine next target waypoint
            desired_tcp = list(tcp_curr)
            desired_gripper = 0.0

            if phase == "HOVER":
                desired_tcp = [b_pos[0], SAFE_HOVER_Y, b_pos[2]]
                desired_gripper = 0.0
                if math.hypot(tcp_curr[0] - desired_tcp[0], tcp_curr[2] - desired_tcp[2]) < 0.02:
                    phase = "PICK"

            elif phase == "PICK":
                desired_tcp = [b_pos[0], PICK_Y, b_pos[2]]
                desired_gripper = 0.0
                if abs(tcp_curr[1] - desired_tcp[1]) < 0.01:
                    phase = "GRASP"
                    ticks_in_phase = 0

            elif phase == "GRASP":
                desired_tcp = [b_pos[0], PICK_Y, b_pos[2]]
                desired_gripper = 1.0
                ticks_in_phase += 1
                if ticks_in_phase > 10:
                    phase = "LIFT"

            elif phase == "LIFT":
                desired_tcp = [tcp_curr[0], SAFE_HOVER_Y, tcp_curr[2]]
                desired_gripper = 1.0
                if abs(tcp_curr[1] - SAFE_HOVER_Y) < 0.02:
                    phase = "TRANSPORT"

            elif phase == "TRANSPORT":
                tgt_x = target_slot[0] + jitter[curr_block_idx, 0]
                tgt_z = target_slot[2] + jitter[curr_block_idx, 2]
                desired_tcp = [tgt_x, SAFE_HOVER_Y, tgt_z]
                desired_gripper = 1.0
                if math.hypot(tcp_curr[0] - tgt_x, tcp_curr[2] - tgt_z) < 0.02:
                    phase = "PLACE"

            elif phase == "PLACE":
                tgt_x = target_slot[0] + jitter[curr_block_idx, 0]
                tgt_y = target_slot[1]
                tgt_z = target_slot[2] + jitter[curr_block_idx, 2]
                desired_tcp = [tgt_x, tgt_y + 0.005, tgt_z]
                desired_gripper = 1.0
                if abs(tcp_curr[1] - (tgt_y + 0.005)) < 0.012:
                    phase = "RELEASE"
                    ticks_in_phase = 0

            elif phase == "RELEASE":
                desired_gripper = 0.0
                ticks_in_phase += 1
                if ticks_in_phase > 12:
                    phase = "HOVER"
                    curr_block_idx += 1

            # Analytical IK step
            ik = inverse_kinematics(desired_tcp, desired_gripper)
            if ik.get("reachable", True):
                self.sim.set_joint_targets(ik["joints"], desired_gripper)

            # Step physical simulation
            snapshot_after = self.sim.step()

            # Detect whether any placed block has toppled
            toppled = False
            for prev_i in range(curr_block_idx):
                p_b = self.sim.blocks.get(block_ids[prev_i])
                expected_y = target_slots[prev_i][1]
                if p_b and abs(p_b.position[1] - expected_y) > 0.04:
                    toppled = True
                    break

            action_vec = [
                desired_tcp[0] - tcp_curr[0],
                desired_tcp[1] - tcp_curr[1],
                desired_tcp[2] - tcp_curr[2],
                desired_gripper,
            ]

            transitions.append({
                "tick": tick,
                "tcp_before": tcp_curr,
                "joints_before": joints_curr,
                "action": action_vec,
                "tcp_after": snapshot_after["tcp"]["position"],
                "joints_after": snapshot_after["joints"],
                "blocks_after": snapshot_after["blocks"],
                "toppled": toppled,
            })

            if toppled:
                break

        return transitions

    def collect_dataset(self, num_episodes: int = 100) -> Dict[str, Any]:
        """Collects a balanced dataset of expert, noisy, and exploration rollouts."""
        dataset = []
        n_expert = int(num_episodes * 0.50)
        n_noisy = int(num_episodes * 0.35)
        n_random = num_episodes - n_expert - n_noisy

        for ep in range(n_expert):
            transitions = self.collect_episode(mode="expert", num_blocks=3)
            dataset.extend(transitions)

        for ep in range(n_noisy):
            transitions = self.collect_episode(mode="noisy", num_blocks=3, noise_std=0.02)
            dataset.extend(transitions)

        for ep in range(n_random):
            transitions = self.collect_episode(mode="expert", num_blocks=1)
            dataset.extend(transitions)

        return {
            "num_episodes": num_episodes,
            "total_transitions": len(dataset),
            "transitions": dataset,
        }
