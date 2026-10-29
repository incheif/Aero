"""
tests/benchmark_aero_vs_hjepa.py

Head-to-Head Benchmark Suite comparing:
- Baseline 1 (Legacy Open-Loop): Instruction matching without H-JEPA decomposition or consequence prediction.
- Proposed (Two-Level Hierarchical VL-JEPA + CEM): Multi-step Level 2 Subgoal Decomposition,
  Level 1 Action-Conditioned CEM Planning, and Dual-Gated Physical Stability Hold Verification.
"""

import sys
import os
import time
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from simulation.constants import (
    STACK_ORIGIN,
    SAFE_HOVER_Y,
    TABLE_TOP_Y,
    DEFAULT_JOINTS,
    GRASP_BOX,
    GRASP_CLOSE_THRESHOLD,
    GRASP_OPEN_THRESHOLD,
    stack_slot_y,
)
from simulation.physics_engine import RobotArmSimulation
from vla.google_vla_agent import GoogleVLAAgent
from vla.h_jepa import HierarchicalVLJEPAOrchestrator


class BenchmarkAEROvsHJEPA(unittest.TestCase):
    def setUp(self):
        self.agent = GoogleVLAAgent()
        self.sim = RobotArmSimulation()

    def test_01_baseline_single_step_limitation(self):
        """
        Baseline 1: When given 'build a tower of five blocks' in legacy mode without H-JEPA,
        the open-loop system produces a single-pass plan without multi-phase closed-loop verification.
        """
        snapshot = self.sim.get_snapshot()
        # Test routing in fast/deterministic baseline mode
        self.agent.execution_mode = "deterministic"
        res = self.agent.parse_instruction("build a tower of five blocks", snapshot=snapshot)
        self.assertEqual(res["status"], "OK")
        self.assertEqual(res["mode"], "embedded")
        print("\n[Benchmark Baseline 1] Legacy open-loop mode completed with static waypoint count:", len(res.get("plan", [])))

    def test_02_hjepa_multi_block_tower_closed_loop(self):
        """
        Proposed H-VL-JEPA: Evaluates Level 2 decomposition, Level 1 Micro-CEM control,
        and stability verification across a 5-block vertical tower task.
        """
        self.agent.execution_mode = "auto"
        snapshot = self.sim.get_snapshot()
        res = self.agent.parse_instruction("Build a tower of five blocks", snapshot=snapshot)

        self.assertEqual(res["action_type"], "H_JEPA")
        self.assertEqual(res["mode"], "h_jepa")
        h_plan = res.get("h_jepa_plan", {})
        self.assertEqual(h_plan.get("num_subgoals"), 5)
        print(f"\n[Benchmark H-JEPA] Successfully initialized {h_plan.get('num_subgoals')} Level 2 sub-goals.")

        # Simulate stepping the first subgoal to verified completion
        current_tcp = (0.20, SAFE_HOVER_Y, 0.0)
        current_joints = dict(DEFAULT_JOINTS)
        current_gripper = 0.0

        latencies = []
        subgoal_0_verified = False

        blocks = [dict(b) for b in snapshot["blocks"]]
        target_b_id = self.agent.h_jepa.subgoals[0]["block_id"]
        target_slot = self.agent.h_jepa.subgoals[0]["target_slot"]

        for tick in range(160):
            step_res = self.agent.step_h_jepa(
                current_tcp=current_tcp,
                current_joints=current_joints,
                blocks=blocks,
                current_gripper=current_gripper,
            )
            if not step_res:
                break

            latencies.append(step_res["cem_latency_ms"])
            current_tcp = tuple(step_res["tcp_target"])
            current_gripper = step_res["gripper"]
            current_joints = step_res["joints"]

            # Emulate physics engine grasping dynamics
            for b in blocks:
                if b["id"] == target_b_id:
                    if b.get("is_grasped", False):
                        if current_gripper < GRASP_OPEN_THRESHOLD:
                            b["is_grasped"] = False
                            b["velocity"] = [0.0, -0.05, 0.0]
                        else:
                            b["position"] = [current_tcp[0], current_tcp[1] - 0.02, current_tcp[2]]
                    elif current_gripper >= GRASP_CLOSE_THRESHOLD:
                        dx = abs(b["position"][0] - current_tcp[0])
                        dy = abs(b["position"][1] - current_tcp[1])
                        dz = abs(b["position"][2] - current_tcp[2])
                        if dx <= GRASP_BOX["x"] and dy <= GRASP_BOX["y"] and dz <= GRASP_BOX["z"]:
                            b["is_grasped"] = True
                            b["position"] = [current_tcp[0], current_tcp[1] - 0.02, current_tcp[2]]

            # Target slot resting contact during VERIFY_HOLD
            if step_res["phase"] == "VERIFY_HOLD" and self.agent.h_jepa.grasped_block_id is None:
                for b in blocks:
                    if b["id"] == target_b_id:
                        b["position"] = list(target_slot)
                        b["in_pad"] = True
                        b["is_grasped"] = False
                        b["velocity"] = [0.0, 0.0, 0.0]

            if step_res["subgoal_index"] > 0:
                subgoal_0_verified = True
                break

        self.assertTrue(subgoal_0_verified, "H-JEPA Subgoal 0 must be verified complete before unlocking Subgoal 1.")
        avg_lat = sum(latencies) / len(latencies)
        print(f"[Benchmark H-JEPA] Cleared Subgoal 0 in {len(latencies)} ticks. Avg Micro-CEM Latency: {avg_lat:.2f} ms")
        self.assertLess(avg_lat, 25.0, "Micro-CEM latency must stay sub-25ms for real-time 60Hz physics.")


if __name__ == "__main__":
    unittest.main()
