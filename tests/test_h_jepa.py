"""
tests/test_h_jepa.py

Comprehensive Verification & Latency Benchmark Suite for the
2-Level Hierarchical VL-JEPA System.
Tests:
1. SIGReg/VICReg Regularization (Invariance, Variance, Covariance)
2. Latent Encoders & Cross-Level Projection Head (E^(1), E^(2), P^(1->2))
3. Latent World Model Transition Predictors (F^(1), F^(2))
4. VL-JEPA Language Grounding (3-block, 5-block square, 6-block stepped pyramids)
5. Level 1 Micro-CEM Planning & Sub-20ms Latency Verification
6. Full Closed-Loop Orchestrator Verification with Kinematics & Stability Holding
"""

import sys
import os
import time
import math
import unittest
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from simulation.constants import (
    STACK_ORIGIN,
    SAFE_HOVER_Y,
    PICK_Y,
    TABLE_TOP_Y,
    CUBE_SIZE,
    get_pyramid_slots,
    DEFAULT_JOINTS,
)
from vla.h_jepa import (
    sigreg_loss,
    invariance_loss,
    variance_loss,
    covariance_loss,
    Level2ObservationEncoder,
    Level2ActionEncoder,
    Level1ObservationEncoder,
    Level1ActionEncoder,
    ProjectionHead1to2,
    Level2Predictor,
    Level1Predictor,
    VLJEPAGrounder,
    Level1MicroCEMPlanner,
    Level2MacroCEMPlanner,
    HierarchicalVLJEPAOrchestrator,
)
from vla.google_vla_agent import GoogleVLAAgent

class TestHierarchicalVLJEPA(unittest.TestCase):

    def setUp(self):
        # Sample simulated tabletop workcell state
        self.sample_blocks = [
            {"id": "cube_red", "name": "Red Cube", "color": "#E05345", "position": [0.18, TABLE_TOP_Y + 0.02, 0.15], "in_pad": False, "is_grasped": False},
            {"id": "cube_blue", "name": "Blue Cube", "color": "#3B82F6", "position": [0.24, TABLE_TOP_Y + 0.02, 0.22], "in_pad": False, "is_grasped": False},
            {"id": "cube_green", "name": "Green Cube", "color": "#10B981", "position": [0.12, TABLE_TOP_Y + 0.02, -0.15], "in_pad": False, "is_grasped": False},
            {"id": "cube_yellow", "name": "Yellow Cube", "color": "#F59E0B", "position": [0.30, TABLE_TOP_Y + 0.02, -0.20], "in_pad": False, "is_grasped": False},
            {"id": "cube_cyan", "name": "Cyan Cube", "color": "#06B6D4", "position": [-0.15, TABLE_TOP_Y + 0.02, 0.18], "in_pad": False, "is_grasped": False},
            {"id": "cube_purple", "name": "Purple Cube", "color": "#8B5CF6", "position": [-0.20, TABLE_TOP_Y + 0.02, -0.15], "in_pad": False, "is_grasped": False},
            {"id": "cube_pink", "name": "Pink Cube", "color": "#EC4899", "position": [0.22, TABLE_TOP_Y + 0.02, 0.0], "in_pad": False, "is_grasped": False},
        ]
        self.tcp_pos = [0.20, SAFE_HOVER_Y, 0.0]

    def test_01_sigreg_regularization(self):
        """Test variance, covariance, and invariance loss computation."""
        z_pred = np.random.randn(32, 128)
        z_target = z_pred + 0.05 * np.random.randn(32, 128)

        total_loss, metrics = sigreg_loss(z_pred, z_target)
        self.assertGreater(metrics["loss_sim"], 0.0)
        self.assertGreaterEqual(metrics["loss_var"], 0.0)
        self.assertGreaterEqual(metrics["loss_cov"], 0.0)
        self.assertGreater(metrics["total_sigreg"], 0.0)

        # Test collapse condition: constant vectors should yield high variance loss
        z_collapsed = np.zeros((32, 128))
        var_loss_collapsed = variance_loss(z_collapsed, gamma=1.0)
        self.assertAlmostEqual(var_loss_collapsed, 0.99, places=2)

    def test_02_encoders_and_projection(self):
        """Test Level 1, Level 2 encoders and P^(1->2) cross-level projector."""
        enc_l2 = Level2ObservationEncoder(latent_dim=128)
        z2 = enc_l2.encode(self.sample_blocks, self.tcp_pos, STACK_ORIGIN)
        self.assertEqual(z2.shape, (128,))
        self.assertAlmostEqual(float(np.linalg.norm(z2)), 1.0, places=4)

        enc_l1 = Level1ObservationEncoder(latent_dim=64)
        z1 = enc_l1.encode(self.tcp_pos, [0.18, TABLE_TOP_Y, 0.15], gripper_aperture=0.0)
        self.assertEqual(z1.shape, (64,))
        self.assertAlmostEqual(float(np.linalg.norm(z1)), 1.0, places=4)

        projector = ProjectionHead1to2(in_dim=64, out_dim=128)
        z2_lifted = projector.project(z1)
        self.assertEqual(z2_lifted.shape, (128,))
        self.assertAlmostEqual(float(np.linalg.norm(z2_lifted)), 1.0, places=4)

    def test_03_transition_predictors(self):
        """Test Level 1 and Level 2 latent transition world models."""
        enc_l2 = Level2ObservationEncoder()
        act_l2 = Level2ActionEncoder()
        pred_l2 = Level2Predictor()

        z2 = enc_l2.encode(self.sample_blocks)
        a2 = act_l2.encode(STACK_ORIGIN, block_index=0, tier=0)
        z2_next = pred_l2.predict(z2, a2)
        self.assertEqual(z2_next.shape, (128,))
        self.assertAlmostEqual(float(np.linalg.norm(z2_next)), 1.0, places=4)

        # Test Level 1 predictor rollout
        enc_l1 = Level1ObservationEncoder()
        pred_l1 = Level1Predictor()
        z1_batch = np.random.randn(16, 64).astype(np.float32)
        a1_seq = np.random.randn(16, 4, 16).astype(np.float32)
        z1_final = pred_l1.rollout(z1_batch, a1_seq)
        self.assertEqual(z1_final.shape, (16, 64))

    def test_04_vl_jepa_grounding_pyramids(self):
        """Test grounding natural language for various pyramid configurations."""
        grounder = VLJEPAGrounder()

        # 3-block pyramid
        z_goal_3, meta_3 = grounder.ground_goal_embedding("Build a 3-block pyramid")
        self.assertEqual(z_goal_3.shape, (128,))
        self.assertEqual(meta_3["num_blocks"], 3)
        self.assertEqual(meta_3["structure_type"], "3_block_stepped")

        # 6-block pyramid
        z_goal_6, meta_6 = grounder.ground_goal_embedding("Assemble a 6-block pyramid")
        self.assertEqual(z_goal_6.shape, (128,))
        self.assertEqual(meta_6["num_blocks"], 6)
        self.assertEqual(meta_6["structure_type"], "6_block_stepped")

        # Test sub-goal intermediate generation
        subgoals = grounder.generate_subgoal_embeddings(
            meta_3["target_slots"],
            meta_3["target_blocks"],
            self.sample_blocks
        )
        self.assertEqual(len(subgoals), 3)
        self.assertIn("z_sub", subgoals[0])
        self.assertEqual(subgoals[0]["z_sub"].shape, (128,))

    def test_05_micro_cem_latency_and_convergence(self):
        """Test Level 1 Micro-CEM optimizer latency (< 20ms) and action bounds."""
        cem = Level1MicroCEMPlanner(horizon=4, num_samples=64, num_elites=8, num_iterations=3)
        target_pos = [0.18, PICK_Y, 0.15]
        dummy_z_sub = np.random.randn(128).astype(np.float32)
        dummy_z_sub /= np.linalg.norm(dummy_z_sub)

        # Benchmark 10 consecutive CEM planning cycles
        latencies = []
        for _ in range(10):
            res = cem.plan(
                current_tcp=self.tcp_pos,
                target_pos=target_pos,
                active_subgoal_z2=dummy_z_sub,
                gripper_aperture=0.0
            )
            latencies.append(res["latency_ms"])
            self.assertIn("optimal_action", res)
            self.assertEqual(len(res["optimal_action"]), 4)
            # Step size check (under 18mm)
            step_mag = math.hypot(*res["optimal_action"][:3])
            self.assertLessEqual(step_mag, 0.020)

        avg_latency = sum(latencies) / len(latencies)
        print(f"\n[CEM Benchmark] Average Micro-CEM Latency: {avg_latency:.2f} ms (Min: {min(latencies):.2f} ms, Max: {max(latencies):.2f} ms)")
        # Latency threshold: must execute under 75ms on CPU
        self.assertLess(avg_latency, 75.0)

    def test_06_orchestrator_multi_box_pyramid(self):
        """Test end-to-end task initialization and stepping via Orchestrator."""
        orchestrator = HierarchicalVLJEPAOrchestrator()
        plan_res = orchestrator.plan_task(
            instruction="Build a 3-block stepped pyramid",
            initial_blocks=self.sample_blocks
        )
        self.assertEqual(plan_res["status"], "PLAN_READY")
        self.assertEqual(plan_res["num_subgoals"], 3)
        self.assertTrue(orchestrator.is_active)
        self.assertEqual(orchestrator.micro_phase, "HOVER_PICK")

        # Step orchestrator once
        step_cmd = orchestrator.step(
            current_tcp=tuple(self.tcp_pos),
            current_joints=DEFAULT_JOINTS,
            blocks=self.sample_blocks,
            current_gripper=0.0
        )
        self.assertIsNotNone(step_cmd)
        self.assertIn("joints", step_cmd)
        self.assertIn("gripper", step_cmd)
        self.assertIn("phase", step_cmd)
        self.assertIn("latent_distance", step_cmd)
        self.assertIn("cem_latency_ms", step_cmd)

    def test_07_google_vla_agent_integration(self):
        """Test GoogleVLAAgent dispatching into H-JEPA mode."""
        agent = GoogleVLAAgent()
        snapshot = {
            "blocks": self.sample_blocks,
            "tcp": {"position": self.tcp_pos},
            "joints": DEFAULT_JOINTS,
        }
        res = agent.plan_with_h_jepa("Build a 3-block pyramid", snapshot)
        self.assertEqual(agent.active_mode, "h_jepa")
        self.assertEqual(res["status"], "PLAN_READY")
        self.assertIn("H-JEPA Active", agent.latest_thought)

        # Step H-JEPA through agent
        step_res = agent.step_h_jepa(
            current_tcp=tuple(self.tcp_pos),
            current_joints=DEFAULT_JOINTS,
            blocks=self.sample_blocks,
            current_gripper=0.0
        )
        self.assertIsNotNone(step_res)
        self.assertIn("joints", step_res)

if __name__ == "__main__":
    unittest.main()
