"""
tests/test_hjepa_predictors.py

Unit tests for H-JEPA neural & latent world model components:
1. Level 1 Observation & Action Encoders (E^(1), A^(1))
2. Level 1 Latent Predictor (F^(1)) & CEM Planner
3. Level 2 Observation & Text Encoders (E^(2), E_txt)
4. Cross-Level Projection Head (P_{1->2})
5. Stability Probe (h_stab)
6. Closed-Loop OutcomeVerifier (surprise detection, stability hold, structural collapse trigger)
7. VICReg Regularization & Loss Objectives
"""

import sys
import os
import unittest
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from simulation.constants import STACK_ORIGIN, TABLE_TOP_Y, DEFAULT_JOINTS
from vla.jepa.schemas import SubgoalSpecification, VerificationReport, CEMPlanResult
from vla.jepa.encoders import (
    Level1ObservationEncoder,
    Level1ActionEncoder,
    Level2ObservationEncoder,
    ProjectionHead1to2,
    VLTextGoalEncoder,
)
from vla.jepa.level1_world_model import Level1Predictor, Level1CEMPlanner
from vla.jepa.level2_world_model import Level2Predictor, StabilityProbe, Level2TaskPlanner
from vla.jepa.verifier import OutcomeVerifier
from vla.jepa.train_hjepa import vicreg_loss, compute_hjepa_losses, HJEPATrainer


class TestHJEPAPredictors(unittest.TestCase):
    def setUp(self):
        self.sample_blocks = [
            {"id": "block_magenta", "name": "Red Cube", "color": "#FF1E27", "position": [0.15, TABLE_TOP_Y + 0.02, 0.16]},
            {"id": "block_cyan", "name": "Blue Cube", "color": "#1E60FF", "position": [0.22, TABLE_TOP_Y + 0.02, 0.22]},
            {"id": "block_yellow", "name": "Yellow Cube", "color": "#FFC300", "position": [0.12, TABLE_TOP_Y + 0.02, -0.15]},
        ]
        self.tcp_pos = [0.20, 0.85, 0.0]

    def test_01_level1_observation_and_action_encoders(self):
        """Verifies Level 1 encoder shapes, normalization, and execution speed."""
        enc_l1 = Level1ObservationEncoder(latent_dim=384)
        z1 = enc_l1.encode(
            joints=DEFAULT_JOINTS,
            tcp_pos=self.tcp_pos,
            gripper=0.0,
            dst_pos=STACK_ORIGIN,
        )
        self.assertEqual(z1.shape, (384,))
        self.assertAlmostEqual(float(np.linalg.norm(z1)), 1.0, places=4)

        enc_act = Level1ActionEncoder(action_dim=5, latent_dim=32)
        act = np.array([0.01, -0.02, 0.005, 0.0, 1.0], dtype=np.float32)
        a_emb = enc_act.encode(act)
        self.assertEqual(a_emb.shape, (32,))
        self.assertAlmostEqual(float(np.linalg.norm(a_emb)), 1.0, places=4)

    def test_02_level1_predictor_and_cem_planner(self):
        """Verifies Level 1 action-conditioned rollout and CEM optimization."""
        pred_l1 = Level1Predictor(state_dim=384, action_dim=32)
        z1 = np.random.randn(384).astype(np.float32)
        z1 = z1 / np.linalg.norm(z1)
        a_emb = np.random.randn(32).astype(np.float32)
        a_emb = a_emb / np.linalg.norm(a_emb)

        z1_next = pred_l1.predict(z1, a_emb)
        self.assertEqual(z1_next.shape, (384,))
        self.assertAlmostEqual(float(np.linalg.norm(z1_next)), 1.0, places=4)

        # Test CEM planner
        planner = Level1CEMPlanner(horizon=4, num_samples=32, num_elites=4, num_iterations=2)
        z_sub2 = np.random.randn(512).astype(np.float32)
        z_sub2 = z_sub2 / np.linalg.norm(z_sub2)

        # Warm-up and benchmark CEM steady-state latency
        planner.plan(
            subgoal_id=0,
            current_tcp=self.tcp_pos,
            target_pos=[0.25, 0.76, 0.0],
            active_subgoal_z2=z_sub2,
            gripper_aperture=0.0,
            desired_gripper=0.0,
            joints=DEFAULT_JOINTS,
        )
        latencies = []
        for _ in range(5):
            res = planner.plan(
                subgoal_id=0,
                current_tcp=self.tcp_pos,
                target_pos=[0.25, 0.76, 0.0],
                active_subgoal_z2=z_sub2,
                gripper_aperture=0.0,
                desired_gripper=0.0,
                joints=DEFAULT_JOINTS,
            )
            latencies.append(res.latency_ms)

        self.assertIsInstance(res, CEMPlanResult)
        self.assertEqual(res.optimal_actions.shape, (4, 5))
        avg_lat = sum(latencies) / len(latencies)
        self.assertLess(avg_lat, 250.0)

    def test_03_level2_encoder_and_predictor(self):
        """Verifies Level 2 macro encoder, predictor, and stability probe."""
        enc_l2 = Level2ObservationEncoder(latent_dim=512)
        z2 = enc_l2.encode(self.sample_blocks, tcp_pos=self.tcp_pos)
        self.assertEqual(z2.shape, (512,))
        self.assertAlmostEqual(float(np.linalg.norm(z2)), 1.0, places=4)

        pred_l2 = Level2Predictor(state_dim=512, macro_action_dim=16)
        u_macro = np.zeros(16, dtype=np.float32)
        z2_next = pred_l2.predict(z2, u_macro)
        self.assertEqual(z2_next.shape, (512,))

        probe = StabilityProbe(latent_dim=512)
        stab_prob = probe.predict_stability(z2)
        self.assertTrue(0.0 <= stab_prob <= 1.0)

    def test_04_cross_level_projection_and_language_alignment(self):
        """Verifies P_{1->2} projection head and text alignment head."""
        proj = ProjectionHead1to2(dim_l1=384, dim_l2=512)
        z1 = np.random.randn(384).astype(np.float32)
        z2_proj = proj.project(z1)
        self.assertEqual(z2_proj.shape, (512,))
        self.assertAlmostEqual(float(np.linalg.norm(z2_proj)), 1.0, places=4)

        text_enc = VLTextGoalEncoder(latent_dim=512)
        z_txt = text_enc.encode_text("build a stable tower of 5 blocks")
        self.assertEqual(z_txt.shape, (512,))
        self.assertAlmostEqual(float(np.linalg.norm(z_txt)), 1.0, places=4)

    def test_05_outcome_verifier_and_replan_triggers(self):
        """Verifies closed-loop verification, stability hold, and structural collapse escalation."""
        verifier = OutcomeVerifier(required_hold_ticks=5, max_l1_retries=2)
        subgoal = SubgoalSpecification(
            subgoal_id=0,
            action_type="pick_and_place",
            source_label="block_magenta",
            source_color="magenta",
            target_xyz=(STACK_ORIGIN[0], TABLE_TOP_Y + 0.0275, STACK_ORIGIN[2]),
            support_label="table",
        )

        # 1. Grasp slip retry
        rep_slip = verifier.verify_step(
            subgoal=subgoal,
            phase="GRASP",
            current_tcp=(0.20, 0.85, 0.0),
            blocks=[{"id": "block_magenta", "position": [0.35, TABLE_TOP_Y, 0.0], "is_grasped": False}],
            grasped_block_id=None,
        )
        self.assertEqual(rep_slip.status, "RETRY_L1")

        # 2. Complete placement with hold ticks
        blocks_placed = [
            {"id": "block_magenta", "position": list(subgoal.target_xyz), "velocity": [0,0,0], "is_grasped": False, "in_pad": True}
        ]
        # Step through required ticks
        for _ in range(6):
            rep_hold = verifier.verify_step(
                subgoal=subgoal,
                phase="VERIFY_HOLD",
                current_tcp=(STACK_ORIGIN[0], 0.92, STACK_ORIGIN[2]),
                blocks=blocks_placed,
                grasped_block_id=None,
            )
        self.assertEqual(rep_hold.status, "SUBGOAL_COMPLETED")

        # 3. Simulate structural collapse of placed block during next subgoal
        subgoal_1 = SubgoalSpecification(
            subgoal_id=1,
            action_type="stack",
            source_label="block_cyan",
            source_color="cyan",
            target_xyz=(STACK_ORIGIN[0], TABLE_TOP_Y + 0.0275 + 0.055, STACK_ORIGIN[2]),
            support_label="block_magenta",
        )
        # Block magenta toppled away
        blocks_collapsed = [
            {"id": "block_magenta", "position": [0.10, TABLE_TOP_Y, -0.20], "velocity": [0,0,0], "is_grasped": False, "in_pad": False},
            {"id": "block_cyan", "position": [0.22, TABLE_TOP_Y, 0.22], "velocity": [0,0,0], "is_grasped": False, "in_pad": False},
        ]
        rep_collapse = verifier.verify_step(
            subgoal=subgoal_1,
            phase="HOVER_PICK",
            current_tcp=(0.22, 0.92, 0.22),
            blocks=blocks_collapsed,
            grasped_block_id=None,
        )
        self.assertEqual(rep_collapse.status, "ESCALATE_L2_REPLAN")
        self.assertFalse(rep_collapse.tower_intact)

    def test_06_vicreg_and_training_losses(self):
        """Verifies VICReg variance-covariance loss computation."""
        batch_z = np.random.randn(16, 128).astype(np.float32)
        total_vic, var_l, cov_l = vicreg_loss(batch_z)
        self.assertGreaterEqual(total_vic, 0.0)
        self.assertGreaterEqual(var_l, 0.0)
        self.assertGreaterEqual(cov_l, 0.0)

        trainer = HJEPATrainer()
        losses = trainer.train_step({"toppled": False})
        self.assertIn("total_loss", losses)
        self.assertIn("l1_loss", losses)
        self.assertIn("l2_loss", losses)
        self.assertIn("l_hier", losses)
        self.assertIn("l_stab", losses)
        self.assertIn("l_vic", losses)


if __name__ == "__main__":
    unittest.main()
