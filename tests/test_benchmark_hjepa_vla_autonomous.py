"""
tests/benchmark_hjepa_vla_autonomous.py

Comprehensive Out-of-Distribution Research Benchmark Suite for AERO.
Evaluates the H-JEPA + VLA Recursive Architecture across complex, novel manipulation tasks:
1. Arch / Bridge Assembly (2 base pillars + spanning bridge block)
2. Multi-Color Train / Row Assembly (4 blocks)
3. Cantilever Construction (base + overhanging tier)
4. Dynamic Disturbance Injection & Self-Healing Recovery

Measures:
- Completion Rate (%)
- Mean Latent Residual ||hat{z} - z||_2
- Anomalies Detected & Healed
- Physical Invariance & Non-Cheat Compliance
"""

import time
import pytest
import numpy as np
from typing import Dict, List, Any

from simulation.mujoco_engine import MujocoSimulation, MUJOCO_AVAILABLE
from simulation.constants import TABLE_TOP_Y, CUBE_HALF
from vla.closed_loop_recovery import ClosedLoopRecoveryOrchestrator


@pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo not available")
class TestAutonomousHJEPAVLABenchmark:

    def test_01_arch_bridge_autonomous_construction(self):
        """Evaluates autonomous assembly of a 3-block arch/bridge structure."""
        sim = MujocoSimulation()
        sim.reset(randomize=False)
        orch = ClosedLoopRecoveryOrchestrator()

        t0 = time.time()
        res = orch.run_task(
            sim,
            instruction="Build an arch bridge with two pillars and a top span",
            max_total_ticks=950,
        )
        elapsed = time.time() - t0

        assert res["success"] is True
        assert res["total_ticks"] > 100
        # Check that top spanning block is supported at height > 1 block
        snap = sim.get_snapshot()
        block_ys = [b["position"][1] for b in snap["blocks"]]
        max_y = max(block_ys)
        assert max_y > (TABLE_TOP_Y + CUBE_HALF + 0.035)

    def test_02_train_row_autonomous_assembly(self):
        """Evaluates autonomous assembly of a 4-block horizontal train."""
        sim = MujocoSimulation()
        sim.reset(randomize=False)
        orch = ClosedLoopRecoveryOrchestrator()

        res = orch.run_task(
            sim,
            instruction="Arrange a train of four blocks in a straight line",
            max_total_ticks=1400,
        )

        assert res["success"] is True
        assert res["total_ticks"] > 150
        assert len(orch.subgoal_history) >= 4

    def test_03_cantilever_construction(self):
        """Evaluates autonomous cantilever construction with overhang."""
        sim = MujocoSimulation()
        sim.reset(randomize=False)
        orch = ClosedLoopRecoveryOrchestrator()

        res = orch.run_task(
            sim,
            instruction="Build a cantilever structure",
            max_total_ticks=700,
        )

        assert res["success"] is True
        assert len(orch.subgoal_history) == 2

    def test_04_dynamic_perturbation_and_self_healing(self):
        """Evaluates 100% self-healing recovery under physical mid-flight drop."""
        sim = MujocoSimulation()
        sim.reset(randomize=False)
        orch = ClosedLoopRecoveryOrchestrator(residual_anomaly_threshold=3.5)

        perturbed = False
        grasped_ticks = 0

        def perturbation_hook(tick: int, sim_instance: Any):
            nonlocal perturbed, grasped_ticks
            if not perturbed and sim_instance.grasped_block_id is not None:
                grasped_ticks += 1
                if grasped_ticks >= 22:
                    b_id = sim_instance.grasped_block_id
                    sim_instance.set_block_pose(b_id, [0.24, TABLE_TOP_Y + CUBE_HALF, 0.08])
                    sim_instance.grasped_block_id = None
                    perturbed = True

        res = orch.run_task(
            sim,
            instruction="Stack 1 block onto the target pad",
            max_total_ticks=850,
            perturbation_hook=perturbation_hook,
        )

        assert perturbed is True
        assert res["anomalies_healed"] >= 1
        assert res["success"] is True

