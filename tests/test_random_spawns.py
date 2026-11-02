import math
import unittest
from simulation.constants import (
    generate_random_block_spawns,
    BLOCK_SPAWNS,
    ARM_MOUNT,
    STACK_ORIGIN,
)
from simulation.physics_engine import RobotArmSimulation
from simulation.mujoco_engine import MujocoSimulation, MUJOCO_AVAILABLE


class TestRandomSpawns(unittest.TestCase):
    def test_random_spawns_count_and_keys(self):
        spawns = generate_random_block_spawns()
        self.assertEqual(len(spawns), len(BLOCK_SPAWNS))
        for s in spawns:
            self.assertIn("id", s)
            self.assertIn("name", s)
            self.assertIn("position", s)
            self.assertEqual(len(s["position"]), 3)

    def test_random_spawns_clearance_and_reach(self):
        for seed in range(10):
            spawns = generate_random_block_spawns(seed=seed)
            positions = [s["position"] for s in spawns]

            # 1. Reach bounds
            arm_x, arm_z = ARM_MOUNT[0], ARM_MOUNT[2]
            for pos in positions:
                d = math.hypot(pos[0] - arm_x, pos[2] - arm_z)
                self.assertGreaterEqual(d, 0.40, f"Block too close to arm mount: {pos}")
                self.assertLessEqual(d, 0.70, f"Block out of reach: {pos}")

            # 2. Separation between blocks (no overlap)
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    dist = math.hypot(positions[i][0] - positions[j][0], positions[i][2] - positions[j][2])
                    self.assertGreaterEqual(dist, 0.075, f"Blocks {i} and {j} overlap: dist={dist:.3f}")

            # 3. Clearance from stack origin
            for pos in positions:
                d_stack = math.hypot(pos[0] - STACK_ORIGIN[0], pos[2] - STACK_ORIGIN[2])
                self.assertGreaterEqual(d_stack, 0.080, f"Block placed on stack pad: {pos}")

    def test_physics_engine_random_reset(self):
        sim = RobotArmSimulation()
        sim.reset(randomize=True)
        snap1 = sim.get_snapshot()
        pos1 = [b["position"] for b in snap1["blocks"]]

        sim.reset(randomize=True)
        snap2 = sim.get_snapshot()
        pos2 = [b["position"] for b in snap2["blocks"]]

        # Verify that positions change
        diffs = [math.hypot(p1[0] - p2[0], p1[2] - p2[2]) for p1, p2 in zip(pos1, pos2)]
        self.assertTrue(any(d > 0.02 for d in diffs), "Consecutive randomized resets should yield different coordinates")

    @unittest.skipUnless(MUJOCO_AVAILABLE, "MuJoCo not installed")
    def test_mujoco_engine_random_reset(self):
        sim = MujocoSimulation()
        sim.reset(randomize=True)
        snap1 = sim.get_snapshot()
        pos1 = [b["position"] for b in snap1["blocks"]]

        sim.reset(randomize=True)
        snap2 = sim.get_snapshot()
        pos2 = [b["position"] for b in snap2["blocks"]]

        diffs = [math.hypot(p1[0] - p2[0], p1[2] - p2[2]) for p1, p2 in zip(pos1, pos2)]
        self.assertTrue(any(d > 0.02 for d in diffs), "MuJoCo randomized resets should yield different coordinates")


if __name__ == "__main__":
    unittest.main()

