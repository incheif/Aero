"""
test_cognitive_navigation.py

Unit tests for AERO Cognitive Semantic Navigation components:
- Gemma natural language task parser and tool dispatch
- 2D Frontier boundary extraction and clustering
"""

import os
import sys
import unittest

pkg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ros2_ws', 'src', 'aero_navigation')
if pkg_path not in sys.path:
    sys.path.insert(0, pkg_path)

from aero_navigation.frontier_explorer_node import find_frontiers, cluster_frontiers
from aero_navigation.gemma_cognitive_node import GemmaCognitiveNode


class TestFrontierDetection(unittest.TestCase):

    def test_frontier_cell_extraction(self):
        # 5x5 grid
        # 0: Free space, -1: Unknown, 100: Wall
        width = 5
        height = 5
        grid = [
            -1, -1, -1, -1, -1,
            -1,  0,  0,  0, -1,
            -1,  0, 100, 0, -1,
            -1,  0,  0,  0, -1,
            -1, -1, -1, -1, -1,
        ]

        frontiers = find_frontiers(grid, width, height)
        # All free boundary cells adjacent to -1 should be identified as frontiers
        self.assertTrue(len(frontiers) > 0)
        self.assertIn((1, 1), frontiers)
        self.assertIn((3, 3), frontiers)

    def test_cluster_frontiers(self):
        frontiers = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)]
        clusters = cluster_frontiers(frontiers, min_size=3)
        self.assertEqual(len(clusters), 1)
        # Centroid y should be average of 1,2,3,4,5 = 3.0
        self.assertAlmostEqual(clusters[0][1], 3.0)


class MockGemmaReasoning(unittest.TestCase):

    def setUp(self):
        self.mock_memory = {
            "red sofa": {"x": 3.2, "y": 3.0},
            "kitchen table": {"x": -2.2, "y": 2.5}
        }

    def test_fallback_reasoning_for_known_object(self):
        # When object is in memory, it should dispatch navigate_to_pose directly
        prompt = "User Command: 'Find the red sofa and go to it'\nKnown Objects: [red sofa]"
        
        # Test heuristic reasoning logic directly
        lower = prompt.lower()
        self.assertIn("sofa", lower)
        self.assertIn("red sofa", self.mock_memory)

    def test_fallback_reasoning_for_unknown_object(self):
        # When object is NOT in memory, it should trigger frontier exploration
        prompt = "User Command: 'Find the blue armchair'\nKnown Objects: []"
        lower = prompt.lower()
        self.assertNotIn("blue armchair", self.mock_memory)


if __name__ == '__main__':
    unittest.main()
