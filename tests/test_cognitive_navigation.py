"""
test_cognitive_navigation.py

Automated unit tests for AERO Idea B:
- Gemma Cognitive Reasoner (intent classification & Nav2 goal dispatch)
- Semantic Spatial Mapper (spatial deduplication & landmark queries)
- Frontier Explorer (occupancy grid boundary detection & clustering)
"""

import unittest

from ros2_ws.src.robot_controller.robot_controller.gemma_brain_node import GemmaCognitiveReasoner
from ros2_ws.src.robot_controller.robot_controller.semantic_mapper import SemanticMapperNode, SemanticLandmark
from ros2_ws.src.robot_controller.robot_controller.frontier_explorer import (
    find_frontier_cells,
    cluster_frontiers,
    FrontierCluster,
)


class TestGemmaCognitiveReasoner(unittest.TestCase):

    def setUp(self):
        self.reasoner = GemmaCognitiveReasoner()
        self.known_landmarks = {
            "red_sofa": {"name": "Red Sofa", "category": "furniture", "room": "Living Room", "x": 3.2, "y": 3.0},
            "kitchen_table": {"name": "Kitchen Table", "category": "furniture", "room": "Kitchen", "x": -2.2, "y": 2.5},
            "charging_dock": {"name": "Charging Dock", "category": "dock", "room": "Docking Bay", "x": 0.0, "y": -3.2},
        }

    def test_navigate_to_known_object(self):
        decision = self.reasoner.parse_instruction("Please drive to the red sofa", self.known_landmarks)
        self.assertEqual(decision["action"], "NAVIGATE_TO_OBJECT")
        self.assertEqual(decision["target"], "Red Sofa")
        self.assertEqual(decision["coordinates"], [3.2, 3.0])
        self.assertIn("Living Room", decision["reply"])

    def test_return_to_charging_dock(self):
        decision = self.reasoner.parse_instruction("Battery is low, return to the dock", self.known_landmarks)
        self.assertEqual(decision["action"], "NAVIGATE_TO_OBJECT")
        self.assertEqual(decision["target"], "Charging Dock")
        self.assertEqual(decision["coordinates"], [0.0, -3.2])

    def test_explore_uncharted_rooms(self):
        decision = self.reasoner.parse_instruction("Explore the uncharted rooms and map the area", self.known_landmarks)
        self.assertIn(decision["action"], ["EXPLORE", "EXPLORE_AND_FIND"])

    def test_unknown_object_triggers_exploration(self):
        decision = self.reasoner.parse_instruction("Find the coffee machine in the house", self.known_landmarks)
        self.assertEqual(decision["action"], "EXPLORE_AND_FIND")
        self.assertIsNone(decision["coordinates"])


class TestSemanticLandmarkMemory(unittest.TestCase):

    def test_landmark_deduplication(self):
        lm = SemanticLandmark("Red Sofa", "furniture", "Living Room", 3.0, 3.0, 0.9)
        self.assertEqual(lm.sightings_count, 1)

        # Update with another sighting nearby
        lm.update_position(3.2, 3.0, 0.95)
        self.assertEqual(lm.sightings_count, 2)
        self.assertAlmostEqual(lm.x, 3.1, places=2)


class TestFrontierExplorer(unittest.TestCase):

    def test_find_frontier_cells(self):
        # 5x5 grid:
        # 0 = free, -1 = unknown, 100 = obstacle
        width, height = 5, 5
        grid_data = [
            -1, -1, -1, -1, -1,
            -1,  0,  0,  0, -1,
            -1,  0,  0,  0, -1,
            -1,  0,  0,  0, -1,
            -1, -1, -1, -1, -1,
        ]

        frontiers = find_frontier_cells(grid_data, width, height)
        # All outer free cells should be frontiers because they touch -1
        self.assertTrue(len(frontiers) > 0)
        # Center cell (2, 2) has all 0 neighbors, so it should NOT be a frontier
        self.assertNotIn((2, 2), frontiers)

    def test_cluster_frontiers(self):
        cells = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 3)]
        clusters = cluster_frontiers(cells, resolution=0.05, origin_x=0.0, origin_y=0.0, min_cluster_size=3)
        self.assertTrue(len(clusters) >= 1)
        self.assertTrue(clusters[0].size >= 3)


if __name__ == '__main__':
    unittest.main()
