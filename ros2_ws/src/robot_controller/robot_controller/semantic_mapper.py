#!/usr/bin/env python3
"""
semantic_mapper.py

Semantic Spatial Perception & Landmark Memory Node for AERO.
Extracts semantic entities from visual perception feeds, projects them into 2D map
coordinates, deduplicates spatial clusters, and maintains an active Semantic Database
for the local Gemma cognitive brain.
"""

import math
import time
from typing import Dict, List, Optional, Tuple, Any

try:
    import rclpy
    from rclpy.node import Node
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import Point
    HAS_RCLPY = True
except ImportError:
    HAS_RCLPY = False
    class Node:  # type: ignore
        def __init__(self, *args, **kwargs): pass
    class Odometry:  # type: ignore
        pass
    class Point:  # type: ignore
        pass


class SemanticLandmark:
    """Represents a discovered physical landmark in the environment."""

    def __init__(
        self,
        name: str,
        category: str,
        room: str,
        x: float,
        y: float,
        confidence: float = 0.95
    ) -> None:
        self.name = name
        self.category = category
        self.room = room
        self.x = x
        self.y = y
        self.confidence = confidence
        self.first_seen = time.time()
        self.sightings_count = 1

    def update_position(self, new_x: float, new_y: float, new_conf: float) -> None:
        """Running average of coordinates to refine localization."""
        total = self.sightings_count + 1
        self.x = (self.x * self.sightings_count + new_x) / total
        self.y = (self.y * self.sightings_count + new_y) / total
        self.confidence = max(self.confidence, new_conf)
        self.sightings_count = total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "room": self.room,
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "confidence": round(self.confidence, 2),
            "sightings": self.sightings_count
        }


class SemanticMapperNode(Node):
    """ROS 2 Node maintaining real-time spatial memory of physical landmarks."""

    def __init__(self) -> None:
        super().__init__('semantic_mapper_node')

        self.landmarks: Dict[str, SemanticLandmark] = {}
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0

        # Subscribe to odometry
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)

        # Periodic memory broadcast & simulated vision detector (10 Hz)
        self.timer = self.create_timer(0.1, self._perception_step)

        self.get_logger().info("SemanticMapperNode initialized. Semantic spatial memory active.")

    def _odom_cb(self, msg: Odometry) -> None:
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        self.robot_x = pos.x
        self.robot_y = pos.y
        # Yaw extraction
        siny = 2.0 * (ori.w * ori.z + ori.x * ori.y)
        cosy = 1.0 - 2.0 * (ori.y * ori.y + ori.z * ori.z)
        self.robot_yaw = math.atan2(siny, cosy)

    def register_detection(
        self,
        name: str,
        category: str,
        room: str,
        world_x: float,
        world_y: float,
        confidence: float = 0.95
    ) -> None:
        """Register or update an identified landmark in semantic memory."""
        key = name.lower().replace(" ", "_")

        # Spatial deduplication: if landmark exists within 0.7m, fuse it
        for existing_key, landmark in self.landmarks.items():
            dist = math.hypot(landmark.x - world_x, landmark.y - world_y)
            if dist < 0.7 and (existing_key == key or landmark.category == category):
                landmark.update_position(world_x, world_y, confidence)
                return

        # New landmark discovery
        self.landmarks[key] = SemanticLandmark(name, category, room, world_x, world_y, confidence)
        self.get_logger().info(
            f"🌟 Discovered Landmark: '{name}' in {room} at ({world_x:.2f}, {world_y:.2f})"
        )

    def query_landmark(self, query: str) -> Optional[Tuple[float, float, str]]:
        """
        Query semantic memory for a landmark matching the query string.
        Returns (x, y, room) or None.
        """
        q = query.lower()
        for key, lm in self.landmarks.items():
            if key in q or lm.name.lower() in q or lm.category.lower() in q or lm.room.lower() in q:
                return (lm.x, lm.y, lm.room)
        return None

    def get_all_landmarks(self) -> Dict[str, Dict[str, Any]]:
        return {k: lm.to_dict() for k, lm in self.landmarks.items()}

    def _perception_step(self) -> None:
        """Simulated visual perception check based on camera field of view."""
        # Simulated ground truth house objects for detection test
        house_objects = [
            ("Red Sofa", "furniture", "Living Room", 3.2, 3.0),
            ("Kitchen Table", "furniture", "Kitchen", -2.2, 2.5),
            ("Storage Boxes", "storage", "Storage Bay", 2.8, -1.8),
            ("Charging Dock", "dock", "Docking Bay", 0.0, -3.2),
        ]

        camera_fov = math.radians(65.0)  # 65 deg horizontal FOV
        camera_range = 3.2              # 3.2 meters max detection range

        for name, cat, room, ox, oy in house_objects:
            dx = ox - self.robot_x
            dy = oy - self.robot_y
            dist = math.hypot(dx, dy)

            if dist <= camera_range:
                angle_to_obj = math.atan2(dy, dx)
                angle_diff = (angle_to_obj - self.robot_yaw + math.pi) % (2.0 * math.pi) - math.pi
                if abs(angle_diff) <= (camera_fov / 2.0):
                    # In camera view! Register detection
                    conf = max(0.65, 1.0 - (dist / camera_range) * 0.35)
                    self.register_detection(name, cat, room, ox, oy, conf)


def main(args=None):
    rclpy.init(args=args)
    node = SemanticMapperNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
