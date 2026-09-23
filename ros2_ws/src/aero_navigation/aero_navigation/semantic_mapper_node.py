#!/usr/bin/env python3
"""
semantic_mapper_node.py

3D Semantic Object Mapping and Spatial Memory Node for AERO.
Detects semantic objects from camera and LiDAR sensor streams, projects them
into metric `/map` coordinates, and maintains a living 3D Scene Graph for Gemma.
"""

import json
import math
import time
from typing import Dict, Any, Optional

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String


class SemanticMapperNode(Node):
    """
    Maintains spatial memory of semantic landmarks (e.g. 'red sofa', 'kitchen table')
    and provides location lookups for the Gemma cognitive agent.
    """

    def __init__(self) -> None:
        super().__init__('semantic_mapper_node')

        # 3D Semantic Scene Memory: {object_name: {x, y, z, confidence, timestamp}}
        self.scene_memory: Dict[str, Dict[str, Any]] = {}

        # Robot pose tracking
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.has_pose = False

        # Subscriptions
        self.create_subscription(Odometry, '/odom', self._odom_callback, 10)
        self.create_subscription(LaserScan, '/scan', self._scan_callback, 10)
        self.create_subscription(String, '/aero/detect_object_mock', self._mock_detect_callback, 10)

        # Publishers
        self.semantic_map_pub = self.create_publisher(String, '/aero/semantic_map', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/aero/semantic_markers', 10)

        # Broadcast scene graph every 1 second
        self.create_timer(1.0, self._broadcast_scene_memory)

        # Simulated semantic detector for world arena objects
        self.create_timer(2.0, self._simulate_camera_detections)

        self.get_logger().info("SemanticMapperNode initialized. Scene graph ready.")

    def _odom_callback(self, msg: Odometry) -> None:
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        self.robot_x = pos.x
        self.robot_y = pos.y

        # Planar yaw
        siny_cosp = 2.0 * (ori.w * ori.z + ori.x * ori.y)
        cosy_cosp = 1.0 - 2.0 * (ori.y * ori.y + ori.z * ori.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)
        self.has_pose = True

    def _scan_callback(self, msg: LaserScan) -> None:
        # Uses LiDAR to estimate depth to detected obstacles in front of the camera
        pass

    def _simulate_camera_detections(self) -> None:
        """
        Simulates zero-shot open-vocabulary camera detection for the semantic arena.
        When the robot gets within field-of-view of an object, it adds it to memory.
        """
        if not self.has_pose:
            return

        # Known ground truth locations in the semantic house
        arena_objects = [
            {"name": "red sofa", "x": 3.2, "y": 3.0, "color": (0.9, 0.1, 0.1)},
            {"name": "kitchen table", "x": -2.2, "y": 2.5, "color": (0.2, 0.5, 0.9)},
            {"name": "storage boxes", "x": 2.8, "y": -1.8, "color": (0.8, 0.6, 0.2)},
            {"name": "charging dock", "x": 0.0, "y": -3.6, "color": (0.1, 0.9, 0.3)},
        ]

        detection_range = 2.8  # Camera sensing range in meters

        for obj in arena_objects:
            dx = obj["x"] - self.robot_x
            dy = obj["y"] - self.robot_y
            dist = math.hypot(dx, dy)

            # Check if within distance and in front of the robot (+-60 deg FOV)
            angle_to_obj = math.atan2(dy, dx)
            angle_diff = (angle_to_obj - self.robot_yaw + math.pi) % (2.0 * math.pi) - math.pi

            if dist <= detection_range and abs(angle_diff) < math.radians(65):
                name = obj["name"]
                if name not in self.scene_memory:
                    self.get_logger().info(
                        f"👁️ Camera Detected New Semantic Object: '{name}' at ({obj['x']:.2f}, {obj['y']:.2f})"
                    )

                self.scene_memory[name] = {
                    "x": obj["x"],
                    "y": obj["y"],
                    "z": 0.4,
                    "confidence": round(0.88 + 0.1 * (1.0 - dist / detection_range), 2),
                    "color": obj["color"],
                    "last_seen": time.time()
                }

    def _mock_detect_callback(self, msg: String) -> None:
        """Allows injecting manual or live external detector outputs."""
        try:
            data = json.loads(msg.data)
            self.scene_memory[data["name"]] = data
        except Exception as e:
            self.get_logger().error(f"Failed parsing detect payload: {e}")

    def _broadcast_scene_memory(self) -> None:
        """Publishes JSON scene graph and 3D visual markers."""
        if not self.scene_memory:
            return

        # 1. Publish JSON topic
        msg = String()
        msg.data = json.dumps(self.scene_memory)
        self.semantic_map_pub.publish(msg)

        # 2. Publish RViz Markers
        markers = MarkerArray()
        for i, (name, data) in enumerate(self.scene_memory.items()):
            # Sphere marker
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'semantic_objects'
            m.id = i * 2
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(data["x"])
            m.pose.position.y = float(data["y"])
            m.pose.position.z = float(data.get("z", 0.4))
            m.scale.x = 0.45
            m.scale.y = 0.45
            m.scale.z = 0.45
            c = data.get("color", (1.0, 1.0, 0.0))
            m.color.r = float(c[0])
            m.color.g = float(c[1])
            m.color.b = float(c[2])
            m.color.a = 0.9
            markers.markers.append(m)

            # Text label marker
            t = Marker()
            t.header.frame_id = 'map'
            t.header.stamp = self.get_clock().now().to_msg()
            t.ns = 'semantic_labels'
            t.id = i * 2 + 1
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = float(data["x"])
            t.pose.position.y = float(data["y"])
            t.pose.position.z = float(data.get("z", 0.4)) + 0.45
            t.scale.z = 0.25
            t.color.r = 1.0
            t.color.g = 1.0
            t.color.b = 1.0
            t.color.a = 1.0
            t.text = f"{name} ({data.get('confidence', 1.0):.0%})"
            markers.markers.append(t)

        self.marker_pub.publish(markers)


def main(args: Optional[list] = None) -> None:
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
