#!/usr/bin/env python3
"""
frontier_explorer_node.py

Autonomous Frontier Exploration Engine for AERO.
Subscribes to `/map` (OccupancyGrid), detects boundaries between explored
free space (0) and unknown space (-1), clusters frontiers, and dispatches
information-gain exploration waypoints to Nav2.
"""

import math
from typing import List, Tuple, Optional

try:
    import rclpy
    from rclpy.node import Node
    from nav_msgs.msg import OccupancyGrid
    from geometry_msgs.msg import PoseStamped, Point
    from visualization_msgs.msg import Marker, MarkerArray
    from std_msgs.msg import Bool
    HAS_RCLPY = True
except ImportError:
    HAS_RCLPY = False
    class Node:
        def __init__(self, *args, **kwargs): pass
    OccupancyGrid = Any = object
    PoseStamped = Point = Marker = MarkerArray = Bool = object


def find_frontiers(grid: List[int], width: int, height: int) -> List[Tuple[int, int]]:
    """Identifies grid cells that are free (0) and adjacent to unknown (-1)."""
    frontiers = []
    # 8-connected neighbor offsets
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for y in range(1, height - 1):
        row_offset = y * width
        for x in range(1, width - 1):
            idx = row_offset + x
            if grid[idx] != 0:
                continue

            # Check if any neighbor is unknown (-1)
            is_frontier = False
            for dx, dy in neighbors:
                n_idx = (y + dy) * width + (x + dx)
                if grid[n_idx] == -1:
                    is_frontier = True
                    break

            if is_frontier:
                frontiers.append((x, y))

    return frontiers


def cluster_frontiers(frontiers: List[Tuple[int, int]], min_size: int = 4) -> List[Tuple[float, float]]:
    """Groups neighboring frontier cells and returns centroids of valid clusters."""
    if not frontiers:
        return []

    visited = set()
    clusters = []

    for cell in frontiers:
        if cell in visited:
            continue

        # BFS to find contiguous cluster
        cluster = []
        queue = [cell]
        visited.add(cell)

        while queue:
            cx, cy = queue.pop(0)
            cluster.append((cx, cy))

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    neighbor = (cx + dx, cy + dy)
                    if neighbor in frontiers and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

        if len(cluster) >= min_size:
            # Centroid
            avg_x = sum(c[0] for c in cluster) / len(cluster)
            avg_y = sum(c[1] for c in cluster) / len(cluster)
            clusters.append((avg_x, avg_y))

    return clusters


class FrontierExplorerNode(Node):
    """ROS 2 Node that calculates and dispatches frontier exploration goals."""

    def __init__(self) -> None:
        super().__init__('frontier_explorer_node')

        self.declare_parameter('min_cluster_size', 5)
        self.declare_parameter('auto_explore', True)

        self.min_cluster_size = self.get_parameter('min_cluster_size').value
        self.is_active = self.get_parameter('auto_explore').value

        self.latest_map: Optional[OccupancyGrid] = None

        # Subscriptions
        self.create_subscription(OccupancyGrid, '/map', self._map_callback, 10)
        self.create_subscription(Bool, '/aero/explore_enable', self._enable_callback, 10)

        # Publishers
        self.frontier_goal_pub = self.create_publisher(PoseStamped, '/frontier_goal', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/aero/frontier_markers', 10)

        # Periodic exploration planner timer (every 3 seconds)
        self.create_timer(3.0, self._exploration_step)

        self.get_logger().info("FrontierExplorerNode initialized. Waiting for /map...")

    def _map_callback(self, msg: OccupancyGrid) -> None:
        self.latest_map = msg

    def _enable_callback(self, msg: Bool) -> None:
        self.is_active = msg.data
        self.get_logger().info(f"Frontier exploration state: {self.is_active}")

    def _exploration_step(self) -> None:
        if not self.is_active or self.latest_map is None:
            return

        map_data = self.latest_map
        width = map_data.info.width
        height = map_data.info.height
        resolution = map_data.info.resolution
        origin_x = map_data.info.origin.position.x
        origin_y = map_data.info.origin.position.y

        # Detect and cluster frontiers
        frontiers = find_frontiers(map_data.data, width, height)
        clusters = cluster_frontiers(frontiers, min_size=self.min_cluster_size)

        if not clusters:
            self.get_logger().info("No valid frontiers detected. Environment mapping complete or unreachable.")
            return

        # Select the largest/best centroid (first cluster)
        target_grid_x, target_grid_y = clusters[0]

        # Convert grid coordinate to real-world metric coordinates
        target_world_x = origin_x + (target_grid_x * resolution)
        target_world_y = origin_y + (target_grid_y * resolution)

        # Publish Nav2 / Frontier Goal
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = 'map'
        goal.pose.position.x = target_world_x
        goal.pose.position.y = target_world_y
        goal.pose.orientation.w = 1.0

        self.frontier_goal_pub.publish(goal)
        self._publish_markers(clusters, resolution, origin_x, origin_y)

        self.get_logger().info(
            f"Dispatched frontier exploration goal: ({target_world_x:.2f}m, {target_world_y:.2f}m) "
            f"[Found {len(clusters)} frontier clusters]"
        )

    def _publish_markers(self, clusters, resolution, origin_x, origin_y):
        marker_array = MarkerArray()
        for i, (cx, cy) in enumerate(clusters):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'frontiers'
            m.id = i
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = origin_x + (cx * resolution)
            m.pose.position.y = origin_y + (cy * resolution)
            m.pose.position.z = 0.2
            m.scale.x = 0.3
            m.scale.y = 0.3
            m.scale.z = 0.4
            m.color.r = 0.0
            m.color.g = 0.9
            m.color.b = 1.0
            m.color.a = 0.8
            marker_array.markers.append(m)

        self.marker_pub.publish(marker_array)


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = FrontierExplorerNode()
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
