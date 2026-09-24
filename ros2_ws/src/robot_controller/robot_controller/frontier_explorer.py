#!/usr/bin/env python3
"""
frontier_explorer.py

Autonomous Frontier Exploration Node for AERO.
Analyzes the SLAM OccupancyGrid (/map) to detect boundaries between known free space (0)
and unmapped space (-1), clusters candidate frontiers, and generates exploration waypoints
to autonomously map the environment.
"""

import math
from typing import List, Tuple, Optional, Dict, Any

try:
    import rclpy
    from rclpy.node import Node
    from nav_msgs.msg import OccupancyGrid, Odometry
    from geometry_msgs.msg import PoseStamped
    HAS_RCLPY = True
except ImportError:
    HAS_RCLPY = False
    class Node:  # type: ignore
        def __init__(self, *args, **kwargs): pass
    class OccupancyGrid:  # type: ignore
        pass
    class Odometry:  # type: ignore
        pass
    class PoseStamped:  # type: ignore
        pass


class FrontierCluster:
    """Represents a connected cluster of unmapped frontier boundary cells."""

    def __init__(self, cells: List[Tuple[int, int]], resolution: float, origin_x: float, origin_y: float) -> None:
        self.cells = cells
        self.size = len(cells)

        # Centroid calculation in world meters
        avg_grid_x = sum(c[0] for c in cells) / float(self.size)
        avg_grid_y = sum(c[1] for c in cells) / float(self.size)

        self.world_x = origin_x + (avg_grid_x * resolution)
        self.world_y = origin_y + (avg_grid_y * resolution)

    def score(self, robot_x: float, robot_y: float) -> float:
        """Heuristic score balancing distance cost against information gain."""
        dist = math.hypot(self.world_x - robot_x, self.world_y - robot_y)
        # Higher score = larger cluster closer to robot
        return (self.size * 1.5) / (dist + 0.8)


def find_frontier_cells(grid_data: List[int], width: int, height: int) -> List[Tuple[int, int]]:
    """
    Finds grid cells that are known free (0) and adjacent to at least one unknown (-1) cell.
    """
    frontiers = []
    # 8-connected neighbor offsets
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            idx = y * width + x
            if grid_data[idx] == 0:  # Free space
                # Check neighbors for unmapped territory
                has_unknown = False
                for dx, dy in neighbors:
                    n_idx = (y + dy) * width + (x + dx)
                    if grid_data[n_idx] == -1:
                        has_unknown = True
                        break
                if has_unknown:
                    frontiers.append((x, y))

    return frontiers


def cluster_frontiers(
    frontier_cells: List[Tuple[int, int]],
    resolution: float,
    origin_x: float,
    origin_y: float,
    min_cluster_size: int = 5
) -> List[FrontierCluster]:
    """Clusters adjacent frontier points using breadth-first connected components."""
    visited = set()
    clusters: List[FrontierCluster] = []
    cell_set = set(frontier_cells)

    for cell in frontier_cells:
        if cell in visited:
            continue

        # BFS queue
        current_cluster = []
        queue = [cell]
        visited.add(cell)

        while queue:
            curr = queue.pop(0)
            current_cluster.append(curr)

            cx, cy = curr
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nb = (cx + dx, cy + dy)
                    if nb in cell_set and nb not in visited:
                        visited.add(nb)
                        queue.append(nb)

        if len(current_cluster) >= min_cluster_size:
            clusters.append(FrontierCluster(current_cluster, resolution, origin_x, origin_y))

    return clusters


class FrontierExplorerNode(Node):
    """ROS 2 Node supervising autonomous frontier exploration."""

    def __init__(self) -> None:
        super().__init__('frontier_explorer_node')

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.current_map: Optional[OccupancyGrid] = None
        self.is_exploring = False

        # Subscriptions
        self.create_subscription(OccupancyGrid, '/map', self._map_cb, 5)
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)

        # Publisher for target exploration goal
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        # Periodic exploration planner timer (1 Hz)
        self.plan_timer = self.create_timer(1.0, self._exploration_step)

        self.get_logger().info("FrontierExplorerNode active. Ready to autonomously map uncharted space.")

    def _map_cb(self, msg: OccupancyGrid) -> None:
        self.current_map = msg

    def _odom_cb(self, msg: Odometry) -> None:
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

    def start_exploration(self) -> None:
        self.is_exploring = True
        self.get_logger().info("🚀 Frontier exploration activated.")

    def stop_exploration(self) -> None:
        self.is_exploring = False
        self.get_logger().info("🛑 Frontier exploration paused.")

    def get_best_frontier(self) -> Optional[Tuple[float, float]]:
        """Calculates the best next exploration waypoint from current map."""
        if not self.current_map:
            return None

        info = self.current_map.info
        cells = find_frontier_cells(self.current_map.data, info.width, info.height)
        clusters = cluster_frontiers(
            cells, info.resolution, info.origin.position.x, info.origin.position.y
        )

        if not clusters:
            return None

        # Pick cluster with highest score
        best = max(clusters, key=lambda c: c.score(self.robot_x, self.robot_y))
        return (best.world_x, best.world_y)

    def _exploration_step(self) -> None:
        if not self.is_exploring or not self.current_map:
            return

        target = self.get_best_frontier()
        if target:
            tx, ty = target
            goal = PoseStamped()
            goal.header.stamp = self.get_clock().now().to_msg()
            goal.header.frame_id = 'map'
            goal.pose.position.x = tx
            goal.pose.position.y = ty
            goal.pose.orientation.w = 1.0
            self.goal_pub.publish(goal)
            self.get_logger().info(f"📍 Dispatched frontier goal: ({tx:.2f}, {ty:.2f})")
        else:
            self.get_logger().info("🎉 Frontier exploration complete: no remaining frontiers.")
            self.is_exploring = False


def main(args=None):
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
