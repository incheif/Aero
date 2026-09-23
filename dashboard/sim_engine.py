"""
sim_engine.py

Real-time 2D differential drive kinematic simulation and raycast LiDAR engine.
Provides sub-millisecond, deterministic simulation frames for the AERO dashboard,
bridging controller logic with the live interactive HTML5 Canvas.
"""

import math
import time
from typing import Dict, List, Any, Optional, Tuple


class SimulationArena:
    """Represents the 10m x 10m navigation benchmark arena."""

    def __init__(self) -> None:
        self.width = 8.0   # x from -2.0 to 6.0
        self.height = 7.0  # y from -2.0 to 5.0
        self.x_min = -2.0
        self.x_max = 6.0
        self.y_min = -2.0
        self.y_max = 5.0

        # Target coordinate
        self.target_x = 3.0
        self.target_y = 3.0
        self.goal_tolerance = 0.10

        # Obstacles: list of dicts {x, y, radius}
        self.obstacles = [
            {"x": 1.5, "y": 1.0, "radius": 0.28},
            {"x": 1.0, "y": 2.2, "radius": 0.28},
            {"x": 2.4, "y": 2.2, "radius": 0.28},
            {"x": 3.8, "y": 1.5, "radius": 0.28},
        ]

        # 3D Semantic Landmarks in the house
        self.semantic_objects = [
            {"name": "Red Sofa", "x": 3.2, "y": 3.0, "radius": 0.45, "color": "#ef4444", "icon": "🛋️"},
            {"name": "Kitchen Table", "x": -2.2, "y": 2.5, "radius": 0.45, "color": "#3b82f6", "icon": "🪑"},
            {"name": "Storage Boxes", "x": 2.8, "y": -1.8, "radius": 0.35, "color": "#f59e0b", "icon": "📦"},
            {"name": "Charging Dock", "x": 0.0, "y": -3.2, "radius": 0.30, "color": "#10b981", "icon": "⚡"},
        ]


class KinematicRobot:
    """Differential drive robot model with 360-degree LiDAR."""

    def __init__(self, x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.yaw = yaw
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.radius = 0.15  # Waffle robot radius

        # LiDAR specs
        self.num_rays = 60
        self.max_lidar_range = 3.5
        self.min_lidar_range = 0.12

    def step(self, linear_cmd: float, angular_cmd: float, dt: float) -> None:
        """Update robot pose using unicycle kinematics."""
        # Clamp inputs to robot physical limits
        self.linear_vel = max(-0.22, min(0.22, linear_cmd))
        self.angular_vel = max(-1.5, min(1.5, angular_cmd))

        # Euler integration
        self.x += self.linear_vel * math.cos(self.yaw) * dt
        self.y += self.linear_vel * math.sin(self.yaw) * dt
        self.yaw += self.angular_vel * dt
        # Normalize yaw to [-pi, pi]
        self.yaw = (self.yaw + math.pi) % (2.0 * math.pi) - math.pi

    def compute_lidar(self, arena: SimulationArena) -> Tuple[List[float], float]:
        """Raycast LiDAR ranges against arena walls and obstacles."""
        ranges = []
        min_clearance = float('inf')

        for i in range(self.num_rays):
            angle = self.yaw + (2.0 * math.pi * i / self.num_rays)
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            ray_dist = self.max_lidar_range

            # Check arena boundary walls
            if cos_a > 1e-5:
                d = (arena.x_max - self.x) / cos_a
                if 0 < d < ray_dist:
                    ray_dist = d
            elif cos_a < -1e-5:
                d = (arena.x_min - self.x) / cos_a
                if 0 < d < ray_dist:
                    ray_dist = d

            if sin_a > 1e-5:
                d = (arena.y_max - self.y) / sin_a
                if 0 < d < ray_dist:
                    ray_dist = d
            elif sin_a < -1e-5:
                d = (arena.y_min - self.y) / sin_a
                if 0 < d < ray_dist:
                    ray_dist = d

            # Check circular obstacles
            for obs in arena.obstacles:
                ox = obs["x"] - self.x
                oy = obs["y"] - self.y
                # Projection of circle center onto ray
                t = ox * cos_a + oy * sin_a
                if t > 0:
                    perp_dist_sq = (ox * ox + oy * oy) - (t * t)
                    r_sq = obs["radius"] * obs["radius"]
                    if perp_dist_sq < r_sq:
                        half_chord = math.sqrt(max(0.0, r_sq - perp_dist_sq))
                        dist_to_surface = t - half_chord
                        if 0 < dist_to_surface < ray_dist:
                            ray_dist = dist_to_surface

            clamped_dist = max(self.min_lidar_range, min(self.max_lidar_range, ray_dist))
            ranges.append(clamped_dist)
            if clamped_dist < min_clearance:
                min_clearance = clamped_dist

        return ranges, min_clearance


class LiveSimulationSession:
    """Manages an active trial simulation session."""

    def __init__(self, target_x: float = 3.0, target_y: float = 3.0) -> None:
        self.arena = SimulationArena()
        self.arena.target_x = target_x
        self.arena.target_y = target_y
        self.robot = KinematicRobot()
        self.sim_time = 0.0
        self.timeout_sec = 30.0
        self.is_active = True
        self.status = "RUNNING"
        self.reason = "IN_PROGRESS"
        self.trajectory: List[Dict[str, float]] = []
        self.min_obstacle_dist = 999.0

    def reset(self) -> None:
        self.robot = KinematicRobot()
        self.sim_time = 0.0
        self.is_active = True
        self.status = "RUNNING"
        self.reason = "IN_PROGRESS"
        self.trajectory = [{"x": 0.0, "y": 0.0}]
        self.min_obstacle_dist = 999.0

    def tick(self, linear_cmd: float, angular_cmd: float, dt: float = 0.05) -> Dict[str, Any]:
        """Advances simulation by dt and computes Oracle verdicts."""
        if not self.is_active:
            return self.get_state()

        self.robot.step(linear_cmd, angular_cmd, dt)
        self.sim_time += dt

        # Record trajectory breadcrumb
        self.trajectory.append({"x": round(self.robot.x, 3), "y": round(self.robot.y, 3)})

        # Compute LiDAR and obstacle clearances
        ranges, clearance = self.robot.compute_lidar(self.arena)
        if clearance < self.min_obstacle_dist:
            self.min_obstacle_dist = clearance

        # Distance to goal
        dx = self.arena.target_x - self.robot.x
        dy = self.arena.target_y - self.robot.y
        goal_distance = math.hypot(dx, dy)

        # Oracle Condition 1: Goal Reached
        if goal_distance <= self.arena.goal_tolerance:
            self.is_active = False
            self.status = "PASSED"
            self.reason = "SUCCESS"

        # Oracle Condition 2: Collision
        elif clearance <= (self.robot.radius + 0.03):
            self.is_active = False
            self.status = "FAILED"
            self.reason = "COLLISION"

        # Oracle Condition 3: Timeout
        elif self.sim_time >= self.timeout_sec:
            self.is_active = False
            self.status = "FAILED"
            self.reason = "TIMEOUT"

        return {
            "type": "sim_frame",
            "sim_time": round(self.sim_time, 2),
            "status": self.status,
            "reason": self.reason,
            "robot": {
                "x": round(self.robot.x, 3),
                "y": round(self.robot.y, 3),
                "yaw": round(self.robot.yaw, 3),
                "linear_vel": round(self.robot.linear_vel, 3),
                "angular_vel": round(self.robot.angular_vel, 3)
            },
            "goal_distance": round(goal_distance, 3),
            "min_obstacle_distance": round(self.min_obstacle_dist, 3) if not math.isinf(self.min_obstacle_dist) else 999.0,
            "lidar_ranges": [round(r, 3) for r in ranges],
            "target": {"x": self.arena.target_x, "y": self.arena.target_y, "tolerance": self.arena.goal_tolerance},
            "obstacles": self.arena.obstacles,
            "semantic_objects": self.arena.semantic_objects,
            "arena": {
                "x_min": self.arena.x_min, "x_max": self.arena.x_max,
                "y_min": self.arena.y_min, "y_max": self.arena.y_max
            }
        }

    def get_state(self) -> Dict[str, Any]:
        dx = self.arena.target_x - self.robot.x
        dy = self.arena.target_y - self.robot.y
        return {
            "type": "sim_frame",
            "sim_time": round(self.sim_time, 2),
            "status": self.status,
            "reason": self.reason,
            "robot": {
                "x": round(self.robot.x, 3),
                "y": round(self.robot.y, 3),
                "yaw": round(self.robot.yaw, 3),
                "linear_vel": round(self.robot.linear_vel, 3),
                "angular_vel": round(self.robot.angular_vel, 3)
            },
            "goal_distance": round(math.hypot(dx, dy), 3),
            "min_obstacle_distance": round(self.min_obstacle_dist, 3) if not math.isinf(self.min_obstacle_dist) else 999.0,
            "target": {"x": self.arena.target_x, "y": self.arena.target_y, "tolerance": self.arena.goal_tolerance},
            "obstacles": self.arena.obstacles,
            "semantic_objects": self.arena.semantic_objects,
            "arena": {
                "x_min": self.arena.x_min, "x_max": self.arena.x_max,
                "y_min": self.arena.y_min, "y_max": self.arena.y_max
            }
        }
