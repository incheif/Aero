"""
sim_engine.py

Real-time 2D multi-room simulation, raycast LiDAR, and semantic perception engine.
Simulates the semantic house with interior walls, doors, physical furniture landmarks,
and camera field-of-view object detection for AERO Idea B.
"""

import math
from typing import Dict, List, Any, Optional, Tuple


class SimulationArena:
    """Represents the multi-room semantic house environment."""

    def __init__(self) -> None:
        # Outer house boundaries: 12m x 10m
        self.x_min = -6.0
        self.x_max = 6.0
        self.y_min = -5.0
        self.y_max = 5.0

        # Target coordinate (dynamically set by Gemma)
        self.target_x = 3.2
        self.target_y = 3.0
        self.target_name = "Red Sofa"
        self.goal_tolerance = 0.15

        # Interior partition walls (line segments: x1, y1, x2, y2)
        self.interior_walls = [
            # Divider between Kitchen (West) & Living Room (East) - Doorway at y: -1.0 to 1.5
            {"x1": 0.0, "y1": 1.5, "x2": 0.0, "y2": 5.0},
            # Divider between Storage (East) & Docking Bay (West)
            {"x1": 0.0, "y1": -5.0, "x2": 0.0, "y2": -1.5},
        ]

        # Circular obstacles / pillar structures
        self.obstacles = [
            {"x": 1.5, "y": 0.8, "radius": 0.25},
            {"x": -1.8, "y": -1.2, "radius": 0.25},
            {"x": 4.0, "y": -3.2, "radius": 0.25},
        ]

        # 3D Semantic Landmarks in the house
        self.semantic_objects = [
            {
                "id": "red_sofa",
                "name": "Red Sofa",
                "category": "furniture",
                "room": "Living Room",
                "x": 3.2,
                "y": 3.0,
                "radius": 0.5,
                "color": "#ef4444",
                "icon": "🛋️",
                "discovered": False
            },
            {
                "id": "kitchen_table",
                "name": "Kitchen Table",
                "category": "furniture",
                "room": "Kitchen",
                "x": -2.2,
                "y": 2.5,
                "radius": 0.5,
                "color": "#3b82f6",
                "icon": "🪑",
                "discovered": False
            },
            {
                "id": "storage_boxes",
                "name": "Storage Boxes",
                "category": "storage",
                "room": "Storage Bay",
                "x": 2.8,
                "y": -1.8,
                "radius": 0.45,
                "color": "#f59e0b",
                "icon": "📦",
                "discovered": False
            },
            {
                "id": "charging_dock",
                "name": "Charging Dock",
                "category": "dock",
                "room": "Docking Bay",
                "x": 0.0,
                "y": -3.2,
                "radius": 0.35,
                "color": "#10b981",
                "icon": "⚡",
                "discovered": True  # Robot starts aware of its charging dock
            },
        ]


class KinematicRobot:
    """Differential drive robot model with 360-degree LiDAR and Camera FOV."""

    def __init__(self, x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.yaw = yaw
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.radius = 0.18  # TurtleBot3 Waffle

        # LiDAR specs
        self.num_rays = 72
        self.max_lidar_range = 4.5
        self.min_lidar_range = 0.12

        # Camera specs
        self.camera_fov = math.radians(65.0)  # 65 deg horizontal FOV
        self.camera_range = 3.5               # 3.5m optical detection range

    def step(self, linear_cmd: float, angular_cmd: float, dt: float) -> None:
        """Update robot pose using differential drive kinematics."""
        self.linear_vel = max(-0.24, min(0.24, linear_cmd))
        self.angular_vel = max(-1.6, min(1.6, angular_cmd))

        self.x += self.linear_vel * math.cos(self.yaw) * dt
        self.y += self.linear_vel * math.sin(self.yaw) * dt
        self.yaw += self.angular_vel * dt
        self.yaw = (self.yaw + math.pi) % (2.0 * math.pi) - math.pi

    def check_camera_sightings(self, arena: SimulationArena) -> List[Dict[str, Any]]:
        """Identifies semantic objects within the camera viewing cone."""
        spotted = []
        for obj in arena.semantic_objects:
            dx = obj["x"] - self.x
            dy = obj["y"] - self.y
            dist = math.hypot(dx, dy)

            if dist <= self.camera_range:
                angle_to_obj = math.atan2(dy, dx)
                angle_diff = (angle_to_obj - self.yaw + math.pi) % (2.0 * math.pi) - math.pi
                if abs(angle_diff) <= (self.camera_fov / 2.0):
                    spotted.append(obj)
                    obj["discovered"] = True
        return spotted

    def compute_lidar(self, arena: SimulationArena) -> Tuple[List[float], float]:
        """Raycast LiDAR against outer boundaries, interior partition walls, and obstacles."""
        ranges = []
        min_clearance = float('inf')

        for i in range(self.num_rays):
            angle = self.yaw + (2.0 * math.pi * i / self.num_rays)
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            ray_dist = self.max_lidar_range

            # Outer boundary check
            if cos_a > 1e-5:
                d = (arena.x_max - self.x) / cos_a
                if 0 < d < ray_dist: ray_dist = d
            elif cos_a < -1e-5:
                d = (arena.x_min - self.x) / cos_a
                if 0 < d < ray_dist: ray_dist = d

            if sin_a > 1e-5:
                d = (arena.y_max - self.y) / sin_a
                if 0 < d < ray_dist: ray_dist = d
            elif sin_a < -1e-5:
                d = (arena.y_min - self.y) / sin_a
                if 0 < d < ray_dist: ray_dist = d

            # Interior wall segments check
            for wall in arena.interior_walls:
                x1, y1, x2, y2 = wall["x1"], wall["y1"], wall["x2"], wall["y2"]
                # Ray-line segment intersection
                denom = cos_a * (y1 - y2) - sin_a * (x1 - x2)
                if abs(denom) > 1e-6:
                    t = ((x1 - self.x) * (y1 - y2) - (y1 - self.y) * (x1 - x2)) / denom
                    u = -((self.x - x1) * sin_a - (self.y - y1) * cos_a) / denom
                    if t > 0 and 0 <= u <= 1 and t < ray_dist:
                        ray_dist = t

            # Circular obstacles check
            all_obstacles = arena.obstacles + [
                {"x": obj["x"], "y": obj["y"], "radius": obj["radius"] * 0.8}
                for obj in arena.semantic_objects
            ]
            for obs in all_obstacles:
                ox = obs["x"] - self.x
                oy = obs["y"] - self.y
                t = ox * cos_a + oy * sin_a
                if t > 0:
                    perp_dist_sq = (ox * ox + oy * oy) - (t * t)
                    r_sq = obs["radius"] * obs["radius"]
                    if perp_dist_sq < r_sq:
                        half_chord = math.sqrt(max(0.0, r_sq - perp_dist_sq))
                        dist_surf = t - half_chord
                        if 0 < dist_surf < ray_dist:
                            ray_dist = dist_surf

            clamped = max(self.min_lidar_range, min(self.max_lidar_range, ray_dist))
            ranges.append(clamped)
            if clamped < min_clearance:
                min_clearance = clamped

        return ranges, min_clearance


class LiveSimulationSession:
    """Manages an active cognitive navigation simulation trial."""

    def __init__(self, target_x: float = 3.2, target_y: float = 3.0, target_name: str = "Red Sofa") -> None:
        self.arena = SimulationArena()
        self.arena.target_x = target_x
        self.arena.target_y = target_y
        self.arena.target_name = target_name
        self.robot = KinematicRobot()
        self.sim_time = 0.0
        self.timeout_sec = 45.0
        self.is_active = True
        self.status = "RUNNING"
        self.reason = "IN_PROGRESS"
        self.trajectory: List[Dict[str, float]] = [{"x": 0.0, "y": 0.0}]
        self.min_obstacle_dist = 999.0

    def set_target(self, x: float, y: float, name: str = "Target") -> None:
        self.arena.target_x = x
        self.arena.target_y = y
        self.arena.target_name = name
        self.is_active = True
        self.status = "RUNNING"
        self.reason = "IN_PROGRESS"

    def reset(self) -> None:
        self.robot = KinematicRobot()
        self.sim_time = 0.0
        self.is_active = True
        self.status = "RUNNING"
        self.reason = "IN_PROGRESS"
        self.trajectory = [{"x": 0.0, "y": 0.0}]
        self.min_obstacle_dist = 999.0

    def tick(self, linear_cmd: float, angular_cmd: float, dt: float = 0.05) -> Dict[str, Any]:
        """Advances simulation by dt and computes spatial perception and Oracle metrics."""
        if not self.is_active:
            return self.get_state()

        self.robot.step(linear_cmd, angular_cmd, dt)
        self.sim_time += dt

        # Record trajectory
        self.trajectory.append({"x": round(self.robot.x, 3), "y": round(self.robot.y, 3)})

        # Camera sightings
        self.robot.check_camera_sightings(self.arena)

        # LiDAR calculation
        ranges, clearance = self.robot.compute_lidar(self.arena)
        if clearance < self.min_obstacle_dist:
            self.min_obstacle_dist = clearance

        # Distance to active target
        dx = self.arena.target_x - self.robot.x
        dy = self.arena.target_y - self.robot.y
        goal_dist = math.hypot(dx, dy)

        # Goal Reached
        if goal_dist <= self.arena.goal_tolerance:
            self.is_active = False
            self.status = "PASSED"
            self.reason = "GOAL_REACHED"

        # Collision
        elif clearance <= (self.robot.radius + 0.02):
            self.is_active = False
            self.status = "FAILED"
            self.reason = "COLLISION"

        # Timeout
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
                "angular_vel": round(self.robot.angular_vel, 3),
                "camera_fov": round(self.robot.camera_fov, 3),
                "camera_range": self.robot.camera_range
            },
            "goal_distance": round(goal_dist, 3),
            "min_obstacle_distance": round(self.min_obstacle_dist, 3),
            "lidar_ranges": [round(r, 3) for r in ranges],
            "target": {
                "name": self.arena.target_name,
                "x": self.arena.target_x,
                "y": self.arena.target_y,
                "tolerance": self.arena.goal_tolerance
            },
            "semantic_objects": self.arena.semantic_objects,
            "interior_walls": self.arena.interior_walls,
            "obstacles": self.arena.obstacles,
            "arena": {
                "x_min": self.arena.x_min, "x_max": self.arena.x_max,
                "y_min": self.arena.y_min, "y_max": self.arena.y_max
            }
        }

    def get_state(self) -> Dict[str, Any]:
        dx = self.arena.target_x - self.robot.x
        dy = self.arena.target_y - self.robot.y
        ranges, clearance = self.robot.compute_lidar(self.arena)
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
                "angular_vel": round(self.robot.angular_vel, 3),
                "camera_fov": round(self.robot.camera_fov, 3),
                "camera_range": self.robot.camera_range
            },
            "goal_distance": round(math.hypot(dx, dy), 3),
            "min_obstacle_distance": round(self.min_obstacle_dist, 3),
            "lidar_ranges": [round(r, 3) for r in ranges],
            "target": {
                "name": self.arena.target_name,
                "x": self.arena.target_x,
                "y": self.arena.target_y,
                "tolerance": self.arena.goal_tolerance
            },
            "semantic_objects": self.arena.semantic_objects,
            "interior_walls": self.arena.interior_walls,
            "obstacles": self.arena.obstacles,
            "arena": {
                "x_min": self.arena.x_min, "x_max": self.arena.x_max,
                "y_min": self.arena.y_min, "y_max": self.arena.y_max
            }
        }
