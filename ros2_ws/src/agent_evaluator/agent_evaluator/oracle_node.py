#!/usr/bin/env python3
"""
oracle_node.py

Impartial Ground Truth Supervisor for AERO (Agentic Environment for Robotic Operations).
Monitors robot state (/odom, /scan, /cmd_vel) against simulation ground truth.
Terminates early on goal arrival (<=0.1m), collision, topic starvation, or 30s timeout.
Produces a structured `test_results.json` artifact upon completion.
"""

import json
import math
import os
import sys
import tempfile
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


class GroundTruthOracle(Node):
    """
    Independent supervisor node that observes robot odometry, sensor scans,
    and velocity commands to evaluate navigation trials.
    """

    def __init__(self) -> None:
        super().__init__('ground_truth_oracle')

        # Declare configurable parameters
        self.declare_parameter('target_x', 3.0)
        self.declare_parameter('target_y', 3.0)
        self.declare_parameter('goal_tolerance', 0.10)
        self.declare_parameter('timeout_sec', 30.0)
        self.declare_parameter('collision_distance', 0.18)
        self.declare_parameter('starvation_timeout_sec', 5.0)
        self.declare_parameter('output_json_path', 'test_results.json')

        # Retrieve parameter values
        self.target_x = float(self.get_parameter('target_x').value)
        self.target_y = float(self.get_parameter('target_y').value)
        self.goal_tolerance = float(self.get_parameter('goal_tolerance').value)
        self.timeout_sec = float(self.get_parameter('timeout_sec').value)
        self.collision_dist_threshold = float(self.get_parameter('collision_distance').value)
        self.starvation_timeout_sec = float(self.get_parameter('starvation_timeout_sec').value)
        self.output_json_path = str(self.get_parameter('output_json_path').value)

        # Internal tracking state
        self.start_sim_time: Optional[Time] = None
        self.current_x: float = 0.0
        self.current_y: float = 0.0
        self.has_received_odom: bool = False
        self.has_received_cmd_vel: bool = False
        self.min_obstacle_distance: float = float('inf')
        self.trial_completed: bool = False
        self.collision_detected: bool = False

        # Subscriptions
        self.create_subscription(Odometry, '/odom', self._odom_callback, 10)
        self.create_subscription(LaserScan, '/scan', self._scan_callback, 10)
        self.create_subscription(Twist, '/cmd_vel', self._cmd_vel_callback, 10)

        # High-frequency evaluation timer (20 Hz)
        self.eval_timer = self.create_timer(0.05, self._evaluate_step)

        self.get_logger().info(
            f"GroundTruthOracle initialized. Target=({self.target_x}, {self.target_y}), "
            f"GoalTol={self.goal_tolerance}m, Timeout={self.timeout_sec}s, "
            f"Output='{self.output_json_path}'"
        )

    def _odom_callback(self, msg: Odometry) -> None:
        """Track current robot Cartesian coordinates from odometry ground truth."""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.has_received_odom = True

    def _scan_callback(self, msg: LaserScan) -> None:
        """Monitor obstacle proximity and detect collision events."""
        valid_ranges = [r for r in msg.ranges if msg.range_min < r < msg.range_max]
        if valid_ranges:
            current_min = min(valid_ranges)
            if current_min < self.min_obstacle_distance:
                self.min_obstacle_distance = current_min

            if current_min <= self.collision_dist_threshold:
                self.collision_detected = True

    def _cmd_vel_callback(self, msg: Twist) -> None:
        """Detect robot controller activity to check against topic starvation."""
        # Non-zero velocity commands mark controller as active
        if abs(msg.linear.x) > 1e-4 or abs(msg.angular.z) > 1e-4:
            self.has_received_cmd_vel = True

    def _evaluate_step(self) -> None:
        """Step function called at 20 Hz to evaluate termination conditions."""
        if self.trial_completed:
            return

        now = self.get_clock().now()
        if self.start_sim_time is None:
            self.start_sim_time = now
            return

        # Elapsed simulation time in seconds
        elapsed_sec = (now - self.start_sim_time).nanoseconds / 1e9

        # Calculate current Euclidean distance to target
        distance_to_target = math.hypot(
            self.target_x - self.current_x,
            self.target_y - self.current_y
        )

        # Condition 1: SUCCESS (Goal reached within tolerance)
        if distance_to_target <= self.goal_tolerance:
            self._finalize_trial(
                status="PASSED",
                reason="SUCCESS",
                final_distance=distance_to_target,
                elapsed_sec=elapsed_sec,
                error_snippet=""
            )
            return

        # Condition 2: COLLISION
        if self.collision_detected:
            self._finalize_trial(
                status="FAILED",
                reason="COLLISION",
                final_distance=distance_to_target,
                elapsed_sec=elapsed_sec,
                error_snippet=(
                    f"Collision detected! Min obstacle clearance {self.min_obstacle_distance:.3f}m "
                    f"breached threshold {self.collision_dist_threshold:.3f}m."
                )
            )
            return

        # Condition 3: TOPIC_STARVATION (No command received within grace period)
        if elapsed_sec >= self.starvation_timeout_sec and not self.has_received_cmd_vel:
            self._finalize_trial(
                status="FAILED",
                reason="TOPIC_STARVATION",
                final_distance=distance_to_target,
                elapsed_sec=elapsed_sec,
                error_snippet=(
                    f"Robot controller failed to publish non-zero /cmd_vel within "
                    f"{self.starvation_timeout_sec}s grace period."
                )
            )
            return

        # Condition 4: TIMEOUT (Exceeded allotted simulation seconds)
        if elapsed_sec >= self.timeout_sec:
            self._finalize_trial(
                status="FAILED",
                reason="TIMEOUT",
                final_distance=distance_to_target,
                elapsed_sec=elapsed_sec,
                error_snippet=(
                    f"Simulation timed out after {self.timeout_sec}s. "
                    f"Robot stopped at ({self.current_x:.2f}, {self.current_y:.2f}), "
                    f"remaining distance to target: {distance_to_target:.3f}m."
                )
            )
            return

    def _finalize_trial(
        self,
        status: str,
        reason: str,
        final_distance: float,
        elapsed_sec: float,
        error_snippet: str
    ) -> None:
        """Write test_results.json atomically and initiate node shutdown."""
        self.trial_completed = True

        results = {
            "status": status,
            "reason": reason,
            "final_distance": round(float(final_distance), 4),
            "time_elapsed_sec": round(float(elapsed_sec), 2),
            "min_obstacle_distance": (
                round(float(self.min_obstacle_distance), 4)
                if not math.isinf(self.min_obstacle_distance) else 999.0
            ),
            "error_log_snippet": error_snippet
        }

        self.get_logger().info(
            f"=== TRIAL FINISHED ===\n"
            f"Status: {status} | Reason: {reason} | "
            f"Distance: {results['final_distance']}m | Time: {results['time_elapsed_sec']}s"
        )

        # Atomic write to avoid partial reads by the harness
        output_dir = os.path.dirname(os.path.abspath(self.output_json_path))
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        tmp_file = tempfile.NamedTemporaryFile(
            mode='w', dir=output_dir, delete=False, suffix='.tmp'
        )
        try:
            json.dump(results, tmp_file, indent=2)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
            tmp_file.close()
            os.replace(tmp_file.name, self.output_json_path)
            self.get_logger().info(f"Results written to '{self.output_json_path}'")
        except Exception as e:
            self.get_logger().error(f"Failed writing results JSON: {e}")
            if os.path.exists(tmp_file.name):
                os.remove(tmp_file.name)

        # Cancel timer to cease execution
        self.eval_timer.cancel()

        # Clean shutdown after short flush delay
        self.create_timer(0.2, self._shutdown_node)

    def _shutdown_node(self) -> None:
        """Gracefully exit node process."""
        self.get_logger().info("Shutting down GroundTruthOracle node.")
        raise SystemExit(0)


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = GroundTruthOracle()
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
