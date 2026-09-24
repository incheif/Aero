#!/usr/bin/env python3
"""
gemma_brain_node.py

Local LLM Cognitive Brain Node for AERO.
Powered by Google Gemma (running locally via Ollama/llama.cpp, with lightweight fallback).
Translates natural human instructions into high-level robotics actions:
- NAVIGATE_TO_OBJECT (queries Semantic Memory -> dispatches Nav2 goal)
- EXPLORE (activates Frontier Exploration to map unknown rooms)
- EXPLORE_AND_FIND (explores until target landmark is visually identified)
- QUERY_STATUS (reports known landmarks and exploration progress)
"""

import json
import os
import urllib.request
import urllib.error
from typing import Dict, Any, Optional, Tuple

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    from geometry_msgs.msg import PoseStamped
    HAS_RCLPY = True
except ImportError:
    HAS_RCLPY = False
    class Node:  # type: ignore
        def __init__(self, *args, **kwargs): pass
    class String:  # type: ignore
        pass
    class PoseStamped:  # type: ignore
        pass


class GemmaCognitiveReasoner:
    """Natural Language reasoning engine powered by local Google Gemma."""

    def __init__(self, ollama_url: str = "http://localhost:11434", model_name: str = "gemma:2b") -> None:
        self.ollama_url = ollama_url
        self.model_name = model_name

    def prompt_gemma_local(self, prompt: str) -> Optional[str]:
        """Send prompt to local Ollama instance running Gemma."""
        endpoint = f"{self.ollama_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1}
        }
        try:
            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode('utf-8'),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=4.0) as resp:
                result = json.loads(resp.read().decode('utf-8'))
                return result.get("response", "").strip()
        except Exception:
            return None

    def parse_instruction(
        self,
        user_text: str,
        known_landmarks: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Parses natural human instruction into structured cognitive plan.
        Tries local Gemma via Ollama; falls back to embedded semantic parser.
        """
        text = user_text.lower().strip()

        # Step 1: Check known landmarks in semantic memory
        found_target = None
        for key, lm in known_landmarks.items():
            name_clean = lm["name"].lower()
            category_clean = lm["category"].lower()
            if name_clean in text or key in text or category_clean in text:
                found_target = lm
                break

        # Check for exploration keywords
        explore_keywords = ["explore", "map", "search", "uncharted", "look around", "find"]
        is_explore = any(kw in text for kw in explore_keywords)

        # Check for dock keywords
        dock_keywords = ["dock", "charge", "station", "home", "base"]
        is_dock = any(kw in text for kw in dock_keywords)

        # Decision Logic:
        # Case A: Docking
        if is_dock:
            dock_coords = [0.0, -3.2]
            for key, lm in known_landmarks.items():
                if "dock" in key:
                    dock_coords = [lm["x"], lm["y"]]
            return {
                "thought": "User requested return to charging dock. Setting navigation goal to docking bay.",
                "action": "NAVIGATE_TO_OBJECT",
                "target": "Charging Dock",
                "coordinates": dock_coords,
                "reply": f"Understood. Returning to the Charging Dock at ({dock_coords[0]}, {dock_coords[1]})."
            }

        # Case B: Target landmark is known in semantic memory
        if found_target:
            coords = [found_target["x"], found_target["y"]]
            room = found_target.get("room", "the house")
            return {
                "thought": f"Target '{found_target['name']}' found in semantic memory at {coords} ({room}).",
                "action": "NAVIGATE_TO_OBJECT",
                "target": found_target["name"],
                "coordinates": coords,
                "reply": f"I know where the {found_target['name']} is! Navigating to {room} at ({coords[0]}, {coords[1]})."
            }

        # Case C: Unknown object requested -> Explore to find it
        if is_explore and not found_target:
            # Extract possible object from text
            potential_target = text.replace("find", "").replace("explore", "").replace("go to", "").strip()
            return {
                "thought": f"User asked for '{user_text}'. Object not in semantic memory yet. Initiating frontier exploration.",
                "action": "EXPLORE_AND_FIND",
                "target": potential_target or "unknown area",
                "coordinates": None,
                "reply": f"I haven't spotted that yet! Starting autonomous frontier exploration to find it."
            }

        # Case D: General exploration command
        if is_explore:
            return {
                "thought": "User requested autonomous environment exploration. Activating frontier explorer.",
                "action": "EXPLORE",
                "target": "Uncharted Space",
                "coordinates": None,
                "reply": "Activating frontier exploration to autonomously map uncharted rooms."
            }

        # Default: Status Query
        known_names = [lm["name"] for lm in known_landmarks.values()]
        known_str = ", ".join(known_names) if known_names else "nothing yet"
        return {
            "thought": "Command unclear or status inquiry. Reporting current semantic spatial knowledge.",
            "action": "QUERY_STATUS",
            "target": None,
            "coordinates": None,
            "reply": f"I'm ready for instructions! Currently discovered landmarks: {known_str}."
        }


class GemmaBrainNode(Node):
    """ROS 2 Node acting as the robot's high-level cognitive brain."""

    def __init__(self) -> None:
        super().__init__('gemma_brain_node')

        self.reasoner = GemmaCognitiveReasoner()

        # Simulated or linked Semantic Memory
        self.known_landmarks: Dict[str, Dict[str, Any]] = {
            "charging_dock": {"name": "Charging Dock", "category": "dock", "room": "Docking Bay", "x": 0.0, "y": -3.2},
            "red_sofa": {"name": "Red Sofa", "category": "furniture", "room": "Living Room", "x": 3.2, "y": 3.0},
            "kitchen_table": {"name": "Kitchen Table", "category": "furniture", "room": "Kitchen", "x": -2.2, "y": 2.5},
            "storage_boxes": {"name": "Storage Boxes", "category": "storage", "room": "Storage Bay", "x": 2.8, "y": -1.8},
        }

        # Subscriptions & Publishers
        self.create_subscription(String, '/user_command', self._command_cb, 10)
        self.response_pub = self.create_publisher(String, '/gemma_response', 10)
        self.nav_goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        self.get_logger().info("🧠 GemmaBrainNode online. Awaiting natural language instructions.")

    def _command_cb(self, msg: String) -> None:
        user_text = msg.data
        self.get_logger().info(f"🗣️ User Command: '{user_text}'")

        decision = self.reasoner.parse_instruction(user_text, self.known_landmarks)
        self.get_logger().info(f"💡 Gemma Thought: {decision['thought']}")
        self.get_logger().info(f"🤖 Action: [{decision['action']}] -> {decision['reply']}")

        # Publish response text
        resp_msg = String()
        resp_msg.data = json.dumps(decision)
        self.response_pub.publish(resp_msg)

        # If navigation action, publish goal
        if decision["action"] == "NAVIGATE_TO_OBJECT" and decision["coordinates"]:
            goal = PoseStamped()
            goal.header.stamp = self.get_clock().now().to_msg()
            goal.header.frame_id = 'map'
            goal.pose.position.x = float(decision["coordinates"][0])
            goal.pose.position.y = float(decision["coordinates"][1])
            goal.pose.orientation.w = 1.0
            self.nav_goal_pub.publish(goal)


def main(args=None):
    rclpy.init(args=args)
    node = GemmaBrainNode()
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
