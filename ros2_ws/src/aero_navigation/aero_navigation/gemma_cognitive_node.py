#!/usr/bin/env python3
"""
gemma_cognitive_node.py

Embodied AI Cognitive Planner for AERO.
Connects to local Google Gemma (via Ollama or local endpoint) to translate
natural language instructions into semantic queries, frontier exploration cycles,
and Nav2 navigation actions.
"""

import json
import os
import re
import urllib.request
from typing import Dict, Any, Optional, List

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import String, Bool
    HAS_RCLPY = True
except ImportError:
    HAS_RCLPY = False
    class Node:
        def __init__(self, *args, **kwargs): pass
    PoseStamped = String = Bool = object


GEMMA_SYSTEM_PROMPT = """You are AERO, an autonomous cognitive robot running Google Gemma.
You control a mobile robot equipped with Nav2, SLAM, frontier exploration, and a 3D semantic memory.

Available Tools:
1. `explore_unmapped_area()`: Triggers autonomous frontier exploration to discover new rooms.
2. `query_semantic_memory(object_name)`: Checks if an object (e.g. 'red sofa', 'kitchen table') has been mapped.
3. `navigate_to_object(object_name, clearance=0.5)`: Navigates to a known semantic object with safe distance.
4. `navigate_to_pose(x, y, yaw)`: Navigates to a specific metric coordinate.
5. `report_status(message)`: Informs the human user about current action or status.

Instructions:
- Analyze the user command.
- If the user wants to find an object, check semantic memory first. If not yet mapped, explore uncharted areas until found.
- Output your reasoning followed by a structured JSON tool call in the format:
```json
{
  "thought": "<your reasoning here>",
  "tool": "<tool_name>",
  "arguments": { ... }
}
```
"""


class GemmaCognitiveNode(Node):
    """Translates natural language into Nav2 goals and semantic queries using Gemma."""

    def __init__(self) -> None:
        super().__init__('gemma_cognitive_node')

        self.declare_parameter('ollama_url', 'http://localhost:11434')
        self.declare_parameter('model_name', 'gemma:2b')

        self.ollama_url = self.get_parameter('ollama_url').value
        self.model_name = self.get_parameter('model_name').value

        # Spatial scene memory received from semantic_mapper_node
        self.semantic_memory: Dict[str, Any] = {}

        # Subscriptions
        self.create_subscription(String, '/user_command', self._command_callback, 10)
        self.create_subscription(String, '/aero/semantic_map', self._semantic_map_callback, 10)

        # Action & Control Publishers
        self.nav_goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.explore_enable_pub = self.create_publisher(Bool, '/aero/explore_enable', 10)
        self.reasoning_pub = self.create_publisher(String, '/gemma_reasoning_trace', 10)

        self.get_logger().info(
            f"GemmaCognitiveNode active. Target model: '{self.model_name}' at {self.ollama_url}"
        )

    def _semantic_map_callback(self, msg: String) -> None:
        try:
            self.semantic_memory = json.loads(msg.data)
        except Exception:
            pass

    def _command_callback(self, msg: String) -> None:
        user_text = msg.data.strip()
        self.get_logger().info(f"🗣️ Received Human Command: '{user_text}'")
        self.process_human_instruction(user_text)

    def process_human_instruction(self, instruction: str) -> None:
        """Invokes Gemma and executes the resulting tool action."""
        # 1. Format context prompt
        known_objects_summary = ", ".join(self.semantic_memory.keys()) if self.semantic_memory else "None yet"
        user_prompt = (
            f"User Command: '{instruction}'\n"
            f"Known Objects in Map: [{known_objects_summary}]\n"
            f"What action should the robot take?"
        )

        self.publish_reasoning(f"🧠 Gemma is deliberating on: '{instruction}'...")

        # 2. Query Local Gemma (or fallback rule-based parser)
        response_text = self._query_gemma(user_prompt)
        parsed_action = self._parse_tool_call(response_text, instruction)

        thought = parsed_action.get("thought", "Executing planned action.")
        tool = parsed_action.get("tool", "report_status")
        args = parsed_action.get("arguments", {})

        self.publish_reasoning(f"💡 Gemma Thought: {thought}")
        self.publish_reasoning(f"⚙️ Tool Dispatched: `{tool}` with {args}")

        # 3. Execute Dispatched Tool
        self._dispatch_tool(tool, args)

    def _query_gemma(self, prompt: str) -> str:
        """Sends request to local Ollama Gemma instance with automatic mock fallback."""
        url = f"{self.ollama_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": f"{GEMMA_SYSTEM_PROMPT}\n\n{prompt}",
            "stream": False
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=4.0) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result.get('response', '')
        except Exception:
            # Fallback cognitive reasoning if Ollama is not locally running
            return self._heuristic_gemma_fallback(prompt)

    def _heuristic_gemma_fallback(self, prompt: str) -> str:
        """Deterministic cognitive parser used when local Ollama is offline."""
        lower = prompt.lower()
        if "sofa" in lower or "couch" in lower:
            if "red sofa" in self.semantic_memory:
                pos = self.semantic_memory["red sofa"]
                return json.dumps({
                    "thought": "I found 'red sofa' in my 3D spatial memory. Navigating to its location.",
                    "tool": "navigate_to_pose",
                    "arguments": {"x": pos["x"] - 0.5, "y": pos["y"], "yaw": 0.0}
                })
            else:
                return json.dumps({
                    "thought": "The 'red sofa' is not in my map yet. I will start frontier exploration to discover it.",
                    "tool": "explore_unmapped_area",
                    "arguments": {}
                })

        elif "table" in lower or "kitchen" in lower:
            if "kitchen table" in self.semantic_memory:
                pos = self.semantic_memory["kitchen table"]
                return json.dumps({
                    "thought": "I know where the kitchen table is. Dispatching Nav2 goal.",
                    "tool": "navigate_to_pose",
                    "arguments": {"x": pos["x"] + 0.6, "y": pos["y"], "yaw": 3.14}
                })
            else:
                return json.dumps({
                    "thought": "Kitchen table not yet mapped. Activating frontier exploration.",
                    "tool": "explore_unmapped_area",
                    "arguments": {}
                })

        elif "dock" in lower or "charge" in lower:
            return json.dumps({
                "thought": "Navigating back to the charging dock origin.",
                "tool": "navigate_to_pose",
                "arguments": {"x": 0.0, "y": -3.2, "yaw": 1.57}
            })

        elif "explore" in lower or "map" in lower or "chart" in lower:
            return json.dumps({
                "thought": "User requested area exploration. Enabling autonomous frontier search.",
                "tool": "explore_unmapped_area",
                "arguments": {}
            })

        return json.dumps({
            "thought": "Received instruction. Monitoring environment.",
            "tool": "report_status",
            "arguments": {"message": "Awaiting specific destination or object command."}
        })

    def _parse_tool_call(self, response_text: str, default_cmd: str) -> Dict[str, Any]:
        """Extracts JSON tool call block from LLM output."""
        try:
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception:
            pass
        return {
            "thought": "Interpreted command directly.",
            "tool": "explore_unmapped_area" if "explore" in default_cmd.lower() else "report_status",
            "arguments": {}
        }

    def _dispatch_tool(self, tool: str, args: Dict[str, Any]) -> None:
        """Executes the tool call by interacting with Nav2 or SLAM services."""
        if tool == "explore_unmapped_area":
            b = Bool()
            b.data = True
            self.explore_enable_pub.publish(b)
            self.publish_reasoning("🗺️ Frontier Exploration Activated: Mapping uncharted boundaries...")

        elif tool in ("navigate_to_pose", "navigate_to_object"):
            # Disable frontier auto-exploration so Nav2 can drive to specific goal
            b = Bool()
            b.data = False
            self.explore_enable_pub.publish(b)

            goal = PoseStamped()
            goal.header.stamp = self.get_clock().now().to_msg()
            goal.header.frame_id = 'map'
            goal.pose.position.x = float(args.get("x", 0.0))
            goal.pose.position.y = float(args.get("y", 0.0))
            goal.pose.position.z = 0.0
            goal.pose.orientation.w = 1.0

            self.nav_goal_pub.publish(goal)
            self.publish_reasoning(
                f"🚀 Nav2 Goal Dispatched: Coordinate ({goal.pose.position.x:.2f}, {goal.pose.position.y:.2f}) in 'map' frame"
            )

        elif tool == "report_status":
            msg = args.get("message", "Ready for commands.")
            self.publish_reasoning(f"📢 Robot Status: {msg}")

    def publish_reasoning(self, text: str) -> None:
        msg = String()
        msg.data = text
        self.reasoning_pub.publish(msg)
        self.get_logger().info(text)


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = GemmaCognitiveNode()
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
