"""
ROS 2 Google VLA Cognitive Node.
Subscribes to high-level natural language instructions, processes visual scene frames
using Google VLA (Gemini 2.5 Flash / 2.0 Pro), and publishes structured action trajectories.
"""

import os
import json
from typing import Optional

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    from sensor_msgs.msg import Image
except ImportError:
    # Graceful fallback when running outside active ROS 2 environment
    class Node:
        def __init__(self, name): self.name = name

# Import VLA agent from project root
from vla.google_vla_agent import GoogleVLAAgent
from vla.semantic_mapper import SemanticSpatialMapper

class VLACognitiveNode(Node):
    def __init__(self):
        super().__init__('vla_cognitive_node')
        self.vla_agent = GoogleVLAAgent(
            api_key=os.environ.get('GEMINI_API_KEY', ''),
            model_name='gemini-2.5-flash'
        )
        self.semantic_mapper = SemanticSpatialMapper()

        # ROS 2 Subscriptions
        self.cmd_sub = self.create_subscription(
            String,
            '/vla/natural_language_command',
            self.command_callback,
            10
        )

        # ROS 2 Publishers
        self.thought_pub = self.create_publisher(String, '/vla/cognitive_thought', 10)
        self.plan_pub = self.create_publisher(String, '/vla/action_plan', 10)

        self.get_logger().info('Google VLA Cognitive Node started. Listening on /vla/natural_language_command.')

    def command_callback(self, msg: String):
        instruction = msg.data.strip()
        self.get_logger().info(f'Received natural-language command: "{instruction}"')

        # Dummy snapshot representation for headless ROS 2 mode
        mock_snapshot = {
            "blocks": [
                {"id": "block_cyan", "name": "Cyan Cube", "color": "#00AEEF", "position": [0.26, 0.75, -0.16]},
                {"id": "block_orange", "name": "Orange Cube", "color": "#F7941E", "position": [0.36, 0.75, 0.0]},
                {"id": "block_magenta", "name": "Magenta Cube", "color": "#E11D8F", "position": [0.28, 0.75, 0.16]},
            ],
            "tcp": {"position": [0.0, 0.85, 0.0]}
        }

        # Plan with Google VLA
        result = self.vla_agent.parse_instruction(
            instruction=instruction,
            image=None,
            snapshot=mock_snapshot,
            semantic_summary=self.semantic_mapper.get_semantic_summary()
        )

        # Publish Thought Stream
        thought_msg = String()
        thought_msg.data = result.get('thought', '')
        self.thought_pub.publish(thought_msg)

        # Publish Action Plan
        plan_msg = String()
        plan_msg.data = json.dumps(result.get('plan', []))
        self.plan_pub.publish(plan_msg)

        self.get_logger().info('Google VLA plan synthesized and published.')

def main(args=None):
    if 'rclpy' in globals():
        rclpy.init(args=args)
        node = VLACognitiveNode()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()
    else:
        print("[!] Note: ROS 2 rclpy environment not sourced. Run within ROS 2 Humble/Jazzy terminal.")

if __name__ == '__main__':
    main()
