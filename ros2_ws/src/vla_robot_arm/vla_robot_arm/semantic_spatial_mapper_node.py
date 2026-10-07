"""
ROS 2 Semantic Spatial Mapper Node.
Publishes 3D tabletop spatial semantic affordances, tracked entity poses,
and topological relationship graphs.
"""

import json
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    from geometry_msgs.msg import PoseArray, Pose
except ImportError:
    class Node:
        def __init__(self, name): self.name = name

from vla.semantic_mapper import SemanticSpatialMapper

class SemanticSpatialMapperNode(Node):
    def __init__(self):
        super().__init__('semantic_spatial_mapper_node')
        self.mapper = SemanticSpatialMapper()

        self.map_json_pub = self.create_publisher(String, '/vla/semantic_map', 10)
        self.poses_pub = self.create_publisher(PoseArray, '/vla/object_poses', 10)

        # Periodic timer (5 Hz)
        self.timer = self.create_timer(0.2, self.timer_callback)
        self.get_logger().info('Semantic Spatial Mapper Node online. Publishing to /vla/semantic_map.')

    def timer_callback(self):
        # Publish JSON state
        msg = String()
        msg.data = json.dumps(self.mapper.to_dict())
        self.map_json_pub.publish(msg)

def main(args=None):
    if 'rclpy' in globals():
        rclpy.init(args=args)
        node = SemanticSpatialMapperNode()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()
    else:
        print("[!] Note: ROS 2 rclpy environment not sourced.")

if __name__ == '__main__':
    main()
