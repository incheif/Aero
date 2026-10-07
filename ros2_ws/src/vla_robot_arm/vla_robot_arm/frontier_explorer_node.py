"""
ROS 2 Tabletop Frontier Explorer Node.
Handles active scanning requests and emits quadrant exploration coverage metrics.
"""

import json
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String, Float32
except ImportError:
    class Node:
        def __init__(self, name): self.name = name

from vla.frontier_explorer import TabletopFrontierExplorer

class FrontierExplorerNode(Node):
    def __init__(self):
        super().__init__('frontier_explorer_node')
        self.explorer = TabletopFrontierExplorer()

        self.coverage_pub = self.create_publisher(Float32, '/vla/frontier_coverage_pct', 10)
        self.status_pub = self.create_publisher(String, '/vla/frontier_status', 10)

        self.timer = self.create_timer(0.5, self.timer_callback)
        self.get_logger().info('Frontier Explorer Node online. Ready for active tabletop mapping.')

    def timer_callback(self):
        cov_msg = Float32()
        cov_msg.data = float(self.explorer.exploration_pct)
        self.coverage_pub.publish(cov_msg)

        status_msg = String()
        status_msg.data = json.dumps(self.explorer.to_dict())
        self.status_pub.publish(status_msg)

def main(args=None):
    if 'rclpy' in globals():
        rclpy.init(args=args)
        node = FrontierExplorerNode()
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
