"""
ROS 2 Launch file for Google VLA Robot Arm Manipulation Architecture.
"""

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='vla_robot_arm',
            executable='vla_cognitive_node',
            name='vla_cognitive_node',
            output='screen',
        ),
        Node(
            package='vla_robot_arm',
            executable='semantic_spatial_mapper_node',
            name='semantic_spatial_mapper_node',
            output='screen',
        ),
        Node(
            package='vla_robot_arm',
            executable='frontier_explorer_node',
            name='frontier_explorer_node',
            output='screen',
        ),
    ])
