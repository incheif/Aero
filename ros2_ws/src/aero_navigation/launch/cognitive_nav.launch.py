#!/usr/bin/env python3
"""
cognitive_nav.launch.py

Master Launch File for AERO Cognitive Explorer.
Brings up:
1. Gazebo simulation with `semantic_house.world`
2. TurtleBot3 Waffle Pi with LiDAR and RGB Camera
3. SLAM Toolbox for online occupancy grid mapping
4. Nav2 Navigation Stack (costmaps, DWB planner, recoveries)
5. Semantic 3D Object Mapper
6. Autonomous Frontier Explorer
7. Local Gemma Cognitive Reasoning Node
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    
    # Path to semantic house world
    current_pkg_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    world_path = os.path.join(current_pkg_dir, 'benchmarks', 'worlds', 'semantic_house.world')

    # Config files
    nav2_params_file = os.path.join(
        current_pkg_dir, 'ros2_ws', 'src', 'aero_navigation', 'config', 'nav2_params.yaml'
    )
    slam_params_file = os.path.join(
        current_pkg_dir, 'ros2_ws', 'src', 'aero_navigation', 'config', 'slam_params.yaml'
    )

    # Launch Arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # 1. Gazebo Server & Client
    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')),
        launch_arguments={'world': world_path, 'verbose': 'false'}.items()
    )

    # 2. TurtleBot3 State Publisher & Spawner
    tb3_launch_dir = ''
    try:
        tb3_gazebo_share = get_package_share_directory('turtlebot3_gazebo')
        tb3_launch_dir = os.path.join(tb3_gazebo_share, 'launch')
    except Exception:
        pass

    robot_state_pub = None
    if tb3_launch_dir and os.path.exists(os.path.join(tb3_launch_dir, 'robot_state_publisher.launch.py')):
        robot_state_pub = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(tb3_launch_dir, 'robot_state_publisher.launch.py')),
            launch_arguments={'use_sim_time': use_sim_time}.items()
        )

    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-entity', 'turtlebot3_waffle_pi', '-topic', 'robot_description', '-x', '0.0', '-y', '-2.5', '-z', '0.01'],
        output='screen'
    )

    # 3. SLAM Toolbox Node
    slam_node = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        parameters=[slam_params_file, {'use_sim_time': use_sim_time}],
        output='screen'
    )

    # 4. Semantic Mapper Node
    semantic_mapper = Node(
        package='aero_navigation',
        executable='semantic_mapper_node',
        name='semantic_mapper_node',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # 5. Frontier Explorer Node
    frontier_explorer = Node(
        package='aero_navigation',
        executable='frontier_explorer_node',
        name='frontier_explorer_node',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # 6. Gemma Cognitive Brain Node
    gemma_cognitive = Node(
        package='aero_navigation',
        executable='gemma_cognitive_node',
        name='gemma_cognitive_node',
        parameters=[{'use_sim_time': use_sim_time, 'model_name': 'gemma:2b'}],
        output='screen'
    )

    ld = LaunchDescription()
    ld.add_action(gzserver)
    if robot_state_pub:
        ld.add_action(robot_state_pub)
    ld.add_action(spawn_robot)
    ld.add_action(slam_node)
    ld.add_action(semantic_mapper)
    ld.add_action(frontier_explorer)
    ld.add_action(gemma_cognitive)

    return ld
