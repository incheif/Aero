#!/usr/bin/env python3
"""
eval_headless.launch.py

Orchestrates headless Gazebo simulation, robot state publishers,
spawns TurtleBot3, launches target controller node, and runs the Ground Truth Oracle.
Cleanly shuts down the entire launch process tree when the Oracle completes.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    RegisterEventHandler,
    EmitEvent,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Workspace & package paths
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    
    # Launch arguments
    world_arg = DeclareLaunchArgument(
        'world',
        default_value=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
            'benchmarks', 'worlds', 'navigation_arena.world'
        ),
        description='Path to the Gazebo simulation world file'
    )

    output_json_path_arg = DeclareLaunchArgument(
        'output_json_path',
        default_value='test_results.json',
        description='Path where test_results.json will be written by the Oracle'
    )

    timeout_sec_arg = DeclareLaunchArgument(
        'timeout_sec',
        default_value='30.0',
        description='Simulation timeout budget in seconds'
    )

    target_x_arg = DeclareLaunchArgument(
        'target_x', default_value='3.0', description='Target X coordinate'
    )

    target_y_arg = DeclareLaunchArgument(
        'target_y', default_value='3.0', description='Target Y coordinate'
    )

    # 1. Gazebo Server (Headless: gzserver only, no GUI gzclient)
    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={'world': LaunchConfiguration('world'), 'verbose': 'false'}.items()
    )

    # 2. TurtleBot3 Robot State Publisher & Spawner
    # Fallback to turtlebot3_gazebo if installed
    tb3_gazebo_launch_dir = ''
    try:
        tb3_gazebo_share = get_package_share_directory('turtlebot3_gazebo')
        tb3_gazebo_launch_dir = os.path.join(tb3_gazebo_share, 'launch')
    except Exception:
        pass

    spawn_turtlebot_cmd = None
    if tb3_gazebo_launch_dir and os.path.exists(os.path.join(tb3_gazebo_launch_dir, 'robot_state_publisher.launch.py')):
        spawn_turtlebot_cmd = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(tb3_gazebo_launch_dir, 'robot_state_publisher.launch.py')
            ),
            launch_arguments={'use_sim_time': 'true'}.items()
        )

    # Spawn entity node at (0, 0, 0)
    spawn_entity_node = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'turtlebot3_waffle',
            '-topic', 'robot_description',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.01'
        ],
        output='screen'
    )

    # 3. Target Robot Controller Node (under test / modified by agent)
    controller_node = Node(
        package='robot_controller',
        executable='controller_node',
        name='robot_controller_node',
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    # 4. Ground Truth Oracle Node
    oracle_node = Node(
        package='agent_evaluator',
        executable='oracle_node',
        name='ground_truth_oracle',
        parameters=[{
            'use_sim_time': True,
            'target_x': LaunchConfiguration('target_x'),
            'target_y': LaunchConfiguration('target_y'),
            'timeout_sec': LaunchConfiguration('timeout_sec'),
            'output_json_path': LaunchConfiguration('output_json_path')
        }],
        output='screen'
    )

    # Event handler: When Oracle finishes, trigger full launch shutdown
    shutdown_on_oracle_exit = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=oracle_node,
            on_exit=[
                EmitEvent(event=Shutdown(reason='GroundTruthOracle completed trial.'))
            ]
        )
    )

    ld = LaunchDescription()
    ld.add_action(world_arg)
    ld.add_action(output_json_path_arg)
    ld.add_action(timeout_sec_arg)
    ld.add_action(target_x_arg)
    ld.add_action(target_y_arg)
    ld.add_action(gzserver_cmd)
    if spawn_turtlebot_cmd:
        ld.add_action(spawn_turtlebot_cmd)
    ld.add_action(spawn_entity_node)
    ld.add_action(controller_node)
    ld.add_action(oracle_node)
    ld.add_action(shutdown_on_oracle_exit)

    return ld
