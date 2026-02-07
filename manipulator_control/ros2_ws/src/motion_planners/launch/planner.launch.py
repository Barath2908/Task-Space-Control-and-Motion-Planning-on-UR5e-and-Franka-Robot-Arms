#!/usr/bin/env python3
"""
Launch file for motion planners (RRT, RRT*, Bi-RRT*).

Author: Barath Kumar JK
Date: 2025
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare arguments
    algorithm_arg = DeclareLaunchArgument(
        'algorithm',
        default_value='birrt_star',
        description='Planning algorithm: rrt, rrt_star, birrt_star'
    )
    
    num_joints_arg = DeclareLaunchArgument(
        'num_joints',
        default_value='6',
        description='Number of robot joints'
    )
    
    timeout_arg = DeclareLaunchArgument(
        'timeout',
        default_value='5.0',
        description='Planning timeout in seconds'
    )
    
    max_iterations_arg = DeclareLaunchArgument(
        'max_iterations',
        default_value='10000',
        description='Maximum planning iterations'
    )
    
    step_size_arg = DeclareLaunchArgument(
        'step_size',
        default_value='0.1',
        description='RRT step size in joint space (rad)'
    )
    
    goal_bias_arg = DeclareLaunchArgument(
        'goal_bias',
        default_value='0.1',
        description='Probability of sampling goal'
    )
    
    # Get config file path
    config_file = PathJoinSubstitution([
        FindPackageShare('motion_planners'),
        'config',
        'planner_params.yaml'
    ])
    
    # Planner node
    planner_node = Node(
        package='motion_planners',
        executable='planner_node',
        name='motion_planner',
        output='screen',
        parameters=[
            config_file,
            {
                'algorithm': LaunchConfiguration('algorithm'),
                'num_joints': LaunchConfiguration('num_joints'),
                'timeout': LaunchConfiguration('timeout'),
                'max_iterations': LaunchConfiguration('max_iterations'),
                'step_size': LaunchConfiguration('step_size'),
                'goal_bias': LaunchConfiguration('goal_bias'),
            }
        ],
        remappings=[
            ('joint_states', '/joint_states'),
            ('planned_trajectory', '/joint_trajectory_controller/joint_trajectory'),
        ]
    )
    
    return LaunchDescription([
        algorithm_arg,
        num_joints_arg,
        timeout_arg,
        max_iterations_arg,
        step_size_arg,
        goal_bias_arg,
        planner_node,
    ])
