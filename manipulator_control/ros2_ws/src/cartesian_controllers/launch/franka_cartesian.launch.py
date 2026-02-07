#!/usr/bin/env python3
"""
Launch file for Franka Panda Cartesian controller with nullspace control.

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
    control_rate_arg = DeclareLaunchArgument(
        'control_rate',
        default_value='1000.0',
        description='Control loop rate in Hz'
    )
    
    max_velocity_arg = DeclareLaunchArgument(
        'max_joint_velocity',
        default_value='1.5',
        description='Maximum joint velocity in rad/s'
    )
    
    enable_nullspace_arg = DeclareLaunchArgument(
        'enable_nullspace',
        default_value='true',
        description='Enable nullspace posture control'
    )
    
    posture_gain_arg = DeclareLaunchArgument(
        'posture_gain',
        default_value='2.0',
        description='Nullspace posture control gain'
    )
    
    # Get config file path
    config_file = PathJoinSubstitution([
        FindPackageShare('cartesian_controllers'),
        'config',
        'franka_params.yaml'
    ])
    
    # Cartesian controller node
    cartesian_controller_node = Node(
        package='cartesian_controllers',
        executable='cartesian_controller_node',
        name='cartesian_controller',
        output='screen',
        parameters=[
            config_file,
            {
                'robot': 'franka',
                'control_rate': LaunchConfiguration('control_rate'),
                'max_joint_velocity': LaunchConfiguration('max_joint_velocity'),
                'enable_nullspace': LaunchConfiguration('enable_nullspace'),
                'posture_gain': LaunchConfiguration('posture_gain'),
            }
        ],
        remappings=[
            ('joint_states', '/franka/joint_states'),
            ('cartesian_pose_cmd', '/cartesian_pose_cmd'),
            ('joint_velocity_command', '/franka/joint_velocity_controller/commands'),
        ]
    )
    
    return LaunchDescription([
        control_rate_arg,
        max_velocity_arg,
        enable_nullspace_arg,
        posture_gain_arg,
        cartesian_controller_node,
    ])
