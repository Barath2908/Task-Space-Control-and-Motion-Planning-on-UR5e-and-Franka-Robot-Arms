#!/usr/bin/env python3
"""
Launch file for UR5e Cartesian controller.

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
        default_value='500.0',
        description='Control loop rate in Hz'
    )
    
    max_velocity_arg = DeclareLaunchArgument(
        'max_joint_velocity',
        default_value='1.5',
        description='Maximum joint velocity in rad/s'
    )
    
    damping_arg = DeclareLaunchArgument(
        'damping_factor',
        default_value='0.05',
        description='DLS damping factor for singularity robustness'
    )
    
    # Get config file path
    config_file = PathJoinSubstitution([
        FindPackageShare('cartesian_controllers'),
        'config',
        'ur5e_params.yaml'
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
                'robot': 'ur5e',
                'control_rate': LaunchConfiguration('control_rate'),
                'max_joint_velocity': LaunchConfiguration('max_joint_velocity'),
                'damping_factor': LaunchConfiguration('damping_factor'),
                'enable_nullspace': False,
            }
        ],
        remappings=[
            ('joint_states', '/joint_states'),
            ('cartesian_pose_cmd', '/cartesian_pose_cmd'),
            ('joint_velocity_command', '/joint_group_vel_controller/commands'),
        ]
    )
    
    return LaunchDescription([
        control_rate_arg,
        max_velocity_arg,
        damping_arg,
        cartesian_controller_node,
    ])
