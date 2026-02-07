#!/usr/bin/env python3
"""
Trajectory Tracker for Cartesian Control

Tracks reference trajectories with real-time performance monitoring.
Computes RMS tracking error, joint rate statistics, and convergence metrics.

Author: Barath Kumar JK
Date: 2025
"""

import numpy as np
from typing import List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Path
import threading
import time


class TrajectoryType(Enum):
    LINE = "line"
    CIRCLE = "circle"
    FIGURE_EIGHT = "figure_eight"
    CUSTOM = "custom"


@dataclass
class TrajectoryPoint:
    """Single point on trajectory."""
    position: np.ndarray
    orientation: np.ndarray  # quaternion (w, x, y, z)
    velocity: Optional[np.ndarray] = None
    timestamp: float = 0.0


@dataclass
class TrackingMetrics:
    """Performance metrics for trajectory tracking."""
    rms_position_error: float = 0.0
    rms_orientation_error: float = 0.0
    max_position_error: float = 0.0
    max_orientation_error: float = 0.0
    mean_joint_velocity: float = 0.0
    max_joint_velocity: float = 0.0
    tracking_duration: float = 0.0
    samples_collected: int = 0
    convergence_time: Optional[float] = None
    
    def __str__(self) -> str:
        return (
            f"Tracking Metrics:\n"
            f"  RMS Position Error: {self.rms_position_error*1000:.2f} mm\n"
            f"  RMS Orientation Error: {np.degrees(self.rms_orientation_error):.2f} deg\n"
            f"  Max Position Error: {self.max_position_error*1000:.2f} mm\n"
            f"  Max Joint Velocity: {self.max_joint_velocity:.3f} rad/s\n"
            f"  Duration: {self.tracking_duration:.2f} s\n"
            f"  Samples: {self.samples_collected}"
        )


class TrajectoryGenerator:
    """Generate reference trajectories."""
    
    @staticmethod
    def line(start: np.ndarray, end: np.ndarray, 
             duration: float, dt: float = 0.002) -> List[TrajectoryPoint]:
        """Generate linear trajectory."""
        n_points = int(duration / dt)
        trajectory = []
        
        for i in range(n_points):
            t = i * dt
            alpha = t / duration
            position = start + alpha * (end - start)
            
            trajectory.append(TrajectoryPoint(
                position=position,
                orientation=np.array([1, 0, 0, 0]),
                velocity=(end - start) / duration,
                timestamp=t
            ))
        
        return trajectory
    
    @staticmethod
    def circle(center: np.ndarray, radius: float, normal: np.ndarray,
               duration: float, dt: float = 0.002) -> List[TrajectoryPoint]:
        """Generate circular trajectory."""
        n_points = int(duration / dt)
        trajectory = []
        
        # Create coordinate frame
        normal = normal / np.linalg.norm(normal)
        if abs(normal[0]) < 0.9:
            u = np.cross(normal, np.array([1, 0, 0]))
        else:
            u = np.cross(normal, np.array([0, 1, 0]))
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        
        omega = 2 * np.pi / duration
        
        for i in range(n_points):
            t = i * dt
            theta = omega * t
            
            position = center + radius * (np.cos(theta) * u + np.sin(theta) * v)
            velocity = radius * omega * (-np.sin(theta) * u + np.cos(theta) * v)
            
            trajectory.append(TrajectoryPoint(
                position=position,
                orientation=np.array([1, 0, 0, 0]),
                velocity=velocity,
                timestamp=t
            ))
        
        return trajectory
    
    @staticmethod
    def figure_eight(center: np.ndarray, size: float,
                     duration: float, dt: float = 0.002) -> List[TrajectoryPoint]:
        """Generate figure-eight (lemniscate) trajectory."""
        n_points = int(duration / dt)
        trajectory = []
        
        omega = 2 * np.pi / duration
        
        for i in range(n_points):
            t = i * dt
            theta = omega * t
            
            # Lemniscate of Bernoulli
            denom = 1 + np.sin(theta)**2
            x = size * np.cos(theta) / denom
            y = size * np.sin(theta) * np.cos(theta) / denom
            
            position = center + np.array([x, y, 0])
            
            trajectory.append(TrajectoryPoint(
                position=position,
                orientation=np.array([1, 0, 0, 0]),
                timestamp=t
            ))
        
        return trajectory


class TrajectoryTracker:
    """Track trajectory and compute metrics."""
    
    def __init__(self, position_tolerance: float = 0.003,
                 orientation_tolerance: float = 0.02):
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
        
        # Data storage
        self.position_errors: List[float] = []
        self.orientation_errors: List[float] = []
        self.joint_velocities: List[np.ndarray] = []
        self.timestamps: List[float] = []
        
        self.start_time: Optional[float] = None
        self.convergence_time: Optional[float] = None
        self.tracking = False
    
    def start(self):
        """Start tracking."""
        self.position_errors.clear()
        self.orientation_errors.clear()
        self.joint_velocities.clear()
        self.timestamps.clear()
        self.start_time = time.time()
        self.convergence_time = None
        self.tracking = True
    
    def stop(self):
        """Stop tracking."""
        self.tracking = False
    
    def record(self, target: TrajectoryPoint, actual_position: np.ndarray,
               actual_orientation: np.ndarray, joint_velocities: np.ndarray):
        """Record tracking data."""
        if not self.tracking:
            return
        
        # Position error
        pos_error = np.linalg.norm(target.position - actual_position)
        self.position_errors.append(pos_error)
        
        # Orientation error (angle between quaternions)
        q_target = target.orientation / np.linalg.norm(target.orientation)
        q_actual = actual_orientation / np.linalg.norm(actual_orientation)
        dot = np.abs(np.dot(q_target, q_actual))
        ori_error = 2 * np.arccos(np.clip(dot, 0, 1))
        self.orientation_errors.append(ori_error)
        
        # Joint velocities
        self.joint_velocities.append(joint_velocities.copy())
        
        # Timestamp
        self.timestamps.append(time.time() - self.start_time)
        
        # Check convergence
        if self.convergence_time is None:
            if pos_error < self.position_tolerance and ori_error < self.orientation_tolerance:
                self.convergence_time = self.timestamps[-1]
    
    def compute_metrics(self) -> TrackingMetrics:
        """Compute tracking performance metrics."""
        if not self.position_errors:
            return TrackingMetrics()
        
        pos_errors = np.array(self.position_errors)
        ori_errors = np.array(self.orientation_errors)
        joint_vels = np.array(self.joint_velocities)
        
        metrics = TrackingMetrics(
            rms_position_error=np.sqrt(np.mean(pos_errors**2)),
            rms_orientation_error=np.sqrt(np.mean(ori_errors**2)),
            max_position_error=np.max(pos_errors),
            max_orientation_error=np.max(ori_errors),
            mean_joint_velocity=np.mean(np.abs(joint_vels)),
            max_joint_velocity=np.max(np.abs(joint_vels)),
            tracking_duration=self.timestamps[-1] if self.timestamps else 0.0,
            samples_collected=len(self.position_errors),
            convergence_time=self.convergence_time
        )
        
        return metrics


class TrajectoryTrackerNode(Node):
    """ROS 2 node for trajectory tracking."""
    
    def __init__(self):
        super().__init__('trajectory_tracker')
        
        # Parameters
        self.declare_parameter('control_rate', 500.0)
        self.declare_parameter('position_tolerance', 0.003)
        self.declare_parameter('orientation_tolerance', 0.02)
        
        rate = self.get_parameter('control_rate').value
        pos_tol = self.get_parameter('position_tolerance').value
        ori_tol = self.get_parameter('orientation_tolerance').value
        
        # Tracker
        self.tracker = TrajectoryTracker(pos_tol, ori_tol)
        self.generator = TrajectoryGenerator()
        
        # Current trajectory
        self.trajectory: List[TrajectoryPoint] = []
        self.trajectory_idx = 0
        self.executing = False
        
        # Current state
        self.current_position = np.zeros(3)
        self.current_orientation = np.array([1, 0, 0, 0])
        self.current_joint_velocities = np.zeros(6)
        
        # Publishers
        self.pose_cmd_pub = self.create_publisher(
            PoseStamped, 'cartesian_pose_cmd', 10)
        self.path_pub = self.create_publisher(Path, 'reference_path', 10)
        
        # Subscribers
        self.pose_sub = self.create_subscription(
            PoseStamped, 'current_pose', self.pose_callback, 10)
        self.joint_vel_sub = self.create_subscription(
            Float64MultiArray, 'joint_velocity_command', self.joint_vel_callback, 10)
        
        # Execution timer
        self.dt = 1.0 / rate
        self.timer = self.create_timer(self.dt, self.execution_loop)
        
        self.get_logger().info('Trajectory tracker ready')
    
    def pose_callback(self, msg: PoseStamped):
        self.current_position = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        self.current_orientation = np.array([
            msg.pose.orientation.w,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z
        ])
    
    def joint_vel_callback(self, msg: Float64MultiArray):
        self.current_joint_velocities = np.array(msg.data)
    
    def execute_trajectory(self, trajectory: List[TrajectoryPoint]):
        """Start executing a trajectory."""
        self.trajectory = trajectory
        self.trajectory_idx = 0
        self.executing = True
        self.tracker.start()
        
        # Publish path for visualization
        self._publish_path()
        
        self.get_logger().info(f'Executing trajectory with {len(trajectory)} points')
    
    def execute_line(self, start: np.ndarray, end: np.ndarray, duration: float):
        """Execute line trajectory."""
        trajectory = self.generator.line(start, end, duration, self.dt)
        self.execute_trajectory(trajectory)
    
    def execute_circle(self, center: np.ndarray, radius: float, 
                      normal: np.ndarray, duration: float):
        """Execute circular trajectory."""
        trajectory = self.generator.circle(center, radius, normal, duration, self.dt)
        self.execute_trajectory(trajectory)
    
    def execution_loop(self):
        """Main execution loop."""
        if not self.executing or self.trajectory_idx >= len(self.trajectory):
            if self.executing:
                self.executing = False
                self.tracker.stop()
                metrics = self.tracker.compute_metrics()
                self.get_logger().info(f'\n{metrics}')
            return
        
        # Get current target
        target = self.trajectory[self.trajectory_idx]
        
        # Publish command
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.pose.position = Point(x=target.position[0], 
                                   y=target.position[1], 
                                   z=target.position[2])
        msg.pose.orientation = Quaternion(w=target.orientation[0],
                                           x=target.orientation[1],
                                           y=target.orientation[2],
                                           z=target.orientation[3])
        self.pose_cmd_pub.publish(msg)
        
        # Record tracking data
        self.tracker.record(
            target,
            self.current_position,
            self.current_orientation,
            self.current_joint_velocities
        )
        
        self.trajectory_idx += 1
    
    def _publish_path(self):
        """Publish trajectory as Path for visualization."""
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        
        for point in self.trajectory[::10]:  # Subsample for efficiency
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position = Point(x=point.position[0],
                                        y=point.position[1],
                                        z=point.position[2])
            pose.pose.orientation = Quaternion(w=point.orientation[0],
                                                x=point.orientation[1],
                                                y=point.orientation[2],
                                                z=point.orientation[3])
            msg.poses.append(pose)
        
        self.path_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryTrackerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
