#!/usr/bin/env python3
"""
Benchmark Analysis for Manipulator Control Stack

Analyzes performance of:
- Cartesian controllers (Jacobian, DLS, Nullspace)
- IK solver (convergence rate, iterations)
- Motion planners (RRT, RRT*, Bi-RRT*)

Author: Barath Kumar JK
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import json
import os
from datetime import datetime
from pathlib import Path


@dataclass
class CartesianControllerMetrics:
    """Metrics for Cartesian controller performance."""
    rms_position_error_mm: float = 0.0
    rms_orientation_error_deg: float = 0.0
    max_joint_velocity: float = 0.0
    mean_joint_velocity: float = 0.0
    tracking_duration: float = 0.0
    samples: int = 0
    rate_limited_percentage: float = 0.0
    near_singularity_percentage: float = 0.0


@dataclass
class IKSolverMetrics:
    """Metrics for IK solver performance."""
    convergence_rate: float = 0.0
    mean_iterations: float = 0.0
    max_iterations: int = 0
    mean_position_error_mm: float = 0.0
    mean_orientation_error_deg: float = 0.0
    total_calls: int = 0


@dataclass
class PlannerMetrics:
    """Metrics for motion planner performance."""
    algorithm: str = ""
    success_rate: float = 0.0
    mean_planning_time_sec: float = 0.0
    std_planning_time_sec: float = 0.0
    mean_iterations: float = 0.0
    mean_path_length: float = 0.0
    mean_tree_size: int = 0
    total_plans: int = 0


@dataclass
class BenchmarkResult:
    """Complete benchmark results."""
    timestamp: str = ""
    robot: str = "ur5e"
    cartesian_metrics: Optional[CartesianControllerMetrics] = None
    ik_metrics: Optional[IKSolverMetrics] = None
    planner_metrics: Dict[str, PlannerMetrics] = field(default_factory=dict)


class CartesianControllerBenchmark:
    """Benchmark Cartesian controller performance."""
    
    def __init__(self):
        self.position_errors: List[float] = []
        self.orientation_errors: List[float] = []
        self.joint_velocities: List[np.ndarray] = []
        self.rate_limited_flags: List[bool] = []
        self.near_singularity_flags: List[bool] = []
    
    def record(self, pos_error: float, ori_error: float, 
               joint_vel: np.ndarray, rate_limited: bool, near_singularity: bool):
        """Record a single control sample."""
        self.position_errors.append(pos_error)
        self.orientation_errors.append(ori_error)
        self.joint_velocities.append(joint_vel.copy())
        self.rate_limited_flags.append(rate_limited)
        self.near_singularity_flags.append(near_singularity)
    
    def compute_metrics(self, dt: float = 0.002) -> CartesianControllerMetrics:
        """Compute benchmark metrics."""
        if not self.position_errors:
            return CartesianControllerMetrics()
        
        pos_errors = np.array(self.position_errors)
        ori_errors = np.array(self.orientation_errors)
        joint_vels = np.array(self.joint_velocities)
        
        return CartesianControllerMetrics(
            rms_position_error_mm=np.sqrt(np.mean(pos_errors**2)) * 1000,
            rms_orientation_error_deg=np.sqrt(np.mean(ori_errors**2)) * 180 / np.pi,
            max_joint_velocity=np.max(np.abs(joint_vels)),
            mean_joint_velocity=np.mean(np.abs(joint_vels)),
            tracking_duration=len(self.position_errors) * dt,
            samples=len(self.position_errors),
            rate_limited_percentage=100 * np.mean(self.rate_limited_flags),
            near_singularity_percentage=100 * np.mean(self.near_singularity_flags)
        )
    
    def clear(self):
        """Clear recorded data."""
        self.position_errors.clear()
        self.orientation_errors.clear()
        self.joint_velocities.clear()
        self.rate_limited_flags.clear()
        self.near_singularity_flags.clear()


class IKSolverBenchmark:
    """Benchmark IK solver performance."""
    
    def __init__(self):
        self.successes: List[bool] = []
        self.iterations: List[int] = []
        self.position_errors: List[float] = []
        self.orientation_errors: List[float] = []
    
    def record(self, success: bool, iterations: int, 
               pos_error: float, ori_error: float):
        """Record a single IK solve."""
        self.successes.append(success)
        self.iterations.append(iterations)
        self.position_errors.append(pos_error)
        self.orientation_errors.append(ori_error)
    
    def compute_metrics(self) -> IKSolverMetrics:
        """Compute IK solver metrics."""
        if not self.successes:
            return IKSolverMetrics()
        
        successes = np.array(self.successes)
        iterations = np.array(self.iterations)
        pos_errors = np.array(self.position_errors)
        ori_errors = np.array(self.orientation_errors)
        
        successful_mask = successes
        
        return IKSolverMetrics(
            convergence_rate=100 * np.mean(successes),
            mean_iterations=np.mean(iterations[successful_mask]) if np.any(successful_mask) else 0,
            max_iterations=int(np.max(iterations[successful_mask])) if np.any(successful_mask) else 0,
            mean_position_error_mm=np.mean(pos_errors[successful_mask]) * 1000 if np.any(successful_mask) else 0,
            mean_orientation_error_deg=np.mean(ori_errors[successful_mask]) * 180 / np.pi if np.any(successful_mask) else 0,
            total_calls=len(self.successes)
        )
    
    def clear(self):
        self.successes.clear()
        self.iterations.clear()
        self.position_errors.clear()
        self.orientation_errors.clear()


class PlannerBenchmark:
    """Benchmark motion planner performance."""
    
    def __init__(self):
        self.results: Dict[str, List[dict]] = {
            'rrt': [], 'rrt_star': [], 'birrt_star': []
        }
    
    def record(self, algorithm: str, success: bool, planning_time: float,
               iterations: int, path_length: float, tree_size: int):
        """Record a single planning attempt."""
        if algorithm not in self.results:
            self.results[algorithm] = []
        
        self.results[algorithm].append({
            'success': success,
            'planning_time': planning_time,
            'iterations': iterations,
            'path_length': path_length,
            'tree_size': tree_size
        })
    
    def compute_metrics(self, algorithm: str) -> PlannerMetrics:
        """Compute metrics for a specific algorithm."""
        if algorithm not in self.results or not self.results[algorithm]:
            return PlannerMetrics(algorithm=algorithm)
        
        data = self.results[algorithm]
        successes = [d['success'] for d in data]
        times = [d['planning_time'] for d in data if d['success']]
        iterations = [d['iterations'] for d in data if d['success']]
        lengths = [d['path_length'] for d in data if d['success']]
        tree_sizes = [d['tree_size'] for d in data if d['success']]
        
        return PlannerMetrics(
            algorithm=algorithm,
            success_rate=100 * np.mean(successes),
            mean_planning_time_sec=np.mean(times) if times else 0,
            std_planning_time_sec=np.std(times) if times else 0,
            mean_iterations=np.mean(iterations) if iterations else 0,
            mean_path_length=np.mean(lengths) if lengths else 0,
            mean_tree_size=int(np.mean(tree_sizes)) if tree_sizes else 0,
            total_plans=len(data)
        )
    
    def compute_all_metrics(self) -> Dict[str, PlannerMetrics]:
        """Compute metrics for all algorithms."""
        return {alg: self.compute_metrics(alg) for alg in self.results}
    
    def clear(self):
        for alg in self.results:
            self.results[alg].clear()


class BenchmarkReporter:
    """Generate benchmark reports and visualizations."""
    
    def __init__(self, output_dir: str = '/tmp/manipulator_benchmark'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(self, result: BenchmarkResult) -> str:
        """Generate text report."""
        lines = [
            "=" * 70,
            "MANIPULATOR CONTROL BENCHMARK REPORT",
            f"Timestamp: {result.timestamp}",
            f"Robot: {result.robot}",
            "=" * 70,
            ""
        ]
        
        if result.cartesian_metrics:
            m = result.cartesian_metrics
            lines.extend([
                "CARTESIAN CONTROLLER PERFORMANCE",
                "-" * 40,
                f"  RMS Position Error:     {m.rms_position_error_mm:.2f} mm",
                f"  RMS Orientation Error:  {m.rms_orientation_error_deg:.2f} deg",
                f"  Max Joint Velocity:     {m.max_joint_velocity:.3f} rad/s",
                f"  Rate Limited:           {m.rate_limited_percentage:.1f}%",
                f"  Near Singularity:       {m.near_singularity_percentage:.1f}%",
                f"  Tracking Duration:      {m.tracking_duration:.2f} s",
                ""
            ])
        
        if result.ik_metrics:
            m = result.ik_metrics
            lines.extend([
                "IK SOLVER PERFORMANCE",
                "-" * 40,
                f"  Convergence Rate:       {m.convergence_rate:.1f}%",
                f"  Mean Iterations:        {m.mean_iterations:.1f}",
                f"  Max Iterations:         {m.max_iterations}",
                f"  Position Error:         {m.mean_position_error_mm:.2f} mm",
                f"  Total Calls:            {m.total_calls}",
                ""
            ])
        
        if result.planner_metrics:
            lines.extend([
                "MOTION PLANNER PERFORMANCE",
                "-" * 40,
                f"{'Algorithm':<15} {'Success':<10} {'Time (s)':<12} {'Iterations':<12} {'Path Len':<10}",
                "-" * 60
            ])
            for alg, m in result.planner_metrics.items():
                lines.append(
                    f"{alg:<15} {m.success_rate:>6.1f}%   "
                    f"{m.mean_planning_time_sec:>8.3f}     "
                    f"{m.mean_iterations:>8.0f}     "
                    f"{m.mean_path_length:>8.2f}"
                )
            lines.append("")
        
        lines.append("=" * 70)
        
        report = "\n".join(lines)
        
        # Save report
        report_path = self.output_dir / f"report_{result.timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        return report
    
    def plot_planner_comparison(self, result: BenchmarkResult):
        """Generate planner comparison plots."""
        if not result.planner_metrics:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        algorithms = list(result.planner_metrics.keys())
        colors = {'rrt': 'blue', 'rrt_star': 'orange', 'birrt_star': 'green'}
        
        # Success rate
        ax = axes[0, 0]
        rates = [result.planner_metrics[a].success_rate for a in algorithms]
        ax.bar(algorithms, rates, color=[colors.get(a, 'gray') for a in algorithms])
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Planning Success Rate')
        ax.set_ylim(0, 105)
        
        # Planning time
        ax = axes[0, 1]
        times = [result.planner_metrics[a].mean_planning_time_sec for a in algorithms]
        stds = [result.planner_metrics[a].std_planning_time_sec for a in algorithms]
        ax.bar(algorithms, times, yerr=stds, color=[colors.get(a, 'gray') for a in algorithms], capsize=5)
        ax.set_ylabel('Planning Time (s)')
        ax.set_title('Mean Planning Time')
        ax.axhline(y=0.5, color='red', linestyle='--', label='Target: 0.5s')
        ax.legend()
        
        # Iterations
        ax = axes[1, 0]
        iters = [result.planner_metrics[a].mean_iterations for a in algorithms]
        ax.bar(algorithms, iters, color=[colors.get(a, 'gray') for a in algorithms])
        ax.set_ylabel('Iterations')
        ax.set_title('Mean Iterations')
        
        # Path length
        ax = axes[1, 1]
        lengths = [result.planner_metrics[a].mean_path_length for a in algorithms]
        ax.bar(algorithms, lengths, color=[colors.get(a, 'gray') for a in algorithms])
        ax.set_ylabel('Path Length (rad)')
        ax.set_title('Mean Path Length')
        
        plt.tight_layout()
        
        plot_path = self.output_dir / f"planner_comparison_{result.timestamp}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        return plot_path
    
    def save_json(self, result: BenchmarkResult):
        """Save results as JSON."""
        data = {
            'timestamp': result.timestamp,
            'robot': result.robot,
            'cartesian_metrics': vars(result.cartesian_metrics) if result.cartesian_metrics else None,
            'ik_metrics': vars(result.ik_metrics) if result.ik_metrics else None,
            'planner_metrics': {
                alg: vars(m) for alg, m in result.planner_metrics.items()
            } if result.planner_metrics else None
        }
        
        json_path = self.output_dir / f"benchmark_{result.timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return json_path


def run_synthetic_benchmark():
    """Run synthetic benchmark for demonstration."""
    print("Running synthetic benchmark...")
    
    # Create benchmarks
    cartesian_bench = CartesianControllerBenchmark()
    ik_bench = IKSolverBenchmark()
    planner_bench = PlannerBenchmark()
    
    # Simulate Cartesian controller data (2-3mm RMS error target)
    np.random.seed(42)
    for _ in range(5000):
        pos_error = np.random.normal(0.002, 0.0005)  # ~2mm RMS
        ori_error = np.random.normal(0.01, 0.003)     # ~0.5 deg RMS
        joint_vel = np.random.uniform(-1.5, 1.5, 6) * 0.5  # < 1.5 rad/s
        rate_limited = np.random.random() < 0.05
        near_singularity = np.random.random() < 0.02
        
        cartesian_bench.record(pos_error, ori_error, joint_vel, rate_limited, near_singularity)
    
    # Simulate IK solver data (95% convergence in <20 iterations)
    for _ in range(1000):
        success = np.random.random() < 0.95
        iterations = int(np.random.exponential(8) + 5) if success else 50
        iterations = min(iterations, 20 if success else 50)
        pos_error = np.random.exponential(0.0005) if success else 0.01
        ori_error = np.random.exponential(0.005) if success else 0.1
        
        ik_bench.record(success, iterations, pos_error, ori_error)
    
    # Simulate planner data (0.3-0.8s planning time)
    for algorithm, base_time, success_rate in [
        ('rrt', 0.35, 0.85),
        ('rrt_star', 0.65, 0.90),
        ('birrt_star', 0.31, 0.95)
    ]:
        for _ in range(100):
            success = np.random.random() < success_rate
            planning_time = np.random.normal(base_time, base_time * 0.2) if success else base_time * 2
            iterations = int(np.random.normal(5000, 1500))
            path_length = np.random.normal(2.0, 0.3) if success else 0
            tree_size = int(np.random.normal(3000, 800))
            
            planner_bench.record(algorithm, success, planning_time, iterations, path_length, tree_size)
    
    # Compute metrics
    result = BenchmarkResult(
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
        robot="ur5e",
        cartesian_metrics=cartesian_bench.compute_metrics(),
        ik_metrics=ik_bench.compute_metrics(),
        planner_metrics=planner_bench.compute_all_metrics()
    )
    
    # Generate reports
    reporter = BenchmarkReporter()
    report = reporter.generate_report(result)
    print(report)
    
    reporter.plot_planner_comparison(result)
    reporter.save_json(result)
    
    print(f"\nResults saved to: {reporter.output_dir}")
    
    return result


if __name__ == '__main__':
    run_synthetic_benchmark()
