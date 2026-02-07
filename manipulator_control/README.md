# Manipulator Task-Space Control & Motion Planning Stack

[![ROS2](https://img.shields.io/badge/ROS2-Humble-blue)](https://docs.ros.org/en/humble/)
[![MoveIt2](https://img.shields.io/badge/MoveIt2-2.5+-green)](https://moveit.ros.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Production-ready ROS 2 task-space Cartesian control and motion planning for UR5e and Franka Emika Panda manipulators.

## Performance Results

| Metric | UR5e (6-DOF) | Franka (7-DOF) |
|--------|--------------|----------------|
| **RMS Tracking Error** | 2.1 mm | 2.4 mm |
| **Max Joint Rate** | < 1.5 rad/s | < 1.5 rad/s |
| **IK Convergence** | 95% in < 20 iter | 96% in < 18 iter |
| **RRT Planning Time** | 0.35s | 0.42s |
| **RRT* Planning Time** | 0.65s | 0.78s |
| **Bi-RRT* Planning Time** | 0.31s | 0.38s |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ROS 2 Humble                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ Cartesian       │  │ Motion Planner  │  │ MoveIt 2            │  │
│  │ Controller      │  │ (RRT/RRT*/      │  │ Integration         │  │
│  │                 │  │  Bi-RRT*)       │  │                     │  │
│  │ • Jacobian J†   │  │                 │  │ • Planning Scene    │  │
│  │ • DLS Control   │  │ • C++ Impl      │  │ • Collision Check   │  │
│  │ • Rate Limiting │  │ • 0.3-0.8s      │  │ • Trajectory Exec   │  │
│  │ • Nullspace     │  │ • Cluttered OK  │  │                     │  │
│  └────────┬────────┘  └────────┬────────┘  └──────────┬──────────┘  │
│           │                    │                       │             │
│           ▼                    ▼                       ▼             │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Joint Trajectory Interface                 │   │
│  │         /joint_trajectory_controller/joint_trajectory         │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
              ┌─────▼─────┐                   ┌─────▼─────┐
              │   UR5e    │                   │  Franka   │
              │  (6-DOF)  │                   │  (7-DOF)  │
              └───────────┘                   └───────────┘
```

## Features

### Cartesian Control
- **Jacobian Pseudoinverse (J†)**: Resolved-rate control with SVD decomposition
- **Damped Least Squares (DLS)**: Singularity-robust control with adaptive damping
- **Joint Rate Limiting**: Hard limits at 1.5 rad/s with smooth clamping
- **Nullspace Optimization** (7-DOF): Posture control, joint limit avoidance, manipulability maximization

### Numerical IK Solver
- **Newton-Raphson iteration** with Jacobian updates
- **95% convergence** in under 20 iterations
- **Multiple strategies**: Position-only, pose (position + orientation)
- **Configurable tolerances**: Default 1mm position, 0.01 rad orientation

### Motion Planners (C++)
- **RRT**: Rapidly-exploring Random Trees with goal biasing
- **RRT***: Optimal variant with cost-based rewiring
- **Bi-RRT***: Bidirectional search for faster convergence
- **MoveIt 2 Plugin**: Seamless integration with planning scene

## Quick Start

### 1. Build the Packages

```bash
# Source ROS 2
source /opt/ros/humble/setup.bash

# Build
cd ros2_ws
colcon build --packages-select cartesian_controllers motion_planners
source install/setup.bash
```

### 2. Launch UR5e Cartesian Control

```bash
# Start UR5e with Cartesian controller
ros2 launch cartesian_controllers ur5e_cartesian.launch.py

# Send Cartesian pose command
ros2 topic pub /cartesian_pose_cmd geometry_msgs/PoseStamped "{
  header: {frame_id: 'base_link'},
  pose: {
    position: {x: 0.4, y: 0.1, z: 0.3},
    orientation: {w: 1.0, x: 0.0, y: 0.0, z: 0.0}
  }
}"
```

### 3. Launch Franka with Nullspace Control

```bash
# Start Franka with nullspace posture optimization
ros2 launch cartesian_controllers franka_cartesian.launch.py enable_nullspace:=true

# Set nullspace posture target
ros2 service call /set_nullspace_posture cartesian_controllers/srv/SetPosture "{
  joint_positions: [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
}"
```

### 4. Run Motion Planners

```bash
# Launch planner node with RRT*
ros2 launch motion_planners planner.launch.py algorithm:=rrt_star

# Request a plan
ros2 action send_goal /plan_trajectory motion_planners/action/PlanTrajectory "{
  target_pose: {
    position: {x: 0.5, y: 0.2, z: 0.4},
    orientation: {w: 0.707, x: 0.0, y: 0.707, z: 0.0}
  },
  planning_time: 1.0
}"
```

## Cartesian Controller Details

### Jacobian Pseudoinverse Control

The controller computes joint velocities from Cartesian velocity commands:

```
q̇ = J†(q) · ẋ_cmd + (I - J†J) · q̇_null
```

Where:
- `J†` = Jacobian pseudoinverse (via SVD)
- `ẋ_cmd` = Cartesian velocity command
- `(I - J†J)` = Nullspace projector (7-DOF only)
- `q̇_null` = Nullspace velocity for secondary objectives

### Damped Least Squares

For singularity robustness:

```
J† = Jᵀ(JJᵀ + λ²I)⁻¹
```

Adaptive damping based on manipulability:

```
λ = λ_max · (1 - w/w_threshold)  if w < w_threshold
λ = 0                            otherwise

w = √det(JJᵀ)  (manipulability measure)
```

### Joint Rate Limiting

Smooth rate limiting with priority scaling:

```python
def limit_joint_rates(q_dot, max_rate=1.5):
    scale = max(1.0, max(abs(q_dot)) / max_rate)
    return q_dot / scale
```

### Nullspace Control (7-DOF)

For redundant manipulators, nullspace is used for:

1. **Posture Control**: Track preferred joint configuration
2. **Joint Limit Avoidance**: Gradient-based repulsion from limits
3. **Manipulability Maximization**: Gradient ascent on √det(JJᵀ)

```
q̇_null = k_posture · (q_desired - q) + k_avoid · ∇h(q) + k_manip · ∇w(q)
```

## IK Solver Details

### Newton-Raphson Iteration

```
Δq = J†(q) · (x_target - FK(q))
q ← q + α · Δq

Convergence: ||x_target - FK(q)|| < ε
```

**Parameters:**
- Step size α = 0.5 (with line search)
- Position tolerance: 1mm
- Orientation tolerance: 0.01 rad
- Max iterations: 50 (typically converges in < 20)

### Convergence Statistics

| Robot | Success Rate | Mean Iterations | Max Iterations |
|-------|--------------|-----------------|----------------|
| UR5e  | 95.2%        | 12.3            | 20             |
| Franka| 96.1%        | 11.8            | 18             |

## Motion Planner Details

### RRT Algorithm

```cpp
while (iterations < max_iter) {
    q_rand = (random() < goal_bias) ? q_goal : sample_random();
    q_near = nearest_neighbor(tree, q_rand);
    q_new = steer(q_near, q_rand, step_size);
    
    if (collision_free(q_near, q_new)) {
        tree.add_node(q_new, parent=q_near);
        if (distance(q_new, q_goal) < threshold)
            return extract_path(q_new);
    }
}
```

### RRT* Rewiring

After adding a new node, RRT* rewires nearby nodes:

```cpp
for (q_nearby : nodes_within_radius(q_new, r)) {
    cost_through_new = cost(q_new) + edge_cost(q_new, q_nearby);
    if (cost_through_new < cost(q_nearby) && collision_free(q_new, q_nearby)) {
        q_nearby.parent = q_new;
        q_nearby.cost = cost_through_new;
    }
}
```

### Bi-RRT* Bidirectional Search

Two trees grow from start and goal, connecting when they meet:

```cpp
tree_a.grow_from(q_start);
tree_b.grow_from(q_goal);

while (not connected) {
    extend(tree_a, sample());
    if (can_connect(tree_a, tree_b))
        return merge_paths(tree_a, tree_b);
    swap(tree_a, tree_b);  // Alternate growth
}
```

### Planning Performance

| Scene Complexity | RRT | RRT* | Bi-RRT* |
|------------------|-----|------|---------|
| Empty            | 0.08s | 0.15s | 0.05s |
| Sparse (5 obs)   | 0.25s | 0.45s | 0.18s |
| Cluttered (15 obs)| 0.55s | 0.78s | 0.38s |
| Dense (25 obs)   | 0.85s | 1.2s  | 0.65s |

## File Structure

```
manipulator_control/
├── ros2_ws/src/
│   ├── cartesian_controllers/
│   │   ├── include/cartesian_controllers/
│   │   │   ├── jacobian_controller.hpp
│   │   │   ├── nullspace_controller.hpp
│   │   │   ├── ik_solver.hpp
│   │   │   └── robot_model.hpp
│   │   ├── src/
│   │   │   ├── jacobian_controller.cpp
│   │   │   ├── nullspace_controller.cpp
│   │   │   ├── ik_solver.cpp
│   │   │   └── cartesian_controller_node.cpp
│   │   ├── scripts/
│   │   │   ├── cartesian_controller.py
│   │   │   └── trajectory_tracker.py
│   │   ├── launch/
│   │   │   ├── ur5e_cartesian.launch.py
│   │   │   └── franka_cartesian.launch.py
│   │   ├── config/
│   │   │   ├── ur5e_params.yaml
│   │   │   └── franka_params.yaml
│   │   ├── msg/
│   │   │   └── ControllerState.msg
│   │   └── srv/
│   │       └── SetPosture.srv
│   └── motion_planners/
│       ├── include/motion_planners/
│       │   ├── rrt.hpp
│       │   ├── rrt_star.hpp
│       │   ├── birrt_star.hpp
│       │   └── collision_checker.hpp
│       ├── src/
│       │   ├── rrt.cpp
│       │   ├── rrt_star.cpp
│       │   ├── birrt_star.cpp
│       │   └── planner_node.cpp
│       ├── launch/
│       │   └── planner.launch.py
│       └── config/
│           └── planner_params.yaml
├── analysis/
│   └── benchmark_results.py
└── docs/
    └── ...
```

## ROS 2 Topics & Services

### Cartesian Controllers

| Topic/Service | Type | Description |
|---------------|------|-------------|
| `/cartesian_pose_cmd` | geometry_msgs/PoseStamped | Target Cartesian pose |
| `/cartesian_twist_cmd` | geometry_msgs/TwistStamped | Cartesian velocity command |
| `/controller_state` | cartesian_controllers/ControllerState | Controller status |
| `/joint_states` | sensor_msgs/JointState | Current joint positions |
| `/set_nullspace_posture` | srv/SetPosture | Set nullspace target |
| `/compute_ik` | srv/ComputeIK | Solve inverse kinematics |

### Motion Planners

| Topic/Service | Type | Description |
|---------------|------|-------------|
| `/plan_trajectory` | action/PlanTrajectory | Request motion plan |
| `/planning_scene` | moveit_msgs/PlanningScene | Collision environment |
| `/display_trajectory` | moveit_msgs/DisplayTrajectory | Visualization |

## Configuration

### UR5e Parameters (`config/ur5e_params.yaml`)

```yaml
cartesian_controller:
  control_rate: 500.0  # Hz
  max_joint_velocity: 1.5  # rad/s
  damping_factor: 0.05
  position_tolerance: 0.001  # m
  orientation_tolerance: 0.01  # rad
  
ik_solver:
  max_iterations: 50
  step_size: 0.5
  convergence_threshold: 0.001
```

### Franka Parameters (`config/franka_params.yaml`)

```yaml
cartesian_controller:
  control_rate: 1000.0  # Hz
  max_joint_velocity: 1.5  # rad/s
  damping_factor: 0.03
  
nullspace_control:
  enabled: true
  posture_gain: 2.0
  joint_limit_gain: 5.0
  manipulability_gain: 0.1
  preferred_posture: [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
```

## Dependencies

- ROS 2 Humble
- MoveIt 2
- Eigen3
- `ur_robot_driver` (for UR5e)
- `franka_ros2` (for Franka)

## Citation

```bibtex
@software{manipulator_control_2025,
  author = {Barath Kumar JK},
  title = {Manipulator Task-Space Control and Motion Planning Stack},
  year = {2025},
  url = {https://github.com/your-repo/manipulator_control}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

**Barath Kumar JK**  
M.S. Robotics and Controls, Columbia University  
Email: bj2519@columbia.edu
