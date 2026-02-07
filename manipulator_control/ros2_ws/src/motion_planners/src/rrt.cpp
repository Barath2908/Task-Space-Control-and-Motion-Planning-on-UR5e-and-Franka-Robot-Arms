/**
 * @file rrt.cpp
 * @brief Implementation of RRT motion planner
 * @author Barath Kumar JK
 * @date 2025
 */

#include "motion_planners/rrt.hpp"
#include <algorithm>

namespace motion_planners {

RRT::RRT(
    int num_joints,
    CollisionChecker collision_check,
    const RRTConfig& config
) : num_joints_(num_joints),
    config_(config),
    collision_check_(collision_check) {
    
    // Initialize random number generator
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    rng_.seed(static_cast<unsigned int>(seed));
    
    // Set default joint limits
    joint_lower_limits_.resize(num_joints_);
    joint_upper_limits_.resize(num_joints_);
    joint_lower_limits_.setConstant(-M_PI);
    joint_upper_limits_.setConstant(M_PI);
    
    // Default distance function (Euclidean)
    distance_func_ = [](const Eigen::VectorXd& q1, const Eigen::VectorXd& q2) {
        return (q1 - q2).norm();
    };
    
    // Default edge collision checker (interpolation-based)
    edge_collision_check_ = [this](const Eigen::VectorXd& q1, const Eigen::VectorXd& q2) {
        int steps = std::max(1, static_cast<int>((q1 - q2).norm() / 0.05));
        for (int i = 0; i <= steps; ++i) {
            double t = static_cast<double>(i) / steps;
            Eigen::VectorXd q = q1 + t * (q2 - q1);
            if (!collision_check_(q)) {
                return false;
            }
        }
        return true;
    };
}

PlanningResult RRT::plan(
    const Eigen::VectorXd& start,
    const Eigen::VectorXd& goal
) {
    PlanningResult result;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Clear tree and add start node
    tree_.clear();
    tree_.emplace_back(start);
    
    // Check if start and goal are valid
    if (!collision_check_(start)) {
        result.success = false;
        return result;
    }
    if (!collision_check_(goal)) {
        result.success = false;
        return result;
    }
    
    // Main loop
    for (int iter = 0; iter < config_.max_iterations; ++iter) {
        // Check timeout
        auto current_time = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(current_time - start_time).count();
        if (elapsed > config_.timeout_sec) {
            break;
        }
        
        // Sample with goal bias
        Eigen::VectorXd q_rand = sampleWithGoalBias(goal);
        
        // Find nearest node
        int nearest_idx = findNearest(q_rand);
        const Eigen::VectorXd& q_near = tree_[nearest_idx].config;
        
        // Steer towards sample
        Eigen::VectorXd q_new = steer(q_near, q_rand);
        
        // Check collision
        if (!collision_check_(q_new) || !isEdgeCollisionFree(q_near, q_new)) {
            continue;
        }
        
        // Add node to tree
        tree_.emplace_back(q_new, nearest_idx);
        int new_idx = tree_.size() - 1;
        
        // Check if goal reached
        if (isAtGoal(q_new, goal)) {
            // Try to connect directly to goal
            if (isEdgeCollisionFree(q_new, goal)) {
                tree_.emplace_back(goal, new_idx);
                
                // Extract path
                result.path = extractPath(tree_.size() - 1);
                if (config_.interpolate_path) {
                    result.path = interpolatePath(result.path);
                }
                
                result.success = true;
                result.iterations = iter + 1;
                result.tree_size = tree_.size();
                result.path_length = computePathLength(result.path);
                
                auto end_time = std::chrono::high_resolution_clock::now();
                result.planning_time_sec = std::chrono::duration<double>(
                    end_time - start_time).count();
                
                return result;
            }
        }
        
        result.iterations = iter + 1;
    }
    
    // Failed to find path
    result.success = false;
    result.tree_size = tree_.size();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.planning_time_sec = std::chrono::duration<double>(end_time - start_time).count();
    
    return result;
}

void RRT::setJointLimits(
    const Eigen::VectorXd& lower,
    const Eigen::VectorXd& upper
) {
    joint_lower_limits_ = lower;
    joint_upper_limits_ = upper;
}

void RRT::setEdgeCollisionChecker(EdgeCollisionChecker checker) {
    edge_collision_check_ = checker;
}

void RRT::setDistanceFunction(DistanceFunc func) {
    distance_func_ = func;
}

Eigen::VectorXd RRT::sampleRandom() {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    Eigen::VectorXd q(num_joints_);
    for (int i = 0; i < num_joints_; ++i) {
        double range = joint_upper_limits_(i) - joint_lower_limits_(i);
        q(i) = joint_lower_limits_(i) + dist(rng_) * range;
    }
    
    return q;
}

Eigen::VectorXd RRT::sampleWithGoalBias(const Eigen::VectorXd& goal) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    if (dist(rng_) < config_.goal_bias) {
        return goal;
    }
    return sampleRandom();
}

int RRT::findNearest(const Eigen::VectorXd& q) const {
    int nearest_idx = 0;
    double min_dist = std::numeric_limits<double>::infinity();
    
    for (size_t i = 0; i < tree_.size(); ++i) {
        double d = distance(tree_[i].config, q);
        if (d < min_dist) {
            min_dist = d;
            nearest_idx = i;
        }
    }
    
    return nearest_idx;
}

Eigen::VectorXd RRT::steer(
    const Eigen::VectorXd& q_near,
    const Eigen::VectorXd& q_rand
) const {
    Eigen::VectorXd direction = q_rand - q_near;
    double dist = direction.norm();
    
    if (dist <= config_.step_size) {
        return q_rand;
    }
    
    return q_near + (direction / dist) * config_.step_size;
}

bool RRT::isEdgeCollisionFree(
    const Eigen::VectorXd& q1,
    const Eigen::VectorXd& q2
) const {
    if (edge_collision_check_) {
        return edge_collision_check_(q1, q2);
    }
    
    // Default: interpolation-based checking
    int steps = std::max(1, static_cast<int>((q1 - q2).norm() / 0.05));
    for (int i = 0; i <= steps; ++i) {
        double t = static_cast<double>(i) / steps;
        Eigen::VectorXd q = q1 + t * (q2 - q1);
        if (!collision_check_(q)) {
            return false;
        }
    }
    return true;
}

double RRT::distance(
    const Eigen::VectorXd& q1,
    const Eigen::VectorXd& q2
) const {
    if (distance_func_) {
        return distance_func_(q1, q2);
    }
    return (q1 - q2).norm();
}

bool RRT::isAtGoal(
    const Eigen::VectorXd& q,
    const Eigen::VectorXd& goal
) const {
    return distance(q, goal) < config_.goal_threshold;
}

std::vector<Eigen::VectorXd> RRT::extractPath(int goal_idx) const {
    std::vector<Eigen::VectorXd> path;
    
    int idx = goal_idx;
    while (idx >= 0) {
        path.push_back(tree_[idx].config);
        idx = tree_[idx].parent_idx;
    }
    
    std::reverse(path.begin(), path.end());
    return path;
}

std::vector<Eigen::VectorXd> RRT::interpolatePath(
    const std::vector<Eigen::VectorXd>& path
) const {
    if (path.size() < 2) {
        return path;
    }
    
    std::vector<Eigen::VectorXd> interpolated;
    
    for (size_t i = 0; i < path.size() - 1; ++i) {
        const auto& q1 = path[i];
        const auto& q2 = path[i + 1];
        
        for (int j = 0; j < config_.interpolation_steps; ++j) {
            double t = static_cast<double>(j) / config_.interpolation_steps;
            interpolated.push_back(q1 + t * (q2 - q1));
        }
    }
    
    interpolated.push_back(path.back());
    return interpolated;
}

double RRT::computePathLength(const std::vector<Eigen::VectorXd>& path) const {
    double length = 0.0;
    
    for (size_t i = 1; i < path.size(); ++i) {
        length += distance(path[i - 1], path[i]);
    }
    
    return length;
}

}  // namespace motion_planners
