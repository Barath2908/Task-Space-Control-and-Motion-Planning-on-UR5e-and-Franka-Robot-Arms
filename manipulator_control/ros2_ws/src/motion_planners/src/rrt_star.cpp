/**
 * @file rrt_star.cpp
 * @brief Implementation of RRT* optimal motion planner
 * @author Barath Kumar JK
 * @date 2025
 */

#include "motion_planners/rrt_star.hpp"

namespace motion_planners {

RRTStar::RRTStar(
    int num_joints,
    CollisionChecker collision_check,
    const RRTStarConfig& config
) : RRT(num_joints, collision_check, config),
    star_config_(config) {}

PlanningResult RRTStar::plan(
    const Eigen::VectorXd& start,
    const Eigen::VectorXd& goal
) {
    PlanningResult result;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Initialize
    tree_.clear();
    tree_.emplace_back(start, -1, 0.0);
    best_cost_ = std::numeric_limits<double>::infinity();
    goal_node_idx_ = -1;
    
    // Check validity
    if (!collision_check_(start) || !collision_check_(goal)) {
        result.success = false;
        return result;
    }
    
    bool found_solution = false;
    
    for (int iter = 0; iter < config_.max_iterations; ++iter) {
        // Check timeout
        auto current_time = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(current_time - start_time).count();
        if (elapsed > config_.timeout_sec) {
            break;
        }
        
        // Sample
        Eigen::VectorXd q_rand = sampleWithGoalBias(goal);
        
        // Find nearest
        int nearest_idx = findNearest(q_rand);
        const Eigen::VectorXd& q_near = tree_[nearest_idx].config;
        
        // Steer
        Eigen::VectorXd q_new = steer(q_near, q_rand);
        
        // Collision check
        if (!collision_check_(q_new)) {
            continue;
        }
        
        // Find nearby nodes for rewiring
        double radius = computeRewireRadius();
        std::vector<int> nearby = findNearbyNodes(q_new, radius);
        
        // If no nearby nodes, use nearest
        if (nearby.empty()) {
            nearby.push_back(nearest_idx);
        }
        
        // Choose best parent
        auto [best_parent, cost_through_parent] = chooseBestParent(q_new, nearby);
        
        if (best_parent < 0) {
            continue;  // No valid parent found
        }
        
        // Add node
        tree_.emplace_back(q_new, best_parent, cost_through_parent);
        int new_idx = tree_.size() - 1;
        
        // Rewire nearby nodes
        rewireNearbyNodes(new_idx, nearby);
        
        // Check if goal reached
        if (isAtGoal(q_new, goal) && isEdgeCollisionFree(q_new, goal)) {
            double cost_to_goal = cost_through_parent + edgeCost(q_new, goal);
            
            if (cost_to_goal < best_cost_) {
                // Add goal node
                tree_.emplace_back(goal, new_idx, cost_to_goal);
                goal_node_idx_ = tree_.size() - 1;
                best_cost_ = cost_to_goal;
                found_solution = true;
            }
        }
        
        // Stop if we have solution and not set to continue
        if (found_solution && !star_config_.continue_after_solution) {
            break;
        }
        
        // Check improvement threshold
        if (found_solution && !canImprove()) {
            break;
        }
        
        result.iterations = iter + 1;
    }
    
    // Extract result
    if (found_solution && goal_node_idx_ >= 0) {
        result.path = extractPath(goal_node_idx_);
        if (config_.interpolate_path) {
            result.path = interpolatePath(result.path);
        }
        result.success = true;
        result.path_length = best_cost_;
    } else {
        result.success = false;
    }
    
    result.tree_size = tree_.size();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.planning_time_sec = std::chrono::duration<double>(end_time - start_time).count();
    
    return result;
}

double RRTStar::computeRewireRadius() const {
    if (!star_config_.adaptive_radius) {
        return star_config_.rewire_radius;
    }
    
    // r = min(γ * (log(n)/n)^(1/d), step_size)
    size_t n = tree_.size();
    if (n < 2) return config_.step_size;
    
    double d = static_cast<double>(num_joints_);
    double log_n = std::log(static_cast<double>(n));
    double radius = star_config_.gamma_rrt_star * std::pow(log_n / n, 1.0 / d);
    
    return std::min(radius, config_.step_size * 3.0);
}

std::vector<int> RRTStar::findNearbyNodes(
    const Eigen::VectorXd& q,
    double radius
) const {
    std::vector<int> nearby;
    nearby.reserve(star_config_.max_rewire_neighbors);
    
    for (size_t i = 0; i < tree_.size(); ++i) {
        if (distance(tree_[i].config, q) < radius) {
            nearby.push_back(i);
            if (nearby.size() >= static_cast<size_t>(star_config_.max_rewire_neighbors)) {
                break;
            }
        }
    }
    
    return nearby;
}

std::pair<int, double> RRTStar::chooseBestParent(
    const Eigen::VectorXd& q_new,
    const std::vector<int>& nearby_indices
) const {
    int best_parent = -1;
    double best_cost = std::numeric_limits<double>::infinity();
    
    for (int idx : nearby_indices) {
        const RRTNode& node = tree_[idx];
        double edge_cost = this->edgeCost(node.config, q_new);
        double total_cost = node.cost + edge_cost;
        
        if (total_cost < best_cost && isEdgeCollisionFree(node.config, q_new)) {
            best_cost = total_cost;
            best_parent = idx;
        }
    }
    
    return {best_parent, best_cost};
}

void RRTStar::rewireNearbyNodes(
    int new_node_idx,
    const std::vector<int>& nearby_indices
) {
    const RRTNode& new_node = tree_[new_node_idx];
    
    for (int idx : nearby_indices) {
        if (idx == new_node.parent_idx) continue;  // Skip parent
        
        RRTNode& nearby_node = tree_[idx];
        double cost_through_new = new_node.cost + edgeCost(new_node.config, nearby_node.config);
        
        if (cost_through_new < nearby_node.cost && 
            isEdgeCollisionFree(new_node.config, nearby_node.config)) {
            
            nearby_node.parent_idx = new_node_idx;
            nearby_node.cost = cost_through_new;
            
            // Propagate cost update to descendants
            propagateCostUpdate(idx);
        }
    }
}

double RRTStar::edgeCost(
    const Eigen::VectorXd& q1,
    const Eigen::VectorXd& q2
) const {
    return distance(q1, q2);
}

void RRTStar::propagateCostUpdate(int node_idx) {
    // BFS to update costs of all descendants
    std::vector<int> to_update;
    to_update.push_back(node_idx);
    
    while (!to_update.empty()) {
        int current_idx = to_update.back();
        to_update.pop_back();
        
        // Find children of current node
        for (size_t i = 0; i < tree_.size(); ++i) {
            if (tree_[i].parent_idx == current_idx) {
                double new_cost = tree_[current_idx].cost + 
                                 edgeCost(tree_[current_idx].config, tree_[i].config);
                if (new_cost < tree_[i].cost) {
                    tree_[i].cost = new_cost;
                    to_update.push_back(i);
                }
            }
        }
    }
}

bool RRTStar::canImprove() const {
    // Check if there's potential for significant improvement
    return best_cost_ == std::numeric_limits<double>::infinity() ||
           best_cost_ > star_config_.improvement_threshold;
}

// InformedRRTStar implementation

InformedRRTStar::InformedRRTStar(
    int num_joints,
    CollisionChecker collision_check,
    const RRTStarConfig& config
) : RRTStar(num_joints, collision_check, config) {}

PlanningResult InformedRRTStar::plan(
    const Eigen::VectorXd& start,
    const Eigen::VectorXd& goal
) {
    start_ = start;
    goal_ = goal;
    c_min_ = distance(start, goal);
    c_best_ = std::numeric_limits<double>::infinity();
    
    // Run standard RRT* until solution found
    PlanningResult result = RRTStar::plan(start, goal);
    
    if (result.success) {
        c_best_ = result.path_length;
    }
    
    return result;
}

Eigen::VectorXd InformedRRTStar::sampleInformed() {
    if (c_best_ == std::numeric_limits<double>::infinity()) {
        return sampleRandom();
    }
    
    // Sample from ellipsoid
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    // Generate random point in unit ball
    Eigen::VectorXd x_ball(num_joints_);
    double r2 = 0.0;
    do {
        r2 = 0.0;
        for (int i = 0; i < num_joints_; ++i) {
            x_ball(i) = dist(rng_);
            r2 += x_ball(i) * x_ball(i);
        }
    } while (r2 > 1.0);
    
    // Scale to ellipsoid
    double c_max = c_best_;
    Eigen::VectorXd radii(num_joints_);
    radii(0) = c_max / 2.0;
    for (int i = 1; i < num_joints_; ++i) {
        radii(i) = std::sqrt(c_max * c_max - c_min_ * c_min_) / 2.0;
    }
    
    Eigen::VectorXd x_ellipse = radii.asDiagonal() * x_ball;
    
    // Rotate and translate to world frame
    Eigen::MatrixXd C = computeRotationToWorldFrame();
    Eigen::VectorXd x_center = (start_ + goal_) / 2.0;
    
    return C * x_ellipse + x_center;
}

Eigen::MatrixXd InformedRRTStar::computeRotationToWorldFrame() const {
    // Compute rotation matrix that aligns first axis with start-goal direction
    Eigen::VectorXd a1 = (goal_ - start_).normalized();
    
    // Simple case: identity-like rotation for single-axis alignment
    Eigen::MatrixXd C = Eigen::MatrixXd::Identity(num_joints_, num_joints_);
    C.col(0) = a1;
    
    // Gram-Schmidt for remaining columns
    for (int i = 1; i < num_joints_; ++i) {
        Eigen::VectorXd ei = Eigen::VectorXd::Zero(num_joints_);
        ei(i) = 1.0;
        
        // Orthogonalize against previous columns
        Eigen::VectorXd vi = ei;
        for (int j = 0; j < i; ++j) {
            vi -= vi.dot(C.col(j)) * C.col(j);
        }
        
        if (vi.norm() > 1e-6) {
            C.col(i) = vi.normalized();
        }
    }
    
    return C;
}

}  // namespace motion_planners
