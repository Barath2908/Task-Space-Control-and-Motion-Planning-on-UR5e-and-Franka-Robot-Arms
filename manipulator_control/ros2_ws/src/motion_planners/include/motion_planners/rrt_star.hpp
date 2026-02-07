/**
 * @file rrt_star.hpp
 * @brief Optimal RRT* Motion Planner with Cost-Based Rewiring
 * @author Barath Kumar JK
 * @date 2025
 *
 * RRT* provides asymptotically optimal paths through:
 * - Cost-aware tree extension
 * - Local rewiring of nearby nodes
 * - Continuous path improvement
 */

#ifndef MOTION_PLANNERS_RRT_STAR_HPP
#define MOTION_PLANNERS_RRT_STAR_HPP

#include "motion_planners/rrt.hpp"
#include <algorithm>
#include <cmath>

namespace motion_planners {

/**
 * @brief Configuration for RRT* planner
 */
struct RRTStarConfig : public RRTConfig {
    double rewire_radius = 0.5;          // Fixed rewiring radius (if not adaptive)
    bool adaptive_radius = true;         // Use adaptive radius based on tree size
    double gamma_rrt_star = 1.5;         // Gamma factor for adaptive radius
    int max_rewire_neighbors = 30;       // Maximum neighbors to consider for rewiring
    bool continue_after_solution = true; // Keep improving after finding solution
    double improvement_threshold = 0.01; // Stop if improvement < threshold
};

/**
 * @brief RRT* Optimal Motion Planner
 *
 * Extends RRT with:
 * 1. Cost tracking for each node
 * 2. Near-neighbor search in radius
 * 3. Choosing best parent for new node
 * 4. Rewiring nearby nodes through new node
 *
 * Asymptotically optimal: converges to optimal path as iterations → ∞
 */
class RRTStar : public RRT {
public:
    explicit RRTStar(
        int num_joints,
        CollisionChecker collision_check,
        const RRTStarConfig& config = RRTStarConfig()
    );
    
    /**
     * @brief Plan optimal path from start to goal
     */
    PlanningResult plan(
        const Eigen::VectorXd& start,
        const Eigen::VectorXd& goal
    ) override;
    
    /**
     * @brief Get RRT* configuration
     */
    const RRTStarConfig& getRRTStarConfig() const { return star_config_; }
    
    /**
     * @brief Set RRT* configuration
     */
    void setRRTStarConfig(const RRTStarConfig& config) { 
        star_config_ = config;
        config_ = config;
    }
    
    /**
     * @brief Get current best cost to goal
     */
    double getBestCost() const { return best_cost_; }

protected:
    RRTStarConfig star_config_;
    double best_cost_ = std::numeric_limits<double>::infinity();
    int goal_node_idx_ = -1;
    
    /**
     * @brief Compute rewiring radius based on tree size
     * r = min(γ * (log(n)/n)^(1/d), step_size)
     */
    double computeRewireRadius() const;
    
    /**
     * @brief Find nodes within radius of configuration
     */
    std::vector<int> findNearbyNodes(
        const Eigen::VectorXd& q,
        double radius
    ) const;
    
    /**
     * @brief Choose best parent among nearby nodes
     * @return Index of best parent, cost through that parent
     */
    std::pair<int, double> chooseBestParent(
        const Eigen::VectorXd& q_new,
        const std::vector<int>& nearby_indices
    ) const;
    
    /**
     * @brief Rewire nearby nodes through new node if cheaper
     */
    void rewireNearbyNodes(
        int new_node_idx,
        const std::vector<int>& nearby_indices
    );
    
    /**
     * @brief Compute cost of edge between two configurations
     */
    virtual double edgeCost(
        const Eigen::VectorXd& q1,
        const Eigen::VectorXd& q2
    ) const;
    
    /**
     * @brief Update costs of descendants after rewiring
     */
    void propagateCostUpdate(int node_idx);
    
    /**
     * @brief Check if current solution can be improved
     */
    bool canImprove() const;
};

/**
 * @brief Informed RRT* with ellipsoidal sampling
 *
 * After finding initial solution, samples from ellipsoidal
 * subset that could contain better solutions
 */
class InformedRRTStar : public RRTStar {
public:
    explicit InformedRRTStar(
        int num_joints,
        CollisionChecker collision_check,
        const RRTStarConfig& config = RRTStarConfig()
    );
    
    /**
     * @brief Plan with informed sampling
     */
    PlanningResult plan(
        const Eigen::VectorXd& start,
        const Eigen::VectorXd& goal
    ) override;

protected:
    Eigen::VectorXd start_;
    Eigen::VectorXd goal_;
    double c_best_ = std::numeric_limits<double>::infinity();
    double c_min_ = 0.0;
    
    /**
     * @brief Sample from informed ellipsoid
     */
    Eigen::VectorXd sampleInformed();
    
    /**
     * @brief Compute rotation matrix for ellipsoid
     */
    Eigen::MatrixXd computeRotationToWorldFrame() const;
};

}  // namespace motion_planners

#endif  // MOTION_PLANNERS_RRT_STAR_HPP
