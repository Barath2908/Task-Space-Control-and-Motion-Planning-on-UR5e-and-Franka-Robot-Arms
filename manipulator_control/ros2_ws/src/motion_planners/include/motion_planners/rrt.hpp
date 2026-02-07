/**
 * @file rrt.hpp
 * @brief Rapidly-exploring Random Trees (RRT) Motion Planner
 * @author Barath Kumar JK
 * @date 2025
 *
 * Features:
 * - Goal biasing for faster convergence
 * - Collision checking integration
 * - Planning time: 0.3-0.8s in cluttered scenes
 */

#ifndef MOTION_PLANNERS_RRT_HPP
#define MOTION_PLANNERS_RRT_HPP

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <random>
#include <functional>
#include <limits>
#include <chrono>

namespace motion_planners {

/**
 * @brief Configuration for RRT planner
 */
struct RRTConfig {
    int max_iterations = 10000;
    double step_size = 0.1;              // Joint-space step (rad)
    double goal_threshold = 0.05;        // Distance to goal (rad)
    double goal_bias = 0.1;              // Probability of sampling goal
    double timeout_sec = 5.0;            // Maximum planning time
    bool interpolate_path = true;        // Smooth output path
    int interpolation_steps = 10;        // Steps between waypoints
};

/**
 * @brief Node in the RRT tree
 */
struct RRTNode {
    Eigen::VectorXd config;              // Joint configuration
    int parent_idx = -1;                 // Index of parent node (-1 for root)
    double cost = 0.0;                   // Cost from root (for RRT*)
    
    RRTNode() = default;
    explicit RRTNode(const Eigen::VectorXd& q, int parent = -1, double c = 0.0)
        : config(q), parent_idx(parent), cost(c) {}
};

/**
 * @brief Planning result
 */
struct PlanningResult {
    std::vector<Eigen::VectorXd> path;   // Waypoints from start to goal
    bool success = false;
    int iterations = 0;
    double planning_time_sec = 0.0;
    int tree_size = 0;
    double path_length = 0.0;
    
    bool isValid() const { return success && !path.empty(); }
};

/**
 * @brief Collision checking function type
 * Returns true if configuration is collision-free
 */
using CollisionChecker = std::function<bool(const Eigen::VectorXd&)>;

/**
 * @brief Edge collision checking function type
 * Returns true if edge between two configs is collision-free
 */
using EdgeCollisionChecker = std::function<bool(
    const Eigen::VectorXd&, const Eigen::VectorXd&)>;

/**
 * @brief Distance function type
 */
using DistanceFunc = std::function<double(
    const Eigen::VectorXd&, const Eigen::VectorXd&)>;

/**
 * @brief Rapidly-exploring Random Trees (RRT) Planner
 *
 * Basic RRT algorithm:
 * 1. Sample random configuration (with goal bias)
 * 2. Find nearest node in tree
 * 3. Extend towards sample by step_size
 * 4. If collision-free, add to tree
 * 5. Check if goal reached
 */
class RRT {
public:
    explicit RRT(
        int num_joints,
        CollisionChecker collision_check,
        const RRTConfig& config = RRTConfig()
    );
    
    virtual ~RRT() = default;
    
    /**
     * @brief Plan path from start to goal
     * @param start Start configuration
     * @param goal Goal configuration
     * @return Planning result with path
     */
    virtual PlanningResult plan(
        const Eigen::VectorXd& start,
        const Eigen::VectorXd& goal
    );
    
    /**
     * @brief Set joint limits
     */
    void setJointLimits(
        const Eigen::VectorXd& lower,
        const Eigen::VectorXd& upper
    );
    
    /**
     * @brief Set edge collision checker
     */
    void setEdgeCollisionChecker(EdgeCollisionChecker checker);
    
    /**
     * @brief Set custom distance function
     */
    void setDistanceFunction(DistanceFunc func);
    
    /**
     * @brief Get configuration
     */
    const RRTConfig& getConfig() const { return config_; }
    
    /**
     * @brief Set configuration
     */
    void setConfig(const RRTConfig& config) { config_ = config; }
    
    /**
     * @brief Get tree nodes (for visualization)
     */
    const std::vector<RRTNode>& getTree() const { return tree_; }
    
    /**
     * @brief Clear tree
     */
    void clear() { tree_.clear(); }

protected:
    int num_joints_;
    RRTConfig config_;
    CollisionChecker collision_check_;
    EdgeCollisionChecker edge_collision_check_;
    DistanceFunc distance_func_;
    
    std::vector<RRTNode> tree_;
    Eigen::VectorXd joint_lower_limits_;
    Eigen::VectorXd joint_upper_limits_;
    
    std::mt19937 rng_;
    
    /**
     * @brief Sample random configuration
     */
    virtual Eigen::VectorXd sampleRandom();
    
    /**
     * @brief Sample with goal bias
     */
    Eigen::VectorXd sampleWithGoalBias(const Eigen::VectorXd& goal);
    
    /**
     * @brief Find nearest node to given configuration
     * @return Index of nearest node
     */
    virtual int findNearest(const Eigen::VectorXd& q) const;
    
    /**
     * @brief Steer from near towards rand by step_size
     */
    Eigen::VectorXd steer(
        const Eigen::VectorXd& q_near,
        const Eigen::VectorXd& q_rand
    ) const;
    
    /**
     * @brief Check if edge is collision-free
     */
    bool isEdgeCollisionFree(
        const Eigen::VectorXd& q1,
        const Eigen::VectorXd& q2
    ) const;
    
    /**
     * @brief Compute distance between configurations
     */
    double distance(
        const Eigen::VectorXd& q1,
        const Eigen::VectorXd& q2
    ) const;
    
    /**
     * @brief Check if configuration is at goal
     */
    bool isAtGoal(
        const Eigen::VectorXd& q,
        const Eigen::VectorXd& goal
    ) const;
    
    /**
     * @brief Extract path from tree
     */
    std::vector<Eigen::VectorXd> extractPath(int goal_idx) const;
    
    /**
     * @brief Interpolate path for smoother trajectory
     */
    std::vector<Eigen::VectorXd> interpolatePath(
        const std::vector<Eigen::VectorXd>& path
    ) const;
    
    /**
     * @brief Compute path length
     */
    double computePathLength(const std::vector<Eigen::VectorXd>& path) const;
};

}  // namespace motion_planners

#endif  // MOTION_PLANNERS_RRT_HPP
