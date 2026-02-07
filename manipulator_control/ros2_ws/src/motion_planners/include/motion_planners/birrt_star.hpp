/**
 * @file birrt_star.hpp
 * @brief Bidirectional RRT* Motion Planner
 * @author Barath Kumar JK
 * @date 2025
 *
 * Combines bidirectional search with RRT* optimization:
 * - Two trees grow from start and goal
 * - Trees attempt to connect at each iteration
 * - Cost optimization via rewiring
 * - Fastest convergence: 0.3-0.4s in cluttered scenes
 */

#ifndef MOTION_PLANNERS_BIRRT_STAR_HPP
#define MOTION_PLANNERS_BIRRT_STAR_HPP

#include "motion_planners/rrt_star.hpp"

namespace motion_planners {

/**
 * @brief Configuration for Bi-RRT* planner
 */
struct BiRRTStarConfig : public RRTStarConfig {
    double connection_threshold = 0.1;   // Distance threshold for tree connection
    int connection_attempts = 5;         // Attempts to connect trees per iteration
    bool alternate_trees = true;         // Alternate between trees each iteration
};

/**
 * @brief Bidirectional RRT tree
 */
struct BiRRTTree {
    std::vector<RRTNode> nodes;
    bool is_start_tree = true;           // true = grows from start, false = from goal
    
    void clear() { nodes.clear(); }
    size_t size() const { return nodes.size(); }
    bool empty() const { return nodes.empty(); }
};

/**
 * @brief Connection between two trees
 */
struct TreeConnection {
    int start_tree_idx = -1;
    int goal_tree_idx = -1;
    double cost = std::numeric_limits<double>::infinity();
    bool valid = false;
    
    bool isValid() const { return valid && start_tree_idx >= 0 && goal_tree_idx >= 0; }
};

/**
 * @brief Bidirectional RRT* Motion Planner
 *
 * Algorithm:
 * 1. Initialize trees at start and goal
 * 2. Alternate growing trees (RRT* extension)
 * 3. After each extension, attempt to connect trees
 * 4. If connected, extract and optimize path
 * 5. Continue searching for better connections
 *
 * Benefits:
 * - Faster than unidirectional RRT*
 * - Better exploration of narrow passages
 * - Continuous path improvement
 */
class BiRRTStar {
public:
    explicit BiRRTStar(
        int num_joints,
        CollisionChecker collision_check,
        const BiRRTStarConfig& config = BiRRTStarConfig()
    );
    
    /**
     * @brief Plan path from start to goal
     */
    PlanningResult plan(
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
    const BiRRTStarConfig& getConfig() const { return config_; }
    
    /**
     * @brief Set configuration
     */
    void setConfig(const BiRRTStarConfig& config) { config_ = config; }
    
    /**
     * @brief Get start tree (for visualization)
     */
    const BiRRTTree& getStartTree() const { return start_tree_; }
    
    /**
     * @brief Get goal tree (for visualization)
     */
    const BiRRTTree& getGoalTree() const { return goal_tree_; }
    
    /**
     * @brief Get best connection
     */
    const TreeConnection& getBestConnection() const { return best_connection_; }
    
    /**
     * @brief Clear both trees
     */
    void clear();

protected:
    int num_joints_;
    BiRRTStarConfig config_;
    CollisionChecker collision_check_;
    EdgeCollisionChecker edge_collision_check_;
    DistanceFunc distance_func_;
    
    BiRRTTree start_tree_;
    BiRRTTree goal_tree_;
    TreeConnection best_connection_;
    
    Eigen::VectorXd joint_lower_limits_;
    Eigen::VectorXd joint_upper_limits_;
    
    std::mt19937 rng_;
    
    /**
     * @brief Initialize both trees
     */
    void initializeTrees(
        const Eigen::VectorXd& start,
        const Eigen::VectorXd& goal
    );
    
    /**
     * @brief Extend tree towards random sample (RRT* style)
     * @return Index of new node, or -1 if extension failed
     */
    int extendTree(
        BiRRTTree& tree,
        const Eigen::VectorXd& q_sample
    );
    
    /**
     * @brief Attempt to connect trees
     * @return Connection if successful
     */
    TreeConnection attemptConnection(
        const BiRRTTree& tree_a,
        const BiRRTTree& tree_b,
        int new_node_idx
    );
    
    /**
     * @brief Find nearest node in tree
     */
    int findNearest(
        const BiRRTTree& tree,
        const Eigen::VectorXd& q
    ) const;
    
    /**
     * @brief Find nodes within radius
     */
    std::vector<int> findNearby(
        const BiRRTTree& tree,
        const Eigen::VectorXd& q,
        double radius
    ) const;
    
    /**
     * @brief Choose best parent for new node
     */
    std::pair<int, double> chooseBestParent(
        const BiRRTTree& tree,
        const Eigen::VectorXd& q_new,
        const std::vector<int>& nearby
    ) const;
    
    /**
     * @brief Rewire nearby nodes
     */
    void rewireNearby(
        BiRRTTree& tree,
        int new_idx,
        const std::vector<int>& nearby
    );
    
    /**
     * @brief Extract path from connected trees
     */
    std::vector<Eigen::VectorXd> extractPath(
        const TreeConnection& connection
    ) const;
    
    /**
     * @brief Sample random configuration
     */
    Eigen::VectorXd sampleRandom();
    
    /**
     * @brief Steer from near towards rand
     */
    Eigen::VectorXd steer(
        const Eigen::VectorXd& q_near,
        const Eigen::VectorXd& q_rand
    ) const;
    
    /**
     * @brief Compute distance
     */
    double distance(
        const Eigen::VectorXd& q1,
        const Eigen::VectorXd& q2
    ) const;
    
    /**
     * @brief Check edge collision
     */
    bool isEdgeCollisionFree(
        const Eigen::VectorXd& q1,
        const Eigen::VectorXd& q2
    ) const;
    
    /**
     * @brief Compute rewiring radius
     */
    double computeRewireRadius(size_t tree_size) const;
    
    /**
     * @brief Compute path cost
     */
    double computePathCost(const std::vector<Eigen::VectorXd>& path) const;
    
    /**
     * @brief Interpolate path
     */
    std::vector<Eigen::VectorXd> interpolatePath(
        const std::vector<Eigen::VectorXd>& path
    ) const;
};

}  // namespace motion_planners

#endif  // MOTION_PLANNERS_BIRRT_STAR_HPP
