/**
 * @file collision_checker.hpp
 * @brief Collision checking interface for motion planners
 * @author Barath Kumar JK
 * @date 2025
 */

#ifndef MOTION_PLANNERS_COLLISION_CHECKER_HPP
#define MOTION_PLANNERS_COLLISION_CHECKER_HPP

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <functional>

namespace motion_planners {

/**
 * @brief Axis-Aligned Bounding Box
 */
struct AABB {
    Eigen::Vector3d min_point;
    Eigen::Vector3d max_point;
    
    AABB() : min_point(Eigen::Vector3d::Zero()), max_point(Eigen::Vector3d::Zero()) {}
    AABB(const Eigen::Vector3d& min_pt, const Eigen::Vector3d& max_pt)
        : min_point(min_pt), max_point(max_pt) {}
    
    bool intersects(const AABB& other) const {
        return (min_point.x() <= other.max_point.x() && max_point.x() >= other.min_point.x()) &&
               (min_point.y() <= other.max_point.y() && max_point.y() >= other.min_point.y()) &&
               (min_point.z() <= other.max_point.z() && max_point.z() >= other.min_point.z());
    }
    
    bool contains(const Eigen::Vector3d& point) const {
        return (point.x() >= min_point.x() && point.x() <= max_point.x()) &&
               (point.y() >= min_point.y() && point.y() <= max_point.y()) &&
               (point.z() >= min_point.z() && point.z() <= max_point.z());
    }
    
    Eigen::Vector3d center() const {
        return (min_point + max_point) / 2.0;
    }
    
    Eigen::Vector3d size() const {
        return max_point - min_point;
    }
};

/**
 * @brief Sphere collision primitive
 */
struct Sphere {
    Eigen::Vector3d center;
    double radius;
    
    Sphere() : center(Eigen::Vector3d::Zero()), radius(0.0) {}
    Sphere(const Eigen::Vector3d& c, double r) : center(c), radius(r) {}
    
    bool intersects(const Sphere& other) const {
        double dist = (center - other.center).norm();
        return dist < (radius + other.radius);
    }
    
    AABB boundingBox() const {
        Eigen::Vector3d r(radius, radius, radius);
        return AABB(center - r, center + r);
    }
};

/**
 * @brief Capsule collision primitive (cylinder with hemispherical caps)
 */
struct Capsule {
    Eigen::Vector3d p1, p2;  // End points
    double radius;
    
    Capsule() : p1(Eigen::Vector3d::Zero()), p2(Eigen::Vector3d::Zero()), radius(0.0) {}
    Capsule(const Eigen::Vector3d& a, const Eigen::Vector3d& b, double r)
        : p1(a), p2(b), radius(r) {}
    
    /**
     * @brief Point-to-capsule distance
     */
    double distanceToPoint(const Eigen::Vector3d& point) const {
        Eigen::Vector3d ab = p2 - p1;
        double t = std::clamp((point - p1).dot(ab) / ab.squaredNorm(), 0.0, 1.0);
        Eigen::Vector3d closest = p1 + t * ab;
        return (point - closest).norm() - radius;
    }
    
    /**
     * @brief Capsule-to-capsule distance
     */
    double distanceToCapsule(const Capsule& other) const;
    
    AABB boundingBox() const {
        Eigen::Vector3d r(radius, radius, radius);
        Eigen::Vector3d min_pt = p1.cwiseMin(p2) - r;
        Eigen::Vector3d max_pt = p1.cwiseMax(p2) + r;
        return AABB(min_pt, max_pt);
    }
};

/**
 * @brief Obstacle in the environment
 */
struct Obstacle {
    enum class Type { SPHERE, BOX, CAPSULE };
    
    Type type;
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;
    Eigen::Vector3d dimensions;  // For box: half-extents, for sphere: (radius, 0, 0)
    
    Obstacle() : type(Type::SPHERE), 
                 position(Eigen::Vector3d::Zero()),
                 orientation(Eigen::Quaterniond::Identity()),
                 dimensions(Eigen::Vector3d::Zero()) {}
    
    AABB boundingBox() const {
        switch (type) {
            case Type::SPHERE:
                return Sphere(position, dimensions.x()).boundingBox();
            case Type::BOX:
                // Conservative AABB for rotated box
                return AABB(position - dimensions.norm() * Eigen::Vector3d::Ones(),
                           position + dimensions.norm() * Eigen::Vector3d::Ones());
            case Type::CAPSULE:
                return Capsule(position, position + orientation * Eigen::Vector3d(0, 0, dimensions.z()),
                              dimensions.x()).boundingBox();
            default:
                return AABB();
        }
    }
};

/**
 * @brief Robot link for collision checking
 */
struct RobotLink {
    std::string name;
    Capsule collision_geometry;
    int parent_joint;  // Joint that controls this link's position
    
    RobotLink() : parent_joint(-1) {}
    RobotLink(const std::string& n, const Capsule& geom, int parent)
        : name(n), collision_geometry(geom), parent_joint(parent) {}
};

/**
 * @brief Forward Kinematics function type
 */
using FKFunction = std::function<std::vector<Eigen::Matrix4d>(const Eigen::VectorXd&)>;

/**
 * @brief Collision Checker for Motion Planning
 */
class CollisionChecker {
public:
    CollisionChecker(int num_joints, FKFunction fk_func);
    
    /**
     * @brief Check if configuration is collision-free
     */
    bool isCollisionFree(const Eigen::VectorXd& q) const;
    
    /**
     * @brief Check if edge between two configs is collision-free
     */
    bool isEdgeCollisionFree(
        const Eigen::VectorXd& q1,
        const Eigen::VectorXd& q2,
        double resolution = 0.05
    ) const;
    
    /**
     * @brief Add obstacle to environment
     */
    void addObstacle(const Obstacle& obstacle);
    
    /**
     * @brief Remove obstacle by index
     */
    void removeObstacle(size_t index);
    
    /**
     * @brief Clear all obstacles
     */
    void clearObstacles();
    
    /**
     * @brief Set robot collision geometry
     */
    void setRobotLinks(const std::vector<RobotLink>& links);
    
    /**
     * @brief Set joint limits
     */
    void setJointLimits(
        const Eigen::VectorXd& lower,
        const Eigen::VectorXd& upper
    );
    
    /**
     * @brief Check if configuration is within joint limits
     */
    bool isWithinJointLimits(const Eigen::VectorXd& q) const;
    
    /**
     * @brief Get minimum distance to obstacles
     */
    double getMinDistance(const Eigen::VectorXd& q) const;
    
    /**
     * @brief Get collision checking statistics
     */
    struct Stats {
        int total_checks = 0;
        int collision_count = 0;
        double avg_check_time_us = 0.0;
    };
    const Stats& getStats() const { return stats_; }
    void resetStats() { stats_ = Stats(); }

private:
    int num_joints_;
    FKFunction fk_func_;
    
    std::vector<Obstacle> obstacles_;
    std::vector<RobotLink> robot_links_;
    
    Eigen::VectorXd joint_lower_limits_;
    Eigen::VectorXd joint_upper_limits_;
    bool limits_set_ = false;
    
    mutable Stats stats_;
    
    /**
     * @brief Check robot self-collision
     */
    bool checkSelfCollision(const std::vector<Eigen::Matrix4d>& link_transforms) const;
    
    /**
     * @brief Check robot-environment collision
     */
    bool checkEnvironmentCollision(const std::vector<Eigen::Matrix4d>& link_transforms) const;
    
    /**
     * @brief Transform robot link geometry to world frame
     */
    Capsule transformLink(const RobotLink& link, const Eigen::Matrix4d& transform) const;
    
    /**
     * @brief Check capsule-obstacle collision
     */
    bool checkCapsuleObstacleCollision(const Capsule& capsule, const Obstacle& obstacle) const;
};

/**
 * @brief MoveIt 2 Planning Scene adapter
 */
class MoveItCollisionChecker {
public:
    MoveItCollisionChecker();
    
    /**
     * @brief Update from MoveIt planning scene
     */
    void updateFromPlanningScene(/* moveit::planning_interface::PlanningSceneInterface */);
    
    /**
     * @brief Check collision using MoveIt
     */
    bool checkCollision(const Eigen::VectorXd& q) const;
    
private:
    // MoveIt planning scene would be stored here
};

}  // namespace motion_planners

#endif  // MOTION_PLANNERS_COLLISION_CHECKER_HPP
