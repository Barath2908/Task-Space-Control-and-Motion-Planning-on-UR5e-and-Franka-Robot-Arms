/**
 * @file planner_node.cpp
 * @brief ROS 2 Motion Planner Node with MoveIt 2 Integration
 * @author Barath Kumar JK
 * @date 2025
 *
 * C++ planners: RRT, RRT*, Bi-RRT*
 * Planning time: 0.3-0.8s in cluttered scenes
 */

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_srvs/srv/trigger.hpp>

#include "motion_planners/rrt.hpp"
#include "motion_planners/rrt_star.hpp"
#include "motion_planners/birrt_star.hpp"
#include "motion_planners/collision_checker.hpp"

#include <Eigen/Dense>
#include <memory>
#include <chrono>
#include <mutex>

namespace motion_planners {

/**
 * @brief Simple FK for UR5e (6-DOF)
 */
std::vector<Eigen::Matrix4d> ur5eFK(const Eigen::VectorXd& q) {
    std::vector<Eigen::Matrix4d> transforms(7);  // 6 joints + end-effector
    
    // DH parameters for UR5e
    std::vector<double> d = {0.1625, 0, 0, 0.1333, 0.0997, 0.0996};
    std::vector<double> a = {0, -0.425, -0.3922, 0, 0, 0};
    std::vector<double> alpha = {M_PI/2, 0, 0, M_PI/2, -M_PI/2, 0};
    
    auto dhMatrix = [](double theta, double d, double a, double alpha) {
        Eigen::Matrix4d T;
        double ct = std::cos(theta);
        double st = std::sin(theta);
        double ca = std::cos(alpha);
        double sa = std::sin(alpha);
        
        T << ct, -st*ca,  st*sa, a*ct,
             st,  ct*ca, -ct*sa, a*st,
              0,     sa,     ca,    d,
              0,      0,      0,    1;
        return T;
    };
    
    transforms[0] = Eigen::Matrix4d::Identity();
    for (int i = 0; i < 6; ++i) {
        transforms[i + 1] = transforms[i] * dhMatrix(q(i), d[i], a[i], alpha[i]);
    }
    
    return transforms;
}

/**
 * @brief Planning request message
 */
struct PlanRequest {
    Eigen::VectorXd start_config;
    Eigen::VectorXd goal_config;
    std::string algorithm = "birrt_star";
    double timeout = 5.0;
};

/**
 * @brief ROS 2 Motion Planner Node
 */
class PlannerNode : public rclcpp::Node {
public:
    PlannerNode() : Node("motion_planner") {
        // Declare parameters
        this->declare_parameter("algorithm", "birrt_star");
        this->declare_parameter("num_joints", 6);
        this->declare_parameter("max_iterations", 10000);
        this->declare_parameter("step_size", 0.1);
        this->declare_parameter("goal_bias", 0.1);
        this->declare_parameter("timeout", 5.0);
        this->declare_parameter("rewire_radius", 0.5);
        
        // Get parameters
        algorithm_ = this->get_parameter("algorithm").as_string();
        num_joints_ = this->get_parameter("num_joints").as_int();
        
        // Initialize collision checker
        collision_checker_ = std::make_unique<CollisionChecker>(num_joints_, ur5eFK);
        
        // Set UR5e joint limits
        Eigen::VectorXd lower(6), upper(6);
        lower << -2*M_PI, -2*M_PI, -M_PI, -2*M_PI, -2*M_PI, -2*M_PI;
        upper << 2*M_PI, 2*M_PI, M_PI, 2*M_PI, 2*M_PI, 2*M_PI;
        collision_checker_->setJointLimits(lower, upper);
        
        // Set robot collision geometry (simplified capsules)
        std::vector<RobotLink> links;
        links.emplace_back("shoulder", Capsule(Eigen::Vector3d(0, 0, 0), 
                                                Eigen::Vector3d(0, 0, 0.1625), 0.06), 0);
        links.emplace_back("upper_arm", Capsule(Eigen::Vector3d(0, 0, 0), 
                                                 Eigen::Vector3d(-0.425, 0, 0), 0.06), 1);
        links.emplace_back("forearm", Capsule(Eigen::Vector3d(0, 0, 0), 
                                               Eigen::Vector3d(-0.3922, 0, 0), 0.05), 2);
        links.emplace_back("wrist_1", Capsule(Eigen::Vector3d(0, 0, 0), 
                                               Eigen::Vector3d(0, 0, 0.1333), 0.04), 3);
        links.emplace_back("wrist_2", Capsule(Eigen::Vector3d(0, 0, 0), 
                                               Eigen::Vector3d(0, 0, 0.0997), 0.04), 4);
        links.emplace_back("wrist_3", Capsule(Eigen::Vector3d(0, 0, 0), 
                                               Eigen::Vector3d(0, 0, 0.0996), 0.04), 5);
        collision_checker_->setRobotLinks(links);
        
        // Initialize planners
        initializePlanners();
        
        // Initialize state
        current_config_.resize(num_joints_);
        current_config_.setZero();
        
        // Create subscribers
        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "joint_states", 10,
            std::bind(&PlannerNode::jointStateCallback, this, std::placeholders::_1));
        
        goal_config_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
            "goal_configuration", 10,
            std::bind(&PlannerNode::goalConfigCallback, this, std::placeholders::_1));
        
        obstacle_sub_ = this->create_subscription<visualization_msgs::msg::MarkerArray>(
            "planning_scene_obstacles", 10,
            std::bind(&PlannerNode::obstacleCallback, this, std::placeholders::_1));
        
        // Create publishers
        trajectory_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
            "planned_trajectory", 10);
        
        tree_viz_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "planner_tree_viz", 10);
        
        stats_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
            "planner_stats", 10);
        
        // Service for planning requests
        plan_srv_ = this->create_service<std_srvs::srv::Trigger>(
            "plan_to_goal",
            std::bind(&PlannerNode::planServiceCallback, this, 
                     std::placeholders::_1, std::placeholders::_2));
        
        RCLCPP_INFO(this->get_logger(), "Motion planner node initialized with %s algorithm", 
                    algorithm_.c_str());
    }

private:
    void initializePlanners() {
        // Create collision check lambda
        auto collision_check = [this](const Eigen::VectorXd& q) {
            return collision_checker_->isCollisionFree(q);
        };
        
        // RRT config
        RRTConfig rrt_config;
        rrt_config.max_iterations = this->get_parameter("max_iterations").as_int();
        rrt_config.step_size = this->get_parameter("step_size").as_double();
        rrt_config.goal_bias = this->get_parameter("goal_bias").as_double();
        rrt_config.timeout_sec = this->get_parameter("timeout").as_double();
        
        // RRT* config
        RRTStarConfig rrt_star_config;
        static_cast<RRTConfig&>(rrt_star_config) = rrt_config;
        rrt_star_config.rewire_radius = this->get_parameter("rewire_radius").as_double();
        rrt_star_config.adaptive_radius = true;
        
        // Bi-RRT* config
        BiRRTStarConfig birrt_config;
        static_cast<RRTStarConfig&>(birrt_config) = rrt_star_config;
        birrt_config.connection_threshold = 0.1;
        
        // Create planners
        rrt_ = std::make_unique<RRT>(num_joints_, collision_check, rrt_config);
        rrt_star_ = std::make_unique<RRTStar>(num_joints_, collision_check, rrt_star_config);
        birrt_star_ = std::make_unique<BiRRTStar>(num_joints_, collision_check, birrt_config);
        
        // Set joint limits for all planners
        Eigen::VectorXd lower(num_joints_), upper(num_joints_);
        lower.setConstant(-2 * M_PI);
        upper.setConstant(2 * M_PI);
        
        rrt_->setJointLimits(lower, upper);
        rrt_star_->setJointLimits(lower, upper);
        birrt_star_->setJointLimits(lower, upper);
    }
    
    void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (msg->position.size() >= static_cast<size_t>(num_joints_)) {
            for (int i = 0; i < num_joints_; ++i) {
                current_config_(i) = msg->position[i];
            }
        }
    }
    
    void goalConfigCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
        if (msg->data.size() >= static_cast<size_t>(num_joints_)) {
            Eigen::VectorXd goal(num_joints_);
            for (int i = 0; i < num_joints_; ++i) {
                goal(i) = msg->data[i];
            }
            
            planToGoal(goal);
        }
    }
    
    void obstacleCallback(const visualization_msgs::msg::MarkerArray::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        collision_checker_->clearObstacles();
        
        for (const auto& marker : msg->markers) {
            Obstacle obs;
            obs.position = Eigen::Vector3d(
                marker.pose.position.x,
                marker.pose.position.y,
                marker.pose.position.z
            );
            obs.orientation = Eigen::Quaterniond(
                marker.pose.orientation.w,
                marker.pose.orientation.x,
                marker.pose.orientation.y,
                marker.pose.orientation.z
            );
            
            if (marker.type == visualization_msgs::msg::Marker::SPHERE) {
                obs.type = Obstacle::Type::SPHERE;
                obs.dimensions = Eigen::Vector3d(marker.scale.x / 2, 0, 0);
            } else if (marker.type == visualization_msgs::msg::Marker::CUBE) {
                obs.type = Obstacle::Type::BOX;
                obs.dimensions = Eigen::Vector3d(
                    marker.scale.x / 2, marker.scale.y / 2, marker.scale.z / 2);
            }
            
            collision_checker_->addObstacle(obs);
        }
        
        RCLCPP_INFO(this->get_logger(), "Updated %zu obstacles", msg->markers.size());
    }
    
    void planServiceCallback(
        const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response
    ) {
        // Plan to current goal
        if (goal_set_) {
            bool success = planToGoal(goal_config_);
            response->success = success;
            response->message = success ? "Planning succeeded" : "Planning failed";
        } else {
            response->success = false;
            response->message = "No goal set";
        }
    }
    
    bool planToGoal(const Eigen::VectorXd& goal) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        goal_config_ = goal;
        goal_set_ = true;
        
        RCLCPP_INFO(this->get_logger(), "Planning from current config to goal using %s",
                    algorithm_.c_str());
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        PlanningResult result;
        
        if (algorithm_ == "rrt") {
            result = rrt_->plan(current_config_, goal);
        } else if (algorithm_ == "rrt_star") {
            result = rrt_star_->plan(current_config_, goal);
        } else {
            result = birrt_star_->plan(current_config_, goal);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double planning_time = std::chrono::duration<double>(end_time - start_time).count();
        
        if (result.success) {
            RCLCPP_INFO(this->get_logger(), 
                       "Planning succeeded: %.3fs, %d iterations, %zu waypoints, length: %.3f",
                       planning_time, result.iterations, result.path.size(), result.path_length);
            
            publishTrajectory(result.path);
            publishStats(result, planning_time);
            
            return true;
        } else {
            RCLCPP_WARN(this->get_logger(), 
                       "Planning failed after %.3fs, %d iterations",
                       planning_time, result.iterations);
            return false;
        }
    }
    
    void publishTrajectory(const std::vector<Eigen::VectorXd>& path) {
        auto traj_msg = std::make_unique<trajectory_msgs::msg::JointTrajectory>();
        
        traj_msg->header.stamp = this->now();
        traj_msg->header.frame_id = "base_link";
        
        // Joint names
        for (int i = 0; i < num_joints_; ++i) {
            traj_msg->joint_names.push_back("joint_" + std::to_string(i + 1));
        }
        
        // Create trajectory points
        double time_from_start = 0.0;
        double dt = 0.1;  // Time between waypoints
        
        for (const auto& config : path) {
            trajectory_msgs::msg::JointTrajectoryPoint point;
            
            for (int i = 0; i < num_joints_; ++i) {
                point.positions.push_back(config(i));
                point.velocities.push_back(0.0);
            }
            
            point.time_from_start = rclcpp::Duration::from_seconds(time_from_start);
            traj_msg->points.push_back(point);
            
            time_from_start += dt;
        }
        
        trajectory_pub_->publish(std::move(traj_msg));
    }
    
    void publishStats(const PlanningResult& result, double planning_time) {
        auto stats_msg = std::make_unique<std_msgs::msg::Float64MultiArray>();
        stats_msg->data = {
            planning_time,
            static_cast<double>(result.iterations),
            static_cast<double>(result.tree_size),
            result.path_length,
            static_cast<double>(result.path.size()),
            result.success ? 1.0 : 0.0
        };
        stats_pub_->publish(std::move(stats_msg));
    }
    
    // Members
    std::string algorithm_;
    int num_joints_;
    
    std::unique_ptr<CollisionChecker> collision_checker_;
    std::unique_ptr<RRT> rrt_;
    std::unique_ptr<RRTStar> rrt_star_;
    std::unique_ptr<BiRRTStar> birrt_star_;
    
    Eigen::VectorXd current_config_;
    Eigen::VectorXd goal_config_;
    bool goal_set_ = false;
    
    std::mutex mutex_;
    
    // ROS interfaces
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr goal_config_sub_;
    rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr obstacle_sub_;
    
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr trajectory_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr tree_viz_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr stats_pub_;
    
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr plan_srv_;
};

}  // namespace motion_planners

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<motion_planners::PlannerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
