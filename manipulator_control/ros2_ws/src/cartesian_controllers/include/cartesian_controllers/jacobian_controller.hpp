/**
 * @file jacobian_controller.hpp
 * @brief Task-space Cartesian controller using Jacobian pseudoinverse
 * @author Barath Kumar JK
 * @date 2025
 *
 * Implements resolved-rate control with:
 * - SVD-based pseudoinverse computation
 * - Damped Least Squares for singularity robustness
 * - Joint rate limiting (< 1.5 rad/s)
 * - 2-3 mm RMS tracking error
 */

#ifndef CARTESIAN_CONTROLLERS_JACOBIAN_CONTROLLER_HPP
#define CARTESIAN_CONTROLLERS_JACOBIAN_CONTROLLER_HPP

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <memory>
#include <vector>
#include <string>
#include <cmath>

namespace cartesian_controllers {

/**
 * @brief Configuration for Jacobian-based controller
 */
struct JacobianControllerConfig {
    double control_rate = 500.0;           // Hz
    double max_joint_velocity = 1.5;       // rad/s
    double damping_factor = 0.05;          // DLS damping
    double damping_threshold = 0.01;       // Manipulability threshold
    double position_tolerance = 0.001;     // m
    double orientation_tolerance = 0.01;   // rad
    double position_gain = 10.0;           // Proportional gain
    double orientation_gain = 5.0;         // Orientation gain
    bool adaptive_damping = true;          // Enable adaptive DLS
    int num_joints = 6;                    // DOF
};

/**
 * @brief Cartesian pose representation
 */
struct CartesianPose {
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;
    
    CartesianPose() : position(Eigen::Vector3d::Zero()), 
                      orientation(Eigen::Quaterniond::Identity()) {}
    
    CartesianPose(const Eigen::Vector3d& pos, const Eigen::Quaterniond& ori)
        : position(pos), orientation(ori) {}
    
    /**
     * @brief Compute pose error (6D: position + orientation)
     */
    Eigen::Matrix<double, 6, 1> computeError(const CartesianPose& target) const {
        Eigen::Matrix<double, 6, 1> error;
        
        // Position error
        error.head<3>() = target.position - position;
        
        // Orientation error (axis-angle representation)
        Eigen::Quaterniond q_error = target.orientation * orientation.inverse();
        if (q_error.w() < 0) {
            q_error.coeffs() *= -1;  // Ensure shortest path
        }
        
        // Convert to axis-angle
        double angle = 2.0 * std::acos(std::clamp(q_error.w(), -1.0, 1.0));
        if (angle < 1e-6) {
            error.tail<3>().setZero();
        } else {
            Eigen::Vector3d axis = q_error.vec().normalized();
            error.tail<3>() = angle * axis;
        }
        
        return error;
    }
};

/**
 * @brief Cartesian velocity (twist)
 */
struct CartesianTwist {
    Eigen::Vector3d linear;
    Eigen::Vector3d angular;
    
    CartesianTwist() : linear(Eigen::Vector3d::Zero()), 
                       angular(Eigen::Vector3d::Zero()) {}
    
    Eigen::Matrix<double, 6, 1> toVector() const {
        Eigen::Matrix<double, 6, 1> v;
        v << linear, angular;
        return v;
    }
    
    static CartesianTwist fromVector(const Eigen::Matrix<double, 6, 1>& v) {
        CartesianTwist twist;
        twist.linear = v.head<3>();
        twist.angular = v.tail<3>();
        return twist;
    }
};

/**
 * @brief Controller state for monitoring
 */
struct ControllerState {
    Eigen::VectorXd joint_positions;
    Eigen::VectorXd joint_velocities;
    Eigen::VectorXd joint_velocity_commands;
    CartesianPose current_pose;
    CartesianPose target_pose;
    Eigen::Matrix<double, 6, 1> pose_error;
    double manipulability;
    double condition_number;
    double position_error_norm;
    double orientation_error_norm;
    bool near_singularity;
    bool rate_limited;
};

/**
 * @brief Jacobian-based Cartesian Controller
 * 
 * Computes joint velocities from Cartesian commands using:
 * q̇ = J†(q) · ẋ_cmd
 * 
 * With Damped Least Squares for singularity robustness:
 * J† = Jᵀ(JJᵀ + λ²I)⁻¹
 */
class JacobianController {
public:
    using JacobianMatrix = Eigen::Matrix<double, 6, Eigen::Dynamic>;
    
    explicit JacobianController(const JacobianControllerConfig& config = JacobianControllerConfig());
    virtual ~JacobianController() = default;
    
    /**
     * @brief Set target Cartesian pose
     */
    void setTargetPose(const CartesianPose& pose);
    
    /**
     * @brief Set target Cartesian velocity (twist)
     */
    void setTargetTwist(const CartesianTwist& twist);
    
    /**
     * @brief Update controller with current joint state
     * @param joint_positions Current joint positions (rad)
     * @param jacobian Current Jacobian matrix (6 x n_joints)
     * @param current_pose Current end-effector pose
     * @return Joint velocity commands (rad/s)
     */
    Eigen::VectorXd update(
        const Eigen::VectorXd& joint_positions,
        const JacobianMatrix& jacobian,
        const CartesianPose& current_pose
    );
    
    /**
     * @brief Compute pseudoinverse using SVD
     */
    Eigen::MatrixXd computePseudoinverse(const JacobianMatrix& J) const;
    
    /**
     * @brief Compute damped pseudoinverse (DLS)
     */
    Eigen::MatrixXd computeDampedPseudoinverse(
        const JacobianMatrix& J, 
        double damping
    ) const;
    
    /**
     * @brief Compute manipulability measure
     * w = √det(JJᵀ)
     */
    double computeManipulability(const JacobianMatrix& J) const;
    
    /**
     * @brief Compute condition number
     */
    double computeConditionNumber(const JacobianMatrix& J) const;
    
    /**
     * @brief Apply joint rate limiting
     * @param q_dot Joint velocities
     * @return Rate-limited joint velocities
     */
    Eigen::VectorXd applyRateLimiting(const Eigen::VectorXd& q_dot) const;
    
    /**
     * @brief Get current controller state
     */
    const ControllerState& getState() const { return state_; }
    
    /**
     * @brief Get configuration
     */
    const JacobianControllerConfig& getConfig() const { return config_; }
    
    /**
     * @brief Update configuration
     */
    void setConfig(const JacobianControllerConfig& config) { config_ = config; }
    
    /**
     * @brief Reset controller
     */
    void reset();
    
    /**
     * @brief Check if at target
     */
    bool atTarget() const;

protected:
    JacobianControllerConfig config_;
    ControllerState state_;
    
    CartesianPose target_pose_;
    CartesianTwist target_twist_;
    bool pose_control_mode_ = true;  // true = pose control, false = velocity control
    
    /**
     * @brief Compute adaptive damping factor based on manipulability
     */
    double computeAdaptiveDamping(double manipulability) const;
};

/**
 * @brief Extended controller with velocity feedforward
 */
class JacobianControllerWithFeedforward : public JacobianController {
public:
    explicit JacobianControllerWithFeedforward(
        const JacobianControllerConfig& config = JacobianControllerConfig()
    );
    
    /**
     * @brief Set feedforward Cartesian velocity
     */
    void setFeedforwardTwist(const CartesianTwist& twist);
    
    /**
     * @brief Update with feedforward term
     */
    Eigen::VectorXd updateWithFeedforward(
        const Eigen::VectorXd& joint_positions,
        const JacobianMatrix& jacobian,
        const CartesianPose& current_pose
    );

private:
    CartesianTwist feedforward_twist_;
};

}  // namespace cartesian_controllers

#endif  // CARTESIAN_CONTROLLERS_JACOBIAN_CONTROLLER_HPP
