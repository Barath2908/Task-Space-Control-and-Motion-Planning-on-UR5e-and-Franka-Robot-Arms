/**
 * @file jacobian_controller.cpp
 * @brief Implementation of Jacobian-based Cartesian controller
 * @author Barath Kumar JK
 * @date 2025
 */

#include "cartesian_controllers/jacobian_controller.hpp"
#include <iostream>

namespace cartesian_controllers {

JacobianController::JacobianController(const JacobianControllerConfig& config)
    : config_(config) {
    state_.joint_positions.resize(config_.num_joints);
    state_.joint_velocities.resize(config_.num_joints);
    state_.joint_velocity_commands.resize(config_.num_joints);
    state_.joint_positions.setZero();
    state_.joint_velocities.setZero();
    state_.joint_velocity_commands.setZero();
}

void JacobianController::setTargetPose(const CartesianPose& pose) {
    target_pose_ = pose;
    pose_control_mode_ = true;
}

void JacobianController::setTargetTwist(const CartesianTwist& twist) {
    target_twist_ = twist;
    pose_control_mode_ = false;
}

Eigen::VectorXd JacobianController::update(
    const Eigen::VectorXd& joint_positions,
    const JacobianMatrix& jacobian,
    const CartesianPose& current_pose
) {
    state_.joint_positions = joint_positions;
    state_.current_pose = current_pose;
    state_.target_pose = target_pose_;
    
    // Compute manipulability and condition number
    state_.manipulability = computeManipulability(jacobian);
    state_.condition_number = computeConditionNumber(jacobian);
    state_.near_singularity = state_.manipulability < config_.damping_threshold;
    
    // Compute Cartesian velocity command
    Eigen::Matrix<double, 6, 1> x_dot_cmd;
    
    if (pose_control_mode_) {
        // Pose control: proportional feedback on error
        state_.pose_error = current_pose.computeError(target_pose_);
        
        // Separate gains for position and orientation
        x_dot_cmd.head<3>() = config_.position_gain * state_.pose_error.head<3>();
        x_dot_cmd.tail<3>() = config_.orientation_gain * state_.pose_error.tail<3>();
        
        state_.position_error_norm = state_.pose_error.head<3>().norm();
        state_.orientation_error_norm = state_.pose_error.tail<3>().norm();
    } else {
        // Velocity control: direct twist command
        x_dot_cmd = target_twist_.toVector();
        state_.pose_error.setZero();
        state_.position_error_norm = 0.0;
        state_.orientation_error_norm = 0.0;
    }
    
    // Compute pseudoinverse with adaptive damping if near singularity
    Eigen::MatrixXd J_pinv;
    if (config_.adaptive_damping && state_.near_singularity) {
        double damping = computeAdaptiveDamping(state_.manipulability);
        J_pinv = computeDampedPseudoinverse(jacobian, damping);
    } else {
        J_pinv = computePseudoinverse(jacobian);
    }
    
    // Compute joint velocities: q̇ = J†·ẋ
    Eigen::VectorXd q_dot = J_pinv * x_dot_cmd;
    
    // Apply rate limiting
    q_dot = applyRateLimiting(q_dot);
    
    state_.joint_velocity_commands = q_dot;
    return q_dot;
}

Eigen::MatrixXd JacobianController::computePseudoinverse(const JacobianMatrix& J) const {
    // SVD-based pseudoinverse
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(J, Eigen::ComputeThinU | Eigen::ComputeThinV);
    
    const auto& singular_values = svd.singularValues();
    Eigen::VectorXd singular_values_inv(singular_values.size());
    
    double tolerance = 1e-6 * std::max(J.rows(), J.cols()) * singular_values.maxCoeff();
    
    for (int i = 0; i < singular_values.size(); ++i) {
        if (singular_values(i) > tolerance) {
            singular_values_inv(i) = 1.0 / singular_values(i);
        } else {
            singular_values_inv(i) = 0.0;
        }
    }
    
    return svd.matrixV() * singular_values_inv.asDiagonal() * svd.matrixU().transpose();
}

Eigen::MatrixXd JacobianController::computeDampedPseudoinverse(
    const JacobianMatrix& J,
    double damping
) const {
    // Damped Least Squares: J† = Jᵀ(JJᵀ + λ²I)⁻¹
    int m = J.rows();
    Eigen::MatrixXd JJt = J * J.transpose();
    Eigen::MatrixXd damped = JJt + damping * damping * Eigen::MatrixXd::Identity(m, m);
    
    return J.transpose() * damped.inverse();
}

double JacobianController::computeManipulability(const JacobianMatrix& J) const {
    // w = √det(JJᵀ)
    Eigen::MatrixXd JJt = J * J.transpose();
    double det = JJt.determinant();
    return det > 0 ? std::sqrt(det) : 0.0;
}

double JacobianController::computeConditionNumber(const JacobianMatrix& J) const {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(J);
    const auto& sv = svd.singularValues();
    
    if (sv.minCoeff() < 1e-10) {
        return std::numeric_limits<double>::infinity();
    }
    return sv.maxCoeff() / sv.minCoeff();
}

Eigen::VectorXd JacobianController::applyRateLimiting(const Eigen::VectorXd& q_dot) const {
    // Find scaling factor to keep all joints within limits
    double max_rate = q_dot.cwiseAbs().maxCoeff();
    
    if (max_rate > config_.max_joint_velocity) {
        state_.rate_limited = true;
        double scale = config_.max_joint_velocity / max_rate;
        return q_dot * scale;
    }
    
    state_.rate_limited = false;
    return q_dot;
}

double JacobianController::computeAdaptiveDamping(double manipulability) const {
    // Adaptive damping: λ = λ_max · (1 - w/w_threshold) when w < threshold
    if (manipulability >= config_.damping_threshold) {
        return 0.0;
    }
    
    double ratio = manipulability / config_.damping_threshold;
    return config_.damping_factor * (1.0 - ratio);
}

void JacobianController::reset() {
    state_.joint_positions.setZero();
    state_.joint_velocities.setZero();
    state_.joint_velocity_commands.setZero();
    state_.pose_error.setZero();
    state_.manipulability = 0.0;
    state_.condition_number = 0.0;
    state_.near_singularity = false;
    state_.rate_limited = false;
}

bool JacobianController::atTarget() const {
    return state_.position_error_norm < config_.position_tolerance &&
           state_.orientation_error_norm < config_.orientation_tolerance;
}

// JacobianControllerWithFeedforward implementation

JacobianControllerWithFeedforward::JacobianControllerWithFeedforward(
    const JacobianControllerConfig& config
) : JacobianController(config) {}

void JacobianControllerWithFeedforward::setFeedforwardTwist(const CartesianTwist& twist) {
    feedforward_twist_ = twist;
}

Eigen::VectorXd JacobianControllerWithFeedforward::updateWithFeedforward(
    const Eigen::VectorXd& joint_positions,
    const JacobianMatrix& jacobian,
    const CartesianPose& current_pose
) {
    // Get feedback component
    Eigen::VectorXd q_dot_fb = update(joint_positions, jacobian, current_pose);
    
    // Add feedforward component
    Eigen::MatrixXd J_pinv = computePseudoinverse(jacobian);
    Eigen::VectorXd q_dot_ff = J_pinv * feedforward_twist_.toVector();
    
    // Combine and apply rate limiting
    Eigen::VectorXd q_dot_total = q_dot_fb + q_dot_ff;
    return applyRateLimiting(q_dot_total);
}

}  // namespace cartesian_controllers
