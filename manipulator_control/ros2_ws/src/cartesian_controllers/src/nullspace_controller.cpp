/**
 * @file nullspace_controller.cpp
 * @brief Implementation of nullspace controller for 7-DOF manipulators
 * @author Barath Kumar JK
 * @date 2025
 */

#include "cartesian_controllers/nullspace_controller.hpp"

namespace cartesian_controllers {

NullspaceController::NullspaceController(
    const JacobianControllerConfig& jac_config,
    const NullspaceConfig& null_config
) : JacobianController(jac_config), null_config_(null_config) {
    nullspace_velocity_.resize(jac_config.num_joints);
    nullspace_velocity_.setZero();
    
    // Initialize with Franka defaults if 7 DOF
    if (jac_config.num_joints == 7 && null_config_.preferred_posture.size() == 0) {
        null_config_.setFrankaDefaults();
    }
}

Eigen::VectorXd NullspaceController::updateWithNullspace(
    const Eigen::VectorXd& joint_positions,
    const JacobianMatrix& jacobian,
    const CartesianPose& current_pose
) {
    // Get primary task velocity (Cartesian control)
    Eigen::VectorXd q_dot_primary = update(joint_positions, jacobian, current_pose);
    
    if (!null_config_.enabled || config_.num_joints <= 6) {
        // No nullspace for non-redundant manipulators
        return q_dot_primary;
    }
    
    // Compute pseudoinverse
    Eigen::MatrixXd J_pinv;
    if (config_.adaptive_damping && state_.near_singularity) {
        double damping = computeAdaptiveDamping(state_.manipulability);
        J_pinv = computeDampedPseudoinverse(jacobian, damping);
    } else {
        J_pinv = computePseudoinverse(jacobian);
    }
    
    // Compute nullspace projector: N = (I - J†J)
    Eigen::MatrixXd N = computeNullspaceProjector(jacobian, J_pinv);
    
    // Compute combined nullspace velocity
    Eigen::VectorXd q_dot_null = computeCombinedNullspaceVelocity(joint_positions, jacobian);
    
    // Project nullspace velocity
    nullspace_velocity_ = N * q_dot_null;
    
    // Combine primary task and nullspace
    Eigen::VectorXd q_dot_total = q_dot_primary + nullspace_velocity_;
    
    // Apply rate limiting to combined velocity
    return applyRateLimiting(q_dot_total);
}

void NullspaceController::setPreferredPosture(const Eigen::VectorXd& posture) {
    null_config_.preferred_posture = posture;
}

void NullspaceController::setJointLimits(
    const Eigen::VectorXd& lower,
    const Eigen::VectorXd& upper
) {
    null_config_.joint_lower_limits = lower;
    null_config_.joint_upper_limits = upper;
}

void NullspaceController::addNullspaceObjective(
    const std::string& name,
    NullspaceObjective objective,
    double gain
) {
    custom_objectives_[name] = {objective, gain};
}

void NullspaceController::removeNullspaceObjective(const std::string& name) {
    custom_objectives_.erase(name);
}

Eigen::MatrixXd NullspaceController::computeNullspaceProjector(
    const JacobianMatrix& J,
    const Eigen::MatrixXd& J_pinv
) const {
    int n = J.cols();
    return Eigen::MatrixXd::Identity(n, n) - J_pinv * J;
}

Eigen::VectorXd NullspaceController::computePostureVelocity(
    const Eigen::VectorXd& joint_positions
) const {
    if (null_config_.preferred_posture.size() != joint_positions.size()) {
        return Eigen::VectorXd::Zero(joint_positions.size());
    }
    
    // Simple proportional control to preferred posture
    return null_config_.posture_gain * (null_config_.preferred_posture - joint_positions);
}

Eigen::VectorXd NullspaceController::computeJointLimitGradient(
    const Eigen::VectorXd& joint_positions
) const {
    int n = joint_positions.size();
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(n);
    
    if (null_config_.joint_lower_limits.size() != n ||
        null_config_.joint_upper_limits.size() != n) {
        return gradient;
    }
    
    // Gradient-based repulsion from joint limits
    // h(q) = -1/2 * Σ ((q_i - q_mid_i) / (q_max_i - q_min_i))^2
    // ∂h/∂q_i pushes towards center of joint range
    
    for (int i = 0; i < n; ++i) {
        double q_min = null_config_.joint_lower_limits(i);
        double q_max = null_config_.joint_upper_limits(i);
        double q_mid = (q_min + q_max) / 2.0;
        double range = q_max - q_min;
        
        double distance_to_lower = joint_positions(i) - q_min;
        double distance_to_upper = q_max - joint_positions(i);
        
        // Apply gradient only when close to limits
        if (distance_to_lower < null_config_.joint_limit_threshold) {
            // Repulsion from lower limit
            double strength = (null_config_.joint_limit_threshold - distance_to_lower) / 
                             null_config_.joint_limit_threshold;
            gradient(i) += null_config_.joint_limit_gain * strength;
        }
        
        if (distance_to_upper < null_config_.joint_limit_threshold) {
            // Repulsion from upper limit
            double strength = (null_config_.joint_limit_threshold - distance_to_upper) /
                             null_config_.joint_limit_threshold;
            gradient(i) -= null_config_.joint_limit_gain * strength;
        }
    }
    
    return gradient;
}

Eigen::VectorXd NullspaceController::computeManipulabilityGradient(
    const Eigen::VectorXd& joint_positions,
    const JacobianMatrix& jacobian,
    double delta
) const {
    int n = joint_positions.size();
    Eigen::VectorXd gradient(n);
    
    double w_current = computeManipulability(jacobian);
    
    // Numerical gradient via finite differences
    // Note: In practice, you'd use the analytical gradient or
    // compute Jacobian at perturbed positions
    
    // For now, use simple central difference approximation
    // This requires FK and Jacobian computation at perturbed positions
    // which we don't have direct access to here
    
    // Simplified: gradient points towards mid-range (heuristic for manipulability)
    for (int i = 0; i < n; ++i) {
        double q_mid = 0.0;  // Assume mid-range is near zero
        if (null_config_.joint_lower_limits.size() == n &&
            null_config_.joint_upper_limits.size() == n) {
            q_mid = (null_config_.joint_lower_limits(i) + 
                    null_config_.joint_upper_limits(i)) / 2.0;
        }
        gradient(i) = null_config_.manipulability_gain * (q_mid - joint_positions(i));
    }
    
    return gradient;
}

Eigen::VectorXd NullspaceController::computeCombinedNullspaceVelocity(
    const Eigen::VectorXd& joint_positions,
    const JacobianMatrix& jacobian
) {
    int n = joint_positions.size();
    Eigen::VectorXd q_dot_null = Eigen::VectorXd::Zero(n);
    
    // 1. Posture control
    if (null_config_.posture_gain > 0) {
        q_dot_null += computePostureVelocity(joint_positions);
    }
    
    // 2. Joint limit avoidance
    if (null_config_.joint_limit_gain > 0) {
        q_dot_null += computeJointLimitGradient(joint_positions);
    }
    
    // 3. Manipulability maximization
    if (null_config_.manipulability_gain > 0) {
        q_dot_null += computeManipulabilityGradient(joint_positions, jacobian);
    }
    
    // 4. Custom objectives
    for (const auto& [name, obj] : custom_objectives_) {
        q_dot_null += obj.gain * obj.function(joint_positions, null_config_);
    }
    
    return q_dot_null;
}

// WeightedNullspaceController implementation

WeightedNullspaceController::WeightedNullspaceController(
    const JacobianControllerConfig& jac_config,
    const NullspaceConfig& null_config
) : NullspaceController(jac_config, null_config) {
    W_.resize(jac_config.num_joints);
    W_inv_.resize(jac_config.num_joints);
    
    // Initialize with identity weights
    for (int i = 0; i < jac_config.num_joints; ++i) {
        W_.diagonal()(i) = 1.0;
        W_inv_.diagonal()(i) = 1.0;
    }
}

void WeightedNullspaceController::setJointWeights(const Eigen::VectorXd& weights) {
    int n = weights.size();
    W_.resize(n);
    W_inv_.resize(n);
    
    for (int i = 0; i < n; ++i) {
        W_.diagonal()(i) = weights(i);
        W_inv_.diagonal()(i) = (weights(i) > 1e-6) ? 1.0 / weights(i) : 0.0;
    }
}

Eigen::MatrixXd WeightedNullspaceController::computeWeightedPseudoinverse(
    const JacobianMatrix& J
) const {
    // J†_w = W⁻¹Jᵀ(JW⁻¹Jᵀ)⁻¹
    Eigen::MatrixXd W_inv_Jt = W_inv_ * J.transpose();
    Eigen::MatrixXd JW_inv_Jt = J * W_inv_Jt;
    
    // Add small regularization for numerical stability
    double reg = 1e-6;
    JW_inv_Jt.diagonal().array() += reg;
    
    return W_inv_Jt * JW_inv_Jt.inverse();
}

}  // namespace cartesian_controllers
