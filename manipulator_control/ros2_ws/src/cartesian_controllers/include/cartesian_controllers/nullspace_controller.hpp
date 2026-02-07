/**
 * @file nullspace_controller.hpp
 * @brief Nullspace controller for redundant manipulators (7-DOF)
 * @author Barath Kumar JK
 * @date 2025
 *
 * Implements nullspace optimization for:
 * - Posture control (track preferred configuration)
 * - Joint limit avoidance
 * - Manipulability maximization
 */

#ifndef CARTESIAN_CONTROLLERS_NULLSPACE_CONTROLLER_HPP
#define CARTESIAN_CONTROLLERS_NULLSPACE_CONTROLLER_HPP

#include "cartesian_controllers/jacobian_controller.hpp"
#include <functional>

namespace cartesian_controllers {

/**
 * @brief Nullspace control configuration
 */
struct NullspaceConfig {
    bool enabled = true;
    double posture_gain = 2.0;          // Posture tracking gain
    double joint_limit_gain = 5.0;      // Joint limit avoidance gain
    double manipulability_gain = 0.1;   // Manipulability maximization gain
    double joint_limit_threshold = 0.1; // Distance from limit to activate (rad)
    Eigen::VectorXd preferred_posture;  // Preferred joint configuration
    Eigen::VectorXd joint_lower_limits;
    Eigen::VectorXd joint_upper_limits;
    
    NullspaceConfig() = default;
    
    void setFrankaDefaults() {
        preferred_posture.resize(7);
        preferred_posture << 0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785;
        
        joint_lower_limits.resize(7);
        joint_lower_limits << -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973;
        
        joint_upper_limits.resize(7);
        joint_upper_limits << 2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973;
    }
};

/**
 * @brief Nullspace objective function type
 */
using NullspaceObjective = std::function<Eigen::VectorXd(
    const Eigen::VectorXd& joint_positions,
    const NullspaceConfig& config
)>;

/**
 * @brief Nullspace Controller for Redundant Manipulators
 *
 * Extends JacobianController with nullspace optimization:
 * q̇ = J†·ẋ + (I - J†J)·q̇_null
 *
 * Where q̇_null combines multiple objectives:
 * q̇_null = k_posture·(q_desired - q) + k_avoid·∇h(q) + k_manip·∇w(q)
 */
class NullspaceController : public JacobianController {
public:
    using JacobianMatrix = Eigen::Matrix<double, 6, Eigen::Dynamic>;
    
    explicit NullspaceController(
        const JacobianControllerConfig& jac_config = JacobianControllerConfig(),
        const NullspaceConfig& null_config = NullspaceConfig()
    );
    
    /**
     * @brief Update with nullspace optimization
     * @param joint_positions Current joint positions
     * @param jacobian Current Jacobian matrix
     * @param current_pose Current end-effector pose
     * @return Joint velocity commands with nullspace optimization
     */
    Eigen::VectorXd updateWithNullspace(
        const Eigen::VectorXd& joint_positions,
        const JacobianMatrix& jacobian,
        const CartesianPose& current_pose
    );
    
    /**
     * @brief Set preferred posture for nullspace control
     */
    void setPreferredPosture(const Eigen::VectorXd& posture);
    
    /**
     * @brief Set joint limits
     */
    void setJointLimits(
        const Eigen::VectorXd& lower,
        const Eigen::VectorXd& upper
    );
    
    /**
     * @brief Enable/disable nullspace control
     */
    void setNullspaceEnabled(bool enabled) { null_config_.enabled = enabled; }
    
    /**
     * @brief Get nullspace configuration
     */
    const NullspaceConfig& getNullspaceConfig() const { return null_config_; }
    
    /**
     * @brief Set nullspace configuration
     */
    void setNullspaceConfig(const NullspaceConfig& config) { null_config_ = config; }
    
    /**
     * @brief Add custom nullspace objective
     */
    void addNullspaceObjective(
        const std::string& name,
        NullspaceObjective objective,
        double gain
    );
    
    /**
     * @brief Remove custom nullspace objective
     */
    void removeNullspaceObjective(const std::string& name);
    
    /**
     * @brief Get nullspace projector (I - J†J)
     */
    Eigen::MatrixXd computeNullspaceProjector(
        const JacobianMatrix& J,
        const Eigen::MatrixXd& J_pinv
    ) const;
    
    /**
     * @brief Compute posture control velocity
     * q̇_posture = k·(q_desired - q)
     */
    Eigen::VectorXd computePostureVelocity(
        const Eigen::VectorXd& joint_positions
    ) const;
    
    /**
     * @brief Compute joint limit avoidance gradient
     * ∇h(q) pushes joints away from limits
     */
    Eigen::VectorXd computeJointLimitGradient(
        const Eigen::VectorXd& joint_positions
    ) const;
    
    /**
     * @brief Compute manipulability gradient
     * ∇w(q) for manipulability maximization
     */
    Eigen::VectorXd computeManipulabilityGradient(
        const Eigen::VectorXd& joint_positions,
        const JacobianMatrix& jacobian,
        double delta = 1e-4
    ) const;
    
    /**
     * @brief Get current nullspace velocity
     */
    const Eigen::VectorXd& getNullspaceVelocity() const { return nullspace_velocity_; }

protected:
    NullspaceConfig null_config_;
    Eigen::VectorXd nullspace_velocity_;
    
    // Custom objectives
    struct CustomObjective {
        NullspaceObjective function;
        double gain;
    };
    std::unordered_map<std::string, CustomObjective> custom_objectives_;
    
    /**
     * @brief Combine all nullspace objectives
     */
    Eigen::VectorXd computeCombinedNullspaceVelocity(
        const Eigen::VectorXd& joint_positions,
        const JacobianMatrix& jacobian
    );
};

/**
 * @brief Weighted Nullspace Controller
 * 
 * Uses weighted pseudoinverse for task prioritization:
 * J†_w = W⁻¹Jᵀ(JW⁻¹Jᵀ)⁻¹
 */
class WeightedNullspaceController : public NullspaceController {
public:
    explicit WeightedNullspaceController(
        const JacobianControllerConfig& jac_config = JacobianControllerConfig(),
        const NullspaceConfig& null_config = NullspaceConfig()
    );
    
    /**
     * @brief Set joint weight matrix
     * Higher weights = less motion for that joint
     */
    void setJointWeights(const Eigen::VectorXd& weights);
    
    /**
     * @brief Compute weighted pseudoinverse
     */
    Eigen::MatrixXd computeWeightedPseudoinverse(
        const JacobianMatrix& J
    ) const;

private:
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> W_;
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> W_inv_;
};

}  // namespace cartesian_controllers

#endif  // CARTESIAN_CONTROLLERS_NULLSPACE_CONTROLLER_HPP
