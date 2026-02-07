# Controller state message for Cartesian controllers

std_msgs/Header header

# Current end-effector pose
geometry_msgs/Pose current_pose

# Target end-effector pose
geometry_msgs/Pose target_pose

# Position error (meters)
float64 position_error

# Orientation error (radians)
float64 orientation_error

# Manipulability measure
float64 manipulability

# Condition number
float64 condition_number

# Joint positions (radians)
float64[] joint_positions

# Joint velocities (rad/s)
float64[] joint_velocities

# Joint velocity commands (rad/s)
float64[] joint_velocity_commands

# Flags
bool at_target
bool near_singularity
bool rate_limited
