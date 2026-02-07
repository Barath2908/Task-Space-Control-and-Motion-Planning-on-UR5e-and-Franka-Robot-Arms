cmake_minimum_required(VERSION 3.8)
project(cartesian_controllers)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(Eigen3 REQUIRED)

# Include directories
include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
)

# Cartesian controller library
add_library(cartesian_controllers_lib SHARED
  src/jacobian_controller.cpp
  src/nullspace_controller.cpp
  src/ik_solver.cpp
)
target_include_directories(cartesian_controllers_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(cartesian_controllers_lib
  Eigen3::Eigen
)
ament_target_dependencies(cartesian_controllers_lib
  rclcpp
  geometry_msgs
  sensor_msgs
  std_msgs
)

# Cartesian controller node
add_executable(cartesian_controller_node
  src/cartesian_controller_node.cpp
)
target_link_libraries(cartesian_controller_node
  cartesian_controllers_lib
)
ament_target_dependencies(cartesian_controller_node
  rclcpp
  geometry_msgs
  sensor_msgs
  std_msgs
  trajectory_msgs
  tf2
  tf2_ros
)

# Install targets
install(TARGETS
  cartesian_controllers_lib
  cartesian_controller_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# Install headers
install(DIRECTORY include/
  DESTINATION include
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install config files
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# Install Python scripts
install(PROGRAMS
  scripts/cartesian_controller.py
  scripts/trajectory_tracker.py
  DESTINATION lib/${PROJECT_NAME}
)

# Export
ament_export_include_directories(include)
ament_export_libraries(cartesian_controllers_lib)
ament_export_dependencies(
  rclcpp
  geometry_msgs
  sensor_msgs
  std_msgs
  Eigen3
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
