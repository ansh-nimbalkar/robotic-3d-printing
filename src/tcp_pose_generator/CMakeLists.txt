cmake_minimum_required(VERSION 3.8)
project(tcp_pose_generator)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

add_executable(tcp_pose_generator_node src/tcp_pose_generator_node.cpp)

ament_target_dependencies(
  tcp_pose_generator_node
  rclcpp
  geometry_msgs
  tf2
  tf2_geometry_msgs
  nav_msgs
  visualization_msgs
)

install(
  TARGETS tcp_pose_generator_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()