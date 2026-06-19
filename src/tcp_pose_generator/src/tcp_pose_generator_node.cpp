#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

// ROS2 message type for individual poses and arrays of poses.
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/pose_array.hpp"

// Main ROS2 C++ client library.
#include "rclcpp/rclcpp.hpp"

// tf2 is used here for rotation matrix and quaternion conversions.
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

using namespace std::chrono_literals;

namespace
{

constexpr double PI = 3.14159265358979323846;

// Simple 3D vector structure used for internal geometry calculations.
struct Vec3
{
  double x;
  double y;
  double z;
};

// Compute the magnitude/length of a 3D vector.
double norm(const Vec3 & v)
{
  return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

// Return a unit-length version of a vector.
Vec3 normalize(const Vec3 & v)
{
  const double n = norm(v);

  // Avoid division by zero if the vector is extremely small.
  if (n < 1e-9) {
    return {0.0, 0.0, 0.0};
  }

  return {v.x / n, v.y / n, v.z / n};
}

// Compute the cross product of two vectors.
//
// The cross product returns a vector perpendicular to both input vectors.
// This is used to generate the third axis of the TCP frame.
Vec3 cross(const Vec3 & a, const Vec3 & b)
{
  return {
    a.y * b.z - a.z * b.y,
    a.z * b.x - a.x * b.z,
    a.x * b.y - a.y * b.x
  };
}

// Compute the dot product of two vectors.
//
// may be useful later
double dot(const Vec3 & a, const Vec3 & b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

}

class TcpPoseGeneratorNode : public rclcpp::Node
{
public:
  TcpPoseGeneratorNode()
  : Node("tcp_pose_generator_node")
  {
    /*
      Declare ROS2 parameters.

      Can be changed at runtime when launching the node using:

        ros2 run tcp_pose_generator tcp_pose_generator_node --ros-args \
          -p radius:=0.40 \
          -p turns:=2.0 \
          -p samples:=120
    */

    // Frame used for visualisation in RViz.
    // Later this should match the actual UR10e base frame or print-bed frame.
    frame_id_ = this->declare_parameter<std::string>("frame_id", "base_link");

    // Simulated cylinder helix path params
    radius_ = this->declare_parameter<double>("radius", 0.30);
    pitch_per_turn_ = this->declare_parameter<double>("pitch_per_turn", 0.12);
    turns_ = this->declare_parameter<double>("turns", 1.5);
    samples_ = this->declare_parameter<int>("samples", 80);

    /*
      normal_sign controls whether the generated TCP Z-axis follows the
      outward surface normal or the inward surface normal.

      normal_sign =  1.0  → TCP Z-axis points outward from cylinder.
      normal_sign = -1.0  → TCP Z-axis points inward toward cylinder.
    */
    normal_sign_ = this->declare_parameter<double>("normal_sign", 1.0);

    //RViz can subscribe to this topic and display all generated TCP poses.
    pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>(
      "tcp_pose_array",
      10
    );

    /*
      Create a timer that calls publishPoses() every 500 milliseconds.

      This repeatedly republishes the same generated path so that RViz can
      receive and display it easily.
    */
    timer_ = this->create_wall_timer(
      500ms,
      std::bind(&TcpPoseGeneratorNode::publishPoses, this)
    );

    RCLCPP_INFO(this->get_logger(), "TCP pose generator started.");
    RCLCPP_INFO(this->get_logger(), "Publishing PoseArray on /tcp_pose_array");
  }

private:
  void publishPoses()
  {
    /*
      Create the PoseArray message.

      A PoseArray contains:
        - one header
        - a vector/list of geometry_msgs::msg::Pose objects

      Each pose contains:
        - position
        - orientation
    */
    geometry_msgs::msg::PoseArray pose_array;

    // Timestamp the message with the current ROS time.
    pose_array.header.stamp = this->now();

    // Set the coordinate frame in which the poses are expressed.
    // RViz Fixed Frame should match this value.
    pose_array.header.frame_id = frame_id_;

    // Ensure at least two samples are generated.
    const int n_samples = std::max(samples_, 2);

    /*
      Main loop.

      For each sample index i:
        1. Compute the position on the helical path.
        2. Compute the tangent direction along the path.
        3. Compute the surface normal of the cylinder.
        4. Build a local TCP frame.
        5. Convert the frame into a quaternion.
        6. Add the pose to the PoseArray.
    */
    for (int i = 0; i < n_samples; ++i) {
      const double s = static_cast<double>(i) / static_cast<double>(n_samples - 1);

      /*
        theta is the angular position around the cylinder.

        For one full turn:
          theta goes from 0 to 2*pi.

        For multiple turns:
          theta goes from 0 to 2*pi*turns.
      */
      const double theta = 2.0 * PI * turns_ * s;

      /*
        Generate a synthetic path on a cylinder.

        Cylinder equation:
          x = R cos(theta)
          y = R sin(theta)

        Helical path:
          z increases gradually as theta increases.

        This gives a simple 3D curved path with known geometry.
      */
      Vec3 position{
        radius_ * std::cos(theta),
        radius_ * std::sin(theta),
        pitch_per_turn_ * turns_ * s
      };

      /*
        Compute the tangent direction along the helix.

        If:
          p(theta) = [R cos(theta), R sin(theta), pitch * theta / 2pi]

        Then:
          dp/dtheta = [-R sin(theta), R cos(theta), pitch / 2pi]

        This derivative points in the direction of travel along the path.
      */
      Vec3 tangent = normalize({
        -radius_ * std::sin(theta),
        radius_ * std::cos(theta),
        pitch_per_turn_ / (2.0 * PI)
      });

      /*
        Compute the analytical surface normal of the cylinder.

        For a cylinder aligned with the Z-axis, the outward normal is radial:

          normal = [cos(theta), sin(theta), 0]

        This points directly away from the cylinder axis.

        normal_sign_ allows the direction to be flipped.
      */
      Vec3 surface_normal = normalize({
        normal_sign_ * std::cos(theta),
        normal_sign_ * std::sin(theta),
        0.0
      });

      /*
        Define the local TCP frame.

        For this first milestone, we use this convention:

          TCP x-axis = tangent along the path
          TCP z-axis = surface normal
          TCP y-axis = z-axis cross x-axis

        This creates a complete right-handed coordinate frame at each point.

        Later, this convention may need to be adjusted depending on the
        actual nozzle/tool frame used by the UR10e setup.
      */
      Vec3 x_axis = tangent;
      Vec3 z_axis = surface_normal;
      Vec3 y_axis = normalize(cross(z_axis, x_axis));

      /*
        Check for a degenerate frame.

        If tangent and normal are parallel, their cross product is zero.
        That would mean we cannot construct a valid y-axis.

        For the synthetic cylinder helix this should not happen, but it is
        important for future real geometry.
      */
      if (norm(y_axis) < 1e-9) {
        RCLCPP_WARN_THROTTLE(
          this->get_logger(),
          *this->get_clock(),
          2000,
          "Degenerate frame detected. Tangent and normal may be parallel."
        );
        continue;
      }

      /*
        Recompute x_axis using y_axis and z_axis.

        This ensures the final frame is orthonormal:
          - all axes have unit length
          - all axes are perpendicular to each other

        This is useful because small numerical errors can otherwise make the
        rotation matrix slightly invalid.
      */
      x_axis = normalize(cross(y_axis, z_axis));

      /*
        Construct the rotation matrix.

        The columns of the matrix are the TCP basis vectors expressed in the
        parent frame:

          R = [x_axis  y_axis  z_axis]

        Expanded:

          R = [ x.x  y.x  z.x
                x.y  y.y  z.y
                x.z  y.z  z.z ]

        This matrix describes the orientation of the TCP frame.
      */
      tf2::Matrix3x3 rotation_matrix(
        x_axis.x, y_axis.x, z_axis.x,
        x_axis.y, y_axis.y, z_axis.y,
        x_axis.z, y_axis.z, z_axis.z
      );

      /*
        Convert the rotation matrix into a quaternion.

        ROS Pose messages store orientation as a quaternion, not as a
        rotation matrix or Euler angles.
      */
      tf2::Quaternion q;
      rotation_matrix.getRotation(q);

      // Normalise the quaternion to avoid small numerical drift.
      q.normalize();

      /*
        Create one geometry_msgs Pose.

        This pose contains:
          - TCP position
          - TCP orientation
      */
      geometry_msgs::msg::Pose pose;

      // Assign TCP position.
      pose.position.x = position.x;
      pose.position.y = position.y;
      pose.position.z = position.z;

      // Assign TCP orientation after converting tf2 quaternion to ROS message.
      pose.orientation = tf2::toMsg(q);

      // Add this pose to the PoseArray.
      pose_array.poses.push_back(pose);
    }

    /*
      Publish the full array of TCP poses.

      RViz receives this message and displays the generated pose frames.
    */
    pose_pub_->publish(pose_array);
  }

  /*
    Stored ROS2 parameters.
  */

  // Coordinate frame in which all generated poses are expressed.
  std::string frame_id_;

  // Synthetic cylinder radius.
  double radius_;

  // Vertical height gained per full revolution.
  double pitch_per_turn_;

  // Number of turns around the cylinder.
  double turns_;

  // Number of pose samples along the path.
  int samples_;

  // Direction multiplier for surface normal.
  double normal_sign_;

  /*
    ROS2 publisher and timer.
  */

  // Publishes the generated TCP poses.
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pose_pub_;

  // Timer that periodically regenerates and republishes the pose array.
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  /*
    Initialise ROS2.
  */
  rclcpp::init(argc, argv);

  /*
    Create and spin the node.

    spin() keeps the node alive so the timer can repeatedly call publishPoses().
  */
  rclcpp::spin(std::make_shared<TcpPoseGeneratorNode>());

  /*
    Shutdown ROS2 cleanly when the node exits.
  */
  rclcpp::shutdown();

  return 0;
}