#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Vector3.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <string>
#include <vector>
#include <chrono>

struct AtomWaypoint
{
  double x;
  double y;
  double z;

  double nx;
  double ny;
  double nz;

  double extrusion;
};

class AtomSlicerPathNode : public rclcpp::Node
{
public:
  AtomSlicerPathNode()
  : Node("atomslicer_path_node")
  {
    declare_parameter<std::string>("gcode_file", "");
    declare_parameter<int>("path_stride", 1);
    declare_parameter<int>("normal_stride", 500);
    declare_parameter<int>("pose_stride", 200);
    declare_parameter<double>("visualization_scale", 1.0);

    gcode_file_ = get_parameter("gcode_file").as_string();
    path_stride_ = get_parameter("path_stride").as_int();
    normal_stride_ = get_parameter("normal_stride").as_int();
    pose_stride_ = get_parameter("pose_stride").as_int();
    visualization_scale_ =
      get_parameter("visualization_scale").as_double();

    if (gcode_file_.empty()) {
      RCLCPP_ERROR(get_logger(), "No G-code file provided.");
      return;
    }

    if (!loadGcode(gcode_file_)) {
      return;
    }

    calculateVisualizationCenter();

    auto qos = rclcpp::QoS(10).reliable();

    path_pub_ =
      create_publisher<visualization_msgs::msg::Marker>(
        "/atomslicer/path", qos);

    normals_pub_ =
      create_publisher<visualization_msgs::msg::Marker>(
        "/atomslicer/normals", qos);

    poses_pub_ =
      create_publisher<geometry_msgs::msg::PoseArray>(
        "/atomslicer/poses", qos);

    tcp_poses_pub_ =
      create_publisher<geometry_msgs::msg::PoseArray>(
        "/atomslicer/tcp_poses", qos);

    // Give RViz/publishers a short time to initialise.
    timer_ = create_wall_timer(
      std::chrono::seconds(1),
      [this]() {
        publishPath();
        publishNormals();
        publishPoses();
      });
  }

private:
  bool parseWaypoint(
    const std::string & line,
    AtomWaypoint & waypoint)
  {
    // Only actual AtomSlicer waypoints have a normal comment.
    const std::size_t comment_pos = line.find("; (");

    if (line.rfind("G1 ", 0) != 0 ||
        comment_pos == std::string::npos)
    {
      return false;
    }

    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double e = 0.0;

    const int position_fields = std::sscanf(
      line.c_str(),
      "G1 X%lf Y%lf Z%lf E%lf",
      &x,
      &y,
      &z,
      &e);

    if (position_fields < 3) {
      return false;
    }

    double nx = 0.0;
    double ny = 0.0;
    double nz = 0.0;

    const int normal_fields = std::sscanf(
      line.c_str() + comment_pos,
      "; (%lf, %lf, %lf)",
      &nx,
      &ny,
      &nz);

    if (normal_fields != 3) {
      return false;
    }

    // AtomSlicer / G-code uses millimetres.
    // ROS uses metres.
    waypoint.x = x * 0.001;
    waypoint.y = y * 0.001;
    waypoint.z = z * 0.001;

    waypoint.nx = nx;
    waypoint.ny = ny;
    waypoint.nz = nz;

    waypoint.extrusion =
      position_fields >= 4 ? e : 0.0;

    return true;
  }

  bool loadGcode(const std::string & filename)
  {
    std::ifstream file(filename);

    if (!file.is_open()) {
      RCLCPP_ERROR(
        get_logger(),
        "Could not open G-code file: %s",
        filename.c_str());

      return false;
    }

    std::string line;

    while (std::getline(file, line)) {
      AtomWaypoint waypoint;

      if (parseWaypoint(line, waypoint)) {
        waypoints_.push_back(waypoint);
      }
    }

    if (waypoints_.empty()) {
      RCLCPP_ERROR(
        get_logger(),
        "No AtomSlicer waypoints were found.");

      return false;
    }

    printStatistics();

    return true;
  }

  void calculateVisualizationCenter()
  {
    double min_x = std::numeric_limits<double>::max();
    double min_y = std::numeric_limits<double>::max();
    double min_z = std::numeric_limits<double>::max();

    double max_x = std::numeric_limits<double>::lowest();
    double max_y = std::numeric_limits<double>::lowest();
    double max_z = std::numeric_limits<double>::lowest();

    for (const auto & wp : waypoints_) {
      min_x = std::min(min_x, wp.x);
      min_y = std::min(min_y, wp.y);
      min_z = std::min(min_z, wp.z);

      max_x = std::max(max_x, wp.x);
      max_y = std::max(max_y, wp.y);
      max_z = std::max(max_z, wp.z);
    }

    visualization_center_x_ = (min_x + max_x) / 2.0;
    visualization_center_y_ = (min_y + max_y) / 2.0;
    visualization_center_z_ = (min_z + max_z) / 2.0;

    // Define print_frame at the bottom-centre of the sliced part.
    // Its axes remain aligned with the AtomSlicer/G-code axes.
    print_origin_x_ = (min_x + max_x) / 2.0;
    print_origin_y_ = (min_y + max_y) / 2.0;
    print_origin_z_ = min_z;
  }

  double scaleForVisualization(
    double value,
    double center) const
  {
    return center +
      (value - center) * visualization_scale_;
  }

  void printStatistics()
  {
    double min_x = std::numeric_limits<double>::max();
    double min_y = std::numeric_limits<double>::max();
    double min_z = std::numeric_limits<double>::max();

    double max_x = std::numeric_limits<double>::lowest();
    double max_y = std::numeric_limits<double>::lowest();
    double max_z = std::numeric_limits<double>::lowest();

    double min_normal_mag =
      std::numeric_limits<double>::max();

    double max_normal_mag = 0.0;

    std::size_t non_vertical_normals = 0;

    for (const auto & wp : waypoints_) {
      min_x = std::min(min_x, wp.x);
      min_y = std::min(min_y, wp.y);
      min_z = std::min(min_z, wp.z);

      max_x = std::max(max_x, wp.x);
      max_y = std::max(max_y, wp.y);
      max_z = std::max(max_z, wp.z);

      const double magnitude =
        std::sqrt(
          wp.nx * wp.nx +
          wp.ny * wp.ny +
          wp.nz * wp.nz);

      min_normal_mag =
        std::min(min_normal_mag, magnitude);

      max_normal_mag =
        std::max(max_normal_mag, magnitude);

      if (std::abs(wp.nx) > 1e-4 ||
          std::abs(wp.ny) > 1e-4 ||
          std::abs(wp.nz - 1.0) > 1e-4)
      {
        ++non_vertical_normals;
      }
    }

    const auto & first = waypoints_.front();
    const auto & last = waypoints_.back();

    RCLCPP_INFO(
      get_logger(),
      "Loaded %zu AtomSlicer waypoints",
      waypoints_.size());

    RCLCPP_INFO(
      get_logger(),
      "Bounds X: %.4f -> %.4f m",
      min_x, max_x);

    RCLCPP_INFO(
      get_logger(),
      "Bounds Y: %.4f -> %.4f m",
      min_y, max_y);

    RCLCPP_INFO(
      get_logger(),
      "Bounds Z: %.4f -> %.4f m",
      min_z, max_z);

    RCLCPP_INFO(
      get_logger(),
      "Normal magnitude: %.6f -> %.6f",
      min_normal_mag,
      max_normal_mag);

    RCLCPP_INFO(
      get_logger(),
      "Non-vertical normals: %zu",
      non_vertical_normals);

    RCLCPP_INFO(
      get_logger(),
      "First waypoint: P=(%.4f, %.4f, %.4f), "
      "N=(%.4f, %.4f, %.4f)",
      first.x,
      first.y,
      first.z,
      first.nx,
      first.ny,
      first.nz);

    RCLCPP_INFO(
      get_logger(),
      "Last waypoint: P=(%.4f, %.4f, %.4f), "
      "N=(%.4f, %.4f, %.4f)",
      last.x,
      last.y,
      last.z,
      last.nx,
      last.ny,
      last.nz);
  }

  void publishPath()
  {
    visualization_msgs::msg::Marker marker;

    marker.header.frame_id = "print_frame";
    marker.header.stamp = now();

    marker.ns = "atomslicer_path";
    marker.id = 0;

    marker.type =
      visualization_msgs::msg::Marker::LINE_LIST;

    marker.action =
      visualization_msgs::msg::Marker::ADD;

    marker.pose.orientation.w = 1.0;

    marker.scale.x = 0.00015;

    marker.color.r = 0.1;
    marker.color.g = 0.8;
    marker.color.b = 1.0;
    marker.color.a = 1.0;

    const std::size_t stride =
      std::max(1, path_stride_);

    for (std::size_t i = 1;
         i < waypoints_.size();
         ++i)
    {
      // In AtomSlicer G-code, positive extrusion means this move
      // is a deposition/printing move.
      if (waypoints_[i].extrusion <= 0.0) {
        continue;
      }

      geometry_msgs::msg::Point start;
      geometry_msgs::msg::Point end;

      // Visualization-only path in print_frame.
      // Convert to print-local coordinates first, then scale for RViz.
      start.x =
        (waypoints_[i - 1].x - print_origin_x_) *
        visualization_scale_;

      start.y =
        (waypoints_[i - 1].y - print_origin_y_) *
        visualization_scale_;

      start.z =
        (waypoints_[i - 1].z - print_origin_z_) *
        visualization_scale_;

      end.x =
        (waypoints_[i].x - print_origin_x_) *
        visualization_scale_;

      end.y =
        (waypoints_[i].y - print_origin_y_) *
        visualization_scale_;

      end.z =
        (waypoints_[i].z - print_origin_z_) *
        visualization_scale_;

      marker.points.push_back(start);
      marker.points.push_back(end);
    }

    path_pub_->publish(marker);
  }

  void publishNormals()
  {
    visualization_msgs::msg::Marker marker;

    marker.header.frame_id = "print_frame";
    marker.header.stamp = now();

    marker.ns = "atomslicer_normals";
    marker.id = 0;

    marker.type =
      visualization_msgs::msg::Marker::LINE_LIST;

    marker.action =
      visualization_msgs::msg::Marker::ADD;

    marker.pose.orientation.w = 1.0;

    marker.scale.x = 0.0001;

    marker.color.r = 1.0;
    marker.color.g = 0.3;
    marker.color.b = 0.1;
    marker.color.a = 1.0;

    constexpr double normal_length = 0.005;

    const std::size_t stride =
      std::max(1, normal_stride_);

    for (std::size_t i = 0;
         i < waypoints_.size();
         i += stride)
    {
      const auto & wp = waypoints_[i];

      const double magnitude =
        std::sqrt(
          wp.nx * wp.nx +
          wp.ny * wp.ny +
          wp.nz * wp.nz);

      if (magnitude < 1e-9) {
        continue;
      }

      const double nx = wp.nx / magnitude;
      const double ny = wp.ny / magnitude;
      const double nz = wp.nz / magnitude;

      geometry_msgs::msg::Point start;
      geometry_msgs::msg::Point end;

      // Visualization-only normal origin in print_frame.
      // Convert to print-local coordinates first, then scale for RViz.
      start.x =
        (wp.x - print_origin_x_) *
        visualization_scale_;

      start.y =
        (wp.y - print_origin_y_) *
        visualization_scale_;

      start.z =
        (wp.z - print_origin_z_) *
        visualization_scale_;

      // Scale the normal length by the same visualization factor
      // so that the normals remain proportional to the geometry.
      end.x =
        start.x +
        normal_length * visualization_scale_ * nx;

      end.y =
        start.y +
        normal_length * visualization_scale_ * ny;

      end.z =
        start.z +
        normal_length * visualization_scale_ * nz;

      marker.points.push_back(start);
      marker.points.push_back(end);
    }

    normals_pub_->publish(marker);
  }

  void publishPoses()
  {
    geometry_msgs::msg::PoseArray pose_array;
    geometry_msgs::msg::PoseArray tcp_pose_array;

    pose_array.header.frame_id = "print_frame";
    pose_array.header.stamp = now();

    tcp_pose_array.header.frame_id = "print_frame";
    tcp_pose_array.header.stamp = pose_array.header.stamp;

    const std::size_t stride =
      std::max(1, pose_stride_);

    std::size_t valid_pose_count = 0;

    // -----------------------------------------------------------------
    // Rotation-minimising / minimum-twist frame state.
    //
    // The first valid deposition pose uses the projected deposition
    // tangent to establish TCP +X.  Every subsequent pose transports
    // the previous frame with the smallest possible rotation that maps
    // the previous surface normal onto the new one.
    //
    // Therefore:
    //   +Z remains exactly aligned with the AtomSlicer surface normal.
    //   +X/+Y do not follow instantaneous tangent fluctuations.
    //   unnecessary roll about +Z is removed.
    //
    // This is intentionally the standard orientation behaviour rather
    // than a selectable mode because the robot should not be commanded
    // to perform avoidable tool-axis rotation during printing.
    // -----------------------------------------------------------------

    bool frame_initialised = false;

    tf2::Vector3 previous_x_axis(1.0, 0.0, 0.0);
    tf2::Vector3 previous_z_axis(0.0, 0.0, 1.0);

    tf2::Quaternion previous_quaternion(
      0.0, 0.0, 0.0, 1.0);

    constexpr double epsilon = 1e-12;

    for (std::size_t i = 1;
         i < waypoints_.size();
         ++i)
    {
      const auto & previous = waypoints_[i - 1];
      const auto & current = waypoints_[i];

      // Positive extrusion means the move ending at the current
      // waypoint is an actual deposition move.
      if (current.extrusion <= 0.0) {
        continue;
      }

      // ---------------------------------------------------------------
      // TCP +Z axis: normalised AtomSlicer surface/tool normal.
      // ---------------------------------------------------------------

      tf2::Vector3 z_axis(
        current.nx,
        current.ny,
        current.nz);

      if (z_axis.length2() < epsilon) {
        // A valid surface-aligned TCP cannot be constructed without a
        // valid normal.  Skip only this malformed waypoint.
        continue;
      }

      z_axis.normalize();

      // ---------------------------------------------------------------
      // Local deposition tangent.
      //
      // This is used to initialise the roll of the first valid TCP and
      // as a robust fallback only.  It is deliberately NOT imposed at
      // every waypoint because doing so caused the large wrist_3/TCP
      // roll discontinuities identified by the continuity diagnostics.
      // ---------------------------------------------------------------

      tf2::Vector3 tangent(
        current.x - previous.x,
        current.y - previous.y,
        current.z - previous.z);

      bool tangent_valid = tangent.length2() >= epsilon;

      if (tangent_valid) {
        tangent.normalize();

        // Project the tangent into the plane perpendicular to +Z.
        tangent -= z_axis * tangent.dot(z_axis);

        tangent_valid = tangent.length2() >= epsilon;

        if (tangent_valid) {
          tangent.normalize();
        }
      }

      tf2::Vector3 x_axis;

      if (!frame_initialised) {
        // -------------------------------------------------------------
        // First valid deposition pose:
        // establish roll from the deposition direction.
        // -------------------------------------------------------------

        if (tangent_valid) {
          x_axis = tangent;
        } else {
          // Extremely rare geometric degeneracy.  Choose a stable
          // world-axis reference least aligned with the tool normal,
          // then project it onto the tangent plane.  This avoids
          // dropping an otherwise valid TCP pose.
          tf2::Vector3 reference_axis =
            std::abs(z_axis.x()) < 0.9 ?
            tf2::Vector3(1.0, 0.0, 0.0) :
            tf2::Vector3(0.0, 1.0, 0.0);

          x_axis =
            reference_axis -
            z_axis * reference_axis.dot(z_axis);

          if (x_axis.length2() < epsilon) {
            continue;
          }

          x_axis.normalize();
        }

        frame_initialised = true;
      } else {
        // -------------------------------------------------------------
        // Minimum-twist transport.
        //
        // Find the smallest 3-D rotation that maps previous +Z onto
        // current +Z.  Applying that same rotation to previous +X
        // parallel-transports the frame without adding any rotation
        // around the tool axis.
        // -------------------------------------------------------------

        const double normal_dot =
          std::clamp(
            previous_z_axis.dot(z_axis),
            -1.0,
            1.0);

        tf2::Quaternion transport_quaternion;

        if (normal_dot > -1.0 + 1e-10) {
          // For two unit vectors a and b, quaternion
          // [a x b, 1 + a.b], after normalisation, represents the
          // minimum rotation from a to b.
          const tf2::Vector3 cross =
            previous_z_axis.cross(z_axis);

          transport_quaternion =
            tf2::Quaternion(
              cross.x(),
              cross.y(),
              cross.z(),
              1.0 + normal_dot);

          if (transport_quaternion.length2() < epsilon) {
            transport_quaternion =
              tf2::Quaternion(0.0, 0.0, 0.0, 1.0);
          } else {
            transport_quaternion.normalize();
          }
        } else {
          // Opposite normals are a mathematically ambiguous 180-degree
          // case.  Rotating around the previous +X axis preserves roll
          // as well as possible and still maps +Z to -Z.
          transport_quaternion.setRotation(
            previous_x_axis,
            M_PI);

          transport_quaternion.normalize();
        }

        const tf2::Matrix3x3 transport_rotation(
          transport_quaternion);

        x_axis =
          transport_rotation * previous_x_axis;

        // Remove tiny numerical components parallel to +Z.
        x_axis -=
          z_axis * x_axis.dot(z_axis);

        if (x_axis.length2() < epsilon) {
          // Robust fallback: use the local deposition tangent if the
          // transported axis became degenerate.
          if (!tangent_valid) {
            continue;
          }

          x_axis = tangent;

          // The tangent has two equivalent signs.  Prefer the sign
          // closest to the previously transported frame to avoid an
          // artificial 180-degree roll flip.
          if (x_axis.dot(previous_x_axis) < 0.0) {
            x_axis = -x_axis;
          }
        }

        x_axis.normalize();
      }

      // ---------------------------------------------------------------
      // Complete a right-handed orthonormal TCP frame.
      //
      // Y = Z x X
      // X = Y x Z  (second pass removes accumulated numerical drift)
      // ---------------------------------------------------------------

      tf2::Vector3 y_axis =
        z_axis.cross(x_axis);

      if (y_axis.length2() < epsilon) {
        continue;
      }

      y_axis.normalize();

      x_axis =
        y_axis.cross(z_axis);

      if (x_axis.length2() < epsilon) {
        continue;
      }

      x_axis.normalize();

      tf2::Matrix3x3 rotation(
        x_axis.x(), y_axis.x(), z_axis.x(),
        x_axis.y(), y_axis.y(), z_axis.y(),
        x_axis.z(), y_axis.z(), z_axis.z());

      tf2::Quaternion quaternion;
      rotation.getRotation(quaternion);
      quaternion.normalize();

      // Quaternions q and -q represent the same physical rotation.
      // Keep a consistent quaternion hemisphere so downstream
      // interpolation/logging does not see artificial sign changes.
      if (valid_pose_count > 0 &&
          quaternion.dot(previous_quaternion) < 0.0)
      {
        quaternion =
          tf2::Quaternion(
            -quaternion.x(),
            -quaternion.y(),
            -quaternion.z(),
            -quaternion.w());
      }

      // ---------------------------------------------------------------
      // Real-scale TCP pose.
      //
      // IMPORTANT: every valid deposition pose is published here.
      // pose_stride must never downsample the robot-planning input.
      // ---------------------------------------------------------------

      geometry_msgs::msg::Pose tcp_pose;

      tcp_pose.position.x =
        current.x - print_origin_x_;

      tcp_pose.position.y =
        current.y - print_origin_y_;

      tcp_pose.position.z =
        current.z - print_origin_z_;

      tcp_pose.orientation.x = quaternion.x();
      tcp_pose.orientation.y = quaternion.y();
      tcp_pose.orientation.z = quaternion.z();
      tcp_pose.orientation.w = quaternion.w();

      tcp_pose_array.poses.push_back(tcp_pose);

      // Save the completed frame for the next minimum-twist transport.
      previous_x_axis = x_axis;
      previous_z_axis = z_axis;
      previous_quaternion = quaternion;

      // ---------------------------------------------------------------
      // Visualization PoseArray only.
      //
      // Downsample AFTER the real TCP pose has been stored.
      // ---------------------------------------------------------------

      if ((valid_pose_count % stride) == 0) {
        geometry_msgs::msg::Pose pose;

        pose.position.x =
          (current.x - print_origin_x_) *
          visualization_scale_;

        pose.position.y =
          (current.y - print_origin_y_) *
          visualization_scale_;

        pose.position.z =
          (current.z - print_origin_z_) *
          visualization_scale_;

        pose.orientation = tcp_pose.orientation;

        pose_array.poses.push_back(pose);
      }

      ++valid_pose_count;
    }

    poses_pub_->publish(pose_array);
    tcp_poses_pub_->publish(tcp_pose_array);
  }

  std::string gcode_file_;

  int path_stride_;
  int normal_stride_;
  int pose_stride_;

  double visualization_scale_;

  double visualization_center_x_;
  double visualization_center_y_;
  double visualization_center_z_;

  double print_origin_x_;
  double print_origin_y_;
  double print_origin_z_;

  std::vector<AtomWaypoint> waypoints_;

  rclcpp::Publisher<
    visualization_msgs::msg::Marker>::SharedPtr
    path_pub_;

  rclcpp::Publisher<
    visualization_msgs::msg::Marker>::SharedPtr
    normals_pub_;

  rclcpp::Publisher<
    geometry_msgs::msg::PoseArray>::SharedPtr
    poses_pub_;

  rclcpp::Publisher<
    geometry_msgs::msg::PoseArray>::SharedPtr
    tcp_poses_pub_;

  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  rclcpp::spin(
    std::make_shared<AtomSlicerPathNode>());

  rclcpp::shutdown();

  return 0;
}