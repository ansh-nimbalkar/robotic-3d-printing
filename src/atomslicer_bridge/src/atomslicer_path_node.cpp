#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>

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

      // -------------------------------------------------------
      // TCP Z axis:
      // AtomSlicer tool normal.
      // -------------------------------------------------------

      double zx = current.nx;
      double zy = current.ny;
      double zz = current.nz;

      const double normal_magnitude =
        std::sqrt(
          zx * zx +
          zy * zy +
          zz * zz);

      if (normal_magnitude < 1e-9) {
        continue;
      }

      zx /= normal_magnitude;
      zy /= normal_magnitude;
      zz /= normal_magnitude;

      // -------------------------------------------------------
      // Local deposition tangent.
      //
      // The extrusion value at the current waypoint describes
      // the move from previous -> current.
      // -------------------------------------------------------

      double dx = current.x - previous.x;
      double dy = current.y - previous.y;
      double dz = current.z - previous.z;

      const double tangent_magnitude =
        std::sqrt(
          dx * dx +
          dy * dy +
          dz * dz);

      if (tangent_magnitude < 1e-9) {
        continue;
      }

      dx /= tangent_magnitude;
      dy /= tangent_magnitude;
      dz /= tangent_magnitude;

      // -------------------------------------------------------
      // TCP X axis:
      // Project the path tangent onto the plane perpendicular
      // to the tool normal.
      // -------------------------------------------------------

      const double tangent_dot_normal =
        dx * zx +
        dy * zy +
        dz * zz;

      double xx =
        dx - tangent_dot_normal * zx;

      double xy =
        dy - tangent_dot_normal * zy;

      double xz =
        dz - tangent_dot_normal * zz;

      const double x_magnitude =
        std::sqrt(
          xx * xx +
          xy * xy +
          xz * xz);

      // Degenerate case where the trajectory tangent is
      // effectively parallel to the normal.
      if (x_magnitude < 1e-9) {
        continue;
      }

      xx /= x_magnitude;
      xy /= x_magnitude;
      xz /= x_magnitude;

      // -------------------------------------------------------
      // TCP Y axis = Z x X.
      // -------------------------------------------------------

      double yx =
        zy * xz - zz * xy;

      double yy =
        zz * xx - zx * xz;

      double yz =
        zx * xy - zy * xx;

      const double y_magnitude =
        std::sqrt(
          yx * yx +
          yy * yy +
          yz * yz);

      if (y_magnitude < 1e-9) {
        continue;
      }

      yx /= y_magnitude;
      yy /= y_magnitude;
      yz /= y_magnitude;

      // -------------------------------------------------------
      // At this point:
      //
      // X = local deposition direction
      // Y = Z x X
      // Z = AtomSlicer tool normal
      //
      // These axes form the complete TCP orientation.
      // -------------------------------------------------------

      tf2::Matrix3x3 rotation(
        xx, yx, zx,
        xy, yy, zy,
        xz, yz, zz);

      tf2::Quaternion quaternion;
      rotation.getRotation(quaternion);
      quaternion.normalize();

      // -------------------------------------------------------
      // Real robot-side TCP pose.
      //
      // IMPORTANT: This path must always contain every valid
      // deposition pose. It must never depend on pose_stride or
      // visualization_scale.
      // -------------------------------------------------------

      geometry_msgs::msg::Pose tcp_pose;

      tcp_pose.position.x = current.x - print_origin_x_;
      tcp_pose.position.y = current.y - print_origin_y_;
      tcp_pose.position.z = current.z - print_origin_z_;

      tcp_pose.orientation.x = quaternion.x();
      tcp_pose.orientation.y = quaternion.y();
      tcp_pose.orientation.z = quaternion.z();
      tcp_pose.orientation.w = quaternion.w();

      tcp_pose_array.poses.push_back(tcp_pose);

      // -------------------------------------------------------
      // RViz-only PoseArray.
      //
      // Downsample only this visualization representation. The
      // real /atomslicer/tcp_poses output above remains full
      // resolution.
      // -------------------------------------------------------

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