#include <chrono>
#include <cmath>
#include <cctype>
#include <exception>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "visualization_msgs/msg/marker.hpp"

using namespace std::chrono_literals;

namespace
{

/*
  3D vector type.

  ROS messages are used for publishing. only used inside the
  node to make the geometry calculations easier to read.
*/
struct Vec3
{
  double x;
  double y;
  double z;
};

/*
  One G-code word.

  Example:
    G1 X10.0 Y5.0 E0.25
  becomes:
    {G, 1.0}, {X, 10.0}, {Y, 5.0}, {E, 0.25}
*/
struct GcodeWord
{
  char letter;
  double value;
};

/*
  One point on the planar extrusion path.

  position:
    TCP position in metres.

  segment_id:
    Consecutive extrusion moves share the same segment_id.
    This prevents drawing false lines
*/
struct ToolpathPoint
{
  Vec3 position;
  int segment_id;
};

double norm(const Vec3 & v)
{
  return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

Vec3 normalize(const Vec3 & v)
{
  const double length = norm(v);

  if (length < 1e-9) {
    return {0.0, 0.0, 0.0};
  }

  return {v.x / length, v.y / length, v.z / length};
}

Vec3 subtract(const Vec3 & a, const Vec3 & b)
{
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 cross(const Vec3 & a, const Vec3 & b)
{
  return {
    a.y * b.z - a.z * b.y,
    a.z * b.x - a.x * b.z,
    a.x * b.y - a.y * b.x
  };
}

/*
  Remove anything after ';'.
*/
std::string stripGcodeComment(const std::string & line)
{
  const std::size_t comment_position = line.find(';');

  if (comment_position == std::string::npos) {
    return line;
  }

  return line.substr(0, comment_position);
}

/*
  Parse a single G-code line into letter/value pairs.
*/
std::vector<GcodeWord> parseGcodeWords(const std::string & raw_line)
{
  std::vector<GcodeWord> words;
  const std::string line = stripGcodeComment(raw_line);

  std::size_t i = 0;

  while (i < line.size()) {
    while (i < line.size() && std::isspace(static_cast<unsigned char>(line[i]))) {
      ++i;
    }

    if (i >= line.size()) {
      break;
    }

    if (!std::isalpha(static_cast<unsigned char>(line[i]))) {
      ++i;
      continue;
    }

    const char letter = static_cast<char>(
      std::toupper(static_cast<unsigned char>(line[i]))
    );

    ++i;

    const std::size_t number_start = i;

    while (i < line.size() && !std::isalpha(static_cast<unsigned char>(line[i]))) {
      ++i;
    }

    const std::string number_string = line.substr(number_start, i - number_start);

    try {
      const double value = std::stod(number_string);
      words.push_back({letter, value});
    } catch (const std::exception &) {
      // Ignore bad words
    }
  }

  return words;
}

bool getWordValue(const std::vector<GcodeWord> & words, char letter, double & value)
{
  const char target = static_cast<char>(
    std::toupper(static_cast<unsigned char>(letter))
  );

  for (const auto & word : words) {
    if (word.letter == target) {
      value = word.value;
      return true;
    }
  }

  return false;
}

bool hasWord(const std::vector<GcodeWord> & words, char letter)
{
  double unused = 0.0;
  return getWordValue(words, letter, unused);
}

geometry_msgs::msg::Point toMsgPoint(const Vec3 & v, double scale)
{
  geometry_msgs::msg::Point point;
  point.x = v.x * scale;
  point.y = v.y * scale;
  point.z = v.z * scale;
  return point;
}

}  // namespace

class TcpPoseGeneratorNode : public rclcpp::Node
{
public:
  TcpPoseGeneratorNode()
  : Node("tcp_pose_generator_node")
  {
    /*
      Parameters.

      Example run command:

        ros2 run tcp_pose_generator tcp_pose_generator_node --ros-args \
          -p gcode_file:=/absolute/path/to/file.gcode \
          -p frame_id:=base_link \
          -p visualization_scale:=10.0
    */
    frame_id_ = this->declare_parameter<std::string>("frame_id", "base_link");
    gcode_file_ = this->declare_parameter<std::string>("gcode_file", "");

    /*
      A G1 move is treated as a print move only if extrusion increases by more
      than this threshold. This filters out travel moves and tiny numerical
      changes in E.
    */
    min_extrusion_delta_ = this->declare_parameter<double>("min_extrusion_delta", 1e-7);

    /*
      This scale only enlarges the displayed path and TCP poses in RViz for visuals.
    */
    visualization_scale_ = this->declare_parameter<double>("visualization_scale", 1.0);

    /*
      Publishers.

      /tcp_pose_array:
        Full list of TCP poses. RViz can display this as pose axes.

      /tcp_path:
        Same poses represented as a nav_msgs Path.

      /tcp_path_marker:
        White line segments showing the planar extrusion path only.
    */
    pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>(
      "tcp_pose_array",
      10
    );

    path_pub_ = this->create_publisher<nav_msgs::msg::Path>(
      "tcp_path",
      10
    );

    marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
      "tcp_path_marker",
      10
    );

    toolpath_points_ = loadGcodeToolpath(gcode_file_);

    timer_ = this->create_wall_timer(
      500ms,
      std::bind(&TcpPoseGeneratorNode::publishToolpath, this)
    );

    RCLCPP_INFO(this->get_logger(), "Milestone 2 planar TCP pose generator started.");
    RCLCPP_INFO(this->get_logger(), "Loaded %zu planar extrusion points.", toolpath_points_.size());
    RCLCPP_INFO(this->get_logger(), "Publishing PoseArray on /tcp_pose_array");
    RCLCPP_INFO(this->get_logger(), "Publishing Path on /tcp_path");
    RCLCPP_INFO(this->get_logger(), "Publishing Marker on /tcp_path_marker");
  }

private:
  std::vector<ToolpathPoint> loadGcodeToolpath(const std::string & file_path)
  {
    std::vector<ToolpathPoint> points;

    if (file_path.empty()) {
      RCLCPP_ERROR(
        this->get_logger(),
        "No G-code file provided. Set the gcode_file parameter."
      );
      return points;
    }

    std::ifstream file(file_path);

    if (!file.is_open()) {
      RCLCPP_ERROR(
        this->get_logger(),
        "Failed to open G-code file: %s",
        file_path.c_str()
      );
      return points;
    }

    RCLCPP_INFO(this->get_logger(), "Reading G-code file: %s", file_path.c_str());

    /*
      Current modal G-code state.

      G-code lines often omit values that have not changed. For example:
        G1 X10 Y10 Z0.2 E0.20
        G1 X12 E0.25

      The second line still uses the previous Y and Z values. These variables
      store the current machine state while reading the file line by line.
    */
    double current_x_mm = 0.0;
    double current_y_mm = 0.0;
    double current_z_mm = 0.0;
    double current_e = 0.0;

    bool absolute_positioning = true;  // G90 = absolute, G91 = relative.
    bool absolute_extrusion = true;    // M82 = absolute E, M83 = relative E.

    int current_motion_command = -1;   // Stores modal G0 or G1 command.
    int segment_id = -1;
    bool extrusion_segment_open = false;

    std::string line;

    while (std::getline(file, line)) {
      const auto words = parseGcodeWords(line);

      if (words.empty()) {
        continue;
      }

      double g_value = 0.0;
      double m_value = 0.0;

      const bool has_g = getWordValue(words, 'G', g_value);
      const bool has_m = getWordValue(words, 'M', m_value);

      /*
        Handle G-code commands that change state.
      */
      if (has_g) {
        const int g_code = static_cast<int>(std::round(g_value));

        if (g_code == 90) {
          absolute_positioning = true;
          continue;
        }

        if (g_code == 91) {
          absolute_positioning = false;
          continue;
        }

        if (g_code == 92) {
          /*
            G92 resets the current coordinate/extruder state without creating
            a physical movement. It is commonly used as G92 E0.
          */
          double value = 0.0;

          if (getWordValue(words, 'X', value)) {
            current_x_mm = value;
          }
          if (getWordValue(words, 'Y', value)) {
            current_y_mm = value;
          }
          if (getWordValue(words, 'Z', value)) {
            current_z_mm = value;
          }
          if (getWordValue(words, 'E', value)) {
            current_e = value;
          }

          extrusion_segment_open = false;
          continue;
        }

        if (g_code == 0 || g_code == 1) {
          current_motion_command = g_code;
        }
      }

      if (has_m) {
        const int m_code = static_cast<int>(std::round(m_value));

        if (m_code == 82) {
          absolute_extrusion = true;
          continue;
        }

        if (m_code == 83) {
          absolute_extrusion = false;
          continue;
        }
      }

      if (current_motion_command != 0 && current_motion_command != 1) {
        continue;
      }

      const bool has_position =
        hasWord(words, 'X') ||
        hasWord(words, 'Y') ||
        hasWord(words, 'Z');

      const bool has_extrusion = hasWord(words, 'E');

      if (!has_position && !has_extrusion) {
        continue;
      }

      const double start_x_mm = current_x_mm;
      const double start_y_mm = current_y_mm;
      const double start_z_mm = current_z_mm;
      const double start_e = current_e;

      double target_x_mm = current_x_mm;
      double target_y_mm = current_y_mm;
      double target_z_mm = current_z_mm;
      double target_e = current_e;

      double value = 0.0;

      if (getWordValue(words, 'X', value)) {
        target_x_mm = absolute_positioning ? value : current_x_mm + value;
      }
      if (getWordValue(words, 'Y', value)) {
        target_y_mm = absolute_positioning ? value : current_y_mm + value;
      }
      if (getWordValue(words, 'Z', value)) {
        target_z_mm = absolute_positioning ? value : current_z_mm + value;
      }

      double extrusion_delta = 0.0;

      if (getWordValue(words, 'E', value)) {
        if (absolute_extrusion) {
          target_e = value;
          extrusion_delta = target_e - current_e;
        } else {
          extrusion_delta = value;
          target_e = current_e + value;
        }
      }

      /*
        A movement is part of the printed path if:
          - it is a G1 linear move,
          - it changes position, and
          - the extrusion value increases.

        Travel moves, retractions, and pure extruder commands are ignored.
      */
      const bool is_linear_extrusion_move =
        current_motion_command == 1 &&
        has_position &&
        extrusion_delta > min_extrusion_delta_;

      if (is_linear_extrusion_move) {
        const Vec3 start_position_m{
          start_x_mm / 1000.0,
          start_y_mm / 1000.0,
          start_z_mm / 1000.0
        };

        const Vec3 target_position_m{
          target_x_mm / 1000.0,
          target_y_mm / 1000.0,
          target_z_mm / 1000.0
        };

        if (!extrusion_segment_open) {
          ++segment_id;
          points.push_back({start_position_m, segment_id});
          extrusion_segment_open = true;
        }

        points.push_back({target_position_m, segment_id});
      } else {
        /*
          Any non-printing movement breaks the current extrusion segment.
          This keeps separate strokes/paths disconnected in RViz.
        */
        if (current_motion_command == 0 || has_position || extrusion_delta <= 0.0) {
          extrusion_segment_open = false;
        }
      }

      current_x_mm = target_x_mm;
      current_y_mm = target_y_mm;
      current_z_mm = target_z_mm;
      current_e = target_e;
    }

    return points;
  }

  Vec3 computeTangentAtIndex(std::size_t i) const
  {
    if (toolpath_points_.size() < 2) {
      return {1.0, 0.0, 0.0};
    }

    const int current_segment = toolpath_points_[i].segment_id;

    const bool has_prev =
      i > 0 &&
      toolpath_points_[i - 1].segment_id == current_segment;

    const bool has_next =
      i + 1 < toolpath_points_.size() &&
      toolpath_points_[i + 1].segment_id == current_segment;

    Vec3 tangent{1.0, 0.0, 0.0};

    if (has_prev && has_next) {
      tangent = subtract(
        toolpath_points_[i + 1].position,
        toolpath_points_[i - 1].position
      );
    } else if (has_next) {
      tangent = subtract(
        toolpath_points_[i + 1].position,
        toolpath_points_[i].position
      );
    } else if (has_prev) {
      tangent = subtract(
        toolpath_points_[i].position,
        toolpath_points_[i - 1].position
      );
    }

    tangent = normalize(tangent);

    if (norm(tangent) < 1e-9) {
      return {1.0, 0.0, 0.0};
    }

    return tangent;
  }

  bool buildPlanarPoseFromPoint(std::size_t index, geometry_msgs::msg::Pose & pose) const
  {
    /*
      planar TCP frame convention.

      x_axis:
        Direction of travel along the extracted G-code path.

      z_axis:
        Fixed planar nozzle normal. For a normal flat print bed, this is +Z.

      y_axis:
        Computed using the cross product to complete a right-handed frame.

      will be replaced by a surface normal extracted from curved geometry.
    */
    Vec3 x_axis = computeTangentAtIndex(index);
    Vec3 z_axis{0.0, 0.0, 1.0};
    Vec3 y_axis = normalize(cross(z_axis, x_axis));

    if (norm(y_axis) < 1e-9) {
      return false;
    }

    x_axis = normalize(cross(y_axis, z_axis));

    tf2::Matrix3x3 rotation_matrix(
      x_axis.x, y_axis.x, z_axis.x,
      x_axis.y, y_axis.y, z_axis.y,
      x_axis.z, y_axis.z, z_axis.z
    );

    tf2::Quaternion q;
    rotation_matrix.getRotation(q);
    q.normalize();

    const Vec3 & position = toolpath_points_[index].position;

    pose.position.x = position.x * visualization_scale_;
    pose.position.y = position.y * visualization_scale_;
    pose.position.z = position.z * visualization_scale_;
    pose.orientation = tf2::toMsg(q);

    return true;
  }

  void publishToolpath()
  {
    if (toolpath_points_.empty()) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        2000,
        "No toolpath points available to publish."
      );
      return;
    }

    /*
      Allow the RViz scale to be changed while the node is running:
        ros2 param set /tcp_pose_generator_node visualization_scale 10.0
    */
    this->get_parameter("visualization_scale", visualization_scale_);

    const auto stamp = this->now();

    geometry_msgs::msg::PoseArray pose_array;
    pose_array.header.stamp = stamp;
    pose_array.header.frame_id = frame_id_;

    nav_msgs::msg::Path tcp_path;
    tcp_path.header.stamp = stamp;
    tcp_path.header.frame_id = frame_id_;

    visualization_msgs::msg::Marker path_marker;
    path_marker.header.stamp = stamp;
    path_marker.header.frame_id = frame_id_;
    path_marker.ns = "tcp_path";
    path_marker.id = 0;
    path_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
    path_marker.action = visualization_msgs::msg::Marker::ADD;
    path_marker.scale.x = 0.003;
    path_marker.color.r = 1.0;
    path_marker.color.g = 1.0;
    path_marker.color.b = 1.0;
    path_marker.color.a = 1.0;

    for (std::size_t i = 0; i < toolpath_points_.size(); ++i) {
      geometry_msgs::msg::Pose pose;

      if (!buildPlanarPoseFromPoint(i, pose)) {
        RCLCPP_WARN_THROTTLE(
          this->get_logger(),
          *this->get_clock(),
          2000,
          "Skipping pose because a valid planar TCP frame could not be built."
        );
        continue;
      }

      pose_array.poses.push_back(pose);

      geometry_msgs::msg::PoseStamped pose_stamped;
      pose_stamped.header.stamp = stamp;
      pose_stamped.header.frame_id = frame_id_;
      pose_stamped.pose = pose;
      tcp_path.poses.push_back(pose_stamped);
    }

    /*
      Draw only connected extrusion segments.
      Do not draw travel moves between separate print segments.
    */
    for (std::size_t i = 1; i < toolpath_points_.size(); ++i) {
      const bool same_segment =
        toolpath_points_[i].segment_id == toolpath_points_[i - 1].segment_id;

      if (!same_segment) {
        continue;
      }

      path_marker.points.push_back(
        toMsgPoint(toolpath_points_[i - 1].position, visualization_scale_)
      );

      path_marker.points.push_back(
        toMsgPoint(toolpath_points_[i].position, visualization_scale_)
      );
    }

    pose_pub_->publish(pose_array);
    path_pub_->publish(tcp_path);
    marker_pub_->publish(path_marker);
  }

  std::string frame_id_;
  std::string gcode_file_;

  double min_extrusion_delta_;
  double visualization_scale_;

  std::vector<ToolpathPoint> toolpath_points_;

  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pose_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;

  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TcpPoseGeneratorNode>());
  rclcpp::shutdown();
  return 0;
}
