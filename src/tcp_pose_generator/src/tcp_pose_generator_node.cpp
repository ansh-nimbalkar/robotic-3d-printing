#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <sstream>
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
  Simple 3D vector type used internally for geometry calculations.
  ROS message types are only used at the publishing boundary.
*/
struct Vec3
{
  double x;
  double y;
  double z;
};

/*
  One parsed G-code word.

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
  One point on the extracted toolpath.

  position:
    TCP position in metres.

  normal:
    Surface normal at this TCP point. For planar mode this is fixed +Z.
    For STL mode this is extracted from the intersected mesh triangle.

  segment_id:
    Consecutive extrusion moves share the same segment_id. This prevents RViz
    from drawing false lines across travel moves or skipped projection regions.
*/
struct ToolpathPoint
{
  Vec3 position;
  Vec3 normal;
  int segment_id;
};

struct Triangle
{
  Vec3 v0;
  Vec3 v1;
  Vec3 v2;
  Vec3 normal;
};

struct Bounds2D
{
  double min_x = std::numeric_limits<double>::infinity();
  double max_x = -std::numeric_limits<double>::infinity();
  double min_y = std::numeric_limits<double>::infinity();
  double max_y = -std::numeric_limits<double>::infinity();
};

double norm(const Vec3 & v)
{
  return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

Vec3 normalize(const Vec3 & v)
{
  const double length = norm(v);

  if (length < 1e-12) {
    return {0.0, 0.0, 0.0};
  }

  return {v.x / length, v.y / length, v.z / length};
}

Vec3 add(const Vec3 & a, const Vec3 & b)
{
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 subtract(const Vec3 & a, const Vec3 & b)
{
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 scale(const Vec3 & v, double s)
{
  return {v.x * s, v.y * s, v.z * s};
}

double dot(const Vec3 & a, const Vec3 & b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 cross(const Vec3 & a, const Vec3 & b)
{
  return {
    a.y * b.z - a.z * b.y,
    a.z * b.x - a.x * b.z,
    a.x * b.y - a.y * b.x
  };
}

Vec3 computeTriangleNormal(const Vec3 & v0, const Vec3 & v1, const Vec3 & v2)
{
  return normalize(cross(subtract(v1, v0), subtract(v2, v0)));
}

geometry_msgs::msg::Point toMsgPoint(const Vec3 & v, double visualization_scale)
{
  geometry_msgs::msg::Point point;
  point.x = v.x * visualization_scale;
  point.y = v.y * visualization_scale;
  point.z = v.z * visualization_scale;
  return point;
}

void updateBounds(Bounds2D & bounds, const Vec3 & p)
{
  bounds.min_x = std::min(bounds.min_x, p.x);
  bounds.max_x = std::max(bounds.max_x, p.x);
  bounds.min_y = std::min(bounds.min_y, p.y);
  bounds.max_y = std::max(bounds.max_y, p.y);
}

Vec3 centreOfBounds(const Bounds2D & bounds)
{
  return {
    0.5 * (bounds.min_x + bounds.max_x),
    0.5 * (bounds.min_y + bounds.max_y),
    0.0
  };
}

/*
  Remove standard ';' comments from a G-code line.
*/
std::string stripGcodeComment(const std::string & line)
{
  const std::size_t comment_position = line.find(';');

  if (comment_position == std::string::npos) {
    return line;
  }

  return line.substr(0, comment_position);
}

std::string trimWhitespace(const std::string & input)
{
  const std::size_t first = input.find_first_not_of(" \t\r\n");

  if (first == std::string::npos) {
    return "";
  }

  const std::size_t last = input.find_last_not_of(" \t\r\n");
  return input.substr(first, last - first + 1);
}

bool startsWith(const std::string & text, const std::string & prefix)
{
  return text.rfind(prefix, 0) == 0;
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
      // Ignore malformed words instead of failing the whole file.
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

uint32_t readLittleEndianUInt32(const std::array<unsigned char, 4> & bytes)
{
  return
    static_cast<uint32_t>(bytes[0]) |
    (static_cast<uint32_t>(bytes[1]) << 8) |
    (static_cast<uint32_t>(bytes[2]) << 16) |
    (static_cast<uint32_t>(bytes[3]) << 24);
}

/*
  Read a 32-bit float from a binary STL file.
  STL files are little-endian; normal desktop machines are also little-endian,
  but this byte-based version keeps the code explicit.
*/
float readLittleEndianFloat(std::istream & stream)
{
  std::array<unsigned char, 4> bytes{};
  stream.read(reinterpret_cast<char *>(bytes.data()), 4);

  const uint32_t value = readLittleEndianUInt32(bytes);

  float result = 0.0f;
  static_assert(sizeof(float) == sizeof(uint32_t), "Unexpected float size");
  std::memcpy(&result, &value, sizeof(float));
  return result;
}

bool binaryStlFileSizeLooksValid(const std::string & file_path)
{
  std::ifstream file(file_path, std::ios::binary | std::ios::ate);

  if (!file.is_open()) {
    return false;
  }

  const std::streamoff file_size = file.tellg();

  if (file_size < 84) {
    return false;
  }

  file.seekg(80, std::ios::beg);
  std::array<unsigned char, 4> count_bytes{};
  file.read(reinterpret_cast<char *>(count_bytes.data()), 4);

  if (!file.good()) {
    return false;
  }

  const uint32_t triangle_count = readLittleEndianUInt32(count_bytes);
  const std::streamoff expected_size = 84 + static_cast<std::streamoff>(triangle_count) * 50;

  return expected_size == file_size;
}

}  // namespace

class TcpPoseGeneratorNode : public rclcpp::Node
{
public:
  TcpPoseGeneratorNode()
  : Node("tcp_pose_generator_node")
  {
    /*
      Basic parameters.

      surface_mode:
        "planar" keeps the Milestone 2 behaviour.
        "stl" projects the G-code XY path onto an STL mesh.

      stl_unit_scale:
        STL files are unitless. Most slicer/CAD STL files are in millimetres,
        so the default converts mm to metres. Use 1.0 if your STL is already
        exported in metres.
    */
    frame_id_ = this->declare_parameter<std::string>("frame_id", "base_link");
    gcode_file_ = this->declare_parameter<std::string>("gcode_file", "");
    surface_mode_ = this->declare_parameter<std::string>("surface_mode", "planar");
    stl_file_ = this->declare_parameter<std::string>("stl_file", "");
    stl_unit_scale_ = this->declare_parameter<double>("stl_unit_scale", 0.001);

    min_extrusion_delta_ = this->declare_parameter<double>("min_extrusion_delta", 1e-7);

    /*
      Cura G-code filtering.

      When enabled, only extrusion belonging to the selected Cura model layer
      and mesh is extracted. This excludes startup purge lines, skirts outside
      the model section, and extrusion from other layers.

      selected_cura_layer:
        -1 = accept every Cura layer
        0 = first model layer

      selected_cura_mesh:
        Empty string = accept any model mesh.
        Otherwise, it must match the Cura ;MESH: comment exactly.
    */
    filter_cura_gcode_ = this->declare_parameter<bool>(
      "filter_cura_gcode",
      false
    );

    selected_cura_layer_ = this->declare_parameter<int>(
      "selected_cura_layer",
      -1
    );

    selected_cura_mesh_ = this->declare_parameter<std::string>(
      "selected_cura_mesh",
      ""
    );

    /*
      Path-to-surface alignment parameters.

      centre_gcode_on_mesh:
        Moves the centre of the extracted G-code path onto the centre of the STL
        footprint before projection. This is useful because slicer G-code often
        uses positive bed coordinates, while an STL mesh is often centred around
        the origin.

      gcode_xy_scale:
        Scales the extracted G-code XY footprint before projection. Use this if
        the G-code path is larger than the STL surface.

      use_gcode_z_as_normal_offset:
        If true, the original G-code Z value is used as an offset along the
        local surface normal. This treats the G-code layer height as distance
        above the curved surface.

      tcp_offset_along_normal:
        Additional constant offset along the local surface normal, in metres.
    */
    centre_gcode_on_mesh_ = this->declare_parameter<bool>("center_gcode_on_mesh", true);
    gcode_xy_scale_ = this->declare_parameter<double>("gcode_xy_scale", 1.0);
    use_gcode_z_as_normal_offset_ = this->declare_parameter<bool>(
      "use_gcode_z_as_normal_offset",
      true
    );
    tcp_offset_along_normal_ = this->declare_parameter<double>("tcp_offset_along_normal", 0.0);

    /*
      STL projection parameters.

      The first STL implementation uses vertical projection:
        G-code X/Y -> intersect STL triangles in the Z direction.

      For top-surface printing, selecting the highest intersected Z is normally
      correct. This does not handle side walls, undercuts, or multi-valued
      geometry yet. Those are later thesis extensions.
    */
    choose_highest_z_intersection_ = this->declare_parameter<bool>(
      "choose_highest_z_intersection",
      true
    );
    force_normals_up_ = this->declare_parameter<bool>("force_normals_up", true);

    /*
      RViz-only parameters.
    */
    visualization_scale_ = this->declare_parameter<double>("visualization_scale", 1.0);
    normal_marker_length_ = this->declare_parameter<double>("normal_marker_length", 0.01);
    normal_marker_stride_ = this->declare_parameter<int>("normal_marker_stride", 10);
    publish_mesh_marker_ = this->declare_parameter<bool>("publish_mesh_marker", true);
    max_mesh_triangles_to_publish_ = this->declare_parameter<int>(
      "max_mesh_triangles_to_publish",
      5000
    );

    pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>(
      "tcp_pose_array",
      10
    );

    path_pub_ = this->create_publisher<nav_msgs::msg::Path>(
      "tcp_path",
      10
    );

    path_marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
      "tcp_path_marker",
      10
    );

    normal_marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
      "tcp_normal_marker",
      10
    );

    mesh_marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
      "surface_mesh_marker",
      10
    );

    raw_toolpath_points_ = loadGcodeToolpath(gcode_file_);

    if (surface_mode_ == "stl") {
      mesh_triangles_ = loadStlMesh(stl_file_);
    }

    toolpath_points_ = buildSurfaceAwareToolpath(raw_toolpath_points_);

    timer_ = this->create_wall_timer(
      500ms,
      std::bind(&TcpPoseGeneratorNode::publishToolpath, this)
    );

    RCLCPP_INFO(this->get_logger(), "Milestone 3 TCP pose generator started.");
    RCLCPP_INFO(this->get_logger(), "Surface mode: %s", surface_mode_.c_str());
    RCLCPP_INFO(this->get_logger(), "Loaded %zu raw G-code extrusion points.", raw_toolpath_points_.size());
    RCLCPP_INFO(this->get_logger(), "Generated %zu publishable TCP poses.", toolpath_points_.size());

    if (surface_mode_ == "stl") {
      RCLCPP_INFO(this->get_logger(), "Loaded %zu STL triangles.", mesh_triangles_.size());
    }
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

    RCLCPP_INFO(
      this->get_logger(),
      "Reading G-code file: %s",
      file_path.c_str()
    );

    double current_x_mm = 0.0;
    double current_y_mm = 0.0;
    double current_z_mm = 0.0;
    double current_e = 0.0;

    bool absolute_positioning = true;
    bool absolute_extrusion = true;

    int current_motion_command = -1;
    int segment_id = -1;
    bool extrusion_segment_open = false;

    /*
      Cura metadata state.

      Before the first ;LAYER comment, current_layer remains -1. Therefore,
      startup purge extrusion is rejected when Cura filtering is enabled.
    */
    int current_layer = -1;
    std::string current_mesh;
    std::string current_feature_type;

    std::size_t accepted_extrusion_moves = 0;
    std::size_t rejected_extrusion_moves = 0;

    auto currentSectionIsSelected = [&]() {
        if (!filter_cura_gcode_) {
          return true;
        }

        /*
          Anything before the first Cura model layer is startup G-code.
        */
        if (current_layer < 0) {
          return false;
        }

        if (
          selected_cura_layer_ >= 0 &&
          current_layer != selected_cura_layer_)
        {
          return false;
        }

        /*
          Cura uses ;MESH:NONMESH outside active model geometry.
        */
        if (current_mesh.empty() || current_mesh == "NONMESH") {
          return false;
        }

        if (
          !selected_cura_mesh_.empty() &&
          current_mesh != selected_cura_mesh_)
        {
          return false;
        }

        return true;
      };

    std::string line;

    while (std::getline(file, line)) {
      const std::string trimmed_line = trimWhitespace(line);

      /*
        Read Cura comments before removing comments from the G-code line.
      */
      if (startsWith(trimmed_line, ";LAYER:")) {
        try {
          current_layer = std::stoi(
            trimWhitespace(trimmed_line.substr(7))
          );
        } catch (const std::exception &) {
          current_layer = -1;
        }

        extrusion_segment_open = false;
        continue;
      }

      if (startsWith(trimmed_line, ";MESH:")) {
        current_mesh = trimWhitespace(trimmed_line.substr(6));
        extrusion_segment_open = false;
        continue;
      }

      if (startsWith(trimmed_line, ";TYPE:")) {
        current_feature_type = trimWhitespace(trimmed_line.substr(6));
        extrusion_segment_open = false;
        continue;
      }

      const auto words = parseGcodeWords(line);

      if (words.empty()) {
        continue;
      }

      double g_value = 0.0;
      double m_value = 0.0;

      const bool has_g = getWordValue(words, 'G', g_value);
      const bool has_m = getWordValue(words, 'M', m_value);

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

      double target_x_mm = current_x_mm;
      double target_y_mm = current_y_mm;
      double target_z_mm = current_z_mm;
      double target_e = current_e;

      double value = 0.0;

      if (getWordValue(words, 'X', value)) {
        target_x_mm =
          absolute_positioning ? value : current_x_mm + value;
      }

      if (getWordValue(words, 'Y', value)) {
        target_y_mm =
          absolute_positioning ? value : current_y_mm + value;
      }

      if (getWordValue(words, 'Z', value)) {
        target_z_mm =
          absolute_positioning ? value : current_z_mm + value;
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

      const bool is_linear_extrusion_move =
        current_motion_command == 1 &&
        has_position &&
        extrusion_delta > min_extrusion_delta_;

      if (is_linear_extrusion_move) {
        if (currentSectionIsSelected()) {
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

            points.push_back({
              start_position_m,
              {0.0, 0.0, 1.0},
              segment_id
            });

            extrusion_segment_open = true;
          }

          points.push_back({
            target_position_m,
            {0.0, 0.0, 1.0},
            segment_id
          });

          ++accepted_extrusion_moves;
        } else {
          /*
            Do not allow a later accepted path to connect to a rejected one.
          */
          extrusion_segment_open = false;
          ++rejected_extrusion_moves;
        }
      } else {
        if (
          current_motion_command == 0 ||
          has_position ||
          extrusion_delta <= 0.0)
        {
          extrusion_segment_open = false;
        }
      }

      /*
        Modal state must always be updated, including for rejected movements.
        Otherwise the first accepted point would use an incorrect start pose or
        extrusion value.
      */
      current_x_mm = target_x_mm;
      current_y_mm = target_y_mm;
      current_z_mm = target_z_mm;
      current_e = target_e;
    }

    if (filter_cura_gcode_) {
      RCLCPP_INFO(
        this->get_logger(),
        "Cura filtering enabled: layer=%d, mesh='%s'.",
        selected_cura_layer_,
        selected_cura_mesh_.empty() ?
        "<any model mesh>" :
        selected_cura_mesh_.c_str()
      );

      RCLCPP_INFO(
        this->get_logger(),
        "Accepted %zu extrusion moves and rejected %zu extrusion moves.",
        accepted_extrusion_moves,
        rejected_extrusion_moves
      );
    }

    return points;
  }

  std::vector<Triangle> loadStlMesh(const std::string & file_path)
  {
    std::vector<Triangle> triangles;

    if (file_path.empty()) {
      RCLCPP_ERROR(
        this->get_logger(),
        "surface_mode is 'stl' but no stl_file parameter was provided."
      );
      return triangles;
    }

    const bool looks_binary = binaryStlFileSizeLooksValid(file_path);

    if (looks_binary) {
      triangles = loadBinaryStl(file_path);
    } else {
      triangles = loadAsciiStl(file_path);
    }

    return triangles;
  }

  std::vector<Triangle> loadBinaryStl(const std::string & file_path)
  {
    std::vector<Triangle> triangles;
    std::ifstream file(file_path, std::ios::binary);

    if (!file.is_open()) {
      RCLCPP_ERROR(this->get_logger(), "Failed to open STL file: %s", file_path.c_str());
      return triangles;
    }

    file.seekg(80, std::ios::beg);

    std::array<unsigned char, 4> count_bytes{};
    file.read(reinterpret_cast<char *>(count_bytes.data()), 4);
    const uint32_t triangle_count = readLittleEndianUInt32(count_bytes);

    triangles.reserve(triangle_count);

    for (uint32_t i = 0; i < triangle_count; ++i) {
      Vec3 normal{
        static_cast<double>(readLittleEndianFloat(file)),
        static_cast<double>(readLittleEndianFloat(file)),
        static_cast<double>(readLittleEndianFloat(file))
      };

      Vec3 v0{
        static_cast<double>(readLittleEndianFloat(file)) * stl_unit_scale_,
        static_cast<double>(readLittleEndianFloat(file)) * stl_unit_scale_,
        static_cast<double>(readLittleEndianFloat(file)) * stl_unit_scale_
      };

      Vec3 v1{
        static_cast<double>(readLittleEndianFloat(file)) * stl_unit_scale_,
        static_cast<double>(readLittleEndianFloat(file)) * stl_unit_scale_,
        static_cast<double>(readLittleEndianFloat(file)) * stl_unit_scale_
      };

      Vec3 v2{
        static_cast<double>(readLittleEndianFloat(file)) * stl_unit_scale_,
        static_cast<double>(readLittleEndianFloat(file)) * stl_unit_scale_,
        static_cast<double>(readLittleEndianFloat(file)) * stl_unit_scale_
      };

      // Attribute byte count, unused.
      file.ignore(2);

      // Prefer the normal calculated directly from the triangle geometry.
      // Stored STL normals may be missing, stale, or incorrectly exported.
      const Vec3 stored_normal = normalize(normal);

      normal = computeTriangleNormal(v0, v1, v2);

      // Use the stored normal only as a fallback for a degenerate triangle.
      if (norm(normal) < 1e-12) {
        normal = stored_normal;
      }

      if (force_normals_up_ && normal.z < 0.0) {
        normal = scale(normal, -1.0);
      }

      triangles.push_back({v0, v1, v2, normal});
    }

    RCLCPP_INFO(this->get_logger(), "Read binary STL: %s", file_path.c_str());
    return triangles;
  }

  std::vector<Triangle> loadAsciiStl(const std::string & file_path)
  {
    std::vector<Triangle> triangles;
    std::ifstream file(file_path);

    if (!file.is_open()) {
      RCLCPP_ERROR(this->get_logger(), "Failed to open STL file: %s", file_path.c_str());
      return triangles;
    }

    Vec3 facet_normal{0.0, 0.0, 0.0};
    std::vector<Vec3> vertices;
    vertices.reserve(3);

    std::string line;

    while (std::getline(file, line)) {
      std::stringstream ss(line);
      std::string first_word;
      ss >> first_word;

      if (first_word == "facet") {
        std::string second_word;
        ss >> second_word;

        if (second_word == "normal") {
          ss >> facet_normal.x >> facet_normal.y >> facet_normal.z;
          facet_normal = normalize(facet_normal);
        }
      } else if (first_word == "vertex") {
        Vec3 vertex{0.0, 0.0, 0.0};
        ss >> vertex.x >> vertex.y >> vertex.z;
        vertex = scale(vertex, stl_unit_scale_);
        vertices.push_back(vertex);

        if (vertices.size() == 3) {
          const Vec3 stored_normal = facet_normal;

          Vec3 normal = computeTriangleNormal(
            vertices[0],
            vertices[1],
            vertices[2]
          );

          // Use the STL normal only if the triangle itself is degenerate.
          if (norm(normal) < 1e-12) {
            normal = stored_normal;
          }

          if (force_normals_up_ && normal.z < 0.0) {
            normal = scale(normal, -1.0);
          }

          triangles.push_back({vertices[0], vertices[1], vertices[2], normal});
          vertices.clear();
          facet_normal = {0.0, 0.0, 0.0};
        }
      }
    }

    RCLCPP_INFO(this->get_logger(), "Read ASCII STL: %s", file_path.c_str());
    return triangles;
  }

  Bounds2D computeToolpathBounds(const std::vector<ToolpathPoint> & points) const
  {
    Bounds2D bounds;

    for (const auto & point : points) {
      updateBounds(bounds, point.position);
    }

    return bounds;
  }

  Bounds2D computeMeshBounds(const std::vector<Triangle> & triangles) const
  {
    Bounds2D bounds;

    for (const auto & triangle : triangles) {
      updateBounds(bounds, triangle.v0);
      updateBounds(bounds, triangle.v1);
      updateBounds(bounds, triangle.v2);
    }

    return bounds;
  }

  bool projectXYOntoTriangle(
    const Vec3 & query,
    const Triangle & triangle,
    double & projected_z,
    Vec3 & normal) const
  {
    /*
      Compute barycentric coordinates in the XY plane.

      If the query XY point lies inside the XY projection of the triangle,
      interpolate the triangle's Z value. This is equivalent to a vertical
      projection onto the triangle.
    */
    const double x = query.x;
    const double y = query.y;

    const double x0 = triangle.v0.x;
    const double y0 = triangle.v0.y;
    const double x1 = triangle.v1.x;
    const double y1 = triangle.v1.y;
    const double x2 = triangle.v2.x;
    const double y2 = triangle.v2.y;

    const double denominator =
      (y1 - y2) * (x0 - x2) +
      (x2 - x1) * (y0 - y2);

    if (std::abs(denominator) < 1e-15) {
      return false;
    }

    const double w0 =
      ((y1 - y2) * (x - x2) +
      (x2 - x1) * (y - y2)) / denominator;

    const double w1 =
      ((y2 - y0) * (x - x2) +
      (x0 - x2) * (y - y2)) / denominator;

    const double w2 = 1.0 - w0 - w1;

    const double tolerance = -1e-9;

    if (w0 < tolerance || w1 < tolerance || w2 < tolerance) {
      return false;
    }

    projected_z =
      w0 * triangle.v0.z +
      w1 * triangle.v1.z +
      w2 * triangle.v2.z;

    normal = triangle.normal;
    return true;
  }

  bool projectXYOntoMesh(const Vec3 & query, Vec3 & surface_point, Vec3 & surface_normal) const
  {
    bool found = false;
    double best_z = choose_highest_z_intersection_ ?
      -std::numeric_limits<double>::infinity() :
      std::numeric_limits<double>::infinity();

    Vec3 best_normal{0.0, 0.0, 1.0};

    for (const auto & triangle : mesh_triangles_) {
      double z = 0.0;
      Vec3 normal{0.0, 0.0, 1.0};

      if (!projectXYOntoTriangle(query, triangle, z, normal)) {
        continue;
      }

      const bool better = choose_highest_z_intersection_ ?
        z > best_z :
        z < best_z;

      if (!found || better) {
        found = true;
        best_z = z;
        best_normal = normal;
      }
    }

    if (!found) {
      return false;
    }

    surface_point = {query.x, query.y, best_z};
    surface_normal = normalize(best_normal);

    if (norm(surface_normal) < 1e-12) {
      surface_normal = {0.0, 0.0, 1.0};
    }

    return true;
  }

  std::vector<ToolpathPoint> buildSurfaceAwareToolpath(
    const std::vector<ToolpathPoint> & raw_points)
  {
    if (raw_points.empty()) {
      return {};
    }

    if (surface_mode_ == "planar") {
      std::vector<ToolpathPoint> planar_points = raw_points;

      for (auto & point : planar_points) {
        point.normal = {0.0, 0.0, 1.0};
      }

      return planar_points;
    }

    if (surface_mode_ != "stl") {
      RCLCPP_ERROR(
        this->get_logger(),
        "Unknown surface_mode '%s'. Use 'planar' or 'stl'.",
        surface_mode_.c_str()
      );
      return {};
    }

    if (mesh_triangles_.empty()) {
      RCLCPP_ERROR(this->get_logger(), "No STL mesh triangles available for projection.");
      return {};
    }

    const Bounds2D toolpath_bounds = computeToolpathBounds(raw_points);
    const Bounds2D mesh_bounds = computeMeshBounds(mesh_triangles_);

    const Vec3 toolpath_centre = centreOfBounds(toolpath_bounds);
    const Vec3 mesh_centre = centreOfBounds(mesh_bounds);

    ///////////////////////////////////// LOGGING
    const double toolpath_width =
      toolpath_bounds.max_x - toolpath_bounds.min_x;

    const double toolpath_height =
      toolpath_bounds.max_y - toolpath_bounds.min_y;

    const double mesh_width =
      mesh_bounds.max_x - mesh_bounds.min_x;

    const double mesh_height =
      mesh_bounds.max_y - mesh_bounds.min_y;

    RCLCPP_INFO(
      this->get_logger(),
      "G-code XY bounds: X=[%.6f, %.6f] m, Y=[%.6f, %.6f] m, "
      "width=%.6f m, height=%.6f m.",
      toolpath_bounds.min_x,
      toolpath_bounds.max_x,
      toolpath_bounds.min_y,
      toolpath_bounds.max_y,
      toolpath_width,
      toolpath_height
    );

    RCLCPP_INFO(
      this->get_logger(),
      "STL XY bounds: X=[%.6f, %.6f] m, Y=[%.6f, %.6f] m, "
      "width=%.6f m, height=%.6f m.",
      mesh_bounds.min_x,
      mesh_bounds.max_x,
      mesh_bounds.min_y,
      mesh_bounds.max_y,
      mesh_width,
      mesh_height
    );

    if (toolpath_width > 1e-12 && toolpath_height > 1e-12) {
      const double x_fit_scale = mesh_width / toolpath_width;
      const double y_fit_scale = mesh_height / toolpath_height;

      const double bounding_box_fit_scale =
        std::min(x_fit_scale, y_fit_scale);

      RCLCPP_INFO(
        this->get_logger(),
        "Bounding-box fit scales: X=%.6f, Y=%.6f. "
        "Maximum uniform fit scale=%.6f.",
        x_fit_scale,
        y_fit_scale,
        bounding_box_fit_scale
      );
    }

    ////////////////////////////////////////// END LOGGING

    std::vector<ToolpathPoint> projected_points;
    projected_points.reserve(raw_points.size());

    int output_segment_id = -1;
    int last_input_segment_id = -1;
    bool output_segment_open = false;
    std::size_t skipped_points = 0;

    for (const auto & raw_point : raw_points) {
      if (raw_point.segment_id != last_input_segment_id) {
        output_segment_open = false;
        last_input_segment_id = raw_point.segment_id;
      }

      Vec3 query = raw_point.position;

      if (centre_gcode_on_mesh_) {
        query.x = (raw_point.position.x - toolpath_centre.x) * gcode_xy_scale_ + mesh_centre.x;
        query.y = (raw_point.position.y - toolpath_centre.y) * gcode_xy_scale_ + mesh_centre.y;
      } else {
        query.x = raw_point.position.x * gcode_xy_scale_;
        query.y = raw_point.position.y * gcode_xy_scale_;
      }

      Vec3 surface_point{0.0, 0.0, 0.0};
      Vec3 surface_normal{0.0, 0.0, 1.0};

      if (!projectXYOntoMesh(query, surface_point, surface_normal)) {
        ++skipped_points;
        output_segment_open = false;
        continue;
      }

      const double normal_offset =
        tcp_offset_along_normal_ +
        (use_gcode_z_as_normal_offset_ ? raw_point.position.z : 0.0);

      const Vec3 tcp_position = add(surface_point, scale(surface_normal, normal_offset));

      if (!output_segment_open) {
        ++output_segment_id;
        output_segment_open = true;
      }

      projected_points.push_back({tcp_position, surface_normal, output_segment_id});
    }

    if (skipped_points > 0) {
      RCLCPP_WARN(
        this->get_logger(),
        "Skipped %zu G-code points that did not project onto the STL surface.",
        skipped_points
      );
    }

    return projected_points;
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

    if (norm(tangent) < 1e-12) {
      return {1.0, 0.0, 0.0};
    }

    return tangent;
  }

  bool buildPoseFromPoint(std::size_t index, geometry_msgs::msg::Pose & pose) const
  {
    /*
      Surface-aware TCP frame convention.

      x_axis:
        Direction of travel along the mapped toolpath.

      z_axis:
        Local surface normal. In planar mode this is fixed +Z. In STL mode this
        comes from the mesh triangle under the G-code point.

      y_axis:
        Cross product completing the right-handed coordinate frame.
    */
    Vec3 z_axis = normalize(toolpath_points_[index].normal);

    if (norm(z_axis) < 1e-12) {
      z_axis = {0.0, 0.0, 1.0};
    }

    Vec3 x_axis = computeTangentAtIndex(index);

    /*
      Make the path tangent perpendicular to the surface normal. This prevents
      small numerical errors from producing a skewed TCP frame.
    */
    x_axis = subtract(x_axis, scale(z_axis, dot(x_axis, z_axis)));
    x_axis = normalize(x_axis);

    if (norm(x_axis) < 1e-12) {
      x_axis = {1.0, 0.0, 0.0};
      x_axis = subtract(x_axis, scale(z_axis, dot(x_axis, z_axis)));
      x_axis = normalize(x_axis);
    }

    if (norm(x_axis) < 1e-12) {
      return false;
    }

    Vec3 y_axis = normalize(cross(z_axis, x_axis));

    if (norm(y_axis) < 1e-12) {
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
      These RViz-only parameters can be changed while the node is running.
    */
    this->get_parameter("visualization_scale", visualization_scale_);
    this->get_parameter("normal_marker_length", normal_marker_length_);
    this->get_parameter("normal_marker_stride", normal_marker_stride_);

    if (normal_marker_stride_ < 1) {
      normal_marker_stride_ = 1;
    }

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

    visualization_msgs::msg::Marker normal_marker;
    normal_marker.header.stamp = stamp;
    normal_marker.header.frame_id = frame_id_;
    normal_marker.ns = "tcp_normals";
    normal_marker.id = 0;
    normal_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
    normal_marker.action = visualization_msgs::msg::Marker::ADD;
    normal_marker.scale.x = 0.0015;
    normal_marker.color.r = 0.1;
    normal_marker.color.g = 0.8;
    normal_marker.color.b = 1.0;
    normal_marker.color.a = 1.0;

    for (std::size_t i = 0; i < toolpath_points_.size(); ++i) {
      geometry_msgs::msg::Pose pose;

      if (!buildPoseFromPoint(i, pose)) {
        RCLCPP_WARN_THROTTLE(
          this->get_logger(),
          *this->get_clock(),
          2000,
          "Skipping pose because a valid TCP frame could not be built."
        );
        continue;
      }

      pose_array.poses.push_back(pose);

      geometry_msgs::msg::PoseStamped pose_stamped;
      pose_stamped.header.stamp = stamp;
      pose_stamped.header.frame_id = frame_id_;
      pose_stamped.pose = pose;
      tcp_path.poses.push_back(pose_stamped);

      if (i % static_cast<std::size_t>(normal_marker_stride_) == 0) {
        const Vec3 start = toolpath_points_[i].position;
        const Vec3 end = add(start, scale(toolpath_points_[i].normal, normal_marker_length_));
        normal_marker.points.push_back(toMsgPoint(start, visualization_scale_));
        normal_marker.points.push_back(toMsgPoint(end, visualization_scale_));
      }
    }

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
    path_marker_pub_->publish(path_marker);
    normal_marker_pub_->publish(normal_marker);

    if (surface_mode_ == "stl" && publish_mesh_marker_) {
      publishMeshMarker(stamp);
    }
  }

  void publishMeshMarker(const rclcpp::Time & stamp)
  {
    visualization_msgs::msg::Marker mesh_marker;
    mesh_marker.header.stamp = stamp;
    mesh_marker.header.frame_id = frame_id_;
    mesh_marker.ns = "surface_mesh";
    mesh_marker.id = 0;
    mesh_marker.type = visualization_msgs::msg::Marker::TRIANGLE_LIST;
    mesh_marker.action = visualization_msgs::msg::Marker::ADD;
    mesh_marker.scale.x = 1.0;
    mesh_marker.scale.y = 1.0;
    mesh_marker.scale.z = 1.0;
    mesh_marker.color.r = 0.35;
    mesh_marker.color.g = 0.35;
    mesh_marker.color.b = 0.35;
    mesh_marker.color.a = 0.35;

    const std::size_t max_triangles = static_cast<std::size_t>(
      std::max(0, max_mesh_triangles_to_publish_)
    );

    const std::size_t triangle_count = std::min(mesh_triangles_.size(), max_triangles);

    for (std::size_t i = 0; i < triangle_count; ++i) {
      mesh_marker.points.push_back(toMsgPoint(mesh_triangles_[i].v0, visualization_scale_));
      mesh_marker.points.push_back(toMsgPoint(mesh_triangles_[i].v1, visualization_scale_));
      mesh_marker.points.push_back(toMsgPoint(mesh_triangles_[i].v2, visualization_scale_));
    }

    mesh_marker_pub_->publish(mesh_marker);
  }

  std::string frame_id_;
  std::string gcode_file_;
  std::string surface_mode_;
  std::string stl_file_;
  std::string selected_cura_mesh_;

  double stl_unit_scale_;
  double min_extrusion_delta_;
  double gcode_xy_scale_;
  double tcp_offset_along_normal_;
  double visualization_scale_;
  double normal_marker_length_;

  bool centre_gcode_on_mesh_;
  bool use_gcode_z_as_normal_offset_;
  bool choose_highest_z_intersection_;
  bool force_normals_up_;
  bool publish_mesh_marker_;
  bool filter_cura_gcode_;

  int normal_marker_stride_;
  int max_mesh_triangles_to_publish_;
  int selected_cura_layer_;

  std::vector<ToolpathPoint> raw_toolpath_points_;
  std::vector<ToolpathPoint> toolpath_points_;
  std::vector<Triangle> mesh_triangles_;

  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pose_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr path_marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr normal_marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr mesh_marker_pub_;

  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TcpPoseGeneratorNode>());
  rclcpp::shutdown();
  return 0;
}
