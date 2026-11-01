"""AtomSlicer-to-UR10e trajectory planning adapter.

Pipeline:

    /atomslicer/tcp_poses (print-local, real scale)
        -> TF into base_link
        -> sequentially seeded nominal UR10e IK
        -> independent FK verification
        -> whole-path joint continuity normalisation
        -> Cartesian/joint-limit-aware time parameterisation
        -> cubic-spline waypoint velocities
        -> /ur10e/target_joint_trajectory

This node is a PLANNER, not a controller.  It never publishes directly to a UR
controller topic.  The existing move_ur node remains the sole execution boundary
and its accept_joint_trajectory_commands safety gate controls whether a validated
trajectory is forwarded to fake or real hardware.
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Pose, PoseArray
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.time import Time
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray
import spatialmath as sm
import tf2_ros

from move_robot.ur10e_kinematics import JOINT_NAMES, UR10eKinematics


RELIABLE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

JOINT_STATE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
)

# Diagnostics are latched so RViz can be opened after analysis completes.
DIAGNOSTIC_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

# builtin_interfaces/Duration.sec is signed int32.
MAX_ROS_DURATION_SECONDS = (2**31) - 1


@dataclass(frozen=True)
class TrajectoryTimingResult:
    """Metrics produced by the final local time-parameterisation pass."""

    trajectory: JointTrajectory
    total_time_sec: float
    effective_print_speed_mm_s: float
    approach_duration_sec: float
    min_print_segment_duration_sec: float
    max_print_segment_duration_sec: float
    max_cartesian_speed_mm_s: float
    max_joint_velocity_rad_s: np.ndarray
    max_joint_acceleration_rad_s2: np.ndarray
    retiming_iterations: int
    retimed_segment_count: int
    max_local_time_scale: float


@dataclass(frozen=True)
class ContinuityDiagnosticEvent:
    """One adjacent joint-space step that exceeds the diagnostic threshold."""

    previous_source_index: int
    source_index: int
    joint_index: int
    joint_name: str
    classification: str

    q_previous_rad: float
    q_current_rad: float
    raw_delta_rad: float
    wrapped_delta_rad: float

    previous_lower_limit_margin_rad: float
    previous_upper_limit_margin_rad: float
    current_lower_limit_margin_rad: float
    current_upper_limit_margin_rad: float
    minimum_limit_margin_rad: float

    tcp_position_step_m: float
    tcp_orientation_step_rad: float

    tcp_x_m: float
    tcp_y_m: float
    tcp_z_m: float


class AtomSlicerTrajectory(Node):
    """Validate AtomSlicer TCP poses and build a controller-ready UR10e trajectory."""

    def __init__(self) -> None:
        super().__init__("atomslicer_trajectory")

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.declare_parameter("tcp_pose_topic", "/atomslicer/tcp_poses")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("tf_timeout_sec", 1.0)

        # Full-resolution analysis is the default.  These two parameters are
        # primarily useful for fast debugging without changing the algorithm.
        self.declare_parameter("analysis_stride", 1)
        self.declare_parameter("max_poses", 0)  # 0 = no limit
        self.declare_parameter("progress_interval", 1000)

        # Numerical IK convergence and independent FK verification are kept
        # separate intentionally.  The solver residual tolerance is tighter
        # than the Cartesian acceptance threshold so an RTB "success" should
        # normally also pass the explicit FK gate.
        self.declare_parameter("ik_residual_tolerance", 1e-8)
        self.declare_parameter("ik_iteration_limit", 100)

        # FK verification protects against accepting a numerical solver result
        # that does not actually reproduce the requested Cartesian target.
        self.declare_parameter("fk_position_tolerance_m", 0.0005)
        self.declare_parameter("fk_orientation_tolerance_deg", 0.5)

        # Continuity is still diagnostic-only in this milestone.  This
        # threshold does not reject any waypoint; it only reports steps that
        # deserve inspection before trajectory timing/execution is introduced.
        self.declare_parameter(
            "continuity_warning_threshold_rad",
            0.35,
        )

        # A large raw revolute-joint step is classified as a joint-limit wrap
        # only when its shortest modulo-2*pi rotation is small AND at least one
        # endpoint is close to the nominal joint limit.
        self.declare_parameter(
            "continuity_joint_limit_margin_rad",
            0.25,
        )

        # Export and console diagnostics.  An empty CSV path disables export.
        self.declare_parameter(
            "continuity_csv_path",
            "~/.ros/atomslicer_continuity_diagnostics.csv",
        )
        self.declare_parameter("continuity_log_count", 20)

        # RViz point sizes for reachability and continuity diagnostics.
        self.declare_parameter("marker_size_m", 0.003)
        self.declare_parameter("continuity_marker_size_m", 0.005)

        # ------------------------------------------------------------------
        # Trajectory generation / existing execution-stack integration
        # ------------------------------------------------------------------
        # move_ur already owns the controller-facing boundary.  This planner
        # publishes only to its safety-gated multi-point input.
        self.declare_parameter(
            "trajectory_command_topic",
            "/ur10e/target_joint_trajectory",
        )
        self.declare_parameter("publish_joint_trajectory", True)

        # The legacy G-code interpreter used distance/speed timing, a 100 mm/s
        # robot Cartesian speed cap, a print-speed multiplier and a 5 s first
        # movement.  Preserve those semantics here, but apply the speed cap
        # after the multiplier so it can never be exceeded accidentally.
        self.declare_parameter("print_speed_mm_s", 20.0)
        self.declare_parameter("print_speed_multiplier", 1.0)
        self.declare_parameter("robot_cartesian_speed_limit_mm_s", 100.0)
        self.declare_parameter("approach_duration_sec", 5.0)

        # Do not generate waypoint intervals below the UR e-series 500 Hz
        # control period.  This is a planning floor, not a request to execute
        # the controller at a different rate.
        self.declare_parameter("min_segment_duration_sec", 0.002)

        # UR10e nominal joint-speed limits used by the previous thesis: 120
        # deg/s for shoulder pan/lift and 180 deg/s for elbow/wrist joints.
        # A safety factor is applied before time parameterisation.
        self.declare_parameter(
            "joint_velocity_limits_rad_s",
            [
                math.radians(120.0),
                math.radians(120.0),
                math.radians(180.0),
                math.radians(180.0),
                math.radians(180.0),
                math.radians(180.0),
            ],
        )
        self.declare_parameter("joint_velocity_safety_factor", 0.8)

        # These are deliberately conservative SOFTWARE PLANNING limits, not
        # claimed UR manufacturer maxima.  They prevent the first generated
        # trajectory from relying on arbitrarily high cubic-spline
        # accelerations.  They can be tuned later from validated hardware
        # limits without changing the planner architecture.
        self.declare_parameter(
            "joint_acceleration_limits_rad_s2",
            [1.0, 1.0, 1.5, 2.0, 2.0, 2.0],
        )
        self.declare_parameter("joint_acceleration_safety_factor", 0.8)

        # Local retiming only increases durations where the cubic
        # interpolation would exceed a configured joint VELOCITY limit.
        #
        # Acceleration is reported diagnostically but is not used as a hard
        # timing gate here.  The legacy execution pipeline did not define a
        # manufacturer-backed UR10e acceleration limit, so enforcing arbitrary
        # software acceleration values would unnecessarily distort print speed.
        self.declare_parameter("retiming_max_iterations", 20)
        self.declare_parameter("retiming_margin", 1.02)

        # Refuse to publish if the robot moves materially while the full-path
        # analysis is running.  The trajectory must start from the state used
        # to construct its approach segment.
        self.declare_parameter("start_state_tolerance_rad", 0.02)

        # Transform from the AtomSlicer process/TCP frame to the nominal robot
        # DH terminal frame used for IK.  Identity is intentional for this
        # milestone and is explicit rather than hidden in the solver.
        self.declare_parameter("tool_offset_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter("tool_offset_rpy_deg", [0.0, 0.0, 0.0])

        self.tcp_pose_topic = str(self.get_parameter("tcp_pose_topic").value)
        self.joint_state_topic = str(self.get_parameter("joint_state_topic").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.tf_timeout_sec = float(self.get_parameter("tf_timeout_sec").value)
        self.analysis_stride = int(self.get_parameter("analysis_stride").value)
        self.max_poses = int(self.get_parameter("max_poses").value)
        self.progress_interval = int(self.get_parameter("progress_interval").value)
        self.ik_residual_tolerance = float(
            self.get_parameter("ik_residual_tolerance").value
        )
        self.ik_iteration_limit = int(
            self.get_parameter("ik_iteration_limit").value
        )
        self.fk_position_tolerance_m = float(
            self.get_parameter("fk_position_tolerance_m").value
        )
        self.fk_orientation_tolerance_rad = math.radians(
            float(self.get_parameter("fk_orientation_tolerance_deg").value)
        )
        self.continuity_warning_threshold_rad = float(
            self.get_parameter(
                "continuity_warning_threshold_rad"
            ).value
        )
        self.continuity_joint_limit_margin_rad = float(
            self.get_parameter(
                "continuity_joint_limit_margin_rad"
            ).value
        )
        self.continuity_csv_path = str(
            self.get_parameter("continuity_csv_path").value
        ).strip()
        self.continuity_log_count = int(
            self.get_parameter("continuity_log_count").value
        )
        self.marker_size_m = float(self.get_parameter("marker_size_m").value)
        self.continuity_marker_size_m = float(
            self.get_parameter("continuity_marker_size_m").value
        )

        self.trajectory_command_topic = str(
            self.get_parameter("trajectory_command_topic").value
        )
        self.publish_joint_trajectory = bool(
            self.get_parameter("publish_joint_trajectory").value
        )
        self.print_speed_mm_s = float(
            self.get_parameter("print_speed_mm_s").value
        )
        self.print_speed_multiplier = float(
            self.get_parameter("print_speed_multiplier").value
        )
        self.robot_cartesian_speed_limit_mm_s = float(
            self.get_parameter("robot_cartesian_speed_limit_mm_s").value
        )
        self.approach_duration_sec = float(
            self.get_parameter("approach_duration_sec").value
        )
        self.min_segment_duration_sec = float(
            self.get_parameter("min_segment_duration_sec").value
        )
        self.joint_velocity_limits_rad_s = self._read_vector_parameter(
            "joint_velocity_limits_rad_s", 6
        )
        self.joint_velocity_safety_factor = float(
            self.get_parameter("joint_velocity_safety_factor").value
        )
        self.joint_acceleration_limits_rad_s2 = self._read_vector_parameter(
            "joint_acceleration_limits_rad_s2", 6
        )
        self.joint_acceleration_safety_factor = float(
            self.get_parameter("joint_acceleration_safety_factor").value
        )
        self.retiming_max_iterations = int(
            self.get_parameter("retiming_max_iterations").value
        )
        self.retiming_margin = float(
            self.get_parameter("retiming_margin").value
        )
        self.start_state_tolerance_rad = float(
            self.get_parameter("start_state_tolerance_rad").value
        )

        tool_offset_xyz = self._read_vector_parameter("tool_offset_xyz", 3)
        tool_offset_rpy_deg = self._read_vector_parameter("tool_offset_rpy_deg", 3)

        self._validate_parameters()
        self.tool_transform = self._make_transform_matrix(
            tool_offset_xyz,
            np.deg2rad(tool_offset_rpy_deg),
        )

        # ------------------------------------------------------------------
        # Kinematics and runtime state
        # ------------------------------------------------------------------
        self.kinematics = UR10eKinematics()
        self.current_q: Optional[np.ndarray] = None
        self.analysis_started = False
        self.analysis_complete = False

        # Stored validated planning outputs.
        self.reachable_source_indices: list[int] = []
        self.reachable_joint_solutions: list[np.ndarray] = []
        self.generated_trajectory: Optional[JointTrajectory] = None

        # ------------------------------------------------------------------
        # TF
        # ------------------------------------------------------------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ------------------------------------------------------------------
        # ROS interfaces
        # ------------------------------------------------------------------
        self.joint_state_sub = self.create_subscription(
            JointState,
            self.joint_state_topic,
            self._joint_state_callback,
            JOINT_STATE_QOS,
        )
        self.tcp_pose_sub = self.create_subscription(
            PoseArray,
            self.tcp_pose_topic,
            self._tcp_pose_callback,
            RELIABLE_QOS,
        )

        self.transformed_pose_pub = self.create_publisher(
            PoseArray,
            "/atomslicer/tcp_poses_base",
            DIAGNOSTIC_QOS,
        )
        self.reachability_marker_pub = self.create_publisher(
            MarkerArray,
            "/atomslicer/reachability_markers",
            DIAGNOSTIC_QOS,
        )
        self.continuity_marker_pub = self.create_publisher(
            MarkerArray,
            "/atomslicer/continuity_markers",
            DIAGNOSTIC_QOS,
        )
        self.trajectory_command_pub = self.create_publisher(
            JointTrajectory,
            self.trajectory_command_topic,
            RELIABLE_QOS,
        )

        self.get_logger().info(
            "AtomSlicer trajectory planner ready; controller execution remains owned by move_ur."
        )
        self.get_logger().info(
            f"Input={self.tcp_pose_topic}, base_frame={self.base_frame}, "
            f"analysis_stride={self.analysis_stride}."
        )
        self.get_logger().info(
            f"IK residual tolerance={self.ik_residual_tolerance:.1e}, "
            f"FK acceptance={self.fk_position_tolerance_m * 1000.0:.3f} mm / "
            f"{math.degrees(self.fk_orientation_tolerance_rad):.3f} deg."
        )
        self.get_logger().info(
            f"Trajectory output={self.trajectory_command_topic}, "
            f"print_speed={self.print_speed_mm_s:.3f} mm/s, "
            f"publish={self.publish_joint_trajectory}."
        )

        if np.allclose(self.tool_transform, np.eye(4), atol=1e-12, rtol=0.0):
            self.get_logger().warn(
                "tool_offset is identity: AtomSlicer TCP frame is currently treated "
                "as the nominal UR10e DH terminal frame. Configure the explicit "
                "tool transform before final physical printing if required."
            )

    # ------------------------------------------------------------------
    # Parameter / representation helpers
    # ------------------------------------------------------------------
    def _read_vector_parameter(self, name: str, expected_size: int) -> np.ndarray:
        value = self.get_parameter(name).value
        try:
            array = np.asarray(value, dtype=float).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a numeric vector") from exc

        if array.size != expected_size or not np.all(np.isfinite(array)):
            raise ValueError(
                f"{name} must contain exactly {expected_size} finite values"
            )
        return array

    def _validate_parameters(self) -> None:
        if not self.base_frame:
            raise ValueError("base_frame must not be empty")
        if not math.isfinite(self.tf_timeout_sec) or self.tf_timeout_sec <= 0.0:
            raise ValueError("tf_timeout_sec must be finite and > 0")
        if self.analysis_stride < 1:
            raise ValueError("analysis_stride must be >= 1")
        if self.max_poses < 0:
            raise ValueError("max_poses must be >= 0")
        if self.progress_interval < 0:
            raise ValueError("progress_interval must be >= 0")
        if (
            not math.isfinite(self.ik_residual_tolerance)
            or self.ik_residual_tolerance <= 0.0
        ):
            raise ValueError("ik_residual_tolerance must be finite and > 0")
        if self.ik_iteration_limit < 1:
            raise ValueError("ik_iteration_limit must be >= 1")
        if (
            not math.isfinite(self.fk_position_tolerance_m)
            or self.fk_position_tolerance_m <= 0.0
        ):
            raise ValueError("fk_position_tolerance_m must be finite and > 0")
        if (
            not math.isfinite(self.continuity_warning_threshold_rad)
            or self.continuity_warning_threshold_rad <= 0.0
        ):
            raise ValueError(
                "continuity_warning_threshold_rad must be finite and > 0"
            )
        if (
            not math.isfinite(self.continuity_joint_limit_margin_rad)
            or self.continuity_joint_limit_margin_rad < 0.0
        ):
            raise ValueError(
                "continuity_joint_limit_margin_rad must be finite and >= 0"
            )
        if self.continuity_log_count < 0:
            raise ValueError("continuity_log_count must be >= 0")
        if (
            not math.isfinite(self.fk_orientation_tolerance_rad)
            or self.fk_orientation_tolerance_rad <= 0.0
        ):
            raise ValueError("fk_orientation_tolerance_deg must be finite and > 0")
        if not math.isfinite(self.marker_size_m) or self.marker_size_m <= 0.0:
            raise ValueError("marker_size_m must be finite and > 0")
        if (
            not math.isfinite(self.continuity_marker_size_m)
            or self.continuity_marker_size_m <= 0.0
        ):
            raise ValueError(
                "continuity_marker_size_m must be finite and > 0"
            )
        if not self.trajectory_command_topic:
            raise ValueError("trajectory_command_topic must not be empty")
        for name, value in (
            ("print_speed_mm_s", self.print_speed_mm_s),
            ("print_speed_multiplier", self.print_speed_multiplier),
            ("robot_cartesian_speed_limit_mm_s", self.robot_cartesian_speed_limit_mm_s),
            ("approach_duration_sec", self.approach_duration_sec),
            ("min_segment_duration_sec", self.min_segment_duration_sec),
            ("start_state_tolerance_rad", self.start_state_tolerance_rad),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")

        if (
            not 0.0 < self.joint_velocity_safety_factor <= 1.0
            or not math.isfinite(self.joint_velocity_safety_factor)
        ):
            raise ValueError(
                "joint_velocity_safety_factor must be finite and in (0, 1]"
            )
        if (
            not 0.0 < self.joint_acceleration_safety_factor <= 1.0
            or not math.isfinite(self.joint_acceleration_safety_factor)
        ):
            raise ValueError(
                "joint_acceleration_safety_factor must be finite and in (0, 1]"
            )
        if np.any(self.joint_velocity_limits_rad_s <= 0.0):
            raise ValueError(
                "joint_velocity_limits_rad_s must contain six values > 0"
            )
        if np.any(self.joint_acceleration_limits_rad_s2 <= 0.0):
            raise ValueError(
                "joint_acceleration_limits_rad_s2 must contain six values > 0"
            )
        if self.retiming_max_iterations < 1:
            raise ValueError("retiming_max_iterations must be >= 1")
        if (
            not math.isfinite(self.retiming_margin)
            or self.retiming_margin < 1.0
        ):
            raise ValueError("retiming_margin must be finite and >= 1.0")

    @staticmethod
    def _rpy_to_rotation(rpy_rad: np.ndarray) -> np.ndarray:
        roll, pitch, yaw = rpy_rad
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)

        # ZYX convention: R = Rz(yaw) * Ry(pitch) * Rx(roll).
        return np.array(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ],
            dtype=float,
        )

    @classmethod
    def _make_transform_matrix(
        cls,
        translation: np.ndarray,
        rpy_rad: np.ndarray,
    ) -> np.ndarray:
        transform = np.eye(4, dtype=float)
        transform[:3, :3] = cls._rpy_to_rotation(rpy_rad)
        transform[:3, 3] = translation
        return transform

    @staticmethod
    def _quaternion_to_rotation(x: float, y: float, z: float, w: float) -> Optional[np.ndarray]:
        q = np.array([x, y, z, w], dtype=float)
        if not np.all(np.isfinite(q)):
            return None

        norm = float(np.linalg.norm(q))
        if norm < 1e-12:
            return None

        x, y, z, w = q / norm

        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=float,
        )

    @staticmethod
    def _rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
        """Convert a proper rotation matrix to [x, y, z, w]."""
        r = np.asarray(rotation, dtype=float)
        trace = float(np.trace(r))

        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            w = 0.25 * s
            x = (r[2, 1] - r[1, 2]) / s
            y = (r[0, 2] - r[2, 0]) / s
            z = (r[1, 0] - r[0, 1]) / s
        elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = math.sqrt(max(0.0, 1.0 + r[0, 0] - r[1, 1] - r[2, 2])) * 2.0
            x = 0.25 * s
            y = (r[0, 1] + r[1, 0]) / s
            z = (r[0, 2] + r[2, 0]) / s
            w = (r[2, 1] - r[1, 2]) / s
        elif r[1, 1] > r[2, 2]:
            s = math.sqrt(max(0.0, 1.0 + r[1, 1] - r[0, 0] - r[2, 2])) * 2.0
            x = (r[0, 1] + r[1, 0]) / s
            y = 0.25 * s
            z = (r[1, 2] + r[2, 1]) / s
            w = (r[0, 2] - r[2, 0]) / s
        else:
            s = math.sqrt(max(0.0, 1.0 + r[2, 2] - r[0, 0] - r[1, 1])) * 2.0
            x = (r[0, 2] + r[2, 0]) / s
            y = (r[1, 2] + r[2, 1]) / s
            z = 0.25 * s
            w = (r[1, 0] - r[0, 1]) / s

        q = np.array([x, y, z, w], dtype=float)
        norm = float(np.linalg.norm(q))
        if norm < 1e-12:
            raise ValueError("Rotation-to-quaternion conversion produced zero norm")
        return q / norm

    @classmethod
    def _pose_to_matrix(cls, pose: Pose) -> Optional[np.ndarray]:
        position = np.array(
            [pose.position.x, pose.position.y, pose.position.z],
            dtype=float,
        )
        if not np.all(np.isfinite(position)):
            return None

        rotation = cls._quaternion_to_rotation(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        if rotation is None:
            return None

        transform = np.eye(4, dtype=float)
        transform[:3, :3] = rotation
        transform[:3, 3] = position
        return transform

    @classmethod
    def _transform_msg_to_matrix(cls, transform_msg) -> Optional[np.ndarray]:
        t = transform_msg.translation
        q = transform_msg.rotation
        rotation = cls._quaternion_to_rotation(q.x, q.y, q.z, q.w)
        if rotation is None:
            return None

        translation = np.array([t.x, t.y, t.z], dtype=float)
        if not np.all(np.isfinite(translation)):
            return None

        transform = np.eye(4, dtype=float)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        return transform

    @classmethod
    def _matrix_to_pose(cls, transform: np.ndarray) -> Pose:
        pose = Pose()
        pose.position.x = float(transform[0, 3])
        pose.position.y = float(transform[1, 3])
        pose.position.z = float(transform[2, 3])

        q = cls._rotation_to_quaternion(transform[:3, :3])
        pose.orientation.x = float(q[0])
        pose.orientation.y = float(q[1])
        pose.orientation.z = float(q[2])
        pose.orientation.w = float(q[3])
        return pose

    @staticmethod
    def _rotation_error_rad(actual: np.ndarray, desired: np.ndarray) -> float:
        error_rotation = actual.T @ desired
        cosine = (float(np.trace(error_rotation)) - 1.0) / 2.0
        cosine = float(np.clip(cosine, -1.0, 1.0))
        return math.acos(cosine)

    @staticmethod
    def _set_ros_duration(duration_msg, seconds: float) -> None:
        """Populate builtin_interfaces/Duration from non-negative seconds."""
        if (
            not math.isfinite(seconds)
            or seconds < 0.0
            or seconds > MAX_ROS_DURATION_SECONDS
        ):
            raise ValueError(f"Invalid ROS duration: {seconds}")

        whole_seconds = int(math.floor(seconds))
        nanoseconds = int(round((seconds - whole_seconds) * 1_000_000_000))

        if nanoseconds >= 1_000_000_000:
            whole_seconds += 1
            nanoseconds -= 1_000_000_000

        duration_msg.sec = whole_seconds
        duration_msg.nanosec = nanoseconds

    def _anchor_equivalent_joint_path_to_initial_state(
        self,
        q_path: list[np.ndarray],
        initial_q: np.ndarray,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Shift complete joint paths by constant +/-2*pi offsets.

        The whole-path continuity optimiser intentionally prioritises printing
        continuity.  Numerically tiny differences can still leave an otherwise
        equivalent entire path on a representation near +/-2*pi, producing an
        unnecessarily long approach from the robot's actual start state.

        A constant 2*pi shift applied to one joint over the COMPLETE print path
        leaves every printing joint delta and every FK pose unchanged.  For each
        joint, choose the feasible constant shift that places the first printing
        waypoint closest to ``initial_q`` while keeping every waypoint inside
        the nominal joint limits.
        """
        path = np.asarray(q_path, dtype=float)
        start = np.asarray(initial_q, dtype=float).reshape(-1)

        if path.ndim != 2 or path.shape[1] != len(JOINT_NAMES):
            raise ValueError("q_path must have shape (N, 6)")
        if path.shape[0] == 0:
            raise ValueError("q_path must not be empty")
        if start.size != len(JOINT_NAMES) or not np.all(np.isfinite(start)):
            raise ValueError("initial_q must contain six finite joint values")
        if not np.all(np.isfinite(path)):
            raise ValueError("q_path contains non-finite joint values")

        qlim = np.asarray(self.kinematics.robot.qlim, dtype=float)
        if qlim.shape != (2, len(JOINT_NAMES)):
            raise RuntimeError("UR10e joint-limit matrix is malformed")

        period = 2.0 * math.pi
        tolerance = 1e-9
        anchored = path.copy()
        applied_shifts = np.zeros(len(JOINT_NAMES), dtype=float)

        for joint_index in range(len(JOINT_NAMES)):
            lower = float(qlim[0, joint_index])
            upper = float(qlim[1, joint_index])

            if not (math.isfinite(lower) and math.isfinite(upper)):
                continue

            joint_values = path[:, joint_index]
            minimum_value = float(np.min(joint_values))
            maximum_value = float(np.max(joint_values))

            k_min = math.ceil(
                (lower - minimum_value - tolerance) / period
            )
            k_max = math.floor(
                (upper - maximum_value + tolerance) / period
            )

            if k_min > k_max:
                raise RuntimeError(
                    "No constant in-limit equivalent representation exists for "
                    f"{JOINT_NAMES[joint_index]}"
                )

            best_k = min(
                range(k_min, k_max + 1),
                key=lambda k: (
                    abs(
                        path[0, joint_index]
                        + k * period
                        - start[joint_index]
                    ),
                    abs(k),
                ),
            )

            shift = best_k * period
            anchored[:, joint_index] += shift
            applied_shifts[joint_index] = shift

        return (
            [row.copy() for row in anchored],
            applied_shifts,
        )

    @staticmethod
    def _compute_monotone_knot_velocities(
        q_path: np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        """Compute smooth, overshoot-resistant knot velocities.

        The interior derivative is a weighted harmonic mean of adjacent
        secant slopes (PCHIP-style).  If adjacent slopes change sign, the knot
        velocity is set to zero.  The first and last trajectory points are
        intentionally at rest.
        """
        q = np.asarray(q_path, dtype=float)
        t = np.asarray(times, dtype=float)

        if q.ndim != 2 or q.shape[1] != len(JOINT_NAMES):
            raise ValueError("q_path must have shape (N, 6)")
        if t.ndim != 1 or t.size != q.shape[0]:
            raise ValueError("times must align one-to-one with q_path")
        if q.shape[0] < 2:
            raise ValueError("trajectory requires at least two joint states")

        h = np.diff(t)
        if not np.all(np.isfinite(h)) or np.any(h <= 0.0):
            raise ValueError("trajectory times must be finite and strictly increasing")

        slopes = np.diff(q, axis=0) / h[:, None]
        velocities = np.zeros_like(q)

        for index in range(1, q.shape[0] - 1):
            h_prev = h[index - 1]
            h_next = h[index]
            slope_prev = slopes[index - 1]
            slope_next = slopes[index]

            same_direction = (slope_prev * slope_next) > 0.0
            nonzero = (np.abs(slope_prev) > 1e-12) & (np.abs(slope_next) > 1e-12)
            use_harmonic = same_direction & nonzero

            if not np.any(use_harmonic):
                continue

            w1 = 2.0 * h_next + h_prev
            w2 = h_next + 2.0 * h_prev

            velocities[index, use_harmonic] = (
                (w1 + w2)
                / (
                    w1 / slope_prev[use_harmonic]
                    + w2 / slope_next[use_harmonic]
                )
            )

        return velocities

    @staticmethod
    def _cubic_hermite_segment_peak_limits(
        q_path: np.ndarray,
        times: np.ndarray,
        velocities: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return per-segment, per-joint peak |velocity| and |acceleration|.

        With position + velocity waypoints the Humble
        joint_trajectory_controller uses cubic interpolation. Velocity is
        quadratic and acceleration is linear inside each segment, so the exact
        extrema can be evaluated analytically without time sampling.
        """
        q = np.asarray(q_path, dtype=float)
        t = np.asarray(times, dtype=float)
        v = np.asarray(velocities, dtype=float)

        if q.shape != v.shape or q.ndim != 2 or q.shape[1] != len(JOINT_NAMES):
            raise ValueError("q_path and velocities must both have shape (N, 6)")
        if t.ndim != 1 or t.size != q.shape[0]:
            raise ValueError("times must align with q_path")
        if q.shape[0] < 2:
            raise ValueError("trajectory requires at least two states")

        segment_velocity = np.zeros(
            (q.shape[0] - 1, len(JOINT_NAMES)),
            dtype=float,
        )
        segment_acceleration = np.zeros_like(segment_velocity)

        for index in range(q.shape[0] - 1):
            h = float(t[index + 1] - t[index])
            if not math.isfinite(h) or h <= 0.0:
                raise ValueError(
                    "trajectory contains a non-positive segment duration"
                )

            q0 = q[index]
            q1 = q[index + 1]
            v0 = v[index]
            v1 = v[index + 1]

            # v(s) = a*s^2 + b*s + c, s in [0, 1].
            a = 6.0 * (q0 - q1) / h + 3.0 * (v0 + v1)
            b = 6.0 * (q1 - q0) / h - 4.0 * v0 - 2.0 * v1
            c = v0

            velocity_max = np.maximum(np.abs(v0), np.abs(v1))

            for joint_index in range(len(JOINT_NAMES)):
                if abs(a[joint_index]) > 1e-14:
                    s_vertex = -b[joint_index] / (
                        2.0 * a[joint_index]
                    )
                    if 0.0 < s_vertex < 1.0:
                        vertex_velocity = (
                            a[joint_index] * s_vertex * s_vertex
                            + b[joint_index] * s_vertex
                            + c[joint_index]
                        )
                        velocity_max[joint_index] = max(
                            velocity_max[joint_index],
                            abs(vertex_velocity),
                        )

            # a_joint(s) = (2*a*s + b) / h, which is linear.
            acceleration_start = b / h
            acceleration_end = (2.0 * a + b) / h

            segment_velocity[index] = velocity_max
            segment_acceleration[index] = np.maximum(
                np.abs(acceleration_start),
                np.abs(acceleration_end),
            )

        return segment_velocity, segment_acceleration

    @classmethod
    def _cubic_hermite_peak_limits(
        cls,
        q_path: np.ndarray,
        times: np.ndarray,
        velocities: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return whole-trajectory per-joint interpolation peaks."""
        segment_velocity, segment_acceleration = (
            cls._cubic_hermite_segment_peak_limits(
                q_path,
                times,
                velocities,
            )
        )

        return (
            np.max(segment_velocity, axis=0),
            np.max(segment_acceleration, axis=0),
        )

    def _build_timed_joint_trajectory(
        self,
        initial_q: np.ndarray,
        joint_path: list[np.ndarray],
        tcp_transforms: list[np.ndarray],
    ) -> TrajectoryTimingResult:
        """Create a smooth locally-retimed trajectory for the move_ur boundary.

        Timing follows the established repository philosophy:

            segment duration = Cartesian distance / requested print speed

        with two additional safeguards required by the full AtomSlicer path:

        1. a joint-velocity floor, and
        2. exact validation of the cubic spline interpolation used by the
           joint_trajectory_controller.

        If cubic interpolation would exceed a joint velocity limit, only that
        segment and its immediate neighbours are lengthened.  Acceleration is
        measured and reported, but is not used as a hard retiming constraint
        because the current project does not yet have a validated physical
        UR10e acceleration policy.
        """
        if not joint_path:
            raise ValueError(
                "Cannot build a trajectory from an empty joint path"
            )
        if len(joint_path) != len(tcp_transforms):
            raise ValueError(
                "joint_path and tcp_transforms must have equal length"
            )

        q_print = np.asarray(joint_path, dtype=float)
        if q_print.ndim != 2 or q_print.shape[1] != len(JOINT_NAMES):
            raise ValueError("joint_path must have shape (N, 6)")
        if not np.all(np.isfinite(q_print)):
            raise ValueError("joint_path contains non-finite values")

        initial_q_array = np.asarray(
            initial_q,
            dtype=float,
        ).reshape(-1)
        if (
            initial_q_array.size != len(JOINT_NAMES)
            or not np.all(np.isfinite(initial_q_array))
        ):
            raise ValueError(
                "initial_q must contain six finite joint positions"
            )

        tcp_positions = np.asarray(
            [transform[:3, 3] for transform in tcp_transforms],
            dtype=float,
        )
        if tcp_positions.shape != (q_print.shape[0], 3):
            raise ValueError("TCP transforms are malformed")

        effective_print_speed_mm_s = min(
            self.print_speed_mm_s * self.print_speed_multiplier,
            self.robot_cartesian_speed_limit_mm_s,
        )
        if effective_print_speed_mm_s <= 0.0:
            raise ValueError("Effective print speed must be > 0")

        velocity_limits = (
            self.joint_velocity_limits_rad_s
            * self.joint_velocity_safety_factor
        )

        # Acceleration limits are retained only for comparison/diagnostics.
        # They are NOT used to accept/reject or retime the trajectory.
        acceleration_reference = (
            self.joint_acceleration_limits_rad_s2
            * self.joint_acceleration_safety_factor
        )

        # Internal state 0 is the actual measured robot state.  It is needed
        # for approach timing and spline derivatives, but it is not published
        # as a t=0 command point.
        q_execution = np.vstack(
            (initial_q_array, q_print)
        )

        segment_durations = np.zeros(
            q_execution.shape[0] - 1,
            dtype=float,
        )

        # --------------------------------------------------------------
        # Approach segment
        # --------------------------------------------------------------
        approach_delta_q = np.abs(
            q_print[0] - initial_q_array
        )

        # For a cubic segment whose endpoint velocities may be zero, peak
        # velocity can be up to 1.5 times the average secant velocity.  The
        # 1.5 factor makes the initial duration conservative before exact
        # spline validation.
        approach_joint_floor = float(
            np.max(
                1.5 * approach_delta_q / velocity_limits
            )
        )

        segment_durations[0] = max(
            self.approach_duration_sec,
            approach_joint_floor,
            self.min_segment_duration_sec,
        )

        # --------------------------------------------------------------
        # Printing segments
        # --------------------------------------------------------------
        cartesian_segment_distances_mm = np.zeros(
            max(0, q_print.shape[0] - 1),
            dtype=float,
        )

        for print_index in range(1, q_print.shape[0]):
            distance_mm = float(
                np.linalg.norm(
                    tcp_positions[print_index]
                    - tcp_positions[print_index - 1]
                )
                * 1000.0
            )

            delta_q = np.abs(
                q_print[print_index]
                - q_print[print_index - 1]
            )

            dt_cartesian = (
                distance_mm / effective_print_speed_mm_s
            )

            # Conservative initial joint-speed floor.  Exact cubic peak
            # velocity is validated below.
            dt_joint = float(
                np.max(
                    1.5 * delta_q / velocity_limits
                )
            )

            duration = max(
                dt_cartesian,
                dt_joint,
                self.min_segment_duration_sec,
            )

            if not math.isfinite(duration) or duration <= 0.0:
                raise ValueError(
                    "Invalid segment duration at print waypoint "
                    f"{print_index}: {duration}"
                )

            segment_durations[print_index] = duration
            cartesian_segment_distances_mm[
                print_index - 1
            ] = distance_mm

        initial_segment_durations = segment_durations.copy()
        retimed_mask = np.zeros(
            segment_durations.size,
            dtype=bool,
        )

        final_velocities = None
        final_segment_velocity = None
        final_segment_acceleration = None
        retiming_iterations = 0

        # --------------------------------------------------------------
        # Stable local spline-velocity retiming.
        #
        # Knot velocities depend on BOTH adjacent segment durations.
        # Therefore when a segment violates a velocity limit, stretch the
        # violating segment and its immediate neighbours together.  This
        # avoids the non-convergent boundary oscillation seen in the previous
        # single-segment retimer.
        # --------------------------------------------------------------
        for iteration in range(
            self.retiming_max_iterations + 1
        ):
            times = np.concatenate(
                (
                    np.array([0.0], dtype=float),
                    np.cumsum(segment_durations),
                )
            )

            velocities = self._compute_monotone_knot_velocities(
                q_execution,
                times,
            )

            (
                segment_velocity,
                segment_acceleration,
            ) = self._cubic_hermite_segment_peak_limits(
                q_execution,
                times,
                velocities,
            )

            velocity_ratio = np.max(
                segment_velocity / velocity_limits[None, :],
                axis=1,
            )

            violating = velocity_ratio > (1.0 + 1e-6)

            if not np.any(violating):
                final_velocities = velocities
                final_segment_velocity = segment_velocity
                final_segment_acceleration = segment_acceleration
                retiming_iterations = iteration
                break

            if iteration >= self.retiming_max_iterations:
                worst_segment = int(
                    np.argmax(velocity_ratio)
                )
                raise RuntimeError(
                    "Joint-velocity retiming did not converge within "
                    f"{self.retiming_max_iterations} iterations; "
                    f"worst segment={worst_segment}, "
                    f"velocity ratio={velocity_ratio[worst_segment]:.4f}"
                )

            direct_scale = np.ones_like(velocity_ratio)
            direct_scale[violating] = (
                velocity_ratio[violating]
                * self.retiming_margin
            )

            # Expand each correction one segment in both directions because
            # PCHIP-style knot velocities are shared by neighbouring segments.
            expanded_scale = direct_scale.copy()

            if expanded_scale.size > 1:
                expanded_scale[:-1] = np.maximum(
                    expanded_scale[:-1],
                    direct_scale[1:],
                )
                expanded_scale[1:] = np.maximum(
                    expanded_scale[1:],
                    direct_scale[:-1],
                )

            segment_durations *= expanded_scale
            retimed_mask |= expanded_scale > (1.0 + 1e-12)

        if (
            final_velocities is None
            or final_segment_velocity is None
            or final_segment_acceleration is None
        ):
            raise RuntimeError(
                "Internal trajectory retiming state is incomplete"
            )

        times = np.concatenate(
            (
                np.array([0.0], dtype=float),
                np.cumsum(segment_durations),
            )
        )

        max_velocity = np.max(
            final_segment_velocity,
            axis=0,
        )
        max_acceleration = np.max(
            final_segment_acceleration,
            axis=0,
        )

        if np.any(
            max_velocity
            > velocity_limits * (1.0 + 1e-6)
        ):
            raise RuntimeError(
                "Local retiming failed to satisfy joint velocity limits"
            )

        if times[-1] > MAX_ROS_DURATION_SECONDS:
            raise RuntimeError(
                "Trajectory duration exceeds ROS Duration.sec representation"
            )

        trajectory = JointTrajectory()
        trajectory.joint_names = list(JOINT_NAMES)

        # The first published point is the first print pose at a strictly
        # positive approach time. The measured current state is internal only.
        for execution_index in range(
            1,
            q_execution.shape[0],
        ):
            point = JointTrajectoryPoint()
            point.positions = (
                q_execution[execution_index].tolist()
            )
            point.velocities = (
                final_velocities[execution_index].tolist()
            )

            self._set_ros_duration(
                point.time_from_start,
                float(times[execution_index]),
            )

            trajectory.points.append(point)

        final_print_durations = segment_durations[1:]

        if final_print_durations.size:
            min_print_dt = float(
                np.min(final_print_durations)
            )
            max_print_dt = float(
                np.max(final_print_durations)
            )

            segment_cartesian_speeds = np.divide(
                cartesian_segment_distances_mm,
                final_print_durations,
                out=np.zeros_like(
                    cartesian_segment_distances_mm
                ),
                where=final_print_durations > 0.0,
            )

            max_cartesian_speed_mm_s = float(
                np.max(segment_cartesian_speeds)
            )
        else:
            min_print_dt = 0.0
            max_print_dt = 0.0
            max_cartesian_speed_mm_s = 0.0

        local_scale = np.divide(
            segment_durations,
            initial_segment_durations,
            out=np.ones_like(segment_durations),
            where=initial_segment_durations > 0.0,
        )

        # Acceleration-reference exceedance is diagnostic only.  This makes it
        # explicit in logs without imposing an arbitrary physical constraint.
        acceleration_reference_ratio = np.divide(
            max_acceleration,
            acceleration_reference,
            out=np.zeros_like(max_acceleration),
            where=acceleration_reference > 0.0,
        )

        if np.any(acceleration_reference_ratio > 1.0):
            self.get_logger().warn(
                "Cubic spline peak acceleration exceeds the current software "
                "reference on at least one joint. This is diagnostic only; "
                "the reference is not a validated UR10e hardware limit."
            )

        return TrajectoryTimingResult(
            trajectory=trajectory,
            total_time_sec=float(times[-1]),
            effective_print_speed_mm_s=(
                effective_print_speed_mm_s
            ),
            approach_duration_sec=float(
                segment_durations[0]
            ),
            min_print_segment_duration_sec=(
                min_print_dt
            ),
            max_print_segment_duration_sec=(
                max_print_dt
            ),
            max_cartesian_speed_mm_s=(
                max_cartesian_speed_mm_s
            ),
            max_joint_velocity_rad_s=max_velocity,
            max_joint_acceleration_rad_s2=max_acceleration,
            retiming_iterations=retiming_iterations,
            retimed_segment_count=int(
                np.count_nonzero(retimed_mask)
            ),
            max_local_time_scale=float(
                np.max(local_scale)
            ),
        )

    def _log_trajectory_summary(
        self,
        timing: TrajectoryTimingResult,
    ) -> None:
        self.get_logger().info("=" * 68)
        self.get_logger().info(
            "UR10e ATOMSLICER JOINT TRAJECTORY SUMMARY"
        )
        self.get_logger().info(
            f"Trajectory points:              {len(timing.trajectory.points)}"
        )
        self.get_logger().info(
            f"Requested print speed:          {self.print_speed_mm_s:.3f} mm/s"
        )
        self.get_logger().info(
            f"Speed multiplier:               {self.print_speed_multiplier:.3f}"
        )
        self.get_logger().info(
            "Effective speed target:         "
            f"{timing.effective_print_speed_mm_s:.3f} mm/s"
        )
        self.get_logger().info(
            "Maximum achieved segment speed: "
            f"{timing.max_cartesian_speed_mm_s:.3f} mm/s"
        )
        self.get_logger().info(
            f"Approach duration:              {timing.approach_duration_sec:.3f} s"
        )
        self.get_logger().info(
            f"Local retiming iterations:      {timing.retiming_iterations}"
        )
        self.get_logger().info(
            f"Retimed segments:               {timing.retimed_segment_count}"
        )
        self.get_logger().info(
            f"Maximum local time scale:       {timing.max_local_time_scale:.4f}"
        )
        self.get_logger().info(
            f"Total trajectory time:          {timing.total_time_sec:.3f} s"
        )
        self.get_logger().info(
            "Print segment dt range:         "
            f"{timing.min_print_segment_duration_sec:.6f} -> "
            f"{timing.max_print_segment_duration_sec:.6f} s"
        )
        self.get_logger().info(
            "Peak joint velocity [rad/s]:    "
            + ", ".join(
                f"{name}={value:.4f}"
                for name, value in zip(
                    JOINT_NAMES,
                    timing.max_joint_velocity_rad_s,
                )
            )
        )
        self.get_logger().info(
            "Peak joint accel [rad/s^2, diagnostic]: "
            + ", ".join(
                f"{name}={value:.4f}"
                for name, value in zip(
                    JOINT_NAMES,
                    timing.max_joint_acceleration_rad_s2,
                )
            )
        )
        self.get_logger().info(
            "Execution boundary topic:       "
            f"{self.trajectory_command_topic}"
        )
        self.get_logger().info("=" * 68)

    def _shortest_angular_delta(delta_rad: float) -> float:
        """Return the signed shortest equivalent angular change in [-pi, pi]."""
        return math.atan2(math.sin(delta_rad), math.cos(delta_rad))

    def _build_continuity_diagnostics(
        self,
        source_indices: list[int],
        joint_path: list[np.ndarray],
        tcp_transforms: list[np.ndarray],
    ) -> tuple[list[ContinuityDiagnosticEvent], int]:
        """Classify every adjacent joint-space step above the warning threshold.

        Classification rules are intentionally conservative:

        JOINT_LIMIT_WRAP
            - raw joint step exceeds the warning threshold;
            - the same revolute motion modulo 2*pi is small enough to satisfy
              the warning threshold; and
            - either endpoint is within ``continuity_joint_limit_margin_rad``
              of a nominal joint limit.

        GENUINE_DISCONTINUITY
            - every other warning.  This includes real large modulo motion,
              branch changes, or large TCP-orientation changes.

        Non-contiguous source poses are not compared because a missing
        reachability point would make the adjacent-step interpretation invalid.
        """
        if not (
            len(source_indices)
            == len(joint_path)
            == len(tcp_transforms)
        ):
            raise ValueError(
                "Continuity diagnostics require aligned source indices, "
                "joint solutions, and TCP transforms."
            )

        qlim = np.asarray(self.kinematics.robot.qlim, dtype=float)
        if qlim.shape != (2, len(JOINT_NAMES)):
            raise RuntimeError("UR10e joint-limit matrix is malformed.")

        events: list[ContinuityDiagnosticEvent] = []
        skipped_noncontiguous_pairs = 0

        for path_index in range(1, len(joint_path)):
            previous_source_index = source_indices[path_index - 1]
            source_index = source_indices[path_index]

            if (
                source_index - previous_source_index
                != self.analysis_stride
            ):
                skipped_noncontiguous_pairs += 1
                continue

            q_previous = np.asarray(
                joint_path[path_index - 1],
                dtype=float,
            )
            q_current = np.asarray(
                joint_path[path_index],
                dtype=float,
            )

            delta = q_current - q_previous
            absolute_delta = np.abs(delta)
            joint_index = int(np.argmax(absolute_delta))
            max_step = float(absolute_delta[joint_index])

            if max_step <= self.continuity_warning_threshold_rad:
                continue

            raw_delta = float(delta[joint_index])
            wrapped_delta = self._shortest_angular_delta(raw_delta)

            lower_limit = float(qlim[0, joint_index])
            upper_limit = float(qlim[1, joint_index])

            previous_lower_margin = (
                float(q_previous[joint_index] - lower_limit)
            )
            previous_upper_margin = (
                float(upper_limit - q_previous[joint_index])
            )
            current_lower_margin = (
                float(q_current[joint_index] - lower_limit)
            )
            current_upper_margin = (
                float(upper_limit - q_current[joint_index])
            )

            minimum_limit_margin = min(
                previous_lower_margin,
                previous_upper_margin,
                current_lower_margin,
                current_upper_margin,
            )

            is_small_modulo_step = (
                abs(wrapped_delta)
                <= self.continuity_warning_threshold_rad
            )
            is_near_joint_limit = (
                minimum_limit_margin
                <= self.continuity_joint_limit_margin_rad
            )

            classification = (
                "JOINT_LIMIT_WRAP"
                if is_small_modulo_step and is_near_joint_limit
                else "GENUINE_DISCONTINUITY"
            )

            previous_tcp = tcp_transforms[path_index - 1]
            current_tcp = tcp_transforms[path_index]

            tcp_position_step_m = float(
                np.linalg.norm(
                    current_tcp[:3, 3] - previous_tcp[:3, 3]
                )
            )
            tcp_orientation_step_rad = self._rotation_error_rad(
                previous_tcp[:3, :3],
                current_tcp[:3, :3],
            )

            events.append(
                ContinuityDiagnosticEvent(
                    previous_source_index=previous_source_index,
                    source_index=source_index,
                    joint_index=joint_index,
                    joint_name=JOINT_NAMES[joint_index],
                    classification=classification,
                    q_previous_rad=float(q_previous[joint_index]),
                    q_current_rad=float(q_current[joint_index]),
                    raw_delta_rad=raw_delta,
                    wrapped_delta_rad=wrapped_delta,
                    previous_lower_limit_margin_rad=previous_lower_margin,
                    previous_upper_limit_margin_rad=previous_upper_margin,
                    current_lower_limit_margin_rad=current_lower_margin,
                    current_upper_limit_margin_rad=current_upper_margin,
                    minimum_limit_margin_rad=minimum_limit_margin,
                    tcp_position_step_m=tcp_position_step_m,
                    tcp_orientation_step_rad=tcp_orientation_step_rad,
                    tcp_x_m=float(current_tcp[0, 3]),
                    tcp_y_m=float(current_tcp[1, 3]),
                    tcp_z_m=float(current_tcp[2, 3]),
                )
            )

        return events, skipped_noncontiguous_pairs

    def _write_continuity_csv(
        self,
        events: list[ContinuityDiagnosticEvent],
    ) -> Optional[Path]:
        """Write continuity warnings to CSV and return the resolved path."""
        if not self.continuity_csv_path:
            return None

        output_path = Path(
            self.continuity_csv_path
        ).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "previous_source_index",
            "source_index",
            "classification",
            "joint_index",
            "joint_name",
            "q_previous_rad",
            "q_current_rad",
            "raw_delta_rad",
            "raw_delta_deg",
            "wrapped_delta_rad",
            "wrapped_delta_deg",
            "previous_lower_limit_margin_rad",
            "previous_upper_limit_margin_rad",
            "current_lower_limit_margin_rad",
            "current_upper_limit_margin_rad",
            "minimum_limit_margin_rad",
            "tcp_position_step_m",
            "tcp_position_step_mm",
            "tcp_orientation_step_rad",
            "tcp_orientation_step_deg",
            "tcp_x_m",
            "tcp_y_m",
            "tcp_z_m",
        ]

        with output_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=fieldnames,
            )
            writer.writeheader()

            for event in events:
                writer.writerow(
                    {
                        "previous_source_index":
                            event.previous_source_index,
                        "source_index":
                            event.source_index,
                        "classification":
                            event.classification,
                        "joint_index":
                            event.joint_index,
                        "joint_name":
                            event.joint_name,
                        "q_previous_rad":
                            f"{event.q_previous_rad:.12f}",
                        "q_current_rad":
                            f"{event.q_current_rad:.12f}",
                        "raw_delta_rad":
                            f"{event.raw_delta_rad:.12f}",
                        "raw_delta_deg":
                            f"{math.degrees(event.raw_delta_rad):.9f}",
                        "wrapped_delta_rad":
                            f"{event.wrapped_delta_rad:.12f}",
                        "wrapped_delta_deg":
                            f"{math.degrees(event.wrapped_delta_rad):.9f}",
                        "previous_lower_limit_margin_rad":
                            f"{event.previous_lower_limit_margin_rad:.12f}",
                        "previous_upper_limit_margin_rad":
                            f"{event.previous_upper_limit_margin_rad:.12f}",
                        "current_lower_limit_margin_rad":
                            f"{event.current_lower_limit_margin_rad:.12f}",
                        "current_upper_limit_margin_rad":
                            f"{event.current_upper_limit_margin_rad:.12f}",
                        "minimum_limit_margin_rad":
                            f"{event.minimum_limit_margin_rad:.12f}",
                        "tcp_position_step_m":
                            f"{event.tcp_position_step_m:.12f}",
                        "tcp_position_step_mm":
                            f"{event.tcp_position_step_m * 1000.0:.9f}",
                        "tcp_orientation_step_rad":
                            f"{event.tcp_orientation_step_rad:.12f}",
                        "tcp_orientation_step_deg":
                            f"{math.degrees(event.tcp_orientation_step_rad):.9f}",
                        "tcp_x_m":
                            f"{event.tcp_x_m:.12f}",
                        "tcp_y_m":
                            f"{event.tcp_y_m:.12f}",
                        "tcp_z_m":
                            f"{event.tcp_z_m:.12f}",
                    }
                )

        return output_path

    def _publish_continuity_markers(
        self,
        events: list[ContinuityDiagnosticEvent],
    ) -> None:
        """Publish real-scale continuity warning locations for RViz."""
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        wrap_marker = Marker()
        wrap_marker.header.frame_id = self.base_frame
        wrap_marker.header.stamp = stamp
        wrap_marker.ns = "atomslicer_continuity"
        wrap_marker.id = 0
        wrap_marker.type = Marker.POINTS
        wrap_marker.action = Marker.ADD
        wrap_marker.pose.orientation.w = 1.0
        wrap_marker.scale.x = self.continuity_marker_size_m
        wrap_marker.scale.y = self.continuity_marker_size_m
        # Orange: joint-limit / +/-2*pi seam.
        wrap_marker.color.r = 1.0
        wrap_marker.color.g = 0.55
        wrap_marker.color.b = 0.05
        wrap_marker.color.a = 1.0

        discontinuity_marker = Marker()
        discontinuity_marker.header.frame_id = self.base_frame
        discontinuity_marker.header.stamp = stamp
        discontinuity_marker.ns = "atomslicer_continuity"
        discontinuity_marker.id = 1
        discontinuity_marker.type = Marker.POINTS
        discontinuity_marker.action = Marker.ADD
        discontinuity_marker.pose.orientation.w = 1.0
        discontinuity_marker.scale.x = self.continuity_marker_size_m
        discontinuity_marker.scale.y = self.continuity_marker_size_m
        # Magenta: genuinely large modulo joint-space/TCP change.
        discontinuity_marker.color.r = 0.95
        discontinuity_marker.color.g = 0.10
        discontinuity_marker.color.b = 0.80
        discontinuity_marker.color.a = 1.0

        for event in events:
            point = Point()
            point.x = event.tcp_x_m
            point.y = event.tcp_y_m
            point.z = event.tcp_z_m

            if event.classification == "JOINT_LIMIT_WRAP":
                wrap_marker.points.append(point)
            else:
                discontinuity_marker.points.append(point)

        marker_array.markers = [
            wrap_marker,
            discontinuity_marker,
        ]
        self.continuity_marker_pub.publish(marker_array)

    def _log_continuity_diagnostics(
        self,
        events: list[ContinuityDiagnosticEvent],
        skipped_noncontiguous_pairs: int,
        csv_path: Optional[Path],
    ) -> None:
        joint_limit_wraps = sum(
            event.classification == "JOINT_LIMIT_WRAP"
            for event in events
        )
        genuine_discontinuities = (
            len(events) - joint_limit_wraps
        )

        self.get_logger().info("-" * 68)
        self.get_logger().info("JOINT CONTINUITY DIAGNOSTICS")
        self.get_logger().info(
            f"Warning steps analysed:         {len(events)}"
        )
        self.get_logger().info(
            f"Joint-limit wrap events:       {joint_limit_wraps}"
        )
        self.get_logger().info(
            f"Genuine discontinuities:       {genuine_discontinuities}"
        )
        self.get_logger().info(
            "Non-contiguous pairs skipped:   "
            f"{skipped_noncontiguous_pairs}"
        )

        if csv_path is not None:
            self.get_logger().info(
                f"Continuity CSV:                {csv_path}"
            )

        if events and self.continuity_log_count > 0:
            worst_events = sorted(
                events,
                key=lambda event: abs(event.raw_delta_rad),
                reverse=True,
            )[: self.continuity_log_count]

            self.get_logger().info(
                "Worst continuity events:"
            )
            for event in worst_events:
                self.get_logger().info(
                    "  "
                    f"{event.previous_source_index}->{event.source_index} "
                    f"{event.joint_name}: "
                    f"raw={event.raw_delta_rad:+.6f} rad "
                    f"({math.degrees(event.raw_delta_rad):+.3f} deg), "
                    f"wrapped={event.wrapped_delta_rad:+.6f} rad "
                    f"({math.degrees(event.wrapped_delta_rad):+.3f} deg), "
                    f"limit_margin={event.minimum_limit_margin_rad:.6f} rad, "
                    f"TCP_step={event.tcp_position_step_m * 1000.0:.4f} mm / "
                    f"{math.degrees(event.tcp_orientation_step_rad):.4f} deg, "
                    f"{event.classification}"
                )

        self.get_logger().info("-" * 68)

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------
    def _joint_state_callback(self, msg: JointState) -> None:
        if len(msg.name) != len(msg.position):
            return

        joint_map = dict(zip(msg.name, msg.position))
        if any(name not in joint_map for name in JOINT_NAMES):
            return

        q = np.array([joint_map[name] for name in JOINT_NAMES], dtype=float)
        if not np.all(np.isfinite(q)):
            return

        self.current_q = q

    def _tcp_pose_callback(self, msg: PoseArray) -> None:
        # AtomSlicer republishes the same path periodically.  Analysis is
        # intentionally one-shot per node process to prevent 16k-pose IK from
        # being restarted every second. Restart this node to re-run after a TF
        # or tool-offset change.
        if self.analysis_started or self.analysis_complete:
            return

        if self.current_q is None:
            self.get_logger().warn(
                "TCP path received, but no complete /joint_states sample is available yet. "
                "Waiting for the next path publication."
            )
            return

        if not msg.header.frame_id:
            self.get_logger().error("Rejected TCP PoseArray with an empty frame_id.")
            return

        if not msg.poses:
            self.get_logger().error("Rejected empty TCP PoseArray.")
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                msg.header.frame_id,
                Time(),
                timeout=Duration(seconds=self.tf_timeout_sec),
            )
        except Exception as exc:
            self.get_logger().warn(
                f"TF {self.base_frame} <- {msg.header.frame_id} is not available yet: {exc}"
            )
            return

        base_from_source = self._transform_msg_to_matrix(transform.transform)
        if base_from_source is None:
            self.get_logger().error("Received invalid TF transform; analysis aborted.")
            return

        self.analysis_started = True
        try:
            self._run_analysis(msg, base_from_source)
            self.analysis_complete = True
        except Exception as exc:
            # Once a full analysis/planning attempt has started, a deterministic
            # failure should not trigger another 16k-pose IK solve every time
            # atomslicer_path_node republishes its PoseArray. Restart this node
            # explicitly after correcting the cause.
            self.analysis_complete = True
            self.get_logger().error(
                f"AtomSlicer planning aborted: {exc}"
            )
            self.get_logger().error(
                "Planner is now latched after failure; restart "
                "atomslicer_trajectory to retry."
            )
            return
        finally:
            self.analysis_started = False

    # ------------------------------------------------------------------
    # Reachability analysis
    # ------------------------------------------------------------------
    def _run_analysis(self, msg: PoseArray, base_from_source: np.ndarray) -> None:
        source_indices = list(range(0, len(msg.poses), self.analysis_stride))
        if self.max_poses > 0:
            source_indices = source_indices[: self.max_poses]

        if not source_indices:
            raise RuntimeError("No poses selected for analysis")

        self.get_logger().info(
            f"Starting UR10e reachability analysis for {len(source_indices)} of "
            f"{len(msg.poses)} TCP poses ({msg.header.frame_id} -> {self.base_frame})."
        )

        start_time = time.perf_counter()
        initial_q = self.current_q.copy()
        previous_q = initial_q.copy()
        previous_reachable_source_index: Optional[int] = None

        # Deterministic fallback seeds are used only if the continuity seed
        # fails. They improve reachability classification without random search.
        fallback_seeds = (
            initial_q,
            np.asarray(self.kinematics.robot.qz, dtype=float),
            np.asarray(self.kinematics.robot.qr, dtype=float),
        )

        transformed_array = PoseArray()
        transformed_array.header.frame_id = self.base_frame
        transformed_array.header.stamp = self.get_clock().now().to_msg()

        reachable_points: list[Point] = []
        unreachable_points: list[Point] = []

        # Kept aligned 1:1 with the accepted source indices and joint
        # solutions so continuity diagnostics can correlate joint behaviour
        # with the Cartesian TCP path.
        reachable_tcp_transforms: list[np.ndarray] = []

        self.reachable_source_indices = []
        self.reachable_joint_solutions = []

        invalid_pose_count = 0
        ik_failure_count = 0
        fk_failure_count = 0
        alternate_seed_success_count = 0
        python_solver_success_count = 0

        max_position_error = 0.0
        max_orientation_error = 0.0
        max_solver_residual = 0.0
        max_raw_joint_step = 0.0
        max_raw_joint_step_index: Optional[int] = None

        failed_indices: list[int] = []

        for selected_number, source_index in enumerate(source_indices, start=1):
            source_pose = msg.poses[source_index]
            source_from_tcp = self._pose_to_matrix(source_pose)

            if source_from_tcp is None:
                invalid_pose_count += 1
                failed_indices.append(source_index)
                continue

            base_from_tool = base_from_source @ source_from_tcp @ self.tool_transform
            transformed_pose = self._matrix_to_pose(base_from_tool)
            transformed_array.poses.append(transformed_pose)

            desired_pose = sm.SE3(base_from_tool)
            ik_result = self.kinematics.solve_ik(
                desired_pose,
                seed_q=previous_q,
                fallback_seeds=fallback_seeds,
                residual_tolerance=self.ik_residual_tolerance,
                iteration_limit=self.ik_iteration_limit,
            )

            if not ik_result.success or ik_result.q is None:
                ik_failure_count += 1
                failed_indices.append(source_index)
                unreachable_points.append(self._point_from_pose(transformed_pose))
                continue

            raw_q = ik_result.q
            q = self.kinematics.nearest_equivalent_configuration(
                raw_q,
                previous_q,
            )

            if ik_result.seed_index > 0:
                alternate_seed_success_count += 1
            if ik_result.solver == "ikine_LM":
                python_solver_success_count += 1

            if (
                ik_result.residual is not None
                and math.isfinite(ik_result.residual)
            ):
                max_solver_residual = max(
                    max_solver_residual,
                    float(ik_result.residual),
                )

            fk_matrix = np.asarray(self.kinematics.robot.fkine(q).A, dtype=float)
            position_error = float(
                np.linalg.norm(fk_matrix[:3, 3] - base_from_tool[:3, 3])
            )
            orientation_error = self._rotation_error_rad(
                fk_matrix[:3, :3],
                base_from_tool[:3, :3],
            )

            max_position_error = max(max_position_error, position_error)
            max_orientation_error = max(max_orientation_error, orientation_error)

            if (
                position_error > self.fk_position_tolerance_m
                or orientation_error > self.fk_orientation_tolerance_rad
            ):
                fk_failure_count += 1
                failed_indices.append(source_index)
                unreachable_points.append(self._point_from_pose(transformed_pose))
                continue

            if (
                previous_reachable_source_index is not None
                and source_index == previous_reachable_source_index + self.analysis_stride
            ):
                raw_joint_step = float(np.max(np.abs(raw_q - previous_q)))
                if raw_joint_step > max_raw_joint_step:
                    max_raw_joint_step = raw_joint_step
                    max_raw_joint_step_index = source_index


            self.reachable_source_indices.append(source_index)
            self.reachable_joint_solutions.append(q.copy())
            reachable_tcp_transforms.append(base_from_tool.copy())
            reachable_points.append(self._point_from_pose(transformed_pose))

            previous_q = q.copy()
            previous_reachable_source_index = source_index

            if (
                self.progress_interval > 0
                and selected_number % self.progress_interval == 0
            ):
                self.get_logger().info(
                    f"Reachability progress: {selected_number}/{len(source_indices)} poses; "
                    f"reachable={len(self.reachable_joint_solutions)}, "
                    f"failed={len(failed_indices)}."
                )

        # ------------------------------------------------------------------
        # Whole-path joint representation optimisation
        # ------------------------------------------------------------------
        # All accepted IK poses are now known.  Choose +/-2*pi equivalent joint
        # representations over the COMPLETE path rather than greedily per
        # waypoint.  This is necessary near the nominal +/-2*pi limits.
        continuity_result = self.kinematics.optimise_equivalent_joint_path(
            self.reachable_joint_solutions,
            initial_q=initial_q,
            warning_threshold_rad=self.continuity_warning_threshold_rad,
        )

        self.reachable_joint_solutions = [
            q.copy() for q in continuity_result.q_path
        ]

        # Whole-print +/-2*pi shifts do not change FK or any printing joint
        # delta.  Use that remaining freedom to place the complete path on the
        # representation closest to the actual robot start state.
        (
            self.reachable_joint_solutions,
            whole_path_wrap_shifts,
        ) = self._anchor_equivalent_joint_path_to_initial_state(
            self.reachable_joint_solutions,
            initial_q,
        )

        if np.any(np.abs(whole_path_wrap_shifts) > 1e-9):
            self.get_logger().info(
                "Applied whole-path equivalent joint shifts [rad]: "
                + ", ".join(
                    f"{name}={shift:+.6f}"
                    for name, shift in zip(
                        JOINT_NAMES,
                        whole_path_wrap_shifts,
                    )
                )
            )

        continuity_events, skipped_noncontiguous_pairs = (
            self._build_continuity_diagnostics(
                self.reachable_source_indices,
                self.reachable_joint_solutions,
                reachable_tcp_transforms,
            )
        )
        continuity_csv_path = self._write_continuity_csv(
            continuity_events
        )

        # ------------------------------------------------------------------
        # Final trajectory generation / handoff to existing move_ur boundary
        # ------------------------------------------------------------------
        timing_result: Optional[TrajectoryTimingResult] = None
        full_path_selected = (
            self.analysis_stride == 1
            and self.max_poses == 0
            and len(source_indices) == len(msg.poses)
        )
        all_selected_poses_valid = (
            len(self.reachable_joint_solutions) == len(source_indices)
            and not failed_indices
        )
        continuity_valid = (
            continuity_result.continuity_warning_count == 0
            and not continuity_events
            and skipped_noncontiguous_pairs == 0
        )

        if not full_path_selected:
            self.get_logger().warn(
                "JointTrajectory not generated: analysis_stride/max_poses selected a "
                "debug subset rather than the complete AtomSlicer path."
            )
        elif not all_selected_poses_valid:
            self.get_logger().error(
                "JointTrajectory not generated: not every AtomSlicer TCP pose passed "
                "IK/FK validation."
            )
        elif not continuity_valid:
            self.get_logger().error(
                "JointTrajectory not generated: joint-path continuity diagnostics "
                "contain warnings or non-contiguous source poses."
            )
        else:
            current_q_now = (
                self.current_q.copy()
                if self.current_q is not None
                else None
            )
            if current_q_now is None:
                self.get_logger().error(
                    "JointTrajectory not generated: /joint_states became unavailable."
                )
            else:
                start_state_drift = float(
                    np.max(np.abs(current_q_now - initial_q))
                )
                if start_state_drift > self.start_state_tolerance_rad:
                    self.get_logger().error(
                        "JointTrajectory not generated: robot moved during planning; "
                        f"max start-state drift={start_state_drift:.6f} rad exceeds "
                        f"{self.start_state_tolerance_rad:.6f} rad."
                    )
                else:
                    timing_result = self._build_timed_joint_trajectory(
                        initial_q,
                        self.reachable_joint_solutions,
                        reachable_tcp_transforms,
                    )
                    self.generated_trajectory = timing_result.trajectory

                    if self.publish_joint_trajectory:
                        self.trajectory_command_pub.publish(
                            timing_result.trajectory
                        )
                        self.get_logger().info(
                            "Published validated AtomSlicer JointTrajectory to the "
                            "existing move_ur execution boundary."
                        )
                    else:
                        self.get_logger().warn(
                            "Validated JointTrajectory generated but publication is disabled "
                            "by publish_joint_trajectory:=false."
                        )

        elapsed = time.perf_counter() - start_time

        # Publish diagnostic geometry only after the complete result is known,
        # so RViz never displays a partial analysis as if it were final.
        self.transformed_pose_pub.publish(transformed_array)
        self._publish_reachability_markers(
            reachable_points,
            unreachable_points,
        )
        self._publish_continuity_markers(continuity_events)

        total = len(source_indices)
        reachable = len(self.reachable_joint_solutions)
        unreachable = total - reachable
        percentage = 100.0 * reachable / total

        self.get_logger().info("=" * 68)
        self.get_logger().info("UR10e ATOMSLICER REACHABILITY SUMMARY")
        self.get_logger().info(f"Input TCP poses selected:      {total}")
        self.get_logger().info(f"Reachable:                     {reachable}")
        self.get_logger().info(f"Unreachable:                   {unreachable}")
        self.get_logger().info(f"Invalid input poses:           {invalid_pose_count}")
        self.get_logger().info(f"IK failures:                   {ik_failure_count}")
        self.get_logger().info(f"FK verification failures:      {fk_failure_count}")
        self.get_logger().info(f"Reachability:                  {percentage:.3f}%")
        self.get_logger().info(
            f"Max FK position error:         {max_position_error * 1000.0:.4f} mm"
        )
        self.get_logger().info(
            f"Max FK orientation error:      {math.degrees(max_orientation_error):.4f} deg"
        )
        self.get_logger().info(
            f"Alternate-seed IK successes:   {alternate_seed_success_count}"
        )
        self.get_logger().info(
            f"Python LM solver successes:     {python_solver_success_count}"
        )
        self.get_logger().info(
            f"Max solver residual:           {max_solver_residual:.3e}"
        )
        self.get_logger().info(
            f"Max raw consecutive joint step: {max_raw_joint_step:.6f} rad"
            + (
                f" (at source pose {max_raw_joint_step_index})"
                if max_raw_joint_step_index is not None
                else ""
            )
        )
        continuity_source_index = (
            self.reachable_source_indices[
                continuity_result.max_step_waypoint
            ]
            if self.reachable_source_indices
            else -1
        )
        continuity_joint_name = JOINT_NAMES[
            continuity_result.max_step_joint_index
        ]

        self.get_logger().info(
            "Max globally-normalised joint step: "
            f"{continuity_result.max_step_rad:.6f} rad "
            f"({math.degrees(continuity_result.max_step_rad):.3f} deg) "
            f"at source pose {continuity_source_index}, "
            f"joint={continuity_joint_name}"
        )
        self.get_logger().info(
            "Per-joint max steps [rad]:       "
            + ", ".join(
                f"{name}={value:.6f}"
                for name, value in zip(
                    JOINT_NAMES,
                    continuity_result.per_joint_max_step_rad,
                )
            )
        )
        self.get_logger().info(
            "Continuity warnings (> "
            f"{self.continuity_warning_threshold_rad:.3f} rad): "
            f"{continuity_result.continuity_warning_count}"
        )

        if (
            continuity_result.continuity_warning_count
            != len(continuity_events)
        ):
            self.get_logger().warn(
                "Continuity warning-count mismatch: global optimiser reported "
                f"{continuity_result.continuity_warning_count}, diagnostic pass "
                f"classified {len(continuity_events)}. This is expected only "
                "when non-contiguous reachable source poses are skipped."
            )

        self._log_continuity_diagnostics(
            continuity_events,
            skipped_noncontiguous_pairs,
            continuity_csv_path,
        )

        if timing_result is not None:
            self._log_trajectory_summary(timing_result)

        self.get_logger().info(f"Planning time:                 {elapsed:.2f} s")

        if failed_indices:
            preview = ", ".join(str(index) for index in failed_indices[:20])
            suffix = " ..." if len(failed_indices) > 20 else ""
            self.get_logger().warn(
                f"First unreachable/invalid source indices: {preview}{suffix}"
            )

        if timing_result is not None:
            if self.publish_joint_trajectory:
                self.get_logger().info(
                    "Planning complete. The JointTrajectory was published only to the "
                    "move_ur execution boundary; controller execution still requires "
                    "accept_joint_trajectory_commands:=true in move_ur."
                )
            else:
                self.get_logger().info(
                    "Planning complete. A validated JointTrajectory is stored in memory; "
                    "publication is disabled."
                )
        else:
            self.get_logger().warn(
                "Planning finished without an executable JointTrajectory; see validation "
                "messages above."
            )
        self.get_logger().info("=" * 68)

    @staticmethod
    def _point_from_pose(pose: Pose) -> Point:
        point = Point()
        point.x = pose.position.x
        point.y = pose.position.y
        point.z = pose.position.z
        return point

    def _publish_reachability_markers(
        self,
        reachable_points: list[Point],
        unreachable_points: list[Point],
    ) -> None:
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        reachable = Marker()
        reachable.header.frame_id = self.base_frame
        reachable.header.stamp = stamp
        reachable.ns = "atomslicer_reachability"
        reachable.id = 0
        reachable.type = Marker.POINTS
        reachable.action = Marker.ADD
        reachable.pose.orientation.w = 1.0
        reachable.scale.x = self.marker_size_m
        reachable.scale.y = self.marker_size_m
        reachable.color.r = 0.1
        reachable.color.g = 0.9
        reachable.color.b = 0.2
        reachable.color.a = 1.0
        reachable.points = reachable_points

        unreachable = Marker()
        unreachable.header.frame_id = self.base_frame
        unreachable.header.stamp = stamp
        unreachable.ns = "atomslicer_reachability"
        unreachable.id = 1
        unreachable.type = Marker.POINTS
        unreachable.action = Marker.ADD
        unreachable.pose.orientation.w = 1.0
        unreachable.scale.x = self.marker_size_m
        unreachable.scale.y = self.marker_size_m
        unreachable.color.r = 0.95
        unreachable.color.g = 0.1
        unreachable.color.b = 0.1
        unreachable.color.a = 1.0
        unreachable.points = unreachable_points

        marker_array.markers = [reachable, unreachable]
        self.reachability_marker_pub.publish(marker_array)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AtomSlicerTrajectory()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
