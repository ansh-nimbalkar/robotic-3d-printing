import math
import sys
from typing import Optional, Sequence

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32, Float32MultiArray, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import roboticstoolbox as rtb
from roboticstoolbox import DHRobot, RevoluteDH
import spatialmath as sm


JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

# builtin_interfaces/Duration.sec is a signed 32-bit integer. This limit is
# far beyond any trajectory duration expected here, but validating it avoids
# creating malformed ROS messages if a bad command is received.
MAX_ROS_DURATION_SECONDS = (2**31) - 1


# Reliable command topics.
RELIABLE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

# Joint states are conventionally published best-effort by the UR driver.
BEST_EFFORT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
)


class URcontrol(Node):
    """UR10e execution node used by the existing printing workflow.

    The legacy single-pose interfaces are intentionally preserved. This node
    additionally exposes an optional, safety-gated multi-point JointTrajectory
    input for the AtomSlicer trajectory planner that will be added next.

    The node remains the sole owner of the controller command topic so that
    legacy G-code execution and future AtomSlicer execution share the same
    robot-control boundary.
    """

    def __init__(self):
        super().__init__("ur_control_node")

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.declare_parameter("ur_type", "UR10e")
        self.declare_parameter("default_duration", 3.0)
        self.declare_parameter("joint_tolerance", 0.001)
        self.declare_parameter(
            "controller_topic",
            "/scaled_joint_trajectory_controller/joint_trajectory",
        )
        self.declare_parameter(
            "trajectory_command_topic",
            "/ur10e/target_joint_trajectory",
        )
        self.declare_parameter("accept_joint_trajectory_commands", False)

        self.ur_type = self.get_parameter("ur_type").value
        self.duration = float(self.get_parameter("default_duration").value)
        self.joint_tolerance = float(self.get_parameter("joint_tolerance").value)
        self.controller_topic = self.get_parameter("controller_topic").value
        self.trajectory_command_topic = self.get_parameter(
            "trajectory_command_topic"
        ).value
        self.accept_joint_trajectory_commands = bool(
            self.get_parameter("accept_joint_trajectory_commands").value
        )

        if not self._is_valid_duration(self.duration):
            raise ValueError("default_duration must be finite and greater than zero")
        if not math.isfinite(self.joint_tolerance) or self.joint_tolerance <= 0.0:
            raise ValueError("joint_tolerance must be finite and greater than zero")

        self.get_logger().info(
            f"UR control node started for {self.ur_type}; controller={self.controller_topic}"
        )

        # ------------------------------------------------------------------
        # Runtime state
        # ------------------------------------------------------------------
        self.current_xyz = np.zeros(3, dtype=float)
        self.is_moving = False
        self.target_q: Optional[np.ndarray] = None
        self.has_joint_state = False

        self.toggle_log = True
        self.shutdown_requested = False
        self.shutdown_delay = 4.0
        self.shutdown_timer = None

        # ------------------------------------------------------------------
        # Robotics Toolbox models
        # ------------------------------------------------------------------
        # RTB's standard model catalogue exposes the legacy UR10 as ``UR10``
        # but does not expose a ``UR10e`` DH model in common releases.  Using
        # the legacy UR10 model as a silent alias would be kinematically wrong:
        # the UR10e has different manufacturer DH dimensions.
        #
        # Build the requested robot explicitly so IK/FK are based on the
        # correct nominal UR10e geometry.  The two model instances preserve the
        # existing repository architecture:
        #   - robot_DH: numerical IK and joint-limit checks
        #   - robot:    current-state FK/reporting
        self.robot_DH, self.robot = self._create_robot_models(self.ur_type)

        initial_q = np.asarray(self.robot_DH.q, dtype=float).copy()
        self.robot.q = initial_q.copy()
        self.robot_DH.q = initial_q.copy()

        self.get_logger().info(
            "Using nominal UR10e DH kinematics from Universal Robots "
            "manufacturer parameters."
        )

        # ------------------------------------------------------------------
        # Subscribers - legacy interfaces remain unchanged
        # ------------------------------------------------------------------
        self.duration_sub = self.create_subscription(
            Float32,
            "/ur10e/movement_duration",
            self.get_duration,
            RELIABLE_QOS,
        )
        self.joint_states_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self.get_current_joint_states,
            BEST_EFFORT_QOS,
        )
        self.xyz_pose_sub = self.create_subscription(
            Twist,
            "/ur10e/point_pose",
            self.get_XYZ_pos,
            RELIABLE_QOS,
        )
        self.joint_pose_sub = self.create_subscription(
            Float32MultiArray,
            "/ur10e/target_joint_pose",
            self.get_joint_pos,
            10,
        )
        self.toggle_log_sub = self.create_subscription(
            Bool,
            "/kb/toggle_log",
            self.toggle_log_callback,
            10,
        )
        self.shutdown_sub = self.create_subscription(
            Bool,
            "/kb/shutdown",
            self.shutdown_callback,
            10,
        )

        # The multi-point interface is deliberately disabled by default. It is
        # enabled explicitly once an AtomSlicer trajectory has passed offline
        # reachability and continuity validation.
        self.trajectory_command_sub = None
        if self.accept_joint_trajectory_commands:
            self.trajectory_command_sub = self.create_subscription(
                JointTrajectory,
                self.trajectory_command_topic,
                self.get_joint_trajectory,
                RELIABLE_QOS,
            )
            self.get_logger().warn(
                "Multi-point JointTrajectory execution input is ENABLED on "
                f"{self.trajectory_command_topic}."
            )

        # ------------------------------------------------------------------
        # Publishers
        # ------------------------------------------------------------------
        self.joint_pose_pub_ = self.create_publisher(
            JointTrajectory,
            self.controller_topic,
            10,
        )
        self.goal_pub_ = self.create_publisher(
            Float32MultiArray,
            "/ur10e/joint_goal",
            10,
        )
        self.calc_pose_pub_ = self.create_publisher(
            Float32MultiArray,
            "/ur10e/calculated_xyz",
            10,
        )
        self.robot_info_pub_ = self.create_publisher(
            String,
            "/ur10e/robot_log",
            10,
        )
        self.is_moving_pub_ = self.create_publisher(
            Bool,
            "/ur10e/is_moving",
            10,
        )

        # Preserve the existing publication rates to avoid changing the legacy
        # G-code interpreter behaviour in this integration step.
        self.log_timer = self.create_timer(0.005, self.send_log_data)
        self.is_moving_timer = self.create_timer(0.002, self.publish_is_moving)

    # ----------------------------------------------------------------------
    # Generic validation helpers
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # Robot-model construction
    # ----------------------------------------------------------------------
    @staticmethod
    def _create_ur10e_dh_model() -> DHRobot:
        """Create a nominal UR10e standard-DH model.

        Robotics Toolbox commonly provides ``DH.UR10`` but not ``DH.UR10e``.
        The legacy UR10 dimensions must not be used as an alias for UR10e
        reachability or IK because the e-Series geometry differs.

        These nominal kinematic dimensions are the manufacturer UR10e/UR12e
        DH parameters, in metres:

            a = [0, -0.6127, -0.57155, 0, 0, 0]
            d = [0.1807, 0, 0, 0.17415, 0.11985, 0.11655]
            alpha = [pi/2, 0, 0, pi/2, -pi/2, 0]

        Joint limits are set to +/- 2*pi as the nominal kinematic range used
        for this solver.  Before real hardware execution, the ROS/UR driver
        model and the robot-specific calibration remain authoritative.
        """
        a = (0.0, -0.6127, -0.57155, 0.0, 0.0, 0.0)
        d = (0.1807, 0.0, 0.0, 0.17415, 0.11985, 0.11655)
        alpha = (
            math.pi / 2.0,
            0.0,
            0.0,
            math.pi / 2.0,
            -math.pi / 2.0,
            0.0,
        )

        nominal_joint_limits = np.array(
            [-2.0 * math.pi, 2.0 * math.pi],
            dtype=float,
        )

        links = [
            RevoluteDH(
                a=a_i,
                d=d_i,
                alpha=alpha_i,
                qlim=nominal_joint_limits.copy(),
            )
            for a_i, d_i, alpha_i in zip(a, d, alpha)
        ]

        robot = DHRobot(
            links,
            name="UR10e",
            manufacturer="Universal Robots",
        )

        # Keep the same convenient nominal configurations used by the RTB UR
        # family while remaining explicit about this being the e-Series model.
        robot.qz = np.zeros(6, dtype=float)
        robot.qr = np.array(
            [math.pi, 0.0, 0.0, 0.0, math.pi / 2.0, 0.0],
            dtype=float,
        )
        robot.q = robot.qz.copy()

        return robot

    @classmethod
    def _create_robot_models(cls, ur_type: str):
        """Return IK and FK model instances for the configured robot.

        The thesis execution pipeline currently targets UR10e only.  Failing
        explicitly for another robot type is safer than silently substituting
        a geometrically different RTB model.
        """
        normalized = str(ur_type).strip().lower()

        if normalized == "ur10e":
            return cls._create_ur10e_dh_model(), cls._create_ur10e_dh_model()

        raise ValueError(
            f"Unsupported ur_type '{ur_type}'. "
            "This implementation currently supports UR10e only."
        )

    @staticmethod
    def _is_valid_duration(seconds: float) -> bool:
        return (
            math.isfinite(seconds)
            and seconds > 0.0
            and seconds <= MAX_ROS_DURATION_SECONDS
        )

    @staticmethod
    def _set_ros_duration(duration_msg, seconds: float) -> None:
        """Populate a builtin_interfaces/Duration from floating-point seconds."""
        if not URcontrol._is_valid_duration(seconds):
            raise ValueError(f"Invalid duration: {seconds}")

        whole_seconds = int(math.floor(seconds))
        nanoseconds = int(round((seconds - whole_seconds) * 1_000_000_000))

        # Rounding can produce exactly 1e9 ns.
        if nanoseconds >= 1_000_000_000:
            whole_seconds += 1
            nanoseconds -= 1_000_000_000

        duration_msg.sec = whole_seconds
        duration_msg.nanosec = nanoseconds

    @staticmethod
    def _duration_to_seconds(duration_msg) -> float:
        return float(duration_msg.sec) + float(duration_msg.nanosec) * 1e-9

    @staticmethod
    def _validate_joint_vector(q: Sequence[float]) -> Optional[np.ndarray]:
        try:
            q_array = np.asarray(q, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return None

        if q_array.size != len(JOINT_NAMES):
            return None
        if not np.all(np.isfinite(q_array)):
            return None

        return q_array

    def _joint_limits_satisfied(self, q: np.ndarray) -> bool:
        """Check configured RTB joint limits when the model provides them."""
        try:
            qlim = np.asarray(self.robot_DH.qlim, dtype=float)
        except (AttributeError, TypeError, ValueError):
            return True

        if qlim.shape != (2, len(JOINT_NAMES)):
            return True

        tolerance = 1e-9
        lower = qlim[0, :]
        upper = qlim[1, :]

        lower_ok = np.logical_or(~np.isfinite(lower), q >= lower - tolerance)
        upper_ok = np.logical_or(~np.isfinite(upper), q <= upper + tolerance)
        return bool(np.all(lower_ok & upper_ok))

    @staticmethod
    def _extract_ik_result(result):
        """Compatibility adapter for old and new Robotics Toolbox IK APIs.

        Older RTB versions returned tuple-like results from the fast solver;
        newer versions return an IKSolution object. The repository does not pin
        an RTB version, so supporting both avoids a fragile environment-specific
        assumption.
        """
        if hasattr(result, "q"):
            q = result.q
            success = bool(getattr(result, "success", q is not None))
            reason = str(getattr(result, "reason", ""))
            return q, success, reason

        if isinstance(result, (tuple, list)) and len(result) >= 2:
            return result[0], bool(result[1]), ""

        return None, False, "Unsupported IK result format"

    # ----------------------------------------------------------------------
    # Existing keyboard / lifecycle callbacks
    # ----------------------------------------------------------------------
    def toggle_log_callback(self, _msg):
        self.toggle_log = not self.toggle_log

    def shutdown_callback(self, msg):
        if not msg.data:
            return

        if self.shutdown_timer is not None:
            return

        self.get_logger().info(
            f"Received shutdown signal. Exiting in {self.shutdown_delay} seconds."
        )
        self.shutdown_timer = self.create_timer(
            self.shutdown_delay,
            self.set_shutdown_flag,
        )

    def set_shutdown_flag(self):
        self.shutdown_requested = True
        if self.shutdown_timer is not None:
            self.destroy_timer(self.shutdown_timer)
            self.shutdown_timer = None

    def send_log_data(self):
        log_msg = String()
        log_msg.data = (
            f"{self.ur_type},{self.current_xyz[0]},{self.current_xyz[1]},"
            f"{self.current_xyz[2]},{self.duration}"
        )
        self.robot_info_pub_.publish(log_msg)

    # ----------------------------------------------------------------------
    # Legacy command callbacks
    # ----------------------------------------------------------------------
    def get_duration(self, data):
        requested_duration = float(data.data)
        if not self._is_valid_duration(requested_duration):
            self.get_logger().error(
                f"Rejected invalid movement duration: {requested_duration}"
            )
            return

        self.duration = requested_duration

    def get_current_joint_states(self, data):
        """Update the RTB models using joint names, never array position order."""
        if len(data.name) != len(data.position):
            self.get_logger().error(
                "Rejected /joint_states message: name and position lengths differ."
            )
            return

        joint_positions = dict(zip(data.name, data.position))
        missing = [name for name in JOINT_NAMES if name not in joint_positions]

        if missing:
            self.get_logger().error(
                "Rejected /joint_states message; missing joints: "
                + ", ".join(missing)
            )
            return

        q = np.array([joint_positions[name] for name in JOINT_NAMES], dtype=float)
        if not np.all(np.isfinite(q)):
            self.get_logger().error(
                "Rejected /joint_states message containing non-finite positions."
            )
            return

        self.robot.q = q.copy()
        self.robot_DH.q = q.copy()
        self.has_joint_state = True

        self.calculate_current_xyz()

        if self.is_moving and self.target_q is not None:
            error = float(np.linalg.norm(q - self.target_q))

            if error < self.joint_tolerance:
                if self.toggle_log:
                    self.get_logger().info(
                        f"Goal reached (joint error norm: {error:.6f} rad)."
                    )

                self.is_moving = False
                self.target_q = None

    def calculate_current_xyz(self):
        """Publish current end-effector translation in millimetres."""
        try:
            translation = np.asarray(self.robot.fkine(self.robot.q).t, dtype=float)
        except Exception as exc:  # RTB model errors should not kill the ROS node.
            self.get_logger().error(f"Forward kinematics failed: {exc}")
            return

        if translation.size != 3 or not np.all(np.isfinite(translation)):
            self.get_logger().error("Forward kinematics returned an invalid translation.")
            return

        self.current_xyz = translation * 1000.0

        pose_msg = Float32MultiArray()
        pose_msg.data = self.current_xyz.tolist()
        self.calc_pose_pub_.publish(pose_msg)

    def get_joint_pos(self, data):
        q_degrees = self._validate_joint_vector(data.data)
        if q_degrees is None:
            self.get_logger().error(
                "Rejected /ur10e/target_joint_pose: expected six finite joint values."
            )
            return

        if self.toggle_log:
            print(
                "RECEIVED JOINT TARGET:\n---\nDuration: {}s".format(self.duration)
            )
            for i, angle in enumerate(q_degrees):
                print(f"Q{i + 1}: {angle:.2f} deg")
            print("---")
            sys.stdout.flush()

        self.move_robot(np.deg2rad(q_degrees))

    def get_XYZ_pos(self, data):
        if not all(
            math.isfinite(value)
            for value in (
                data.linear.x,
                data.linear.y,
                data.linear.z,
                data.angular.x,
                data.angular.y,
                data.angular.z,
            )
        ):
            self.get_logger().error(
                "Rejected /ur10e/point_pose containing non-finite values."
            )
            return

        if data.linear.z < 0.0:
            self.get_logger().error(
                "Rejected target pose with negative Z coordinate."
            )
            return

        # Legacy point-pose interface uses millimetres and Euler degrees.
        x, y, z = [
            value / 1000.0
            for value in (data.linear.x, data.linear.y, data.linear.z)
        ]

        q = self.find_q_pose(
            x,
            y,
            z,
            data.angular.x,
            data.angular.y,
            data.angular.z,
        )

        if q is None:
            self.get_logger().error(
                "Target Cartesian pose is not reachable with the configured UR10e model."
            )
            return

        if self.toggle_log:
            print(
                "RECEIVED CARTESIAN TARGET:\n---\n"
                f"Duration: {self.duration}s\n"
                f"X: {data.linear.x}\nY: {data.linear.y}\nZ: {data.linear.z}\n"
                f"RX: {data.angular.x}\nRY: {data.angular.y}\nRZ: {data.angular.z}\n---"
            )
            sys.stdout.flush()

        self.move_robot(q)

    # ----------------------------------------------------------------------
    # Single-point execution - legacy behaviour preserved
    # ----------------------------------------------------------------------
    def move_robot(self, q):
        q_array = self._validate_joint_vector(q)
        if q_array is None:
            self.get_logger().error(
                "Rejected robot command: expected six finite joint positions."
            )
            return False

        if not self._joint_limits_satisfied(q_array):
            self.get_logger().error("Rejected robot command outside model joint limits.")
            return False

        if not self._is_valid_duration(self.duration):
            self.get_logger().error(
                f"Rejected robot command with invalid duration: {self.duration}"
            )
            return False

        trajectory = JointTrajectory()
        trajectory.joint_names = list(JOINT_NAMES)

        point = JointTrajectoryPoint()
        point.positions = q_array.tolist()
        self._set_ros_duration(point.time_from_start, self.duration)
        trajectory.points = [point]

        self._publish_validated_trajectory(trajectory)

        if self.toggle_log:
            print(f"SENT ROBOT:\n---\nDuration: {self.duration}s")
            for i, angle in enumerate(q_array):
                print(f"Q{i + 1}: {np.rad2deg(angle):.2f} deg")
            print("---")
            sys.stdout.flush()

        return True

    # ----------------------------------------------------------------------
    # Multi-point execution boundary for the upcoming AtomSlicer planner
    # ----------------------------------------------------------------------
    def get_joint_trajectory(self, msg):
        """Validate and forward a complete joint trajectory to the UR controller."""
        trajectory = self._normalise_joint_trajectory(msg)
        if trajectory is None:
            return

        self._publish_validated_trajectory(trajectory)
        self.get_logger().info(
            "Forwarded validated multi-point trajectory with "
            f"{len(trajectory.points)} points to the UR controller."
        )

    def _normalise_joint_trajectory(
        self,
        msg: JointTrajectory,
    ) -> Optional[JointTrajectory]:
        if not msg.points:
            self.get_logger().error("Rejected empty JointTrajectory command.")
            return None

        if len(msg.joint_names) != len(JOINT_NAMES):
            self.get_logger().error(
                "Rejected JointTrajectory: expected exactly six joint names."
            )
            return None

        if len(set(msg.joint_names)) != len(msg.joint_names):
            self.get_logger().error(
                "Rejected JointTrajectory containing duplicate joint names."
            )
            return None

        if set(msg.joint_names) != set(JOINT_NAMES):
            self.get_logger().error(
                "Rejected JointTrajectory with unexpected joint names: "
                + ", ".join(msg.joint_names)
            )
            return None

        source_index = {name: i for i, name in enumerate(msg.joint_names)}
        reorder = [source_index[name] for name in JOINT_NAMES]

        normalised = JointTrajectory()
        normalised.header = msg.header
        normalised.joint_names = list(JOINT_NAMES)

        previous_time = -1.0

        for point_index, source_point in enumerate(msg.points):
            if len(source_point.positions) != len(JOINT_NAMES):
                self.get_logger().error(
                    f"Rejected JointTrajectory point {point_index}: "
                    "positions must contain six values."
                )
                return None

            def reorder_optional(values, field_name):
                if len(values) == 0:
                    return []
                if len(values) != len(JOINT_NAMES):
                    raise ValueError(
                        f"{field_name} must be empty or contain six values"
                    )
                reordered = [float(values[i]) for i in reorder]
                if not np.all(np.isfinite(reordered)):
                    raise ValueError(f"{field_name} contains non-finite values")
                return reordered

            try:
                positions = [float(source_point.positions[i]) for i in reorder]
                if not np.all(np.isfinite(positions)):
                    raise ValueError("positions contains non-finite values")

                velocities = reorder_optional(source_point.velocities, "velocities")
                accelerations = reorder_optional(
                    source_point.accelerations,
                    "accelerations",
                )
                effort = reorder_optional(source_point.effort, "effort")
            except (TypeError, ValueError, IndexError) as exc:
                self.get_logger().error(
                    f"Rejected JointTrajectory point {point_index}: {exc}"
                )
                return None

            q = np.asarray(positions, dtype=float)
            if not self._joint_limits_satisfied(q):
                self.get_logger().error(
                    f"Rejected JointTrajectory point {point_index}: "
                    "joint limits exceeded."
                )
                return None

            if source_point.time_from_start.nanosec >= 1_000_000_000:
                self.get_logger().error(
                    f"Rejected JointTrajectory point {point_index}: invalid nanoseconds."
                )
                return None

            point_time = self._duration_to_seconds(source_point.time_from_start)
            if point_time < 0.0 or not math.isfinite(point_time):
                self.get_logger().error(
                    f"Rejected JointTrajectory point {point_index}: invalid timestamp."
                )
                return None

            if point_index > 0 and point_time <= previous_time:
                self.get_logger().error(
                    f"Rejected JointTrajectory point {point_index}: "
                    "time_from_start must be strictly increasing."
                )
                return None

            previous_time = point_time

            target_point = JointTrajectoryPoint()
            target_point.positions = positions
            target_point.velocities = velocities
            target_point.accelerations = accelerations
            target_point.effort = effort
            target_point.time_from_start = source_point.time_from_start
            normalised.points.append(target_point)

        if previous_time <= 0.0:
            self.get_logger().error(
                "Rejected JointTrajectory: final time_from_start must be greater than zero."
            )
            return None

        return normalised

    def _publish_validated_trajectory(self, trajectory: JointTrajectory) -> None:
        """Publish an already-validated trajectory and update node state."""
        final_q = np.asarray(trajectory.points[-1].positions, dtype=float)
        self.target_q = final_q.copy()
        self.is_moving = True

        self.joint_pose_pub_.publish(trajectory)

        # Preserve the legacy final-goal topic for existing logging/monitoring.
        goal_msg = Float32MultiArray()
        goal_msg.data = final_q.tolist()
        self.goal_pub_.publish(goal_msg)

        if self.toggle_log:
            self.get_logger().info(
                "New trajectory command published; robot marked as moving."
            )

    # ----------------------------------------------------------------------
    # IK helpers used by the legacy Cartesian interface
    # ----------------------------------------------------------------------
    def find_q_pose(self, x, y, z, rx, ry, rz) -> Optional[np.ndarray]:
        """Solve one Cartesian target using the repository's UR10e DH model.

        Position is supplied in metres and RPY orientation in degrees, matching
        the legacy caller. The current robot state is used as the IK seed to
        favour continuity with the existing configuration.
        """
        roll = np.deg2rad(rx)
        pitch = np.deg2rad(ry)
        yaw = np.deg2rad(rz)

        desired_pose = (
            sm.SE3(x, y, z)
            * sm.SE3.RPY(roll, pitch, yaw, order="zyx")
        )

        starting_q = np.asarray(self.robot.q, dtype=float).copy()
        return self.solve_ik(desired_pose, starting_q)

    def solve_ik(
        self,
        desired_pose,
        seed_q: Optional[Sequence[float]] = None,
    ) -> Optional[np.ndarray]:
        """Solve IK with compatibility across Robotics Toolbox API versions."""
        if seed_q is None:
            seed = np.asarray(self.robot.q, dtype=float).copy()
        else:
            seed = self._validate_joint_vector(seed_q)
            if seed is None:
                self.get_logger().error("IK rejected: invalid seed configuration.")
                return None

        # Fast/C++-backed solver first, preserving the existing repository's
        # preferred behaviour.
        try:
            result = self.robot_DH.ik_LM(
                desired_pose,
                q0=seed,
                joint_limits=True,
            )
            q, success, reason = self._extract_ik_result(result)
            q_array = self._validate_joint_vector(q) if q is not None else None

            if success and q_array is not None and self._joint_limits_satisfied(q_array):
                return q_array

            if self.toggle_log:
                detail = f" ({reason})" if reason else ""
                self.get_logger().warn(
                    "Fast IK solver did not return a valid solution" + detail + "."
                )
        except Exception as exc:
            self.get_logger().warn(f"Fast IK solver failed: {exc}")

        # Portable Python solver fallback.
        try:
            result = self.robot_DH.ikine_LM(
                desired_pose,
                q0=seed,
                joint_limits=True,
            )
            q, success, reason = self._extract_ik_result(result)
            q_array = self._validate_joint_vector(q) if q is not None else None

            if success and q_array is not None and self._joint_limits_satisfied(q_array):
                return q_array

            detail = f" ({reason})" if reason else ""
            self.get_logger().error(
                "Python IK solver did not return a valid solution" + detail + "."
            )
        except Exception as exc:
            self.get_logger().error(f"Python IK solver failed: {exc}")

        return None

    def publish_is_moving(self):
        msg = Bool()
        msg.data = self.is_moving
        self.is_moving_pub_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = URcontrol()

    try:
        while rclpy.ok() and not node.shutdown_requested:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
