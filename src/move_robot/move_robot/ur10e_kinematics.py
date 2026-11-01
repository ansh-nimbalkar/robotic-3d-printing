"""Shared nominal UR10e kinematics helpers for the thesis motion pipeline.

This module intentionally contains no ROS node logic.  It provides one source
of truth for the nominal UR10e DH model and numerical IK used by both the
existing execution node and the AtomSlicer planning/validation nodes.

The model is nominal.  Robot-specific UR calibration remains authoritative for
final physical execution accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Optional, Sequence

import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH


JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


@dataclass(frozen=True)
class IKSolveResult:
    """Result of one deterministic IK search."""

    success: bool
    q: Optional[np.ndarray]
    solver: str
    seed_index: int
    reason: str
    residual: Optional[float] = None


@dataclass(frozen=True)
class JointPathContinuityResult:
    """Globally selected equivalent revolute-joint representation."""

    q_path: np.ndarray
    max_step_rad: float
    max_step_waypoint: int
    max_step_joint_index: int
    per_joint_max_step_rad: np.ndarray
    continuity_warning_count: int


class UR10eKinematics:
    """Nominal UR10e DH model plus robust numerical IK helpers."""

    def __init__(self) -> None:
        self.robot = self.create_model()

    @staticmethod
    def create_model() -> DHRobot:
        """Create a nominal UR10e standard-DH model.

        Dimensions are the Universal Robots nominal UR10e/UR12e kinematic DH
        parameters, expressed in metres.  The model is intentionally built
        explicitly because common Robotics Toolbox releases expose the legacy
        ``UR10`` model but not a distinct ``UR10e`` model.
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

        robot.qz = np.zeros(6, dtype=float)
        robot.qr = np.array(
            [math.pi, 0.0, 0.0, 0.0, math.pi / 2.0, 0.0],
            dtype=float,
        )
        robot.q = robot.qz.copy()
        return robot

    @staticmethod
    def validate_joint_vector(q: Sequence[float]) -> Optional[np.ndarray]:
        """Return a finite six-joint vector, or ``None`` when invalid."""
        try:
            q_array = np.asarray(q, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return None

        if q_array.size != len(JOINT_NAMES):
            return None
        if not np.all(np.isfinite(q_array)):
            return None

        return q_array

    def joint_limits_satisfied(self, q: Sequence[float]) -> bool:
        """Check the nominal model's joint limits."""
        q_array = self.validate_joint_vector(q)
        if q_array is None:
            return False

        try:
            qlim = np.asarray(self.robot.qlim, dtype=float)
        except (AttributeError, TypeError, ValueError):
            return True

        if qlim.shape != (2, len(JOINT_NAMES)):
            return True

        tolerance = 1e-9
        lower = qlim[0, :]
        upper = qlim[1, :]

        lower_ok = np.logical_or(~np.isfinite(lower), q_array >= lower - tolerance)
        upper_ok = np.logical_or(~np.isfinite(upper), q_array <= upper + tolerance)
        return bool(np.all(lower_ok & upper_ok))

    @staticmethod
    def extract_ik_result(result):
        """Normalise older tuple and newer IKSolution RTB return formats.

        Returns:
            (q, success, reason, residual)

        Older Robotics Toolbox tuple APIs commonly expose residual as the fifth
        element.  Newer IKSolution objects expose it as ``result.residual``.
        """
        if hasattr(result, "q"):
            q = result.q
            success = bool(getattr(result, "success", q is not None))
            reason = str(getattr(result, "reason", ""))
            residual_value = getattr(result, "residual", None)
            try:
                residual = (
                    float(residual_value)
                    if residual_value is not None and np.isfinite(residual_value)
                    else None
                )
            except (TypeError, ValueError):
                residual = None
            return q, success, reason, residual

        if isinstance(result, (tuple, list)) and len(result) >= 2:
            residual = None
            if len(result) >= 5:
                try:
                    candidate = float(result[4])
                    if np.isfinite(candidate):
                        residual = candidate
                except (TypeError, ValueError):
                    residual = None
            return result[0], bool(result[1]), "", residual

        return None, False, "Unsupported IK result format", None

    @staticmethod
    def _unique_seeds(seeds: Iterable[Sequence[float]]) -> list[np.ndarray]:
        """Validate and de-duplicate seeds while preserving order."""
        unique: list[np.ndarray] = []

        for seed in seeds:
            q = UR10eKinematics.validate_joint_vector(seed)
            if q is None:
                continue

            if any(np.allclose(q, existing, atol=1e-12, rtol=0.0) for existing in unique):
                continue

            unique.append(q.copy())

        return unique

    def solve_ik(
        self,
        desired_pose,
        seed_q: Sequence[float],
        fallback_seeds: Iterable[Sequence[float]] = (),
        *,
        residual_tolerance: float = 1e-8,
        iteration_limit: int = 100,
    ) -> IKSolveResult:
        """Solve IK using deterministic seeds and both RTB LM APIs.

        The first seed should normally be the previous valid trajectory joint
        solution.  Fallback seeds are only tried if that continuity-preserving
        solve fails.

        ``slimit=1`` is intentional.  Robotics Toolbox otherwise permits
        additional internally generated search seeds after ``q0``.  That would
        undermine deterministic branch tracking and make the wrapper's explicit
        fallback seed order meaningless.

        ``residual_tolerance`` is deliberately tighter than RTB's common
        default of 1e-6 because reachability is independently verified using
        Cartesian FK tolerances after every solve.
        """
        if (
            not math.isfinite(residual_tolerance)
            or residual_tolerance <= 0.0
        ):
            raise ValueError("residual_tolerance must be finite and > 0")
        if iteration_limit < 1:
            raise ValueError("iteration_limit must be >= 1")

        seeds = self._unique_seeds((seed_q, *tuple(fallback_seeds)))
        if not seeds:
            return IKSolveResult(
                False, None, "", -1, "No valid IK seed", None
            )

        failure_reasons: list[str] = []

        for seed_index, seed in enumerate(seeds):
            # Fast C++ Levenberg-Marquardt interface.
            try:
                result = self.robot.ik_LM(
                    desired_pose,
                    q0=seed,
                    ilimit=iteration_limit,
                    slimit=1,
                    tol=residual_tolerance,
                    joint_limits=True,
                )
                q, success, reason, residual = self.extract_ik_result(result)
                q_array = self.validate_joint_vector(q) if q is not None else None

                if (
                    success
                    and q_array is not None
                    and self.joint_limits_satisfied(q_array)
                ):
                    return IKSolveResult(
                        True,
                        q_array,
                        "ik_LM",
                        seed_index,
                        reason,
                        residual,
                    )

                failure_reasons.append(
                    f"ik_LM(seed={seed_index}): {reason or 'no valid solution'}"
                )
            except Exception as exc:
                # RTB API/version differences must not kill analysis.
                failure_reasons.append(f"ik_LM(seed={seed_index}): {exc}")

            # Portable Python LM interface retained as deterministic fallback.
            try:
                result = self.robot.ikine_LM(
                    desired_pose,
                    q0=seed,
                    ilimit=iteration_limit,
                    slimit=1,
                    tol=residual_tolerance,
                    joint_limits=True,
                )
                q, success, reason, residual = self.extract_ik_result(result)
                q_array = self.validate_joint_vector(q) if q is not None else None

                if (
                    success
                    and q_array is not None
                    and self.joint_limits_satisfied(q_array)
                ):
                    return IKSolveResult(
                        True,
                        q_array,
                        "ikine_LM",
                        seed_index,
                        reason,
                        residual,
                    )

                failure_reasons.append(
                    f"ikine_LM(seed={seed_index}): {reason or 'no valid solution'}"
                )
            except Exception as exc:
                failure_reasons.append(f"ikine_LM(seed={seed_index}): {exc}")

        return IKSolveResult(
            False,
            None,
            "",
            -1,
            "; ".join(failure_reasons),
            None,
        )


    def optimise_equivalent_joint_path(
        self,
        q_path: Sequence[Sequence[float]],
        *,
        initial_q: Optional[Sequence[float]] = None,
        warning_threshold_rad: float = 0.35,
    ) -> JointPathContinuityResult:
        """Select a globally continuous equivalent representation of a joint path.

        Revolute joint angles separated by integer multiples of 2*pi represent
        the same kinematic configuration.  A greedy waypoint-by-waypoint wrap
        choice can fail near the +/-2*pi joint limits because the best choice
        at one waypoint may require shifting an earlier section of the path.

        This method therefore solves each joint's wrap-selection problem over
        the *entire path* using dynamic programming.  Candidate values are
        restricted to the nominal joint limits.  The optimisation is
        lexicographic:

          1. minimise the largest adjacent joint step;
          2. then minimise total squared joint motion.

        If ``initial_q`` is supplied, it is used only to break otherwise
        equivalent whole-path branches by preferring a first waypoint closer
        to the current robot state.  The reported maximum step covers printing
        waypoints only, not the approach from ``initial_q``.

        This operation changes representation only; adding/subtracting 2*pi
        does not change forward kinematics.
        """
        path = np.asarray(q_path, dtype=float)

        if path.ndim != 2 or path.shape[1] != len(JOINT_NAMES):
            raise ValueError(
                f"q_path must have shape (N, {len(JOINT_NAMES)})"
            )
        if path.shape[0] == 0:
            raise ValueError("q_path must not be empty")
        if not np.all(np.isfinite(path)):
            raise ValueError("q_path contains non-finite joint values")
        if (
            not math.isfinite(warning_threshold_rad)
            or warning_threshold_rad <= 0.0
        ):
            raise ValueError(
                "warning_threshold_rad must be finite and > 0"
            )

        anchor = None
        if initial_q is not None:
            anchor = self.validate_joint_vector(initial_q)
            if anchor is None:
                raise ValueError(
                    "initial_q must be a finite six-joint vector"
                )

        qlim = np.asarray(self.robot.qlim, dtype=float)
        if qlim.shape != (2, len(JOINT_NAMES)):
            raise RuntimeError(
                "UR10e joint limits are unavailable or malformed"
            )

        period = 2.0 * math.pi
        tolerance = 1e-9
        output = np.empty_like(path)

        # Each joint can be optimised independently because +/-2*pi changes on
        # one revolute joint do not constrain equivalent representations of the
        # other joints.
        for joint_index in range(len(JOINT_NAMES)):
            lower = float(qlim[0, joint_index])
            upper = float(qlim[1, joint_index])

            candidates_per_waypoint: list[np.ndarray] = []
            for value in path[:, joint_index]:
                if not (math.isfinite(lower) and math.isfinite(upper)):
                    candidates = np.array([value], dtype=float)
                else:
                    k_min = math.ceil(
                        (lower - value - tolerance) / period
                    )
                    k_max = math.floor(
                        (upper - value + tolerance) / period
                    )
                    candidates = np.array(
                        [
                            value + k * period
                            for k in range(k_min, k_max + 1)
                            if (
                                value + k * period >= lower - tolerance
                                and value + k * period <= upper + tolerance
                            )
                        ],
                        dtype=float,
                    )

                if candidates.size == 0:
                    raise RuntimeError(
                        f"No equivalent in-limit representation exists for "
                        f"{JOINT_NAMES[joint_index]} at waypoint "
                        f"{len(candidates_per_waypoint)}"
                    )

                # Remove numerically duplicated candidates and keep a stable
                # ascending ordering for deterministic backtracking.
                candidates = np.unique(
                    np.round(candidates, decimals=12)
                )
                candidates_per_waypoint.append(candidates)

            # State cost is (maximum_step_so_far, sum_squared_steps,
            # first_waypoint_anchor_distance).  Python tuple ordering gives the
            # desired deterministic lexicographic optimisation.
            previous_costs: list[tuple[float, float, float]] = []
            first_candidates = candidates_per_waypoint[0]
            for candidate in first_candidates:
                anchor_distance = (
                    abs(candidate - anchor[joint_index])
                    if anchor is not None
                    else abs(candidate - path[0, joint_index])
                )
                previous_costs.append((0.0, 0.0, anchor_distance))

            backpointers: list[np.ndarray] = []

            for waypoint_index in range(1, path.shape[0]):
                previous_candidates = candidates_per_waypoint[
                    waypoint_index - 1
                ]
                current_candidates = candidates_per_waypoint[
                    waypoint_index
                ]

                current_costs: list[tuple[float, float, float]] = []
                current_backpointers = np.full(
                    current_candidates.size,
                    -1,
                    dtype=int,
                )

                for current_index, current_value in enumerate(
                    current_candidates
                ):
                    best_cost = None
                    best_previous_index = -1

                    for previous_index, previous_value in enumerate(
                        previous_candidates
                    ):
                        step = abs(current_value - previous_value)
                        previous_cost = previous_costs[previous_index]

                        candidate_cost = (
                            max(previous_cost[0], step),
                            previous_cost[1] + step * step,
                            previous_cost[2],
                        )

                        if (
                            best_cost is None
                            or candidate_cost < best_cost
                        ):
                            best_cost = candidate_cost
                            best_previous_index = previous_index

                    current_costs.append(best_cost)
                    current_backpointers[current_index] = (
                        best_previous_index
                    )

                backpointers.append(current_backpointers)
                previous_costs = current_costs

            final_index = min(
                range(len(previous_costs)),
                key=lambda index: previous_costs[index],
            )

            selected_indices = np.empty(path.shape[0], dtype=int)
            selected_indices[-1] = final_index

            for waypoint_index in range(
                path.shape[0] - 1,
                0,
                -1,
            ):
                selected_indices[waypoint_index - 1] = (
                    backpointers[waypoint_index - 1][
                        selected_indices[waypoint_index]
                    ]
                )

            for waypoint_index, candidate_index in enumerate(
                selected_indices
            ):
                output[waypoint_index, joint_index] = (
                    candidates_per_waypoint[waypoint_index][candidate_index]
                )

        if path.shape[0] == 1:
            per_joint_max = np.zeros(len(JOINT_NAMES), dtype=float)
            return JointPathContinuityResult(
                q_path=output,
                max_step_rad=0.0,
                max_step_waypoint=0,
                max_step_joint_index=0,
                per_joint_max_step_rad=per_joint_max,
                continuity_warning_count=0,
            )

        steps = np.abs(np.diff(output, axis=0))
        per_joint_max = np.max(steps, axis=0)

        flat_index = int(np.argmax(steps))
        step_waypoint_zero_based, joint_index = np.unravel_index(
            flat_index,
            steps.shape,
        )

        # np.diff row i is the movement from waypoint i -> i+1.
        max_step_waypoint = int(step_waypoint_zero_based + 1)
        max_step = float(steps[step_waypoint_zero_based, joint_index])
        warning_count = int(
            np.count_nonzero(
                np.max(steps, axis=1) > warning_threshold_rad
            )
        )

        return JointPathContinuityResult(
            q_path=output,
            max_step_rad=max_step,
            max_step_waypoint=max_step_waypoint,
            max_step_joint_index=int(joint_index),
            per_joint_max_step_rad=per_joint_max,
            continuity_warning_count=warning_count,
        )

    def nearest_equivalent_configuration(
        self,
        q: Sequence[float],
        reference_q: Sequence[float],
    ) -> np.ndarray:
        """Choose the equivalent 2π representation nearest ``reference_q``.

        UR joints are revolute and the nominal model permits approximately
        ±2π.  Numerical IK can therefore return two values that describe the
        same physical joint angle but differ by almost 2π.  This helper chooses
        an equivalent representation within the configured joint limits that
        minimises each joint's change from the previous trajectory state.

        This is representation normalisation only; it does not alter the
        physical FK pose.
        """
        q_array = self.validate_joint_vector(q)
        reference = self.validate_joint_vector(reference_q)

        if q_array is None or reference is None:
            raise ValueError("q and reference_q must be valid six-joint vectors")

        qlim = np.asarray(self.robot.qlim, dtype=float)
        if qlim.shape != (2, len(JOINT_NAMES)):
            return q_array.copy()

        result = q_array.copy()
        period = 2.0 * math.pi

        for index in range(len(JOINT_NAMES)):
            candidates = []
            for turns in (-2, -1, 0, 1, 2):
                candidate = q_array[index] + turns * period
                if (
                    candidate >= qlim[0, index] - 1e-9
                    and candidate <= qlim[1, index] + 1e-9
                ):
                    candidates.append(candidate)

            if candidates:
                result[index] = min(
                    candidates,
                    key=lambda value: abs(value - reference[index]),
                )

        return result