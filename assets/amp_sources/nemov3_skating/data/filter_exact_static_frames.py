"""Deduplicate exact motion states in the initial Fast06 standing prefix.

The source clip is sampled at a fixed rate and may contain a measured standing
hold.  This utility removes a frame only when it is an *exact* copy of the
preceding complete motion state and only while the contiguous initial static
prefix is active.  It deliberately does not use a stride, a distance
threshold, or a global resampling pass.  Every transition and gait frame is
copied in its original order.

All frame-valued fields are copied without recomputing derivatives.  Comparing
the complete state (poses and velocities) makes that policy explicit: a row is
removed only when retaining its predecessor cannot change any motion value.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = (
    REPOSITORY_ROOT
    / "external/amp/data/npz/Robot_Model0530_V2_SkatingFast06_IK_control.npz"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "external/amp/data/npz/Robot_Model0530_V2_SkatingFast06_exact_static_dedup_v2.npz"
)

LEG_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_calf_joint",
    "right_calf_joint",
)

# These thresholds identify where the contiguous measured standing prefix ends;
# they are not used to decide whether a frame is a duplicate.
DEFAULT_STATIC_JOINT_SPEED = 0.1
DEFAULT_STATIC_ROOT_SPEED = 0.01

POSE_FIELDS = ("joint_pos", "body_pos_w", "body_quat_w")
STATE_FIELDS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)
DROP_ON_OUTPUT_FIELDS = ("sample_weights",)


def _validate_source(motion: dict[str, np.ndarray]) -> None:
    required = (
        "fps",
        *STATE_FIELDS,
        "joint_names",
        "body_names",
    )
    missing = [name for name in required if name not in motion]
    if missing:
        raise ValueError(f"Input motion is missing fields: {missing}")

    frame_count = int(motion["joint_pos"].shape[0])
    if frame_count < 4:
        raise ValueError("Input motion must contain at least three unique frames.")
    for name in STATE_FIELDS:
        values = motion[name]
        if values.ndim < 2 or values.shape[0] != frame_count:
            raise ValueError(
                f"Motion field {name!r} must have {frame_count} frame rows; "
                f"got shape {values.shape}."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Motion field {name!r} contains non-finite values.")

    fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"Invalid input fps: {fps}")
    # The existing motion-loader contract treats the final row as a periodic
    # endpoint.  Do not alter it, even if other pose fields have small source
    # conversion error at the boundary.
    if not np.allclose(
        motion["joint_pos"][-1], motion["joint_pos"][0], rtol=0.0, atol=1.0e-5
    ):
        raise ValueError("Input joint trajectory does not have a duplicated endpoint.")


def _find_static_prefix(
    motion: dict[str, np.ndarray],
    joint_speed_threshold: float,
    root_speed_threshold: float,
) -> int:
    """Return the first frame outside the contiguous initial static prefix."""

    joint_names = [str(name) for name in motion["joint_names"].tolist()]
    leg_columns = [
        joint_names.index(name) for name in LEG_JOINT_NAMES if name in joint_names
    ]
    if not leg_columns:
        leg_columns = list(range(motion["joint_pos"].shape[1]))

    unique_count = motion["joint_pos"].shape[0] - 1
    joint_speed = np.linalg.norm(
        motion["joint_vel"][:unique_count, leg_columns], axis=-1
    )
    body_names = [str(name) for name in motion["body_names"].tolist()]
    root_column = (
        body_names.index("link_leg2pelvis")
        if "link_leg2pelvis" in body_names
        else 0
    )
    root_speed = np.linalg.norm(
        motion["body_lin_vel_w"][:unique_count, root_column, :2], axis=-1
    )
    static = (joint_speed <= float(joint_speed_threshold)) & (
        root_speed <= float(root_speed_threshold)
    )

    prefix_length = 0
    while prefix_length < unique_count and bool(static[prefix_length]):
        prefix_length += 1
    if prefix_length < 1:
        raise ValueError(
            "Could not find an initial static prefix; adjust the static thresholds."
        )
    return prefix_length


def _exact_adjacent_mask(
    motion: dict[str, np.ndarray], fields: tuple[str, ...]
) -> np.ndarray:
    """Return exact-equality flags for each adjacent pair (i - 1, i)."""

    frame_count = int(motion["joint_pos"].shape[0])
    equal = np.ones(frame_count - 1, dtype=bool)
    for name in fields:
        values = motion[name]
        element_equal = values[1:] == values[:-1]
        equal &= np.all(
            element_equal,
            axis=tuple(range(1, element_equal.ndim)),
        )
    return equal


def build_exact_static_deduplicated_motion(
    input_path: str | Path,
    output_path: str | Path,
    *,
    static_joint_speed_threshold: float = DEFAULT_STATIC_JOINT_SPEED,
    static_root_speed_threshold: float = DEFAULT_STATIC_ROOT_SPEED,
) -> dict[str, int | float | str]:
    """Create an exact-prefix-deduplicated clip and return audit statistics."""

    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if input_path == output_path:
        raise ValueError("Input and output NPZ paths must be different.")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input motion NPZ not found: {input_path}")
    for threshold_name, threshold in (
        ("static_joint_speed_threshold", static_joint_speed_threshold),
        ("static_root_speed_threshold", static_root_speed_threshold),
    ):
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError(f"{threshold_name} must be finite and non-negative")
    with np.load(input_path, allow_pickle=False) as source:
        motion = {name: np.asarray(source[name]) for name in source.files}
    _validate_source(motion)

    frame_count = int(motion["joint_pos"].shape[0])
    unique_count = frame_count - 1
    static_prefix_end = _find_static_prefix(
        motion,
        static_joint_speed_threshold,
        static_root_speed_threshold,
    )

    # Compute both masks for auditability.  Only complete-state exact matches
    # in the initial prefix are eligible for removal.  A pose-only match is not
    # enough because its velocity fields could still encode a real change.
    pose_duplicates = _exact_adjacent_mask(motion, POSE_FIELDS)
    state_duplicates = _exact_adjacent_mask(motion, STATE_FIELDS)
    prefix_pair_mask = np.arange(1, frame_count) < static_prefix_end
    removable = prefix_pair_mask & state_duplicates
    removed_source_indices = np.flatnonzero(removable).astype(np.int64) + 1
    keep_mask = np.ones(frame_count, dtype=bool)
    keep_mask[1:] = ~removable
    source_indices = np.flatnonzero(keep_mask).astype(np.int64)

    filtered: dict[str, np.ndarray] = {}
    for name, values in motion.items():
        # A weighted source (for example the legacy stride-filtered v1 file)
        # must not silently carry its distribution into an exact-only output.
        # Its interval count also differs whenever a frame is removed.
        if name in DROP_ON_OUTPUT_FIELDS:
            continue
        if values.ndim >= 1 and values.shape[0] == frame_count:
            filtered[name] = values[source_indices].copy()
        else:
            filtered[name] = values.copy()

    # These metadata fields are scalar/auxiliary and do not alter the motion
    # state consumed by the loader.  In particular, no velocity is recomputed.
    filtered["source_frame_indices"] = source_indices
    filtered["removed_source_frame_indices"] = removed_source_indices
    filtered["source_frame_times_s"] = (
        np.arange(frame_count, dtype=np.float32) / float(motion["fps"].reshape(-1)[0])
    )[source_indices]
    filtered["static_prefix_end_frame"] = np.asarray(
        static_prefix_end, dtype=np.int64
    )
    filtered["static_prefix_frames"] = np.asarray(static_prefix_end, dtype=np.int64)
    filtered["static_prefix_output_end_frame"] = np.asarray(
        int(np.searchsorted(source_indices, static_prefix_end, side="left")),
        dtype=np.int64,
    )
    filtered["static_prefix_exact_pose_duplicate_pairs"] = np.asarray(
        int(np.sum(prefix_pair_mask & pose_duplicates)), dtype=np.int64
    )
    filtered["static_prefix_exact_state_duplicate_pairs"] = np.asarray(
        int(np.sum(removable)), dtype=np.int64
    )
    filtered["static_frames_removed"] = np.asarray(
        int(np.sum(removable)), dtype=np.int64
    )
    filtered["exact_duplicate_tolerance"] = np.asarray(0.0, dtype=np.float32)
    filtered["exact_duplicate_scope"] = np.asarray("complete_motion_state")
    filtered["exact_duplicate_fields"] = np.asarray(STATE_FIELDS)
    filtered["pose_audit_fields"] = np.asarray(POSE_FIELDS)
    filtered["static_detection_fields"] = np.asarray(
        "leg_joint_speed_and_root_planar_speed"
    )
    filtered["static_joint_speed_threshold"] = np.asarray(
        float(static_joint_speed_threshold), dtype=np.float32
    )
    filtered["static_root_speed_threshold"] = np.asarray(
        float(static_root_speed_threshold), dtype=np.float32
    )
    filtered["velocity_policy"] = np.asarray("preserve_source_rows_no_recompute")
    filtered["sampling_weight_policy"] = np.asarray("uniform_source_intervals")
    filtered["dropped_input_sampling_fields"] = np.asarray(
        tuple(name for name in DROP_ON_OUTPUT_FIELDS if name in motion)
    )
    filtered["filter_version"] = np.asarray("exact_initial_static_dedup_v2")
    filtered["source_period_frames"] = np.asarray(unique_count, dtype=np.int64)
    filtered["output_period_frames"] = np.asarray(
        int(source_indices.size - 1), dtype=np.int64
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a sibling temporary file so an interrupted compression cannot
    # leave a partially valid NPZ at the requested path.
    temporary_path = output_path.with_name(output_path.name + ".tmp.npz")
    try:
        np.savez_compressed(temporary_path, **filtered)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return {
        "input": str(input_path),
        "output": str(output_path),
        "fps": float(np.asarray(motion["fps"]).reshape(-1)[0]),
        "source_frames": frame_count,
        "output_frames": int(source_indices.size),
        "source_period_frames": unique_count,
        "output_period_frames": int(source_indices.size - 1),
        "static_prefix_end_frame": static_prefix_end,
        "pose_duplicate_pairs_whole_clip": int(np.sum(pose_duplicates)),
        "state_duplicate_pairs_whole_clip": int(np.sum(state_duplicates)),
        "pose_duplicate_pairs_static_prefix": int(
            np.sum(prefix_pair_mask & pose_duplicates)
        ),
        "state_duplicate_pairs_static_prefix": int(np.sum(removable)),
        "static_frames_removed": int(np.sum(removable)),
        "velocity_policy": "preserve_source_rows_no_recompute",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Source NPZ")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output NPZ")
    parser.add_argument(
        "--static-joint-speed-threshold",
        type=float,
        default=DEFAULT_STATIC_JOINT_SPEED,
        help="Initial-prefix leg speed threshold in rad/s",
    )
    parser.add_argument(
        "--static-root-speed-threshold",
        type=float,
        default=DEFAULT_STATIC_ROOT_SPEED,
        help="Initial-prefix planar root speed threshold in m/s",
    )
    args = parser.parse_args()
    summary = build_exact_static_deduplicated_motion(
        args.input,
        args.output,
        static_joint_speed_threshold=args.static_joint_speed_threshold,
        static_root_speed_threshold=args.static_root_speed_threshold,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
