"""Build the legacy fixed-stride filtered periodic AMP clip from a motion NPZ.

This module is retained to reproduce the earlier ``v1`` experiment.  It uses
``static_keep_stride`` and therefore is *not* the exact duplicate filter used
for the corrected data preparation.  For the requested behavior (remove only
consecutive, exactly identical frames in the initial standing prefix), use
``filter_exact_static_frames.py``.

The Fast06 clip starts with a measured standing hold.  Keeping every sample in
that hold gives the discriminator many nearly identical positive examples and
does not add useful temporal information.  This utility compresses only the
redundant prefix, keeps a few anchor poses for a smooth stand-to-gait start,
and stores optional per-frame sampling weights for the AMP expert sampler.

The source file is never modified.  The last source row is a periodic endpoint
duplicate and is retained in the output so existing motion-loader semantics
remain valid.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = (
    REPOSITORY_ROOT
    / "external/amp/data/npz/Robot_Model0530_V2_SkatingFast06_IK_control.npz"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "external/amp/data/npz/Robot_Model0530_V2_SkatingFast06_static_prefix_trim_transition_v1.npz"
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

DEFAULT_STATIC_JOINT_SPEED = 0.1
DEFAULT_STATIC_ROOT_SPEED = 0.01
DEFAULT_STATIC_KEEP_STRIDE = 8
DEFAULT_TRANSITION_END_FRAME = 180
DEFAULT_TRANSITION_WEIGHT = 3.0


def _periodic_derivative(values: np.ndarray, fps: float) -> np.ndarray:
    """Differentiate an array whose last row duplicates its first row."""

    if values.shape[0] < 4:
        raise ValueError("A periodic clip needs at least three unique frames.")
    unique = values[:-1]
    derivative = np.empty_like(unique)
    dt = 1.0 / float(fps)
    derivative[1:-1] = (unique[2:] - unique[:-2]) / (2.0 * dt)
    derivative[0] = (unique[1] - unique[-1]) / (2.0 * dt)
    derivative[-1] = (unique[0] - unique[-2]) / (2.0 * dt)
    return np.concatenate((derivative, derivative[:1]), axis=0)


def _find_static_prefix(
    motion: dict[str, np.ndarray],
    joint_speed_threshold: float,
    root_speed_threshold: float,
) -> int:
    """Return the first frame outside the contiguous initial static prefix."""

    joint_names = [str(name) for name in motion["joint_names"].tolist()]
    leg_columns = [joint_names.index(name) for name in LEG_JOINT_NAMES if name in joint_names]
    if not leg_columns:
        leg_columns = list(range(motion["joint_pos"].shape[1]))

    unique_count = motion["joint_pos"].shape[0] - 1
    joint_speed = np.linalg.norm(
        motion["joint_vel"][:unique_count, leg_columns], axis=-1
    )
    body_names = [str(name) for name in motion["body_names"].tolist()]
    root_column = body_names.index("link_leg2pelvis") if "link_leg2pelvis" in body_names else 0
    root_speed = np.linalg.norm(
        motion["body_lin_vel_w"][:unique_count, root_column, :2], axis=-1
    )
    static = (joint_speed <= float(joint_speed_threshold)) & (
        root_speed <= float(root_speed_threshold)
    )

    prefix_length = 0
    while prefix_length < unique_count and bool(static[prefix_length]):
        prefix_length += 1
    if prefix_length < 2:
        raise ValueError(
            "Could not find an initial static prefix; adjust the static thresholds."
        )
    return prefix_length


def _validate_source(motion: dict[str, np.ndarray]) -> None:
    required = (
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "joint_names",
        "body_names",
    )
    missing = [name for name in required if name not in motion]
    if missing:
        raise ValueError(f"Input motion is missing fields: {missing}")
    frame_count = int(motion["joint_pos"].shape[0])
    if frame_count < 4:
        raise ValueError("Input motion must contain at least three unique frames.")
    for name in required:
        if name in {"fps", "joint_names", "body_names"}:
            continue
        if motion[name].shape[0] != frame_count:
            raise ValueError(
                f"Motion field {name!r} has {motion[name].shape[0]} rows; expected {frame_count}."
            )
    if not np.allclose(motion["joint_pos"][-1], motion["joint_pos"][0], atol=1.0e-5):
        raise ValueError("Input joint trajectory does not have a duplicated endpoint.")


def build_filtered_motion(
    input_path: str | Path,
    output_path: str | Path,
    *,
    static_joint_speed_threshold: float = DEFAULT_STATIC_JOINT_SPEED,
    static_root_speed_threshold: float = DEFAULT_STATIC_ROOT_SPEED,
    static_keep_stride: int = DEFAULT_STATIC_KEEP_STRIDE,
    transition_end_frame: int = DEFAULT_TRANSITION_END_FRAME,
    transition_weight: float = DEFAULT_TRANSITION_WEIGHT,
) -> dict[str, int | float | str]:
    """Create the filtered clip and return an auditable summary."""

    if static_keep_stride <= 0:
        raise ValueError("static_keep_stride must be positive")
    if transition_weight < 1.0 or not np.isfinite(transition_weight):
        raise ValueError("transition_weight must be finite and at least one")

    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    with np.load(input_path, allow_pickle=False) as source:
        motion = {name: np.asarray(source[name]) for name in source.files}
    _validate_source(motion)

    fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"Invalid input fps: {fps}")
    unique_count = int(motion["joint_pos"].shape[0] - 1)
    static_prefix_end = _find_static_prefix(
        motion, static_joint_speed_threshold, static_root_speed_threshold
    )
    transition_end_frame = min(int(transition_end_frame), unique_count)
    if transition_end_frame <= static_prefix_end:
        raise ValueError(
            "transition_end_frame must be after the detected static prefix."
        )

    # Keep sparse static anchors, all transition/gait frames, and the original
    # periodic endpoint.  This removes redundant hold samples without making a
    # discontinuity at the cycle boundary or losing the frame-0 stand pose.
    static_indices = np.arange(0, static_prefix_end, static_keep_stride, dtype=np.int64)
    last_static = np.asarray([static_prefix_end - 1], dtype=np.int64)
    if static_indices.size == 0 or static_indices[-1] != last_static[0]:
        static_indices = np.concatenate((static_indices, last_static))
    source_indices = np.concatenate(
        (static_indices, np.arange(static_prefix_end, unique_count + 1, dtype=np.int64))
    )
    if source_indices[0] != 0 or source_indices[-1] != unique_count:
        raise AssertionError("Filtered source indices must retain both endpoints.")
    if np.any(np.diff(source_indices) <= 0):
        raise AssertionError("Filtered source indices must be strictly increasing.")

    filtered: dict[str, np.ndarray] = {}
    frame_count = int(motion["joint_pos"].shape[0])
    for name, values in motion.items():
        if values.ndim >= 1 and values.shape[0] == frame_count:
            filtered[name] = values[source_indices]
        else:
            filtered[name] = values.copy()

    # Recompute joint and relative-body linear velocities for the compressed
    # timeline.  The runtime loader performs the same periodic repair; keeping
    # the NPZ fields consistent also makes it safe for standalone replay tools.
    filtered["joint_vel"] = _periodic_derivative(filtered["joint_pos"], fps).astype(
        np.float32, copy=False
    )
    body_pos = filtered["body_pos_w"].astype(np.float32, copy=False)
    body_names = [str(name) for name in filtered["body_names"].tolist()]
    root_column = body_names.index("link_leg2pelvis") if "link_leg2pelvis" in body_names else 0
    root_pos = body_pos[:, root_column]
    root_velocity = _periodic_derivative(root_pos, fps)
    source_root_velocity = filtered["body_lin_vel_w"][:, root_column].copy()
    root_velocity[:, 0] = source_root_velocity[:, 0]
    root_velocity[-1, 0] = root_velocity[0, 0]
    relative_body_pos = body_pos - root_pos[:, None, :]
    relative_body_velocity = _periodic_derivative(relative_body_pos, fps)
    filtered["body_lin_vel_w"] = (
        relative_body_velocity + root_velocity[:, None, :]
    ).astype(np.float32, copy=False)

    # The loader samples starting frames using these optional weights.  The
    # endpoint has no independent sampling interval and is therefore omitted.
    weights = np.ones(source_indices.shape[0] - 1, dtype=np.float32)
    transition_mask = (
        source_indices[:-1] >= static_prefix_end
    ) & (source_indices[:-1] < transition_end_frame)
    weights[transition_mask] = float(transition_weight)
    filtered["sample_weights"] = weights
    filtered["source_frame_indices"] = source_indices
    filtered["static_prefix_end_frame"] = np.asarray(static_prefix_end, dtype=np.int64)
    filtered["static_frames_removed"] = np.asarray(
        static_prefix_end - static_indices.size, dtype=np.int64
    )
    filtered["transition_start_frame"] = np.asarray(static_prefix_end, dtype=np.int64)
    filtered["transition_end_frame"] = np.asarray(transition_end_frame, dtype=np.int64)
    filtered["transition_oversample_factor"] = np.asarray(float(transition_weight), dtype=np.float32)
    filtered["filter_version"] = np.asarray("static_prefix_trim_transition_v1")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **filtered)
    summary: dict[str, int | float | str] = {
        "input": str(input_path),
        "output": str(output_path),
        "fps": fps,
        "source_frames": frame_count,
        "output_frames": int(source_indices.size),
        "source_period_frames": unique_count,
        "output_period_frames": int(source_indices.size - 1),
        "static_prefix_end_frame": static_prefix_end,
        "static_frames_removed": int(static_prefix_end - static_indices.size),
        "transition_end_frame": transition_end_frame,
        "transition_oversample_factor": float(transition_weight),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Source NPZ")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Filtered NPZ")
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
    parser.add_argument(
        "--static-keep-stride",
        type=int,
        default=DEFAULT_STATIC_KEEP_STRIDE,
        help="Keep one static anchor every N source frames",
    )
    parser.add_argument(
        "--transition-end-frame",
        type=int,
        default=DEFAULT_TRANSITION_END_FRAME,
        help="Exclusive source-frame end of the oversampled startup interval",
    )
    parser.add_argument(
        "--transition-weight",
        type=float,
        default=DEFAULT_TRANSITION_WEIGHT,
        help="AMP expert sampling weight for startup frames",
    )
    args = parser.parse_args()
    summary = build_filtered_motion(
        args.input,
        args.output,
        static_joint_speed_threshold=args.static_joint_speed_threshold,
        static_root_speed_threshold=args.static_root_speed_threshold,
        static_keep_stride=args.static_keep_stride,
        transition_end_frame=args.transition_end_frame,
        transition_weight=args.transition_weight,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
