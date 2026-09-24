"""Convert NemoV3 joint CSV files to replayable whole-body NPZ files.

Example:

    python data/csv_to_npz.py data/csv/Robot_Model0530_V2_SkatingFast03_IK_control.csv \
        --output_dir data/npz --root_orientation_mode bind_relative_inverse

Fast06 has a small measured endpoint mismatch. Close it explicitly while
keeping the strict default for other clips:

    python data/csv_to_npz.py data/csv/Robot_Model0530_V2_SkatingFast06_IK_control.csv \
        --output_dir data/npz --root_orientation_mode bind_relative_inverse \
        --periodic_endpoint_tolerance 0.04
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT_POSITION_COLUMNS = ("root_x", "root_y", "root_z")
ROOT_QUATERNION_COLUMNS = ("root_qx", "root_qy", "root_qz", "root_qw")
TIME_COLUMNS = ("time_sec", "time_from_start")
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MJCF_FILE = (
    REPOSITORY_ROOT
    / "lab/flamingo/assets/data/Robots/nemo/nemov3_trainv2/mjcf/nemov3_601.xml"
)
FK_MJCF_BODY_NAMES = (
    "link_leg2pelvis",
    "left_calf_link",
    "left_wheel_link",
    "right_calf_link",
    "right_wheel_link",
)

NEMOV3_NPZ_JOINT_NAMES = [
    "joint_pelvisYaw",
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "joint_pelvisPitch",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "joint_pelvisRoll",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "joint_lShoulderPitch",
    "joint_rShoulderPitch",
    "left_calf_joint",
    "right_calf_joint",
    "joint_lShoulderRoll",
    "joint_rShoulderRoll",
    "left_wheel_joint",
    "right_wheel_joint",
    "joint_lShoulderYaw",
    "joint_rShoulderYaw",
    "joint_lElbow",
    "joint_rElbow",
    "joint_lWristRoll",
    "joint_rWristRoll",
    "joint_lWristPitch",
    "joint_rWristPitch",
]
NEMOV3_MUJOCO_JOINT_SIGNS = tuple(
    -1.0 if name in {"joint_lShoulderRoll", "joint_rShoulderRoll"} else 1.0
    for name in NEMOV3_NPZ_JOINT_NAMES
)
NEMOV3_AMP_LEG_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_calf_joint",
    "right_calf_joint",
)
NEMOV3_AMP_JOINT_POSITION_LIMITS = {
    "left_calf_joint": (0.0, 1.5708),
    "right_calf_joint": (0.0, 1.5708),
}
NEMOV3_AMP_JOINT_VELOCITY_LIMIT_RAD_S = 10.0
RATE_LIMIT_NUMERICAL_MARGIN = 1.0e-5
PERIODIC_JOINT_ENDPOINT_TOLERANCE_RAD = 5.0e-3

# The Fast03 source uses X=right, Y=up, Z=forward, while Isaac/MuJoCo use
# X=forward, Y=left, Z=up.  Its polar-vector positions therefore map as
# (x, y, z) -> (z, -x, y).  The legacy ``axis_remap`` path retains its old
# positive lateral mapping so existing datasets remain reproducible; Fast03
# uses the explicit bind-relative mode.  Move the scalar component first for
# the legacy quaternion path:
# (qx, qy, qz, qw) -> (qw, qz, qx, qy).
SOURCE_XYZW_TO_ISAAC_WXYZ = (3, 2, 0, 1)
FORWARD_HEADING_EPSILON = 1.0e-6
ROOT_ORIENTATION_MODE_AXIS_REMAP = "axis_remap"
ROOT_ORIENTATION_MODE_BIND_RELATIVE_INVERSE = "bind_relative_inverse"
ROOT_ORIENTATION_MODES = (
    ROOT_ORIENTATION_MODE_AXIS_REMAP,
    ROOT_ORIENTATION_MODE_BIND_RELATIVE_INVERSE,
)


@dataclass(frozen=True)
class CsvMotion:
    time: np.ndarray
    fps: int
    joint_pos: np.ndarray
    root_pos: np.ndarray
    root_quat_wxyz: np.ndarray
    root_orientation_mode: str
    root_bind_quat_xyzw: np.ndarray


def joint_qpos_addresses(model: mujoco.MjModel, joint_names: list[str]) -> np.ndarray:
    addresses = []
    missing = []
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            missing.append(joint_name)
        else:
            addresses.append(model.jnt_qposadr[joint_id])
    if missing:
        raise ValueError(f"MJCF is missing joints: {missing}")
    return np.asarray(addresses, dtype=np.int32)


def root_qpos_address(model: mujoco.MjModel, joint_name: str = "root") -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"MJCF is missing root joint: {joint_name}")
    if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(f"Root joint is not a freejoint: {joint_name}")
    return int(model.jnt_qposadr[joint_id])


def load_kinematic_model(mjcf_path: Path) -> mujoco.MjModel:
    """Load a mesh-free copy of the tracked MJCF for deterministic FK."""
    root = ET.parse(mjcf_path).getroot()
    for asset in root.findall("asset"):
        for child in list(asset):
            if child.tag in {"mesh", "material", "texture"}:
                asset.remove(child)
    for body in root.iter("body"):
        for child in list(body):
            if child.tag == "geom":
                body.remove(child)
    worldbody = root.find("worldbody")
    if worldbody is not None:
        for child in list(worldbody):
            if child.tag in {"geom", "camera", "light"}:
                worldbody.remove(child)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    missing_bodies = [
        name
        for name in FK_MJCF_BODY_NAMES
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) < 0
    ]
    if missing_bodies:
        raise ValueError(f"MJCF is missing AMP bodies: {missing_bodies}")
    return model


def _read_numeric_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"CSV is empty: {path}") from exc
        if len(header) != len(set(header)):
            raise ValueError(f"CSV contains duplicate column names: {path}")
        rows = []
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(f"CSV row {line_number} has {len(row)} values; expected {len(header)}")
            try:
                rows.append([float(value) for value in row])
            except ValueError as exc:
                raise ValueError(f"CSV row {line_number} contains a non-numeric value") from exc

    if len(rows) < 2:
        raise ValueError(f"CSV must contain at least two frames: {path}")
    data = np.asarray(rows, dtype=np.float64)
    if not np.all(np.isfinite(data)):
        raise ValueError(f"CSV contains non-finite values: {path}")
    return header, data


def _normalized_quaternions_wxyz(quat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("CSV contains a near-zero root quaternion")
    normalized = quat / norms
    for frame in range(1, normalized.shape[0]):
        if np.dot(normalized[frame - 1], normalized[frame]) < 0.0:
            normalized[frame] *= -1.0
    return normalized


def _quat_multiply_wxyz(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(lhs, -1, 0)
    rw, rx, ry, rz = np.moveaxis(rhs, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def _quat_conjugate_wxyz(quaternion: np.ndarray) -> np.ndarray:
    conjugate = quaternion.copy()
    conjugate[..., 1:] *= -1.0
    return conjugate


def _convert_root_quaternion_xyzw(
    root_quat_xyzw: np.ndarray,
    root_orientation_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert the complete source root orientation to MuJoCo ``wxyz``.

    ``bind_relative_inverse`` cancels the source root-bone bind rotation and
    converts its local rotation direction to the active MuJoCo convention:
    ``q_target(t) = inverse(q_source(t)) * q_source(0)``.  The operation uses
    the complete quaternion; no component is interpreted as an Euler angle.
    """
    if root_orientation_mode not in ROOT_ORIENTATION_MODES:
        raise ValueError(
            f"Unknown root orientation mode {root_orientation_mode!r}; "
            f"expected one of {ROOT_ORIENTATION_MODES}"
        )
    source_wxyz = _normalized_quaternions_wxyz(
        root_quat_xyzw[:, (3, 0, 1, 2)]
    )
    bind_wxyz = source_wxyz[0].copy()
    if root_orientation_mode == ROOT_ORIENTATION_MODE_BIND_RELATIVE_INVERSE:
        target_wxyz = _normalized_quaternions_wxyz(
            _quat_multiply_wxyz(
                _quat_conjugate_wxyz(source_wxyz),
                np.broadcast_to(bind_wxyz, source_wxyz.shape),
            )
        )
    else:
        target_wxyz = _normalized_quaternions_wxyz(
            root_quat_xyzw[:, SOURCE_XYZW_TO_ISAAC_WXYZ]
        )
    return target_wxyz, bind_wxyz[[1, 2, 3, 0]]


def _align_mean_heading_to_displacement(
    root_pos: np.ndarray, root_quat_wxyz: np.ndarray
) -> np.ndarray:
    """Remove one global yaw offset while preserving the clip's yaw dynamics."""
    displacement_xy = root_pos[-1, :2] - root_pos[0, :2]
    displacement_norm = float(np.linalg.norm(displacement_xy))
    if displacement_norm <= FORWARD_HEADING_EPSILON:
        return root_quat_wxyz
    travel_yaw = float(np.arctan2(displacement_xy[1], displacement_xy[0]))
    w, x, y, z = np.moveaxis(root_quat_wxyz, -1, 0)
    heading_yaw = np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    frame_travel = np.gradient(root_pos[:, :2], axis=0)
    travel_weights = np.linalg.norm(frame_travel, axis=1)
    weighted_heading = np.sum(travel_weights * np.exp(1j * heading_yaw))
    if abs(weighted_heading) <= FORWARD_HEADING_EPSILON:
        return root_quat_wxyz
    mean_heading_yaw = float(np.angle(weighted_heading))
    half_yaw_offset = 0.5 * (travel_yaw - mean_heading_yaw)
    correction = np.array(
        (
            np.cos(half_yaw_offset),
            0.0,
            0.0,
            np.sin(half_yaw_offset),
        ),
        dtype=np.float64,
    )
    correction = np.broadcast_to(correction, root_quat_wxyz.shape)
    return _normalized_quaternions_wxyz(
        _quat_multiply_wxyz(correction, root_quat_wxyz)
    )


def _causal_rate_limit(values: np.ndarray, maximum_step: float) -> np.ndarray:
    limited = values.copy()
    for frame in range(1, limited.shape[0]):
        limited[frame] = np.clip(
            values[frame],
            limited[frame - 1] - maximum_step,
            limited[frame - 1] + maximum_step,
        )
    return limited


def _make_amp_joint_trajectory_reachable(
    joint_pos: np.ndarray,
    fps: int,
    periodic_endpoint_tolerance: float = PERIODIC_JOINT_ENDPOINT_TOLERANCE_RAD,
) -> np.ndarray:
    """Apply NemoV3 pose/rate limits before FK so all AMP features agree."""
    if not np.isfinite(periodic_endpoint_tolerance) or periodic_endpoint_tolerance < 0.0:
        raise ValueError(
            "periodic_endpoint_tolerance must be finite and non-negative, "
            f"got {periodic_endpoint_tolerance}"
        )
    endpoint_error = float(np.max(np.abs(joint_pos[-1] - joint_pos[0])))
    if endpoint_error > periodic_endpoint_tolerance:
        raise ValueError(
            "AMP CSV joint trajectory is not periodic: maximum endpoint error "
            f"is {endpoint_error:g} rad (limit {periodic_endpoint_tolerance:g})"
        )
    reachable = joint_pos.copy()
    name_to_index = {
        name: index for index, name in enumerate(NEMOV3_NPZ_JOINT_NAMES)
    }
    for name, (lower, upper) in NEMOV3_AMP_JOINT_POSITION_LIMITS.items():
        np.clip(reachable[:, name_to_index[name]], lower, upper, out=reachable[:, name_to_index[name]])

    # The CSV contains a duplicate endpoint.  Close it before filtering so the
    # output remains a loop and central differences agree at both boundaries.
    reachable[-1] = reachable[0]
    maximum_step = (
        NEMOV3_AMP_JOINT_VELOCITY_LIMIT_RAD_S
        * (1.0 - RATE_LIMIT_NUMERICAL_MARGIN)
        / float(fps)
    )
    for name in NEMOV3_AMP_LEG_JOINT_NAMES:
        index = name_to_index[name]
        values = reachable[:, index]
        forward = _causal_rate_limit(values, maximum_step)
        backward = _causal_rate_limit(values[::-1], maximum_step)[::-1]
        reachable[:, index] = 0.5 * (forward + backward)

    reachable[-1] = reachable[0]
    leg_indices = [name_to_index[name] for name in NEMOV3_AMP_LEG_JOINT_NAMES]
    maximum_observed_step = float(
        np.max(np.abs(np.diff(reachable[:, leg_indices], axis=0)))
    )
    if maximum_observed_step > maximum_step * (1.0 + 1.0e-9):
        raise ValueError(
            "CSV joint trajectory cannot be made periodic at the configured "
            f"{NEMOV3_AMP_JOINT_VELOCITY_LIMIT_RAD_S:g} rad/s limit"
        )
    return reachable


def load_csv_motion(
    path: Path,
    root_height: float = 0.51,
    root_orientation_mode: str = ROOT_ORIENTATION_MODE_AXIS_REMAP,
    periodic_endpoint_tolerance: float = PERIODIC_JOINT_ENDPOINT_TOLERANCE_RAD,
) -> CsvMotion:
    header, data = _read_numeric_csv(path)
    column_index = {name: index for index, name in enumerate(header)}

    time_column = next((name for name in TIME_COLUMNS if name in column_index), None)
    if time_column is None:
        raise ValueError(f"CSV is missing a time column; expected one of {TIME_COLUMNS}")
    missing_joints = [name for name in NEMOV3_NPZ_JOINT_NAMES if name not in column_index]
    if missing_joints:
        raise ValueError(f"CSV is missing motion joints: {missing_joints}")

    time_values = data[:, column_index[time_column]]
    time_values = time_values - time_values[0]
    frame_dt = np.diff(time_values)
    if np.any(frame_dt <= 0.0):
        raise ValueError("CSV timestamps must be strictly increasing")
    median_dt = float(np.median(frame_dt))
    fps = int(round(1.0 / median_dt))
    if fps <= 0 or not np.allclose(frame_dt, 1.0 / fps, rtol=1e-3, atol=1e-6):
        raise ValueError(f"CSV timestamps do not have a stable integer frame rate: median dt={median_dt:g}")

    joint_pos = np.column_stack([data[:, column_index[name]] for name in NEMOV3_NPZ_JOINT_NAMES])
    joint_pos = _make_amp_joint_trajectory_reachable(
        joint_pos,
        fps,
        periodic_endpoint_tolerance=periodic_endpoint_tolerance,
    )

    root_columns = (*ROOT_POSITION_COLUMNS, *ROOT_QUATERNION_COLUMNS)
    present_root_columns = [name for name in root_columns if name in column_index]
    if present_root_columns and len(present_root_columns) != len(root_columns):
        missing_root = [name for name in root_columns if name not in column_index]
        raise ValueError(f"CSV has an incomplete root pose; missing columns: {missing_root}")
    if present_root_columns:
        source_root_pos = np.column_stack([data[:, column_index[name]] for name in ROOT_POSITION_COLUMNS])
        lateral_sign = (
            -1.0
            if root_orientation_mode == ROOT_ORIENTATION_MODE_BIND_RELATIVE_INVERSE
            else 1.0
        )
        root_pos = np.column_stack(
            (
                source_root_pos[:, 2],
                lateral_sign * source_root_pos[:, 0],
                root_height + source_root_pos[:, 1] - source_root_pos[0, 1],
            )
        )
        root_quat_xyzw = np.column_stack([data[:, column_index[name]] for name in ROOT_QUATERNION_COLUMNS])
        root_quat_wxyz, root_bind_quat_xyzw = _convert_root_quaternion_xyzw(
            root_quat_xyzw,
            root_orientation_mode,
        )
        if root_orientation_mode == ROOT_ORIENTATION_MODE_AXIS_REMAP:
            root_quat_wxyz = _align_mean_heading_to_displacement(
                root_pos, root_quat_wxyz
            )
    else:
        root_pos = np.zeros((data.shape[0], 3), dtype=np.float64)
        root_pos[:, 2] = root_height
        root_quat_wxyz = np.zeros((data.shape[0], 4), dtype=np.float64)
        root_quat_wxyz[:, 0] = 1.0
        root_bind_quat_xyzw = np.asarray((0.0, 0.0, 0.0, 1.0))

    return CsvMotion(
        time=time_values,
        fps=fps,
        joint_pos=joint_pos,
        root_pos=root_pos,
        root_quat_wxyz=root_quat_wxyz,
        root_orientation_mode=root_orientation_mode,
        root_bind_quat_xyzw=root_bind_quat_xyzw,
    )


def _angular_velocity_w(quat_wxyz: np.ndarray, time_values: np.ndarray) -> np.ndarray:
    continuous = quat_wxyz.copy()
    for frame in range(1, continuous.shape[0]):
        flip = np.sum(continuous[frame - 1] * continuous[frame], axis=-1) < 0.0
        continuous[frame, flip] *= -1.0
    quat_dot = np.gradient(continuous, time_values, axis=0)
    conjugate = continuous.copy()
    conjugate[..., 1:] *= -1.0
    return 2.0 * _quat_multiply_wxyz(quat_dot, conjugate)[..., 1:]


def build_npz_arrays(motion: CsvMotion, model: mujoco.MjModel) -> dict[str, np.ndarray]:
    qpos_addresses = joint_qpos_addresses(model, NEMOV3_NPZ_JOINT_NAMES)
    root_qpos_adr = root_qpos_address(model)
    mujoco_joint_pos = motion.joint_pos * np.asarray(NEMOV3_MUJOCO_JOINT_SIGNS)
    sim_data = mujoco.MjData(model)

    frames = motion.joint_pos.shape[0]
    body_count = model.nbody - 1
    body_pos_w = np.empty((frames, body_count, 3), dtype=np.float64)
    body_quat_w = np.empty((frames, body_count, 4), dtype=np.float64)

    for frame in range(frames):
        sim_data.qpos[:] = 0.0
        sim_data.qpos[root_qpos_adr : root_qpos_adr + 3] = motion.root_pos[frame]
        sim_data.qpos[root_qpos_adr + 3 : root_qpos_adr + 7] = motion.root_quat_wxyz[frame]
        sim_data.qpos[qpos_addresses] = mujoco_joint_pos[frame]
        mujoco.mj_forward(model, sim_data)
        body_pos_w[frame] = sim_data.xpos[1:]
        body_quat_w[frame] = sim_data.xquat[1:]

    joint_vel = np.gradient(motion.joint_pos, motion.time, axis=0)
    body_lin_vel_w = np.gradient(body_pos_w, motion.time, axis=0)
    body_ang_vel_w = _angular_velocity_w(body_quat_w, motion.time)
    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
        for body_id in range(1, model.nbody)
    ]

    arrays = {
        "fps": np.asarray([motion.fps], dtype=np.int64),
        "joint_pos": motion.joint_pos.astype(np.float32),
        "joint_vel": joint_vel.astype(np.float32),
        "body_pos_w": body_pos_w.astype(np.float32),
        "body_quat_w": body_quat_w.astype(np.float32),
        "body_lin_vel_w": body_lin_vel_w.astype(np.float32),
        "body_ang_vel_w": body_ang_vel_w.astype(np.float32),
        "joint_names": np.asarray(NEMOV3_NPZ_JOINT_NAMES),
        "body_names": np.asarray(body_names),
    }
    if motion.root_orientation_mode == ROOT_ORIENTATION_MODE_BIND_RELATIVE_INVERSE:
        arrays["root_orientation_mode"] = np.asarray(motion.root_orientation_mode)
        arrays["root_bind_quat_xyzw"] = motion.root_bind_quat_xyzw.astype(np.float32)
    return arrays


def convert_csv(
    input_path: Path,
    output_path: Path,
    model: mujoco.MjModel,
    root_height: float = 0.51,
    root_orientation_mode: str = ROOT_ORIENTATION_MODE_AXIS_REMAP,
    periodic_endpoint_tolerance: float = PERIODIC_JOINT_ENDPOINT_TOLERANCE_RAD,
) -> None:
    motion = load_csv_motion(
        input_path,
        root_height=root_height,
        root_orientation_mode=root_orientation_mode,
        periodic_endpoint_tolerance=periodic_endpoint_tolerance,
    )
    arrays = build_npz_arrays(motion, model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert NemoV3 CSV motion files to replayable NPZ files.")
    parser.add_argument("input_files", nargs="+", type=Path, help="One or more input CSV files.")
    parser.add_argument("--output_dir", type=Path, default=Path("data/npz"))
    parser.add_argument("--mjcf_file", type=Path, default=DEFAULT_MJCF_FILE)
    parser.add_argument(
        "--root_height",
        type=float,
        default=0.51,
        help="Initial MuJoCo root height; CSV Y-axis height changes are preserved relative to the first frame.",
    )
    parser.add_argument(
        "--root_orientation_mode",
        choices=ROOT_ORIENTATION_MODES,
        default=ROOT_ORIENTATION_MODE_AXIS_REMAP,
        help=(
            "Root quaternion convention. Fast03 uses bind_relative_inverse; "
            "axis_remap preserves the legacy converter behavior."
        ),
    )
    parser.add_argument(
        "--periodic_endpoint_tolerance",
        type=float,
        default=PERIODIC_JOINT_ENDPOINT_TOLERANCE_RAD,
        help=(
            "Maximum source joint endpoint mismatch, in radians, accepted before "
            "closing and rate-limiting the periodic trajectory."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not np.isfinite(args.root_height):
        raise ValueError(f"root_height must be finite, got {args.root_height}")
    if not args.mjcf_file.is_file():
        raise FileNotFoundError(f"MJCF file not found: {args.mjcf_file}")

    model = load_kinematic_model(args.mjcf_file)
    for input_path in args.input_files:
        if not input_path.is_file():
            raise FileNotFoundError(f"CSV file not found: {input_path}")
        output_path = args.output_dir / f"{input_path.stem}.npz"
        convert_csv(
            input_path,
            output_path,
            model=model,
            root_height=args.root_height,
            root_orientation_mode=args.root_orientation_mode,
            periodic_endpoint_tolerance=args.periodic_endpoint_tolerance,
        )
        print(f"[INFO] Converted {input_path} -> {output_path}")


if __name__ == "__main__":
    main()
