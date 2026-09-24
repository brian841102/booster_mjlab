"""Replay a converted whole-body tracking NPZ in MuJoCo.

Example:

    python data/replay_npz_mujoco.py \
      --robot nemov3 \
      --motion_file data/npz/nemov3_skating2.npz

For non-window validation:

    python data/replay_npz_mujoco.py \
      --robot nemov3 \
      --motion_file data/npz/nemov3_skating2.npz \
      --headless --max_frames 120
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


# The converted NPZ stores joint_pos in IsaacLab articulation order, not in the
# MJCF/XML declaration order. Keep this list aligned with the NPZ columns and
# resolve MuJoCo qpos addresses by joint name.
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


ROBOT_CONFIGS = {
    "nemov3": {
        "mjcf_file": REPO_ROOT / "asset/nemov3_trainv2/mjcf/nemov3_601.xml",
        "root_body_name": "link_leg2pelvis",
        "root_joint_name": "root",
        "npz_joint_names": NEMOV3_NPZ_JOINT_NAMES,
        "mujoco_joint_signs": NEMOV3_MUJOCO_JOINT_SIGNS,
    },
    "nemov3_old": {
        "mjcf_file": REPO_ROOT / "asset/nemov3_old/mjcf/full_body_nemov3.xml",
        "root_body_name": "link_pelvisTorso",
        "root_joint_name": "root",
        "npz_joint_names": NEMOV3_NPZ_JOINT_NAMES,
        "mujoco_joint_signs": (1.0,) * len(NEMOV3_NPZ_JOINT_NAMES),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay converted NPZ motion data in MuJoCo.")
    parser.add_argument("--robot", choices=sorted(ROBOT_CONFIGS), required=True)
    parser.add_argument("--motion_file", type=Path, required=True)
    parser.add_argument("--mjcf_file", type=Path, default=None, help="Override the robot MJCF file.")
    parser.add_argument("--max_frames", type=int, default=-1, help="Maximum frames to replay. Negative means all.")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--headless", action="store_true", help="Validate playback without opening the viewer.")
    parser.add_argument("--print_every", type=int, default=120, help="Frame interval for progress logs.")
    return parser.parse_args()


def joint_qpos_addresses(model: mujoco.MjModel, joint_names: list[str]) -> np.ndarray:
    addresses = []
    missing = []
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            missing.append(joint_name)
            continue
        addresses.append(model.jnt_qposadr[joint_id])
    if missing:
        raise ValueError(f"MJCF is missing joints: {missing}")
    return np.asarray(addresses, dtype=np.int32)


def root_qpos_address(model: mujoco.MjModel, joint_name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"MJCF is missing root joint: {joint_name}")
    if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(f"Root joint is not a freejoint: {joint_name}")
    return int(model.jnt_qposadr[joint_id])


def map_joint_positions(joint_pos: np.ndarray, joint_signs: tuple[float, ...]) -> np.ndarray:
    values = np.asarray(joint_pos, dtype=np.float64)
    signs = np.asarray(joint_signs, dtype=np.float64)
    if values.ndim == 0 or values.shape[-1] != signs.shape[0]:
        raise ValueError(f"Expected joint positions ending in shape ({signs.shape[0]},), got {values.shape}")
    return values * signs


def validate_motion(data: np.lib.npyio.NpzFile, joint_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    required_keys = ["fps", "joint_pos", "body_pos_w", "body_quat_w"]
    missing = [key for key in required_keys if key not in data.files]
    if missing:
        raise ValueError(f"Motion NPZ is missing keys: {missing}")

    joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
    if joint_pos.ndim != 2 or joint_pos.shape[0] == 0:
        raise ValueError(f"Motion must contain at least one frame, got joint_pos shape {joint_pos.shape}")
    root_pos = np.asarray(data["body_pos_w"][:, 0, :], dtype=np.float64)
    root_quat = np.asarray(data["body_quat_w"][:, 0, :], dtype=np.float64)
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])

    if joint_pos.shape[1] != joint_count:
        raise ValueError(f"Expected joint_pos shape (frames, {joint_count}), got {joint_pos.shape}")
    if root_pos.shape != (joint_pos.shape[0], 3):
        raise ValueError(f"Expected root_pos shape {(joint_pos.shape[0], 3)}, got {root_pos.shape}")
    if root_quat.shape != (joint_pos.shape[0], 4):
        raise ValueError(f"Expected root_quat shape {(joint_pos.shape[0], 4)}, got {root_quat.shape}")
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"Invalid fps: {fps}")
    if not np.all(np.isfinite(joint_pos)) or not np.all(np.isfinite(root_pos)) or not np.all(np.isfinite(root_quat)):
        raise ValueError("Motion contains non-finite values.")

    quat_norm = np.linalg.norm(root_quat, axis=1, keepdims=True)
    if np.any(quat_norm < 1e-6):
        raise ValueError("Motion contains near-zero root quaternions.")
    root_quat = root_quat / quat_norm

    return joint_pos, root_pos, root_quat, fps


def apply_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    root_qpos_adr: int,
    qpos_addresses: np.ndarray,
    root_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,
    joint_pos: np.ndarray,
) -> None:
    data.qpos[:] = 0.0
    data.qpos[root_qpos_adr : root_qpos_adr + 3] = root_pos
    data.qpos[root_qpos_adr + 3 : root_qpos_adr + 7] = root_quat_wxyz
    data.qpos[qpos_addresses] = joint_pos
    mujoco.mj_forward(model, data)


def frame_range(num_frames: int, start_frame: int, max_frames: int) -> range:
    if num_frames <= 0:
        raise ValueError("Motion must contain at least one frame.")
    if start_frame < 0 or start_frame >= num_frames:
        raise ValueError(f"start_frame must be in [0, {num_frames - 1}], got {start_frame}")
    if max_frames == 0:
        raise ValueError("max_frames must be negative or greater than zero.")
    start = start_frame
    stop = num_frames if max_frames < 0 else min(num_frames, start + max_frames)
    return range(start, stop)


def run_headless(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    root_qpos_adr: int,
    qpos_addresses: np.ndarray,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    joint_pos: np.ndarray,
    frames: range,
    root_body_id: int,
) -> None:
    for frame_idx in frames:
        apply_frame(
            model,
            data,
            root_qpos_adr,
            qpos_addresses,
            root_pos[frame_idx],
            root_quat[frame_idx],
            joint_pos[frame_idx],
        )
        if not np.all(np.isfinite(data.qpos)):
            raise ValueError(f"Non-finite qpos at frame {frame_idx}")
        if not np.all(np.isfinite(data.xpos[root_body_id])):
            raise ValueError(f"Non-finite root body position at frame {frame_idx}")
    print(f"[INFO] Headless validation completed for {len(frames)} frame(s).")


def run_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    root_qpos_adr: int,
    qpos_addresses: np.ndarray,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    joint_pos: np.ndarray,
    frames: range,
    fps: float,
    speed: float,
    loop: bool,
    root_body_id: int,
    print_every: int,
) -> None:
    import mujoco.viewer

    frame_ids = list(frames)
    if not frame_ids:
        raise ValueError("No frames selected for playback.")

    frame_dt = 1.0 / (fps * max(speed, 1e-6))
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.distance = 2.8
        viewer.cam.elevation = -12
        viewer.cam.azimuth = 180

        while viewer.is_running():
            for local_idx, frame_idx in enumerate(frame_ids):
                start_time = time.perf_counter()
                apply_frame(
                    model,
                    data,
                    root_qpos_adr,
                    qpos_addresses,
                    root_pos[frame_idx],
                    root_quat[frame_idx],
                    joint_pos[frame_idx],
                )
                viewer.cam.lookat[:] = data.xpos[root_body_id]
                viewer.sync()

                if print_every > 0 and local_idx % print_every == 0:
                    print(f"[INFO] frame {frame_idx}/{len(root_pos) - 1}")

                elapsed = time.perf_counter() - start_time
                sleep_time = frame_dt - elapsed
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
                if not viewer.is_running():
                    break
            if not loop:
                break


def main() -> None:
    args = parse_args()
    if args.speed <= 0.0:
        raise ValueError(f"speed must be greater than zero, got {args.speed}")
    robot_cfg = ROBOT_CONFIGS[args.robot]
    mjcf_file = args.mjcf_file or robot_cfg["mjcf_file"]

    if not args.motion_file.exists():
        raise FileNotFoundError(f"Motion file not found: {args.motion_file}")
    if not mjcf_file.exists():
        raise FileNotFoundError(f"MJCF file not found: {mjcf_file}")

    model = mujoco.MjModel.from_xml_path(str(mjcf_file))
    data = mujoco.MjData(model)
    root_qpos_adr = root_qpos_address(model, robot_cfg["root_joint_name"])
    qpos_addresses = joint_qpos_addresses(model, robot_cfg["npz_joint_names"])

    root_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot_cfg["root_body_name"])
    if root_body_id < 0:
        raise ValueError(f"MJCF is missing root body: {robot_cfg['root_body_name']}")

    with np.load(args.motion_file) as motion:
        joint_pos, root_pos, root_quat, fps = validate_motion(motion, len(robot_cfg["npz_joint_names"]))
    mujoco_joint_pos = map_joint_positions(joint_pos, robot_cfg["mujoco_joint_signs"])
    frames = frame_range(joint_pos.shape[0], args.start_frame, args.max_frames)

    print(f"[INFO] Robot: {args.robot}")
    print(f"[INFO] MJCF: {mjcf_file}")
    print(f"[INFO] Motion: {args.motion_file}")
    print(f"[INFO] Frames: {joint_pos.shape[0]}, selected: {len(frames)}")
    print(f"[INFO] FPS: {fps:g}, speed: {args.speed:g}")
    print(f"[INFO] Root z min/max: {root_pos[:, 2].min():.4f} / {root_pos[:, 2].max():.4f}")

    if args.headless:
        run_headless(
            model,
            data,
            root_qpos_adr,
            qpos_addresses,
            root_pos,
            root_quat,
            mujoco_joint_pos,
            frames,
            root_body_id,
        )
    else:
        run_viewer(
            model,
            data,
            root_qpos_adr,
            qpos_addresses,
            root_pos,
            root_quat,
            mujoco_joint_pos,
            frames,
            fps,
            args.speed,
            args.loop,
            root_body_id,
            args.print_every,
        )


if __name__ == "__main__":
    main()
