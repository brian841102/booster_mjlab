"""Convert Pino v1 retargeted skating CSV files to replayable whole-body NPZ files.

The CSVs have no header: 30 fps rows of root x y z, root quaternion qx qy qz qw (scalar last) and the 21 joints of
pino_joint21_nowheel_retarget.xml in its order (PINOV1_RETARGET_JOINT_ORDER). Forward kinematics on pinov1.xml (wheels
held at 0) gives the body poses. The NPZ keys match the nemov3 converter (fps, joint_names, joint_pos, joint_vel,
body_names, body_pos_w, body_quat_w wxyz, body_lin_vel_w, body_ang_vel_w), so
``booster_mjlab.scripts.pinov1_skating_to_pkl`` turns it into an AMP clip the same way.

Example (run from the repository root; drops the 1.2 s standstill / start-up lead-in, user 2026-11-03):

    uv run python assets/amp_sources/pinov1_skating/data/csv_to_npz.py \
        assets/amp_sources/pinov1_skating/data/csv/skating_op_fps30.csv --start_s 1.2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from booster_mjlab.robots.pinov1.pinov1_constants import (
    PINOV1_RETARGET_JOINT_ORDER,
    PINOV1_WHEEL_BODY_RADII,
    PINOV1_XML,
)

DATA_DIR = Path(__file__).resolve().parent
LIFT_THRESHOLD_M = 0.01


def load_csv(path: Path, start_s: float, fps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Root position, root quaternion (wxyz) and joint positions from ``start_s`` on."""
    rows = np.loadtxt(path, delimiter=",", ndmin=2)
    expected = 7 + len(PINOV1_RETARGET_JOINT_ORDER)
    if rows.shape[1] != expected:
        raise ValueError(f"{path}: expected {expected} columns, got {rows.shape[1]}")
    rows = rows[int(round(start_s * fps)) :]
    quat_xyzw = rows[:, 3:7]
    norm = np.linalg.norm(quat_xyzw, axis=1)
    if np.abs(norm - 1.0).max() > 1e-3:
        raise ValueError(f"{path}: root quaternion norm {norm.min():.4f}..{norm.max():.4f}")
    quat_wxyz = (quat_xyzw / norm[:, None])[:, [3, 0, 1, 2]]
    for frame in range(1, len(quat_wxyz)):  # keep the quaternion sign continuous for the velocity estimate
        if np.dot(quat_wxyz[frame - 1], quat_wxyz[frame]) < 0.0:
            quat_wxyz[frame] *= -1.0
    return rows[:, :3], quat_wxyz, rows[:, 7:]


def _angular_velocity_w(quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
    """World angular velocity from a (frames, bodies, 4) wxyz quaternion track: w = 2 q_dot q^-1."""
    q_dot = np.gradient(quat_wxyz, dt, axis=0)
    w, x, y, z = np.moveaxis(quat_wxyz * [1, -1, -1, -1], -1, 0)  # conjugate
    a, b, c, d = np.moveaxis(q_dot, -1, 0)
    return 2.0 * np.stack(
        [a * x + b * w + c * z - d * y, a * y - b * z + c * w + d * x, a * z + b * y - c * x + d * w], axis=-1
    )


def forward_kinematics(model: mujoco.MjModel, root_pos, root_quat_wxyz, joint_pos) -> tuple[np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    adr = [model.jnt_qposadr[model.joint(n).id] for n in PINOV1_RETARGET_JOINT_ORDER]
    body_pos_w = np.empty((len(joint_pos), model.nbody - 1, 3))
    body_quat_w = np.empty((len(joint_pos), model.nbody - 1, 4))
    for frame in range(len(joint_pos)):
        data.qpos[:] = 0.0
        data.qpos[:3] = root_pos[frame]
        data.qpos[3:7] = root_quat_wxyz[frame]
        data.qpos[adr] = joint_pos[frame]
        mujoco.mj_kinematics(model, data)
        body_pos_w[frame] = data.xpos[1:]
        body_quat_w[frame] = data.xquat[1:]
    return body_pos_w, body_quat_w


def report(name: str, model: mujoco.MjModel, fps: float, joint_pos, body_names, body_pos_w) -> None:
    """Print what decides whether a clip is usable: duration, speed, joint limits, wheel lift / penetration, seam."""
    root = body_pos_w[:, 0]
    speed = np.linalg.norm(np.gradient(root[:, :2], 1.0 / fps, axis=0), axis=1)
    print(f"{name}: {len(joint_pos)} frames = {len(joint_pos) / fps:.2f} s | speed {speed.min():.2f}..{speed.max():.2f} "
          f"m/s (mean {speed.mean():.2f}) | root height {root[:, 2].min():.3f}..{root[:, 2].max():.3f} m")
    for i, joint in enumerate(PINOV1_RETARGET_JOINT_ORDER):
        lo, hi = model.jnt_range[model.joint(joint).id]
        q = joint_pos[:, i]
        if q.min() < lo - 1e-3 or q.max() > hi + 1e-3:
            print(f"  WARNING {joint} {q.min():+.3f}..{q.max():+.3f} outside the limit [{lo:+.2f}, {hi:+.2f}]")
        elif min(q.min() - lo, hi - q.max()) < 1e-3:
            print(f"  note: {joint} reaches its limit [{lo:+.2f}, {hi:+.2f}]")
    for body, radius in PINOV1_WHEEL_BODY_RADII.items():
        bottom = body_pos_w[:, body_names.index(body), 2] - radius
        print(f"  {body:24s} bottom {bottom.min():+.3f}..{bottom.max():+.3f} m, "
              f"> {100 * LIFT_THRESHOLD_M:.0f} cm off the ground in {100 * np.mean(bottom > LIFT_THRESHOLD_M):.0f}% of frames")
    seam = np.abs(joint_pos[-1] - joint_pos[0])
    print(f"  seam last -> first frame: max joint jump {seam.max():.3f} rad ({PINOV1_RETARGET_JOINT_ORDER[seam.argmax()]})")


def convert(path: Path, output_dir: Path, start_s: float, fps: float, model: mujoco.MjModel) -> Path:
    root_pos, root_quat_wxyz, joint_pos = load_csv(path, start_s, fps)
    body_pos_w, body_quat_w = forward_kinematics(model, root_pos, root_quat_wxyz, joint_pos)
    body_names = [model.body(i).name for i in range(1, model.nbody)]
    dt = 1.0 / fps
    suffix = f"_from{start_s:g}s" if start_s > 0 else ""
    output_path = output_dir / f"{path.stem}{suffix}.npz"
    report(output_path.name, model, fps, joint_pos, body_names, body_pos_w)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        fps=np.asarray([fps]),
        joint_names=np.asarray(PINOV1_RETARGET_JOINT_ORDER),
        joint_pos=joint_pos.astype(np.float32),
        joint_vel=np.gradient(joint_pos, dt, axis=0).astype(np.float32),
        body_names=np.asarray(body_names),
        body_pos_w=body_pos_w.astype(np.float32),
        body_quat_w=body_quat_w.astype(np.float32),
        body_lin_vel_w=np.gradient(body_pos_w, dt, axis=0).astype(np.float32),
        body_ang_vel_w=_angular_velocity_w(body_quat_w, dt).astype(np.float32),
        source_csv=np.asarray(path.name),
        start_s=np.asarray([start_s]),
    )
    print(f"  -> {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input_files", nargs="+", type=Path)
    parser.add_argument("--output_dir", type=Path, default=DATA_DIR / "npz")
    parser.add_argument("--start_s", type=float, default=0.0, help="Drop the frames before this time (s).")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(PINOV1_XML))
    for path in args.input_files:
        convert(path, args.output_dir, args.start_s, args.fps, model)


if __name__ == "__main__":
    main()
