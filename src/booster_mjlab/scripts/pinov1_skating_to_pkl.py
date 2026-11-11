"""Convert the Pino v1 skating NPZ into an AMP-ready GMR-format .pkl (same format as nemov3_skating_to_pkl.py).

The NPZ (``assets/amp_sources/pinov1_skating/data/csv_to_npz.py``) holds the 21 retargeted joints in the retarget
model's order and body poses from forward kinematics on pinov1.xml. Here:

* ``dof_pos`` is reordered into PINOV1_JOINT_ORDER (the XML order the simulated robot uses); the four wheel joints were
  not retargeted and are 0 (AMP features never include them);
* the root orientation goes from wxyz (MuJoCo) to xyzw (scipy / MotionFile);
* joint angles outside the soft joint limits (PINOV1_ARTICULATION, 95 % of the range, what the dof_pos_limits
  penalty uses) are clamped onto them: the retargeting clipped hip roll and ankle pitch at their hard limits, and a
  reference sitting there would make imitating it cost the limit penalty -- moving the reference in by 1-3 deg keeps
  the policy 5 % away from the mechanical stops instead (user 2026-11-03: do not let the motion hit the limits);
* only the original clip is converted -- left/right balance comes from the mirror augmentation at training time, as
  for nemov3. The upstream mirrored CSV is used to check the mirror table (pinov1_mirror_maps) instead.

Datasets (``--dataset``):

* ``straight`` (default): the nemov3 Exp 33 clip set retargeted to pinov1 (``*_from_nemov3.csv``, user 2026-11-04) --
  Skating01, Skating02 and SkatingFast06 with its static lead-in trimmed by the same 1.133 s as nemov3's
  ``..._static_prefix_trim_transition_v1`` (68 of 1201 frames at 60 fps) -> ``assets/amp/pinov1_skating_straight``;
* ``op``: the first pinov1 clip (skating_op, 1.2 s trimmed) -> ``assets/amp/pinov1_skating``.

Usage (after csv_to_npz.py on the CSVs, with ``--start_s 1.1333`` for SkatingFast06 / ``--start_s 1.2`` for skating_op):

    uv run python -m booster_mjlab.scripts.pinov1_skating_to_pkl [--dataset straight|op]
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from booster_mjlab.robots.pinov1.pinov1_constants import (
    PINOV1_ARTICULATION,
    PINOV1_JOINT_ORDER,
    PINOV1_XML,
    pinov1_mirror_maps,
)

SOURCE_DIR = Path("assets/amp_sources/pinov1_skating/data/npz")
ROOT_BODY_NAME = "pelvis"

# The upstream mirrored CSV keeps the wrist angle's sign, but both wrist axes point along +x at the zero pose, so a
# true left/right reflection flips it: by forward kinematics our table reproduces the reflected wrist orientation to
# 0.015 (the left axis is tilted 0.8 deg), the upstream clip is off by up to 0.71 (checked 2026-11-03). Wrists are
# therefore left out of the comparison.
UPSTREAM_MIRROR_WRONG = ("left_wrist_joint", "right_wrist_joint")

# dataset -> (output dir, clips); clip = (output name, source NPZ, its mirrored counterpart used only to check the mirror
# table, or None)
_NEMO = "Robot_Model0530_V2_{}_IK_control_from_nemov3"
DATASETS = {
    "straight": (
        Path("assets/amp/pinov1_skating_straight"),
        (
            ("skating01", _NEMO.format("Skating01") + ".npz", None),
            ("skating02", _NEMO.format("Skating02") + ".npz", None),
            ("skatingfast06", _NEMO.format("SkatingFast06") + "_from1.1333s.npz", None),
        ),
    ),
    "op": (
        Path("assets/amp/pinov1_skating"),
        (("skating_op", "skating_op_fps30_from1.2s.npz", "skating_op_mirror_fps30_from1.2s.npz"),),
    ),
}


def load_dof_pos(path: Path) -> tuple[np.ndarray, dict]:
    """dof_pos in PINOV1_JOINT_ORDER (wheels 0) plus the raw NPZ arrays."""
    with np.load(path) as npz:
        data = {k: npz[k] for k in npz.files}
    names = [str(n) for n in data["joint_names"]]
    dof_pos = np.zeros((len(data["joint_pos"]), len(PINOV1_JOINT_ORDER)), dtype=np.float32)
    for col, name in enumerate(PINOV1_JOINT_ORDER):
        if name in names:
            dof_pos[:, col] = data["joint_pos"][:, names.index(name)]
    return dof_pos, data


def check_mirror(dof_pos: np.ndarray, mirrored_path: Path) -> None:
    index_map, sign_map = pinov1_mirror_maps(PINOV1_JOINT_ORDER)
    expected = dof_pos[:, index_map] * np.asarray(sign_map, dtype=np.float32)
    actual, _ = load_dof_pos(mirrored_path)
    err = np.abs(expected - actual).max(0)
    err[[PINOV1_JOINT_ORDER.index(n) for n in UPSTREAM_MIRROR_WRONG]] = 0.0
    print(f"  mirror table vs {mirrored_path.name}: max error {err.max():.4f} rad (wrists excluded)")
    if err.max() > 1e-3:
        raise ValueError("pinov1_mirror_maps disagrees with the mirrored clip")


def clamp_to_soft_limits(dof_pos: np.ndarray) -> np.ndarray:
    """Clamp every limited joint onto its soft limits (mjlab: range centre +- half range x factor)."""
    model = mujoco.MjModel.from_xml_path(str(PINOV1_XML))
    factor = PINOV1_ARTICULATION.soft_joint_pos_limit_factor
    out = dof_pos.copy()
    for col, name in enumerate(PINOV1_JOINT_ORDER):
        joint = model.joint(name).id
        if not model.jnt_limited[joint]:
            continue
        lo, hi = model.jnt_range[joint]
        mid, half = (lo + hi) / 2, (hi - lo) / 2 * factor
        out[:, col] = np.clip(dof_pos[:, col], mid - half, mid + half)
        moved = np.abs(out[:, col] - dof_pos[:, col])
        if moved.max() > 0:
            print(f"  clamped {name} onto [{mid - half:+.3f}, {mid + half:+.3f}] in {np.mean(moved > 0) * 100:.0f}% of "
                  f"frames (max {moved.max():.3f} rad)")
    return out


def convert(output_dir: Path, output_name: str, source: str, mirrored: str | None) -> None:
    dof_pos, data = load_dof_pos(SOURCE_DIR / source)
    if mirrored is not None:
        check_mirror(dof_pos, SOURCE_DIR / mirrored)
    dof_pos = clamp_to_soft_limits(dof_pos)
    body_names = [str(n) for n in data["body_names"]]
    root_idx = body_names.index(ROOT_BODY_NAME)
    body_pos_w = data["body_pos_w"].astype(np.float64)
    root_pos = body_pos_w[:, root_idx].astype(np.float32)
    root_quat_xyzw = data["body_quat_w"][:, root_idx][:, [1, 2, 3, 0]].astype(np.float32)
    inv_rot = Rotation.from_quat(root_quat_xyzw).inv()
    relative = body_pos_w - root_pos[:, None, :]
    local_body_pos = np.stack([inv_rot[t].apply(relative[t]) for t in range(len(dof_pos))]).astype(np.float32)
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    out = {
        "fps": fps,
        "root_pos": root_pos,
        "root_rot": list(root_quat_xyzw),
        "dof_pos": dof_pos,
        "local_body_pos": local_body_pos,
        "link_body_list": body_names,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{output_name}.pkl"
    with out_path.open("wb") as f:
        pickle.dump(out, f)
    print(f"[{output_name}] {source} -> {out_path} ({len(dof_pos)} frames @ {fps:g} fps, {len(dof_pos) / fps:.2f} s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="straight")
    output_dir, clips = DATASETS[parser.parse_args().dataset]
    for clip in clips:
        convert(output_dir, *clip)


if __name__ == "__main__":
    main()
