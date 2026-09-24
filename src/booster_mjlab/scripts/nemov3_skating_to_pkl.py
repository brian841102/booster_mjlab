"""Convert selected Nemov3 skating NPZ clips into AMP-ready GMR-format .pkl files.

The source NPZs (see ``assets/amp_sources/nemov3_skating/``) are already IK
joint trajectories for the Nemov3 wheeled-legged robot, but:

* their ``joint_pos`` columns are ordered by the *source* articulation
  (``joint_names`` stored in the npz), not by ``NEMOV3_JOINT_ORDER`` (the
  MJCF declaration order that ``MotionFile.dof_pos`` / the simulated robot's
  ``default_joint_pos`` use);
* the two shoulder-roll joints have their axis flipped relative to nemov3.xml
  (``NEMOV3_NPZ_TO_MJCF_JOINT_SIGNS``, same as the reference
  ``replay_npz_mujoco.py``); without it the arms end up raised;
* they don't cover the head or telescoping-arm ("Linear") joints at all
  (those weren't part of the original skating capture), so those columns
  are held at the robot's keyframe default for the whole clip;
* the root orientation (``body_quat_w``) is stored wxyz (IsaacLab/MuJoCo
  convention); ``MotionFile.root_rot`` expects scipy's xyzw convention.

Only the three clips agreed on for the first straight-line-forward (0-1.3
m/s) training pass are converted here: Skating01 (fast/large-stride),
Skating02 (slow/small-stride), and the *trimmed* SkatingFast06 (static
lead-in already removed) -- not the raw SkatingFast06_IK_control, which
still has ~76 static frames before the skating cycle starts.

Usage:

    uv run python -m booster_mjlab.scripts.nemov3_skating_to_pkl
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from booster_mjlab.robots.nemov3.nemov3_constants import (
    HOME_KEYFRAME,
    NEMOV3_JOINT_ORDER,
    NEMOV3_NPZ_TO_MJCF_JOINT_SIGNS,
)

SOURCE_DIR = Path("assets/amp_sources/nemov3_skating/data/npz")
OUTPUT_DIR = Path("assets/amp/nemov3_skating_straight")

ROOT_BODY_NAME = "link_leg2pelvis"


@dataclass(frozen=True)
class ClipSpec:
    output_name: str
    source_file: str


CLIPS: tuple[ClipSpec, ...] = (
    ClipSpec("skating01", "Robot_Model0530_V2_Skating01_IK_control.npz"),
    ClipSpec("skating02", "Robot_Model0530_V2_Skating02_IK_control.npz"),
    ClipSpec(
        "skatingfast06",
        "Robot_Model0530_V2_SkatingFast06_static_prefix_trim_transition_v1.npz",
    ),
)


def convert(spec: ClipSpec) -> None:
    src_path = SOURCE_DIR / spec.source_file
    with np.load(src_path, allow_pickle=True) as data:
        src_joint_names = [str(n) for n in data["joint_names"]]
        body_names = [str(n) for n in data["body_names"]]
        joint_pos_src = np.asarray(data["joint_pos"], dtype=np.float64)
        body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float64)
        body_quat_w_wxyz = np.asarray(data["body_quat_w"], dtype=np.float64)
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])

    num_frames = joint_pos_src.shape[0]
    root_idx = body_names.index(ROOT_BODY_NAME)

    # Reorder into NEMOV3_JOINT_ORDER (MJCF declaration order). Joints absent
    # from the source capture (head, telescoping arm) hold their keyframe
    # default for every frame.
    dof_pos = np.zeros((num_frames, len(NEMOV3_JOINT_ORDER)), dtype=np.float32)
    missing: list[str] = []
    for col, name in enumerate(NEMOV3_JOINT_ORDER):
        if name in src_joint_names:
            sign = NEMOV3_NPZ_TO_MJCF_JOINT_SIGNS.get(name, 1.0)
            dof_pos[:, col] = sign * joint_pos_src[:, src_joint_names.index(name)]
        else:
            dof_pos[:, col] = HOME_KEYFRAME.joint_pos.get(name, 0.0)
            missing.append(name)

    root_pos = body_pos_w[:, root_idx, :].astype(np.float32)

    # wxyz (IsaacLab/MuJoCo) -> xyzw (scipy / MotionFile convention).
    root_quat_wxyz = body_quat_w_wxyz[:, root_idx, :]
    root_quat_xyzw = root_quat_wxyz[:, [1, 2, 3, 0]].astype(np.float32)
    root_rot = Rotation.from_quat(root_quat_xyzw)

    # Per-body positions in the root-local frame (SMP-compatible extra; AMP
    # ignores this field).
    inv_rot = root_rot.inv()
    relative = body_pos_w - root_pos[:, None, :]
    local_body_pos = np.stack(
        [inv_rot[t].apply(relative[t]) for t in range(num_frames)], axis=0
    ).astype(np.float32)

    out = {
        "fps": fps,
        "root_pos": root_pos,
        "root_rot": list(root_quat_xyzw),
        "dof_pos": dof_pos,
        "local_body_pos": local_body_pos,
        "link_body_list": body_names,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{spec.output_name}.pkl"
    with out_path.open("wb") as f:
        pickle.dump(out, f)

    print(
        f"[{spec.output_name}] {src_path.name} -> {out_path} "
        f"({num_frames} frames @ {fps:g} fps, {num_frames / fps:.2f}s). "
        f"Held-at-default joints: {missing}"
    )


def main() -> None:
    for spec in CLIPS:
        convert(spec)


if __name__ == "__main__":
    main()
