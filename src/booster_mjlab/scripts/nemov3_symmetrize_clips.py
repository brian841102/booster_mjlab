"""Make every Nemov3 skating reference clip left/right symmetric on its own.

The mirror augmentation makes the reference set symmetric as a whole, but each clip (and so each discriminator
window) stays lopsided: Skating01 swings the right hip / calf / elbow 1.29 / 1.59 / 2.3x more than the left,
Skating02 alternates with an uneven R->L / L->R timing (0.56 of the period). The discriminator then accepts a
per-episode asymmetric gait, and a left/right-equivariant policy settles into one of the two mirror images.

For a gait of period P, a symmetric stride satisfies q(t) = M q(t - P/2). Each clip is projected onto that
subspace, which keeps the stride's frequency, mean amplitudes and arm/leg coordination and removes the
left/right-asymmetric part:
    q_sym(t) = 0.5 * (q(t) + M q(t - P/2))
Looping clips (Skating01: P = 80 frames, Skating02: P = 75) use a cyclic shift (linear interpolation for a half
frame); the long SkatingFast06 clip (P = 80) uses the non-cyclic shift and drops its first P/2 frames. Only the
joint angles are symmetrised (the AMP features are joint positions); the root pose of each frame is kept for the
reset pose pool.

Usage:
    uv run python -m booster_mjlab.scripts.nemov3_symmetrize_clips
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from booster_mjlab.robots.nemov3.nemov3_constants import NEMOV3_JOINT_ORDER, nemov3_mirror_maps

SRC = Path("assets/amp/nemov3_skating_straight")
DST = Path("assets/amp/nemov3_skating_symmetric")
# clip -> (period in frames at 60 fps, cyclic). Periods measured with a best-self-match search (the best
# half-period mirror lag is 40 frames for all three clips).
CLIPS = {"skating01": (80, True), "skating02": (75, True), "skatingfast06": (80, False)}
REPORT = {
    "hip_pitch": ("right_hip_pitch_joint", "left_hip_pitch_joint"),
    "calf": ("right_calf_joint", "left_calf_joint"),
    "shoulder_pitch": ("joint_rShoulderPitch", "joint_lShoulderPitch"),
    "elbow": ("joint_rElbow", "joint_lElbow"),
}


def _mirror(q: np.ndarray) -> np.ndarray:
    idx, sign = nemov3_mirror_maps(NEMOV3_JOINT_ORDER)
    return q[:, idx] * np.asarray(sign, dtype=q.dtype)


def _cyclic_shift(q: np.ndarray, shift: float) -> np.ndarray:
    """q(t - shift) on a loop of len(q) frames, linear interpolation for fractional shifts."""
    n = len(q)
    t = (np.arange(n) - shift) % n
    lo = np.floor(t).astype(int)
    w = (t - lo)[:, None]
    return (1.0 - w) * q[lo] + w * q[(lo + 1) % n]


def _stats(q: np.ndarray) -> str:
    j = NEMOV3_JOINT_ORDER.index
    parts = [f"{k} {q[:, j(r)].std() / (q[:, j(l)].std() + 1e-9):.2f}" for k, (r, l) in REPORT.items()]
    off = max(abs(q[:, j(r)].mean() - q[:, j(l)].mean()) for r, l in REPORT.values())
    return "R/L std " + ", ".join(parts) + f" | max |mean_R - mean_L| {off:.3f} rad"


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    for name, (period, cyclic) in CLIPS.items():
        d = pickle.load(open(SRC / f"{name}.pkl", "rb"))
        q = np.asarray(d["dof_pos"], dtype=np.float64)
        if cyclic:
            loop = q[:period]
            sym = 0.5 * (loop + _mirror(_cyclic_shift(loop, period / 2)))
            q_sym = np.concatenate([sym, sym[:1]])  # close the loop like the source clip
            # root pose of the original frames 0..period: the root keeps moving forward across the loop
            # (taking frame `period` from frame 0 would teleport the root back -> ~-48 m/s reset velocity)
            keep = np.arange(len(q_sym))
        else:
            h = period // 2
            q_sym = 0.5 * (q[h:] + _mirror(q[:-h]))
            keep = np.arange(h, len(q))
        out = {
            "fps": d["fps"],
            "root_pos": np.asarray(d["root_pos"])[keep],
            "root_rot": [d["root_rot"][i] for i in keep],
            "dof_pos": q_sym.astype(np.asarray(d["dof_pos"]).dtype),
            "local_body_pos": None,
            "link_body_list": None,
        }
        with open(DST / f"{name}.pkl", "wb") as f:
            pickle.dump(out, f)
        print(f"{name}: {len(q)} -> {len(q_sym)} frames (P = {period}, {'cyclic' if cyclic else 'non-cyclic'})")
        print(f"   before: {_stats(q)}")
        print(f"   after:  {_stats(q_sym)}")
        print(f"   removed asymmetric part: rms {np.sqrt(np.mean((q[keep] - q_sym) ** 2)):.3f} rad")
        v = np.linalg.norm(np.diff(out["root_pos"], axis=0), axis=1) * d["fps"]
        print(f"   root speed max {v.max():.2f} m/s (source {np.linalg.norm(np.diff(np.asarray(d['root_pos']), axis=0), axis=1).max() * d['fps']:.2f})")


if __name__ == "__main__":
    main()
