"""Left/right gait symmetry across the speed range, judged PER EPISODE, with a pass/fail summary.

usage: TASK=<task> [EP_STEPS=750] asymmetry_eval.py <onnx> [episodes=6] [cmds=0.3,0.6,0.8,1.0,1.15,1.3] [tag]

Why per episode: a left/right-equivariant policy can still settle into an asymmetric gait / stance (a mirror
pair of attractors, e.g. "left foot forward" and "right foot forward"); which one depends on the initial state.
Pooling several episodes then averages the two mirror images and looks symmetric although every single run
(what one sees in the viewer) is asymmetric. So every statistic is computed per episode (first 100 steps
dropped) and the magnitudes are averaged over episodes:
  * timing: peaks of the smoothed right / left hip-pitch signals; t_RL = right peak -> next left peak,
    t_LR = left -> right; ratio_e = mean t_RL / (mean t_RL + mean t_LR); reported: mean_e |ratio_e - 0.5|.
  * amplitude: std_R / std_L per joint; reported: mean_e |log(ratio_e)| per joint (as a ratio exp(.)).
  * static offset: |mean_R - mean_L| per joint and episode; reported: mean over episodes.
    (All these joints have mirror sign +1 on nemov3, so a symmetric gait has equal L/R means.)
Pass (per speed): timing mean |dev| <= 0.05, every amplitude ratio in [0.85, 1.18] (i.e. exp(mean|log|) <= 1.18),
every static offset <= 0.05 rad. Episodes without a detectable alternation only enter the static-offset test.
The pooled (all episodes together) numbers are printed too, for comparison with older results.
"""

import os
import sys

import numpy as np
import onnxruntime as ort
import torch
from scipy.signal import find_peaks

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

import booster_mjlab.tasks.velocity.config.nemov3  # noqa: F401

PAIRS = {
    "hip_pitch": ("right_hip_pitch_joint", "left_hip_pitch_joint"),
    "calf": ("right_calf_joint", "left_calf_joint"),
    "hip_roll": ("right_hip_roll_joint", "left_hip_roll_joint"),
    "hip_yaw": ("right_hip_yaw_joint", "left_hip_yaw_joint"),
    "shoulder_pitch": ("joint_rShoulderPitch", "joint_lShoulderPitch"),
    "elbow": ("joint_rElbow", "joint_lElbow"),
}
AMP_KEYS = ("hip_pitch", "calf", "shoulder_pitch", "elbow")
TIMING_TOL, AMP_TOL, OFFSET_TOL = 0.05, np.log(1.18), 0.05


def smooth(x, k):
    return np.convolve(x, np.ones(k) / k, mode="same")


def intervals(r, l, dt):
    k = max(3, int(round(0.08 / dt)))
    r = smooth(r - r.mean(), k)
    l = smooth(l - l.mean(), k)
    if min(r.std(), l.std()) < 0.03:  # no real swing
        return [], []
    md = int(0.4 / dt)
    pr, _ = find_peaks(r, distance=md, prominence=0.4 * r.std())
    pl, _ = find_peaks(l, distance=md, prominence=0.4 * l.std())
    trl = [(pl[pl > a][0] - a) * dt for a in pr if (pl > a).any()]
    tlr = [(pr[pr > a][0] - a) * dt for a in pl if (pr > a).any()]
    return trl, tlr


def main():
    onnx_path = sys.argv[1]
    n_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    cmds = [float(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "0.3,0.6,0.8,1.0,1.15,1.3").split(",")]
    tag = sys.argv[4] if len(sys.argv) > 4 else os.path.basename(onnx_path)
    ep_steps = int(os.environ.get("EP_STEPS", 750))

    cfg = load_env_cfg(os.environ.get("TASK", "Mjlab-Velocity-Flat-Amp-Clock-Arms-Booster-Nemov3"), play=True)
    cfg.scene.num_envs = 1
    tw = cfg.commands["twist"]
    tw.ranges.lin_vel_y = (0.0, 0.0)
    tw.ranges.ang_vel_z = (0.0, 0.0)
    tw.rel_standing_envs = 0.0
    tw.rel_forward_envs = 0.0
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    robot = env.scene["robot"]
    term = env.command_manager.get_term("twist")
    dt = env.step_dt
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    nm = sess.get_inputs()[0].name
    ids = {k: [robot.joint_names.index(v[0]), robot.joint_names.index(v[1])] for k, v in PAIRS.items()}

    n_pass = 0
    for c in cmds:
        term.cfg.ranges.lin_vel_x = (c, c)
        ep_ratio, ep_amp, ep_off, speed = [], [], [], []
        pool_r = {k: [] for k in PAIRS}
        pool_l = {k: [] for k in PAIRS}
        pool_trl, pool_tlr = [], []
        for _ in range(n_ep):
            obs, _ = env.reset()
            er = {k: [] for k in PAIRS}
            el = {k: [] for k in PAIRS}
            for t in range(ep_steps):
                a = sess.run(None, {nm: obs["actor"].cpu().numpy().astype(np.float32)})[0]
                obs, *_ = env.step(torch.as_tensor(a, device="cuda:0"))
                if t >= 100:
                    q = robot.data.joint_pos[0].cpu().numpy()
                    for k, (ir, il) in ids.items():
                        er[k].append(q[ir])
                        el[k].append(q[il])
                    speed.append(robot.data.root_link_lin_vel_b[0, 0].item())
            er = {k: np.array(v) for k, v in er.items()}
            el = {k: np.array(v) for k, v in el.items()}
            for k in PAIRS:
                pool_r[k].append(er[k])
                pool_l[k].append(el[k])
            ep_off.append({k: abs(er[k].mean() - el[k].mean()) for k in PAIRS})
            trl, tlr = intervals(er["hip_pitch"], el["hip_pitch"], dt)
            if len(trl) >= 3 and len(tlr) >= 3:
                ep_ratio.append(np.mean(trl) / (np.mean(trl) + np.mean(tlr)))
                ep_amp.append({k: er[k].std() / (el[k].std() + 1e-9) for k in AMP_KEYS})
                pool_trl += trl
                pool_tlr += tlr

        timing_dev = np.mean([abs(x - 0.5) for x in ep_ratio]) if ep_ratio else float("nan")
        amp = {k: float(np.exp(np.mean([abs(np.log(e[k])) for e in ep_amp]))) for k in AMP_KEYS} if ep_amp else {}
        off = {k: float(np.mean([e[k] for e in ep_off])) for k in PAIRS}
        ok_timing = (not ep_ratio) or timing_dev <= TIMING_TOL
        ok_amp = all(np.log(v) <= AMP_TOL for v in amp.values())
        ok_off = all(v <= OFFSET_TOL for v in off.values())
        ok = ok_timing and ok_amp and ok_off
        n_pass += ok
        pr = {k: np.concatenate(v) for k, v in pool_r.items()}
        pl = {k: np.concatenate(v) for k, v in pool_l.items()}
        pooled_ratio = np.mean(pool_trl) / (np.mean(pool_trl) + np.mean(pool_tlr)) if pool_trl else float("nan")
        pooled_off = max(abs(pr[k].mean() - pl[k].mean()) for k in PAIRS)
        timing = (
            f"timing ratio per episode {' '.join(f'{x:.2f}' for x in ep_ratio)} (mean|dev| {timing_dev:.3f}, "
            f"{len(ep_ratio)}/{n_ep} episodes alternate)"
            if ep_ratio
            else "no alternation detected"
        )
        worst_amp = max(amp, key=lambda k: amp[k]) if amp else None
        worst_off = max(off, key=off.get)
        print(
            f"SYM {tag} cmd {c:.2f} (speed {np.mean(speed):.2f}) {'PASS' if ok else 'FAIL'} | {timing} | "
            + (f"amp |R/L| " + " ".join(f"{k} {v:.2f}" for k, v in amp.items()) + f" (worst {worst_amp}) | " if amp else "")
            + "offset rad " + " ".join(f"{k} {v:.3f}" for k, v in off.items()) + f" (worst {worst_off}) | "
            f"pooled: ratio {pooled_ratio:.2f}, max offset {pooled_off:.3f}",
            flush=True,
        )
    print(f"SYM {tag} SUMMARY: {n_pass}/{len(cmds)} speeds pass", flush=True)
    env.close()


if __name__ == "__main__":
    main()
