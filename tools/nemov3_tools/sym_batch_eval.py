"""Batched, per-episode left/right symmetry evaluation (many environments in parallel).

usage: sym_batch_eval.py <task> <checkpoint.pt> [envs_per_speed=32] [cmds=0.3,0.6,0.8,1.0,1.15,1.3] [steps=750] [tag]

Loads the checkpoint the same way `play` does (runner.load(actor only) -> get_inference_policy, deterministic
mean actions) and runs `envs_per_speed` robots at once for every commanded speed. With one environment the AMP
reset pose pool holds only ~5 poses, so single-env evaluations start every episode from nearly the same state;
here the pool is envs x 5 poses, like in training.

Every robot = one episode (first 100 steps dropped). Per episode: timing ratio R->L / period from hip-pitch
peaks, amplitude ratio std_R/std_L (hip pitch, calf, shoulder pitch, elbow), static offset |mean_R - mean_L|
(hip pitch, calf, hip roll, hip yaw, shoulder pitch, elbow), hip-pitch swing std. Reported per speed:
  timing: mean |ratio - 0.5| and the fraction of episodes with ratio > 0.5 (about 0.5 with a large mean |dev|
          = a mirror pair of lopsided gaits; about 0 or 1 = a systematic bias);
  amplitude: exp(mean |log ratio|) per joint; offset: mean per joint; swing: mean hip-pitch std.
Pass (per speed): timing mean|dev| <= 0.05 (the 0.45..0.55 band on average; the share of episodes inside the band
is printed too), amplitude ratios <= 1.18, offsets <= 0.05 rad. At commanded speeds <= 0.3 m/s the amplitude test is
mean |std_R - std_L| <= 0.02 rad per joint instead of the ratio (user decision 2026-09-25: the swing there is only
~0.06 rad, so a ratio of 1.25 is a sub-degree difference that cannot be seen).
A second line ("SB-signed") gives the signed means: R/L amplitude exp(mean log ratio) and R-L offset, plus the
absolute swing difference mean |std_R - std_L| (rad; the ratio is ill-conditioned when the swing is tiny). Signed ~ the
|.| value = the same side in every episode (systematic: robot model / policy / env not mirror-symmetric); signed
~ 1 / 0 with a large |.| value = episodes split between a mirror pair of lopsided gaits.
"""

import os
import sys
from dataclasses import asdict

import numpy as np
import torch
from scipy.signal import find_peaks

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

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
LOW_SPEED, AMP_ABS_TOL = 0.3, 0.02  # at cmd <= LOW_SPEED: amplitude judged by |std_R - std_L| <= AMP_ABS_TOL rad


def smooth(x, k):
    return np.convolve(x, np.ones(k) / k, mode="same")


def timing_ratio(r, l, dt):
    k = max(3, int(round(0.08 / dt)))
    r = smooth(r - r.mean(), k)
    l = smooth(l - l.mean(), k)
    if min(r.std(), l.std()) < 0.03:
        return None
    md = int(0.4 / dt)
    pr, _ = find_peaks(r, distance=md, prominence=0.4 * r.std())
    pl, _ = find_peaks(l, distance=md, prominence=0.4 * l.std())
    trl = [(pl[pl > a][0] - a) * dt for a in pr if (pl > a).any()]
    tlr = [(pr[pr > a][0] - a) * dt for a in pl if (pr > a).any()]
    if len(trl) < 3 or len(tlr) < 3:
        return None
    return np.mean(trl) / (np.mean(trl) + np.mean(tlr)), np.mean(trl) + np.mean(tlr)


def main():
    task, ckpt = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    cmds = [float(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "0.3,0.6,0.8,1.0,1.15,1.3").split(",")]
    steps = int(sys.argv[5]) if len(sys.argv) > 5 else 750
    tag = sys.argv[6] if len(sys.argv) > 6 else os.path.basename(os.path.dirname(ckpt))[-24:]

    env_cfg = load_env_cfg(task, play=True)
    env_cfg.scene.num_envs = n
    tw = env_cfg.commands["twist"]
    tw.ranges.lin_vel_y = (0.0, 0.0)
    tw.ranges.ang_vel_z = (0.0, 0.0)
    tw.rel_standing_envs = 0.0
    tw.rel_forward_envs = 0.0
    agent_cfg = load_rl_cfg(task)
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
    venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = (load_runner_cls(task))(venv, asdict(agent_cfg), device="cuda:0")
    runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
    policy = runner.get_inference_policy(device="cuda:0")
    robot = env.scene["robot"]
    term = env.command_manager.get_term("twist")
    dt = env.step_dt
    idx_r = [robot.joint_names.index(v[0]) for v in PAIRS.values()]
    idx_l = [robot.joint_names.index(v[1]) for v in PAIRS.values()]
    keys = list(PAIRS)

    n_pass = 0
    for c in cmds:
        term.cfg.ranges.lin_vel_x = (c, c)
        qr, ql, spd = [], [], []
        with torch.inference_mode():
            obs, _ = venv.reset()
            alive = torch.ones(n, dtype=torch.bool, device="cuda:0")
            for t in range(steps):
                obs, _, dones, _ = venv.step(policy(obs))
                alive &= ~dones.bool()
                if t >= 100:
                    q = robot.data.joint_pos
                    qr.append(q[:, idx_r].cpu().numpy())
                    ql.append(q[:, idx_l].cpu().numpy())
                    spd.append(robot.data.root_link_lin_vel_b[:, 0].cpu().numpy())
        qr, ql = np.stack(qr, 1), np.stack(ql, 1)  # [env, time, joint]
        ok_env = alive.cpu().numpy()
        ratios, amps, offs, swing, periods = [], [], [], [], []
        for e in np.flatnonzero(ok_env):
            offs.append(np.abs(qr[e].mean(0) - ql[e].mean(0)))
            swing.append(min(qr[e, :, 0].std(), ql[e, :, 0].std()))
            res = timing_ratio(qr[e, :, 0], ql[e, :, 0], dt)
            if res is not None:
                r, period = res
                ratios.append(r)
                periods.append(period)
                amps.append([qr[e, :, keys.index(k)].std() / (ql[e, :, keys.index(k)].std() + 1e-9) for k in AMP_KEYS])
        offs = np.mean(offs, 0)
        ratios = np.array(ratios)
        dev = np.abs(ratios - 0.5).mean() if len(ratios) else float("nan")
        amp = np.exp(np.abs(np.log(np.array(amps))).mean(0)) if amps else np.array([])
        amp_signed = np.exp(np.log(np.array(amps)).mean(0)) if amps else np.array([])
        # |std_R - std_L| in rad: with a tiny swing (0.3 m/s) the ratio blows up on sub-degree differences
        amp_absdiff = np.mean(
            [[abs(qr[e, :, keys.index(k)].std() - ql[e, :, keys.index(k)].std()) for k in AMP_KEYS] for e in np.flatnonzero(ok_env)], 0
        )
        offs_signed = np.mean([qr[e].mean(0) - ql[e].mean(0) for e in np.flatnonzero(ok_env)], 0)
        low = c <= LOW_SPEED + 1e-6
        amp_ok = all(amp_absdiff <= AMP_ABS_TOL) if low else all(np.log(amp) <= AMP_TOL)
        ok = ((not len(ratios)) or dev <= TIMING_TOL) and amp_ok and all(offs <= OFFSET_TOL)
        n_pass += ok
        print(
            f"SB {tag} cmd {c:.2f} (speed {np.mean(spd):.2f}, fell {int((~ok_env).sum())}/{n}) {'PASS' if ok else 'FAIL'} | "
            + (
                f"timing mean|dev| {dev:.3f} (ratio p10/p50/p90 {np.percentile(ratios,10):.2f}/{np.median(ratios):.2f}/{np.percentile(ratios,90):.2f}, "
                f">0.5 in {np.mean(ratios > 0.5)*100:.0f}%, in 0.45..0.55 {np.mean(np.abs(ratios - 0.5) <= 0.05)*100:.0f}%, "
                f"{len(ratios)}/{ok_env.sum()} alternate) | "
                if len(ratios)
                else "no alternation | "
            )
            + ("amp |R/L| " + " ".join(f"{k} {v:.2f}" for k, v in zip(AMP_KEYS, amp)) + " | " if len(amp) else "")
            + ("amp |std_R-std_L| rad (low-speed test) " + " ".join(f"{k} {v:.3f}" for k, v in zip(AMP_KEYS, amp_absdiff)) + " | " if low else "")
            + "offset rad " + " ".join(f"{k} {v:.3f}" for k, v in zip(keys, offs))
            + f" | hip-pitch swing std {np.mean(swing):.3f}"
            + (f" | gait period {np.median(periods):.2f} s (clock 1.35 s)" if periods else ""),
            flush=True,
        )
        print(
            f"SB-signed {tag} cmd {c:.2f} | "
            + ("amp R/L " + " ".join(f"{k} {v:.2f}" for k, v in zip(AMP_KEYS, amp_signed)) + " | " if len(amp_signed) else "")
            + "offset R-L rad " + " ".join(f"{k} {v:+.3f}" for k, v in zip(keys, offs_signed))
            + " | |std_R-std_L| rad " + " ".join(f"{k} {v:.3f}" for k, v in zip(AMP_KEYS, amp_absdiff)),
            flush=True,
        )
    print(f"SB {tag} SUMMARY: {n_pass}/{len(cmds)} speeds pass", flush=True)
    env.close()


if __name__ == "__main__":
    main()
