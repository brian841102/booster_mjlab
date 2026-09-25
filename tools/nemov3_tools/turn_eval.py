"""Turning evaluation: does the robot track the yaw-rate command, and does it keep skating while it turns?

usage: turn_eval.py <task> <checkpoint.pt> [envs=32] [vx=0,0.6,1.0,1.3] [wz=-1.3,-0.6,0,0.6,1.3] [steps=600] [tag]

Loads the checkpoint like `play` (PyTorch policy, deterministic) and runs `envs` robots at once for every
(forward speed, yaw rate) command of the grid; every robot = one episode, first 100 steps dropped. Per command:
  speed / yaw     mean forward speed (body x) and mean yaw rate (world z), m/s and rad/s
  swing           mean per-episode hip-pitch swing std (min of right / left), and its ratio to the straight
                  (wz = 0) swing at the same forward speed ("skating while turning" = ratio close to 1); plus the
                  right / left leg separately (in a left turn, +wz, the right leg is the outer one)
  alt             share of episodes whose hip-pitch peaks alternate right / left (at least 3 of each)
  timing          mean |R->L interval / period - 0.5| over the alternating episodes
  R-L             signed mean right-minus-left offset of hip pitch / calf / hip roll / hip yaw (rad): a turn to
                  one side should show the mirror image (opposite sign) of the turn to the other side
  fell            robots that terminated
A summary per forward speed compares each left / right pair (+wz vs -wz): yaw-rate magnitude, swing, and the
R-L offsets (mirror consistency: offsets(+wz) ~ -offsets(-wz)).
Rotating in place (vx = 0) is allowed without leg swing.
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
}


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
    return np.mean(trl) / (np.mean(trl) + np.mean(tlr))


def main():
    task, ckpt = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    vxs = [float(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "0,0.6,1.0,1.3").split(",")]
    wzs = [float(x) for x in (sys.argv[5] if len(sys.argv) > 5 else "-1.3,-0.6,0,0.6,1.3").split(",")]
    steps = int(sys.argv[6]) if len(sys.argv) > 6 else 600
    tag = sys.argv[7] if len(sys.argv) > 7 else os.path.basename(os.path.dirname(ckpt))[-24:]

    env_cfg = load_env_cfg(task, play=True)
    env_cfg.scene.num_envs = n
    tw = env_cfg.commands["twist"]
    tw.ranges.lin_vel_y = (0.0, 0.0)
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

    res = {}
    for vx in vxs:
        for wz in wzs:
            term.cfg.ranges.lin_vel_x = (vx, vx)
            term.cfg.ranges.ang_vel_z = (wz, wz)
            qr, ql, spd, yaw = [], [], [], []
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
                        yaw.append(robot.data.root_link_ang_vel_w[:, 2].cpu().numpy())
            qr, ql = np.stack(qr, 1), np.stack(ql, 1)
            spd, yaw = np.stack(spd, 1), np.stack(yaw, 1)
            ok = alive.cpu().numpy()
            e_ok = np.flatnonzero(ok)
            swing = np.mean([min(qr[e, :, 0].std(), ql[e, :, 0].std()) for e in e_ok]) if len(e_ok) else float("nan")
            swing_r = np.mean([qr[e, :, 0].std() for e in e_ok]) if len(e_ok) else float("nan")
            swing_l = np.mean([ql[e, :, 0].std() for e in e_ok]) if len(e_ok) else float("nan")
            ratios = [timing_ratio(qr[e, :, 0], ql[e, :, 0], dt) for e in e_ok]
            ratios = np.array([r for r in ratios if r is not None])
            off = np.mean([qr[e].mean(0) - ql[e].mean(0) for e in e_ok], 0) if len(e_ok) else np.full(4, np.nan)
            res[(vx, wz)] = dict(
                speed=spd[ok].mean() if ok.any() else np.nan,
                yaw=yaw[ok].mean() if ok.any() else np.nan,
                swing=swing,
                swing_r=swing_r,
                swing_l=swing_l,
                alt=len(ratios) / max(len(e_ok), 1),
                timing=np.abs(ratios - 0.5).mean() if len(ratios) else np.nan,
                off=off,
                fell=int((~ok).sum()),
            )
    for vx in vxs:
        straight = res.get((vx, 0.0), {}).get("swing", np.nan)
        for wz in wzs:
            r = res[(vx, wz)]
            print(
                f"TURN {tag} vx {vx:.2f} wz {wz:+.2f} | speed {r['speed']:.2f} yaw {r['yaw']:+.2f} (err {r['yaw'] - wz:+.2f}) | "
                f"swing {r['swing']:.3f} ({r['swing'] / straight if straight > 0 else float('nan'):.2f}x straight; R {r['swing_r']:.3f} L {r['swing_l']:.3f}) | "
                f"alt {r['alt'] * 100:.0f}% timing {r['timing']:.3f} | R-L " + " ".join(f"{k} {v:+.3f}" for k, v in zip(PAIRS, r["off"]))
                + f" | fell {r['fell']}/{n}",
                flush=True,
            )
        for wz in wzs:
            if wz <= 0 or (vx, -wz) not in res:
                continue
            a, b = res[(vx, wz)], res[(vx, -wz)]
            mirror_err = np.abs(a["off"] + b["off"]).max()
            print(
                f"TURN-LR {tag} vx {vx:.2f} |wz| {wz:.2f} | yaw left {a['yaw']:+.2f} right {b['yaw']:+.2f} | "
                f"swing left {a['swing']:.3f} right {b['swing']:.3f} | max |off(+)+off(-)| {mirror_err:.3f} rad",
                flush=True,
            )
    env.close()


if __name__ == "__main__":
    main()
