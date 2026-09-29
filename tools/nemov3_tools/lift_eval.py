"""Foot (wheel) lift in policy rollouts, batched, vs the SkatingFast06 reference.

usage: lift_eval.py <task> <checkpoint.pt> [envs=32] [vx=0.6,1.0,1.3] [wz=0] [steps=600] [tag]

Loads the checkpoint like `play` (PyTorch policy, deterministic), runs `envs` robots per (forward speed, yaw
rate) command, drops the first 100 steps. Wheel height h = wheel-center z - 0.075 (the wheel bottom; the tread is
~a sphere around the center, so this holds while the wheel tilts).
  above 2/5 cm   share of time a wheel is more than 2 / 5 cm up (mean of both wheels)
  lifts          contiguous h > 2 cm segments longer than 2 steps: events per second per wheel, mean / p90 / max
                 peak height, mean duration
  single         share of time one wheel is > 3 cm higher than the other
  alt            share of consecutive lift events that switch wheel (1 = strictly alternating)
  rear           share of lift events whose wheel is behind the other one at lift-off
  behind         how far the lifted wheel is behind the support wheel at lift-off / at its peak (m; Fast06 0.22 /
                 0.20 -- the rear leg stretched back; ~0 = stamping in place)
  bounce         share of time a wheel has lost contact while both wheels are below 2 cm (micro-bounces)
  cycle          stride cycle (s): from one right-wheel lift-off (right > 3 cm above left) to the next, with exactly
                 one left lift-off in between (left-start cycles counted too). Fast06 1.33 s; the user accepts +-10 %
                 (1.20-1.46 s)
  yaw            mean yaw rate (rad/s; a lean without a matching yaw rate means the robot circles)
  fell           robots that terminated
A BAL line per command describes how the robot balances while on one wheel (frames with one wheel > 3 cm higher),
in the frame of the base velocity (travel direction), "inward" = toward the body midline from the support wheel:
  com            whole-body CoM offset from the support wheel, inward positive (cm)
  steer          support-wheel heading vs travel, inward positive (deg); share of frames steering toward the CoM
                 side and corr(steer, CoM offset) -- the bicycle-style balance of the reference
  lean           base roll toward the support side (deg)
  hip roll/yaw   support-leg hip roll (outward positive) and hip yaw, pelvis roll / yaw joints (rad)
Fast06 reference (steady part): peak 8.3-8.5 cm, 0.37-0.42 s per lift (> 3 cm above the other
wheel), single support ~57% of the time, strictly alternating, lifted wheel 0.22-0.24 m behind at lift-off.
"""

import os
import sys
from dataclasses import asdict

import mujoco
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import booster_mjlab.tasks.velocity.config.nemov3  # noqa: F401

R_WHEEL, LIFT, SINGLE = 0.075, 0.02, 0.03
XML = "src/booster_mjlab/robots/nemov3/xmls/nemov3.xml"


def segments(mask):
    """(start, end) index pairs of the True runs of a 1-D bool array."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return list(zip(edges[::2], edges[1::2]))


def stride_cycles(hd, dt):
    """Stride cycle lengths (s) from (T,) right-minus-left wheel heights (see ``cycle`` in the module docstring)."""
    out = []
    for d in (hd, -hd):
        on_a = np.flatnonzero(np.diff(np.r_[0, (d > SINGLE).astype(int)]) == 1)
        on_b = np.flatnonzero(np.diff(np.r_[0, (-d > SINGLE).astype(int)]) == 1)
        for a, b in zip(on_a[:-1], on_a[1:]):
            if np.sum((on_b > a) & (on_b < b)) == 1 and 0.6 <= (b - a) * dt <= 3.0:
                out.append((b - a) * dt)
    return out


def balance(H, COM, VEL, WXY, AX, ROLL, HR):
    """Balance statistics over the single-support frames (see the module docstring); '' if there are none."""
    diff = H[..., 0] - H[..., 1]
    sup = np.where(diff < -SINGLE, 0, np.where(diff > SINGLE, 1, -1))  # support wheel: 0 right, 1 left
    e, t = np.nonzero(sup >= 0)
    if len(e) < 10:
        return ""
    s = sup[e, t]
    inward = np.where(s == 0, 1.0, -1.0)  # the midline is left (+lat) of the right wheel
    v = VEL[e, t]
    fwd = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-6)
    lat = np.c_[-fwd[:, 1], fwd[:, 0]]
    off = np.sum((COM[e, t] - WXY[e, t, s]) * lat, 1) * inward
    ax = AX[e, t, s]
    roll_dir = np.c_[ax[:, 1], -ax[:, 0]]  # z x axle
    roll_dir *= np.sign(np.sum(roll_dir * fwd, 1, keepdims=True) + 1e-9)
    steer_raw = np.arctan2(np.sum(roll_dir * lat, 1), np.sum(roll_dir * fwd, 1))
    steer = steer_raw * inward
    lean = ROLL[e, t] * np.where(s == 0, -1.0, 1.0)
    hr = HR[e, t, s]
    toward_com = np.mean(np.sign(steer_raw) == np.sign(off * inward))
    corr = np.corrcoef(steer_raw, off * inward)[0, 1] if np.std(steer_raw) > 1e-6 else float("nan")
    return (f"single-support frames {len(e)} | com inward {np.median(off) * 100:+.1f} cm (p5 {np.percentile(off, 5) * 100:+.1f}, "
            f"p95 {np.percentile(off, 95) * 100:+.1f}) | steer inward {np.degrees(np.median(steer)):+.1f} deg, toward CoM side "
            f"{toward_com * 100:.0f}%, corr {corr:+.2f} | lean to support {np.degrees(np.median(lean)):+.1f} deg | "
            f"support hip roll {np.median(hr):+.3f} rad")


def main():
    task, ckpt = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    vxs = [float(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "0.6,1.0,1.3").split(",")]
    wzs = [float(x) for x in (sys.argv[5] if len(sys.argv) > 5 else "0").split(",")]
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
    runner.load(ckpt, load_cfg={"actor": True, "discriminator": False}, strict=True, map_location="cuda:0")
    policy = runner.get_inference_policy(device="cuda:0")
    robot = env.scene["robot"]
    sensor = env.scene["feet_ground_contact"]
    dt = env.step_dt
    ids = [robot.body_names.index("right_wheel_link"), robot.body_names.index("left_wheel_link")]
    xml = mujoco.MjModel.from_xml_path(XML)
    mass = torch.tensor([xml.body(b).mass[0] for b in robot.body_names], device="cuda:0")
    hip_roll = [robot.joint_names.index(f"{s}_hip_roll_joint") for s in ("right", "left")]

    for vx in vxs:
        for wz in wzs:
            tw.ranges.lin_vel_x = (vx, vx)
            tw.ranges.ang_vel_z = (wz, wz)
            H, C, X, V, W = [], [], [], [], []
            COM, VEL, WXY, AX, ROLL, HR = [], [], [], [], [], []
            with torch.inference_mode():
                obs, _ = venv.reset()
                alive = torch.ones(n, dtype=torch.bool, device="cuda:0")
                for t in range(steps):
                    obs, _, dones, _ = venv.step(policy(obs))
                    alive &= ~dones.bool()
                    if t >= 100:
                        p = robot.data.body_link_pos_w[:, ids]
                        H.append((p[..., 2] - R_WHEEL).cpu().numpy())
                        C.append(sensor.data.found.cpu().numpy() > 0)
                        q = robot.data.root_link_quat_w
                        yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
                        d = p[:, 0, :2] - p[:, 1, :2]  # right - left, world
                        X.append((torch.cos(yaw) * d[:, 0] + torch.sin(yaw) * d[:, 1]).cpu().numpy())
                        V.append(robot.data.root_link_lin_vel_b[:, 0].cpu().numpy())
                        W.append(robot.data.root_link_ang_vel_w[:, 2].cpu().numpy())
                        com = (mass[None, :, None] * robot.data.body_com_pos_w).sum(1) / mass.sum()
                        COM.append(com[:, :2].cpu().numpy())
                        VEL.append(robot.data.root_link_lin_vel_w[:, :2].cpu().numpy())
                        WXY.append(p[..., :2].cpu().numpy())
                        wq = robot.data.body_link_quat_w[:, ids]  # (n, 2, 4) wxyz; the axle is the wheel body's y axis
                        w_, x_, y_, z_ = wq.unbind(-1)
                        AX.append(torch.stack([2 * (x_ * y_ - w_ * z_), 1 - 2 * (x_**2 + z_**2)], -1).cpu().numpy())
                        g = robot.data.projected_gravity_b
                        ROLL.append(torch.atan2(g[:, 1], -g[:, 2]).cpu().numpy())  # + = leaning left
                        HR.append(robot.data.joint_pos[:, hip_roll].cpu().numpy())
            H, C, X, V = np.stack(H, 1), np.stack(C, 1), np.stack(X, 1), np.stack(V, 1)  # (n, T, 2) / (n, T)
            ok = np.flatnonzero(alive.cpu().numpy())
            T = H.shape[1]
            peaks, durs, n_ev, alt, rear, behind, cycles = [], [], 0, [], [], [], []
            for e in ok:
                cycles += stride_cycles(H[e, :, 0] - H[e, :, 1], dt)
                ev = []
                for w in range(2):
                    for a, b in segments(H[e, :, w] > LIFT):
                        if b - a > 2:
                            peaks.append(H[e, a:b, w].max() * 100)
                            durs.append((b - a) * dt)
                            ev.append((a, w))
                            rel = X[e, a] if w == 0 else -X[e, a]  # lifted wheel minus the other, forward positive
                            rear.append(rel < 0)
                            pk_t = a + int(np.argmax(H[e, a:b, w]))
                            rel_pk = X[e, pk_t] if w == 0 else -X[e, pk_t]
                            behind.append((-rel, -rel_pk))
                n_ev += len(ev)
                ev.sort()
                alt += [ev[i][1] != ev[i - 1][1] for i in range(1, len(ev))]
            if len(ok):
                hh = H[ok]
                single = np.mean(np.abs(hh[..., 0] - hh[..., 1]) > SINGLE)
                above2, above5 = np.mean(hh > LIFT), np.mean(hh > 0.05)
                bounce = np.mean((~C[ok]).any(-1) & (hh < LIFT).all(-1))
                speed = V[ok].mean()
                yaw = np.stack(W, 1)[ok].mean()
            else:
                single = above2 = above5 = bounce = speed = yaw = float("nan")
            bal = balance(H[ok], np.stack(COM, 1)[ok], np.stack(VEL, 1)[ok], np.stack(WXY, 1)[ok], np.stack(AX, 1)[ok],
                          np.stack(ROLL, 1)[ok], np.stack(HR, 1)[ok])
            pk = np.array(peaks) if peaks else np.zeros(1)
            rate = n_ev / max(len(ok) * 2 * T * dt, 1e-9)
            print(
                f"LIFT {tag} vx {vx:.2f} wz {wz:+.2f} | speed {speed:.2f} yaw {yaw:+.2f} | above 2 cm {above2 * 100:.0f}% 5 cm {above5 * 100:.0f}% | "
                f"lifts {rate:.2f}/s/wheel, peak {pk.mean():.1f} / p90 {np.percentile(pk, 90):.1f} / max {pk.max():.1f} cm, "
                f"{np.mean(durs) if durs else 0:.2f} s | single {single * 100:.0f}% | alt {np.mean(alt) * 100 if alt else float('nan'):.0f}% "
                f"rear {np.mean(rear) * 100 if rear else float('nan'):.0f}% behind {np.mean([b[0] for b in behind]) if behind else float('nan'):+.2f} / "
                f"{np.mean([b[1] for b in behind]) if behind else float('nan'):+.2f} m | bounce {bounce * 100:.1f}% | "
                f"cycle {np.mean(cycles) if cycles else float('nan'):.2f} s | fell {n - len(ok)}/{n}",
                flush=True,
            )
            if bal:
                print(f"BAL  {tag} vx {vx:.2f} wz {wz:+.2f} | " + bal, flush=True)
    env.close()


if __name__ == "__main__":
    main()
