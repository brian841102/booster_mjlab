"""Straight-line gait summary of a Pino v1 policy: per commanded speed, 32 robots, 10 s each (first 3 s dropped).

usage: gait_eval.py <task> <checkpoint.pt> [speeds=0,0.3,0.6,1.0,1.3]

Per speed: actual forward speed, falls, share of time each foot has a wheel > 1 cm off the ground and the mean peak
height of those lifts, outward hip roll (p95), left/right hip-pitch correlation (-1 = clean alternation) and swing
amplitude, torso pitch; and how still the robot is (joint velocity RMS of the 21 position joints, yaw rate std) --
the standing-still check at 0 m/s (also the mean |joint - default| of the 21 position joints). A second line per speed ("GF") splits each foot's air time into toe up (front wheel
> 1 cm, rear wheel down), heel up (rear up, front down) and whole foot up, with the ankle pitch. The last line
("PASS") applies the goal of the autonomous loop (user 2026-11-03): no toe raising (<= 10 % of the time at speeds >= 0.3),
clean alternation (L/R hip pitch correlation <= -0.6 at speeds >= 0.3), speed within 0.25 m/s of the command, <= 3 falls,
and standing still in the default pose (joint velocity RMS <= 0.2 rad/s, a foot raised <= 5 % of the time, yaw rate std
<= 6 deg/s, mean |joint - default| <= 0.12 rad).
"""

import sys
from dataclasses import asdict

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from scipy import signal

import booster_mjlab.tasks.velocity.config.pinov1  # noqa: F401
from booster_mjlab.robots.pinov1.pinov1_constants import PINOV1_POSITION_JOINT_NAMES, PINOV1_WHEEL_BODY_RADII

N = 32
task, ckpt = sys.argv[1], sys.argv[2]
speeds = [float(v) for v in (sys.argv[3] if len(sys.argv) > 3 else "0,0.3,0.6,1.0,1.3").split(",")]
# "rest": also stand still starting at rest in the default pose (rest-standing tasks), besides stopping from a skating
# reference frame (the 0 m/s row).
if 0.0 in speeds:
    speeds = speeds + ["rest"]

cfg = load_env_cfg(task, play=True)
cfg.scene.num_envs = N
tw = cfg.commands["twist"]
tw.rel_standing_envs = tw.rel_forward_envs = 0.0
tw.ranges.lin_vel_y = tw.ranges.ang_vel_z = (0.0, 0.0)
tw.resampling_time_range = (1e9, 1e9)
agent = load_rl_cfg(task)
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
venv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
runner = load_runner_cls(task)(venv, asdict(agent), device="cuda:0")
runner.load(ckpt, load_cfg={"actor": True, "discriminator": False}, strict=True, map_location="cuda:0")
policy = runner.get_inference_policy(device="cuda:0")
robot = env.scene["robot"]
term = env.command_manager.get_term("twist")
wheels = [robot.body_names.index(n) for n in PINOV1_WHEEL_BODY_RADII]
radii = torch.tensor(list(PINOV1_WHEEL_BODY_RADII.values()), device="cuda:0")
hp = [robot.joint_names.index(n) for n in ("left_hip_pitch_joint", "right_hip_pitch_joint")]
hr = [robot.joint_names.index(n) for n in ("left_hip_roll_joint", "right_hip_roll_joint")]
pos_joints = [robot.joint_names.index(n) for n in PINOV1_POSITION_JOINT_NAMES]
ank = [robot.joint_names.index(n) for n in ("left_ankle_pitch_joint", "right_ankle_pitch_joint")]
default_q = robot.data.default_joint_pos[0, pos_joints].cpu().numpy()
summary = {"toe": [], "corr": [], "track": [], "falls": 0, "stand": None, "rest": None}
sos = signal.butter(2, [0.2, 3.0], btype="band", fs=50.0, output="sos")

print(f"== {task} {ckpt.split('/')[-2]}/{ckpt.split('/')[-1]}")
print("GE cmd  | speed | fell | lift>1cm L/R foot | lift peak cm | hip roll out p95 | L/R hip pitch corr | swing | torso pitch "
      "| joint vel rms | yaw rate std")
with torch.inference_mode():
    for vx in speeds:
        rest = vx == "rest"
        vx = 0.0 if rest else vx
        tw.ranges.lin_vel_x = (vx, vx)
        tw.rel_standing_envs = 1.0 if rest else 0.0
        obs, _ = venv.reset()
        term.vel_command_b[:, 0] = vx
        term.is_standing_env[:] = rest
        fell = torch.zeros(N, dtype=torch.bool, device="cuda:0")
        rec = {k: [] for k in ("v", "h", "hp", "hr", "pitch", "qd", "wz", "ank", "q")}
        for t in range(500):
            term.vel_command_b[:, 0] = vx
            obs, _, done, _ = venv.step(policy(obs))
            fell |= done.bool()
            if t < 150:
                continue
            rec["v"].append(robot.data.root_link_lin_vel_b[:, 0].clone())
            rec["h"].append((robot.data.body_link_pos_w[:, wheels, 2] - radii).clamp_min(0).clone())
            rec["hp"].append(robot.data.joint_pos[:, hp].clone())
            rec["hr"].append(robot.data.joint_pos[:, hr].clone())
            g = robot.data.projected_gravity_b
            rec["pitch"].append(torch.rad2deg(torch.atan2(g[:, 0], -g[:, 2])).clone())
            rec["qd"].append(robot.data.joint_vel[:, pos_joints].clone())
            rec["wz"].append(robot.data.root_link_ang_vel_w[:, 2].clone())
            rec["ank"].append(robot.data.joint_pos[:, ank].clone())
            rec["q"].append(robot.data.joint_pos[:, pos_joints].clone())
        ok = (~fell).cpu().numpy()
        if not ok.any():
            print(f"GE {vx:4.1f}  | all {N} robots fell within the 10 s")
            summary["falls"] += int(fell.sum())
            summary["track"].append(vx)
            continue
        r = {k: torch.stack(v).cpu().numpy()[:, ok] for k, v in rec.items()}
        foot_h = np.stack([r["h"][..., :2].max(-1), r["h"][..., 2:].max(-1)], -1)  # (T, n, L/R)
        lifted = foot_h > 0.01
        peak = foot_h[lifted].mean() * 100 if lifted.any() else 0.0
        out = np.stack([r["hr"][..., 0], -r["hr"][..., 1]], -1)
        x = signal.sosfiltfilt(sos, r["hp"], axis=0)
        corr = np.median([np.corrcoef(x[:, e, 0], x[:, e, 1])[0, 1] for e in range(x.shape[1])])
        label = "rest" if rest else f"{vx:4.1f}"
        print(f"GE {label}  | {r['v'].mean():5.2f} | {int(fell.sum()):2d}/{N} | {100 * lifted[..., 0].mean():3.0f}% / "
              f"{100 * lifted[..., 1].mean():3.0f}%       | {peak:5.1f}        | {np.percentile(out, 95):+.3f}           | "
              f"{corr:+.2f}              | {x.std(0).mean():.3f} | {r['pitch'].mean():+5.1f} deg  "
              f"| {np.sqrt((r['qd'] ** 2).mean()):.3f} rad/s   | {np.degrees(r['wz'].std(0).mean()):.1f} deg/s "
              f"| dev {np.abs(r['q'] - default_q).mean():.3f} rad")
        parts = []
        for k, side in enumerate(("L", "R")):  # wheel order: left rear, left front, right rear, right front
            rear, front = r["h"][..., 2 * k] > 0.01, r["h"][..., 2 * k + 1] > 0.01
            toe, heel, whole = front & ~rear, rear & ~front, rear & front
            a = r["ank"][..., k]
            parts.append(f"{side}: toe up {100 * toe.mean():3.0f}% heel up {100 * heel.mean():3.0f}% whole {100 * whole.mean():3.0f}% "
                         f"ankle {a.mean():+.2f} (toe up {a[toe].mean() if toe.any() else float('nan'):+.2f})")
        print(f"GF {label}  | " + " | ".join(parts))
        toe = np.mean([((r["h"][..., 2 * k + 1] > 0.01) & ~(r["h"][..., 2 * k] > 0.01)).mean() for k in (0, 1)])
        summary["falls"] += int(fell.sum())
        if vx >= 0.3:
            summary["toe"].append(toe)
            summary["corr"].append(corr)
        if not rest:
            summary["track"].append(abs(r["v"].mean() - vx))
        if rest:
            summary["rest"] = (np.sqrt((r["qd"] ** 2).mean()), lifted.mean(), np.degrees(r["wz"].std(0).mean()),
                               np.abs(r["q"] - default_q).mean(), int(fell.sum()))
        elif vx == 0:
            summary["stand"] = (np.sqrt((r["qd"] ** 2).mean()), lifted.mean(), np.degrees(r["wz"].std(0).mean()),
                                np.abs(r["q"] - default_q).mean())
env.close()
toe, corr = float(np.mean(summary["toe"] or [1.0])), float(np.max(summary["corr"] or [0.0]))
flags = {"toe<=0.10": toe <= 0.10, "alt<=-0.6": corr <= -0.6, "track<=0.25": max(summary["track"]) <= 0.25,
         "falls<=3": summary["falls"] <= 3}
if summary["stand"] is not None:
    v, lift, yaw, dev = summary["stand"]
    flags.update({"stand_vel<=0.2": v <= 0.2, "stand_lift<=0.05": lift <= 0.05, "stand_yaw<=6": yaw <= 6.0,
                  "stand_dev<=0.12": dev <= 0.12})
if summary["rest"] is not None:
    v, lift, yaw, dev, nf = summary["rest"]
    print(f"REST stand from the default pose: joint vel {v:.3f} rad/s, foot raised {100 * lift:.0f}%, yaw std {yaw:.1f} deg/s, "
          f"|q - default| {dev:.3f} rad, falls {nf}")
print(f"PASS {int(all(flags.values()))} | toe {100 * toe:.0f}% worst-corr {corr:+.2f} falls {summary['falls']} | "
      + " ".join(f"{k}:{'ok' if v else 'NO'}" for k, v in flags.items()))
