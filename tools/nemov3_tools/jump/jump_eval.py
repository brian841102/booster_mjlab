"""Press the jump button while skating and measure the jump (task nemov3_velocity_amp_jump).

Per forward speed: n robots skate straight for 4 s, the button is pressed once (each robot at its own random time in
[4, 5] s), then they keep skating for 4 s. Reported per speed:
  - success: both wheels off the ground inside the jump window for >= 2 policy steps and >= 2 cm at the peak;
  - clearance: peak of the lower wheel's bottom height while both are off the ground (target 5 cm), and flight time;
  - falls (episode ended) within 4 s of the press;
  - speed after landing (1.5..4 s after the press) vs. the command, and the hip-pitch swing (band-passed std, both legs)
    after landing vs. before the press -- "keeps skating";
  - unasked hops: both wheels > 1 cm off the ground outside the window and its recovery, per robot and minute.
usage: [JUMP_FF_FULL=1] jump_eval.py <task> <ckpt> [n=64] [speeds=0.6,1.0,1.3] [tag]
(JUMP_FF_FULL=1: tasks with the J06 leg feed-forward keep it fully on, as at the start of training.)
Prints one "JUMP" line per speed (parsed by the run notes).
"""

import os
import sys
from dataclasses import asdict

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from scipy import signal

import booster_mjlab.tasks.velocity.config.nemov3  # noqa: F401

task, ckpt = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 64
speeds = [float(v) for v in (sys.argv[4] if len(sys.argv) > 4 else "0.6,1.0,1.3").split(",")]
tag = sys.argv[5] if len(sys.argv) > 5 else ckpt

cfg = load_env_cfg(task, play=True)
cfg.scene.num_envs = n
if os.environ.get("JUMP_FF_FULL"):  # J06+: keep the leg feed-forward at alpha = 1 (play has it off)
    cfg.actions["joint_pos"].anneal_steps = (10**9, 10**9 + 1)
tw = cfg.commands["twist"]
tw.rel_standing_envs = tw.rel_forward_envs = 0.0
tw.resampling_time_range = (1e9, 1e9)
agent = load_rl_cfg(task)
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
venv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
runner = load_runner_cls(task)(venv, asdict(agent), device="cuda:0")
runner.load(ckpt, load_cfg={"actor": True, "discriminator": False}, strict=True, map_location="cuda:0")
pol = runner.get_inference_policy(device="cuda:0")
robot = env.scene["robot"]
term = env.command_manager.get_term("twist")
jump = env.command_manager.get_term("jump")
sensor = env.scene["feet_ground_contact"]
wheel_ids = [robot.body_names.index(b) for b in ("left_wheel_link", "right_wheel_link")]
hp = [robot.joint_names.index(j) for j in ("right_hip_pitch_joint", "left_hip_pitch_joint")]
calf = [robot.joint_names.index(j) for j in ("right_calf_joint", "left_calf_joint")]
dt = env.step_dt
win, rec = jump.cfg.duration_s, jump.cfg.recovery_s
sos = signal.butter(2, [0.3, 3.0], btype="band", fs=1.0 / dt, output="sos")
print(f"== {tag}: {n} robots, button once per robot at 4..5 s, window {win} s")
for vx in speeds:
    T = int(9.0 / dt)
    press = torch.randint(int(4.0 / dt), int(5.0 / dt), (n,), device="cuda:0")
    air, h, vxs, hips, zs, calfs = [], [], [], [], [], []
    ended = torch.zeros(n, dtype=torch.bool, device="cuda:0")
    with torch.inference_mode():
        obs, _ = venv.reset()
        for t in range(T):
            fire = press == t
            if fire.any():
                jump.trigger(fire.nonzero().flatten())
                obs = venv.get_observations()
            term.vel_command_b[:, 0] = vx
            term.vel_command_b[:, 1:] = 0.0
            obs, _, d, _ = venv.step(pol(obs))
            # Play has no illegal-contact termination and a stochastic tilt one: count a fall as a body touching the
            # ground or a tilt beyond 60 degrees.
            down = d.bool() | (env.scene["non_foot_ground_contact"].data.found > 0).any(dim=1)
            down |= robot.data.projected_gravity_b[:, 2] > -0.5
            ended |= down & (t >= press) & (t < press + int(4.0 / dt))
            air.append((sensor.data.found == 0).all(dim=1).cpu().numpy())
            h.append((robot.data.body_link_pos_w[:, wheel_ids, 2] - 0.075).clamp_min(0).min(dim=1).values.cpu().numpy())
            vxs.append(robot.data.root_link_lin_vel_b[:, 0].cpu().numpy())
            hips.append(robot.data.joint_pos[:, hp].cpu().numpy())
            zs.append(robot.data.root_link_pos_w[:, 2].cpu().numpy())
            calfs.append(robot.data.joint_pos[:, calf].mean(dim=1).cpu().numpy())
    air, h, vxs, hips, zs, calfs = map(np.array, (air, h, vxs, hips, zs, calfs))
    p = press.cpu().numpy()
    ok = ~ended.cpu().numpy()
    succ, peak, flight, unasked, v_after, swing_b, swing_a, rise, calf_min, drop, calf_max, v_before, hold = [], [], [], 0, [], [], [], [], [], [], [], [], []
    x = signal.sosfiltfilt(sos, hips, axis=0)
    for e in range(n):
        w0, w1 = p[e], p[e] + int(win / dt)
        a = air[w0:w1, e]
        rise.append(zs[w0:w1, e].max() - zs[w0, e])  # trunk rise above its height at the press
        drop.append(zs[w0, e] - zs[w0:w1, e].min())  # crouch: trunk drop below its height at the press
        dz = zs[w0, e] - zs[w0:w1, e]
        hold.append(np.sum(dz > 0.7 * max(dz.max(), 1e-6)) * dt if dz.max() > 0.02 else 0.0)  # time crouched >= 70 % deep
        calf_min.append(calfs[w0:w1, e].min())  # straightest knee in the window (0 = straight, default 0.43)
        calf_max.append(calfs[w0:w1, e].max())  # most bent knee in the window (crouch)
        succ.append(a.sum() >= 2 and h[w0:w1, e][a].max() >= 0.02)  # a real hop: 2+ steps free, >= 2 cm
        peak.append(h[w0:w1, e][a].max() if a.any() else 0.0)
        flight.append(a.sum() * dt)
        outside = np.ones(T, dtype=bool)
        outside[w0 : w1 + int(rec / dt)] = False
        outside[: int(1.0 / dt)] = False  # settling after reset
        unasked += int(np.sum(np.diff((air[:, e] & (h[:, e] > 0.01)).astype(int) * outside) == 1))
        v_after.append(vxs[p[e] + int(1.5 / dt) : p[e] + int(4.0 / dt), e].mean())
        v_before.append(vxs[int(1.5 / dt) : p[e], e].mean())
        swing_b.append(x[int(1.5 / dt) : p[e], e].std(0).mean())
        swing_a.append(x[p[e] + int(1.5 / dt) : p[e] + int(4.0 / dt), e].std(0).mean())
    succ, peak, flight, v_after = map(np.array, (succ, peak, flight, v_after))
    sb, sa = np.array(swing_b), np.array(swing_a)
    print(
        f"JUMP vx {vx:.1f} | success {100 * succ.mean():3.0f}% | clearance median {100 * np.median(peak[succ]) if succ.any() else 0:4.1f} cm"
        f" (p10 {100 * np.percentile(peak[succ], 10) if succ.any() else 0:4.1f}, p90 {100 * np.percentile(peak[succ], 90) if succ.any() else 0:4.1f})"
        f" | flight {np.median(flight[succ]) if succ.any() else 0:.2f} s | falls {int((~ok).sum())}/{n}"
        f" | speed before {np.mean(v_before):.2f} after {np.mean(v_after[ok]) if ok.any() else 0:.2f} (cmd {vx}) | hip swing before {sb.mean():.3f} after {sa[ok].mean() if ok.any() else 0:.3f}"
        f" | unasked hops {unasked / (n * 9.0 / 60):.2f}/robot/min"
        f" | trunk drop median {100 * np.median(drop):.1f} cm (held {np.median(hold):.2f} s), rise {100 * np.median(rise):.1f} cm | calf most bent / straightest {np.median(calf_max):.2f} / {np.median(calf_min):.2f} rad",
        flush=True,
    )
env.close()
