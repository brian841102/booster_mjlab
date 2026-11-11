"""Can this robot jump 5 cm at all? Scripted crouch -> push-off of the legs while a skating policy balances.

The Exp 90 seed (or any checkpoint of the task) skates straight at ``vx``; at t = 3 s each robot's hip pitch and calf
targets of both legs are replaced by a script: crouch to (hip, calf) = default + depth * (-1.0, +1.6) over ``t_crouch``,
then the extension target (hip ``hip_ext``, calf ``calf_ext``) at once, held for 0.3 s, then the policy takes the legs
back. The policy keeps the wheels, arms and hip roll / yaw. Robots differ in crouch depth and crouch time. Prints the peak
clearance of the lower wheel while both are off the ground, the flight time, and whether the robot fell within 3 s.
(An earlier open-loop version without the policy fell over every time and was inconclusive.)
usage: jump_feasibility.py <task> <ckpt> [vx=1.0] [calf_ext=0.0] [hip_ext=0.0] [depths=0,0.2,0.35,0.5] [crouch times=0.1,0.2,0.35]
"""

import sys
from dataclasses import asdict

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import booster_mjlab.tasks.velocity.config.nemov3  # noqa: F401

task, ckpt = sys.argv[1], sys.argv[2]
vx = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
calf_ext = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
hip_ext = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
depths = [float(v) for v in (sys.argv[6] if len(sys.argv) > 6 else "0.0,0.2,0.35,0.5").split(",")]
t_crouches = [float(v) for v in (sys.argv[7] if len(sys.argv) > 7 else "0.1,0.2,0.35").split(",")]
reps = 4
grid = [(d, tc) for d in depths for tc in t_crouches for _ in range(reps)]
n = len(grid)
cfg = load_env_cfg(task, play=True)
cfg.scene.num_envs = n
cfg.events.pop("push_robot", None)
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
term = env.action_manager.get_term("joint_pos")
names = list(term.target_names)
scale, offset = term._scale, term._offset
slots = {s: (names.index(f"{s}_hip_pitch_joint"), names.index(f"{s}_calf_joint")) for s in ("left", "right")}
twist = env.command_manager.get_term("twist")
dt = env.step_dt
wheel_ids = [robot.body_names.index(b) for b in ("left_wheel_link", "right_wheel_link")]
D = torch.tensor([g[0] for g in grid], device="cuda:0")
TC = torch.tensor([g[1] for g in grid], device="cuda:0")
T0 = 3.0
peak, air_t, fallen = np.zeros(n), np.zeros(n), np.zeros(n, dtype=bool)


def target_to_action(q, i):
    sc = scale[:, i] if torch.is_tensor(scale) and scale.dim() == 2 else (scale[i] if torch.is_tensor(scale) else scale)
    off = offset[:, i] if offset.dim() == 2 else offset[i]
    return (q - off) / sc


with torch.inference_mode():
    obs, _ = venv.reset()
    for k in range(int(6.0 / dt)):
        t = k * dt
        twist.vel_command_b[:, 0] = vx
        twist.vel_command_b[:, 1:] = 0.0
        a = pol(obs).clone()
        s = ((t - T0) / TC).clamp(0.0, 1.0)
        crouch = (t >= T0) & (t < T0 + TC)
        push = (t >= T0 + TC) & (t < T0 + TC + 0.3)
        for side, (ih, ic) in slots.items():
            off_h = offset[:, ih] if offset.dim() == 2 else offset[ih]
            off_c = offset[:, ic] if offset.dim() == 2 else offset[ic]
            qh = torch.where(push, torch.full_like(D, hip_ext), off_h - 1.0 * D * s)
            qc = torch.where(push, torch.full_like(D, calf_ext), off_c + 1.6 * D * s)
            m = crouch | push
            a[:, ih] = torch.where(m, target_to_action(qh, ih), a[:, ih])
            a[:, ic] = torch.where(m, target_to_action(qc, ic), a[:, ic])
        obs, *_ = venv.step(a)
        h = (robot.data.body_link_pos_w[:, wheel_ids, 2] - 0.075).clamp_min(0).min(dim=1).values
        air = (env.scene["feet_ground_contact"].data.found == 0).all(dim=1)
        upright = robot.data.projected_gravity_b[:, 2] < -0.85
        win = (t >= T0) & (t < T0 + TC + 0.8)
        ok = (air & upright & win).cpu().numpy()
        peak = np.maximum(peak, h.cpu().numpy() * ok)
        air_t += ok * dt
        fallen |= ((robot.data.projected_gravity_b[:, 2] > -0.5) | (env.scene["non_foot_ground_contact"].data.found > 0).any(dim=1)).cpu().numpy() & (t >= T0)
print(f"== scripted legs, policy balances, vx {vx}, extend to hip {hip_ext} calf {calf_ext}")
for gi in range(0, n, reps):
    d, tc = grid[gi]
    p, at, f = peak[gi : gi + reps], air_t[gi : gi + reps], fallen[gi : gi + reps]
    print(f"FEAS crouch depth {d:.2f} (hip -{d:.2f}, calf +{1.6 * d:.2f}) over {tc:.2f} s -> peak clearance {100 * np.median(p):5.1f} cm"
          f" (max {100 * p.max():4.1f}), airborne {np.median(at):.2f} s, fell {int(f.sum())}/{reps}")
env.close()
