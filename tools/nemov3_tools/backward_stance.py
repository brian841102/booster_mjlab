"""Backward driving: do the feet (wheels) drift apart?  usage: backward_stance.py <task> <ckpt> <tag> [vx=-0.3] [seconds=12] [envs=64] [start=pool|default]
Holds 0 m/s for 3 s, then vx. Every 1 s prints mean wheel lateral gap (base frame, cm), mean hip roll of both legs, actual vx and falls.
The training command range is vx in [0, 1.3], so negative vx is out of distribution."""
import sys
from dataclasses import asdict
import numpy as np, torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
import booster_mjlab.tasks.velocity.config.nemov3  # noqa
import booster_mjlab.tasks.velocity.config.pinov1  # noqa
task, ckpt, tag = sys.argv[1:4]
vx = float(sys.argv[4]) if len(sys.argv) > 4 else -0.3
seconds = float(sys.argv[5]) if len(sys.argv) > 5 else 12.0
n = int(sys.argv[6]) if len(sys.argv) > 6 else 64
start = sys.argv[7] if len(sys.argv) > 7 else "pool"
cfg = load_env_cfg(task, play=True); cfg.scene.num_envs = n
tw = cfg.commands["twist"]
for r in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
    setattr(tw.ranges, r, (0.0, 0.0))
tw.rel_standing_envs = 0.0; tw.rel_forward_envs = 0.0
agent = load_rl_cfg(task)
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0"); venv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
runner = load_runner_cls(task)(venv, asdict(agent), device="cuda:0")
runner.load(ckpt, load_cfg={"actor": True, "discriminator": False}, strict=True, map_location="cuda:0")
pol = runner.get_inference_policy(device="cuda:0")
robot = env.scene["robot"]
cmdterm = env.command_manager.get_term("twist")
# pinov1 has two wheels per foot: the gap is measured between the driven rear wheels.
wheel_names = ("left_rear_wheel_link", "right_rear_wheel_link") if "Pinov1" in task else ("left_wheel_link", "right_wheel_link")
bi = [robot.body_names.index(b) for b in wheel_names]
hj = [robot.joint_names.index(j) for j in ("left_hip_roll_joint", "right_hip_roll_joint")]
fell = torch.zeros(n, dtype=torch.bool, device="cuda:0")
rows = []
with torch.inference_mode():
    obs, _ = venv.reset()
    if start == "default":
        robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), torch.zeros_like(robot.data.joint_vel))
    for t in range(int(round(seconds / env.step_dt))):
        sec = t * env.step_dt
        cmdterm.vel_command_b[:, 0] = 0.0 if sec < 3.0 else vx
        obs, _, d, _ = venv.step(pol(obs))
        fell |= d.bool()
        if t % 50 == 49:
            from mjlab.utils.lab_api.math import quat_apply_inverse
            p = robot.data.body_link_pos_w[:, bi]; r = p - robot.data.root_link_pos_w[:, None]
            q = robot.data.root_link_quat_w[:, None].expand(-1, 2, -1)
            rb = quat_apply_inverse(q.reshape(-1, 4), r.reshape(-1, 3)).reshape(n, 2, 3)
            gap = (rb[:, 0, 1] - rb[:, 1, 1]).abs().mean().item() * 100
            hr = robot.data.joint_pos[:, hj]
            rows.append((sec + env.step_dt, gap, hr[:, 0].mean().item(), hr[:, 1].mean().item(), robot.data.root_link_lin_vel_b[:, 0].mean().item(), int(fell.sum())))
print(f"{tag} vx={vx} start={start}")
print("  t(s)  wheelgap(cm)  hipRoll L/R(rad)   vx    falls")
for r in rows: print(f"  {r[0]:4.0f}  {r[1]:8.1f}    {r[2]:6.3f} {r[3]:6.3f}   {r[4]:5.2f}  {r[5]}")
env.close()
