import sys
from dataclasses import asdict
import numpy as np, torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
import booster_mjlab.tasks.velocity.config.nemov3  # noqa
task, ckpt, tag = sys.argv[1:4]
n = 256
cfg = load_env_cfg(task, play=True); cfg.scene.num_envs = n
tw = cfg.commands["twist"]; tw.ranges.lin_vel_x = (0.0, 0.0); tw.ranges.lin_vel_y = (0.0, 0.0); tw.ranges.ang_vel_z = (0.0, 0.0)
tw.rel_standing_envs = 0.0; tw.rel_forward_envs = 0.0
agent = load_rl_cfg(task)
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0"); venv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
runner = load_runner_cls(task)(venv, asdict(agent), device="cuda:0")
runner.load(ckpt, load_cfg={"actor": True, "discriminator": False}, strict=True, map_location="cuda:0")
pol = runner.get_inference_policy(device="cuda:0")
robot = env.scene["robot"]
first = torch.full((n,), -1, device="cuda:0")
with torch.inference_mode():
    obs, _ = venv.reset()
    v0 = robot.data.root_link_lin_vel_b[:, 0].clone()
    for t in range(1500):
        obs, _, d, _ = venv.step(pol(obs))
        new = d.bool() & (first < 0)
        first[new] = t
f = first.cpu().numpy(); v0 = v0.cpu().numpy()
fell = f >= 0
print(f"{tag}: fell {fell.sum()}/{n} in 30 s; fall step (0.02 s) quantiles:", np.percentile(f[fell], [0, 25, 50, 75, 100]).round() if fell.any() else "-")
print(f"  bins: <1 s {np.sum(fell & (f < 50))}, 1-3 s {np.sum(fell & (f >= 50) & (f < 150))}, 3-10 s {np.sum(fell & (f >= 150) & (f < 500))}, 10-30 s {np.sum(fell & (f >= 500))}")
print(f"  initial fwd speed of fallers {np.abs(v0[fell]).mean() if fell.any() else 0:.2f} vs survivors {np.abs(v0[~fell]).mean():.2f} m/s")
env.close()
