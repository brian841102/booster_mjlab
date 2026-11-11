"""Compare an exported jump-task ONNX with the PyTorch policy on the observations of a simulated rollout that skates at
0.6 / 1.0 / 1.3 m/s and presses the jump button (the ONNX input has batch 1, so every robot is fed separately).
Also prints the ONNX metadata keys and the last two observation values during a jump (jump_command = flag, phase).
usage: onnx_check.py <task> <ckpt.pt> <model.onnx> [n=8]
"""

import sys
from dataclasses import asdict

import numpy as np
import onnx
import onnxruntime as ort
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import booster_mjlab.tasks.velocity.config.nemov3  # noqa: F401

task, ckpt, path = sys.argv[1], sys.argv[2], sys.argv[3]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 8
cfg = load_env_cfg(task, play=True)
cfg.scene.num_envs = n
tw = cfg.commands["twist"]
tw.rel_standing_envs = tw.rel_forward_envs = 0.0
tw.resampling_time_range = (1e9, 1e9)
agent = load_rl_cfg(task)
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
venv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
runner = load_runner_cls(task)(venv, asdict(agent), device="cuda:0")
runner.load(ckpt, load_cfg={"actor": True, "discriminator": False}, strict=True, map_location="cuda:0")
pol = runner.get_inference_policy(device="cuda:0")
sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
name = sess.get_inputs()[0].name
meta = {p.key: p.value for p in onnx.load(path).metadata_props}
print("onnx input", sess.get_inputs()[0].shape, "output", sess.get_outputs()[0].shape, "| metadata keys:", sorted(meta))
twist = env.command_manager.get_term("twist")
jump = env.command_manager.get_term("jump")
worst, steps, jump_obs = 0.0, 0, None
with torch.inference_mode():
    obs, _ = venv.reset()
    for vx in (0.6, 1.0, 1.3):
        for t in range(int(4.0 / env.step_dt)):
            if t == int(2.0 / env.step_dt):
                jump.trigger()
                obs = venv.get_observations()
            twist.vel_command_b[:, 0] = vx
            twist.vel_command_b[:, 1:] = 0.0
            a = pol(obs)
            x = obs["actor"].cpu().numpy().astype(np.float32)
            if bool(jump.active[0]) and jump_obs is None and float(jump.phase[0]) > 0.3:
                jump_obs = x[0, -2:]
            y = np.concatenate([sess.run(None, {name: x[i : i + 1]})[0] for i in range(n)])
            worst = max(worst, float(np.abs(y - a.cpu().numpy()).max()))
            steps += 1
            obs, *_ = venv.step(a)
print(f"max |onnx - pytorch| over {steps} steps x {n} envs (with jump presses): {worst:.2e}")
print("observation tail (jump_command flag, phase) mid-jump:", jump_obs)
env.close()
