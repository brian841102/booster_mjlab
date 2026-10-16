"""Arm swing while skating straight: how far the arms swing and whether they swing with the opposite leg.

usage: arm_swing.py <task> <checkpoint.pt> [envs=32] [cmds=0.6,1.0,1.3] [steps=600] [tag]

Per commanded speed (first 100 steps dropped, deterministic policy): mean per-episode std (rad) of shoulder pitch and
elbow (right / left), hip-pitch std for scale, and the correlation of each shoulder pitch with the opposite hip pitch
(> 0 = the arm swings forward with the opposite leg, the diagonal swing of the reference skating; < 0 = same side).
Also the mean elbow / shoulder pitch angle (a soft arm sags under gravity). Written for the deployment-PD runs (Exp 81+:
elbow kp 1.5) to check the arms still swing; works for any Nemov3 task.
"""

import sys
from dataclasses import asdict

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import booster_mjlab.tasks.velocity.config.nemov3  # noqa: F401

JOINTS = (
    "joint_rShoulderPitch", "joint_lShoulderPitch", "joint_rElbow", "joint_lElbow",
    "right_hip_pitch_joint", "left_hip_pitch_joint",
)


def main():
    task, ckpt = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    cmds = [float(v) for v in (sys.argv[4] if len(sys.argv) > 4 else "0.6,1.0,1.3").split(",")]
    steps = int(sys.argv[5]) if len(sys.argv) > 5 else 600
    tag = sys.argv[6] if len(sys.argv) > 6 else ""

    env_cfg = load_env_cfg(task, play=True)
    env_cfg.scene.num_envs = n
    tw = env_cfg.commands["twist"]
    tw.rel_standing_envs = tw.rel_forward_envs = 0.0
    tw.ranges.lin_vel_y = tw.ranges.ang_vel_z = (0.0, 0.0)
    agent_cfg = load_rl_cfg(task)
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
    venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = load_runner_cls(task)(venv, asdict(agent_cfg), device="cuda:0")
    runner.load(ckpt, load_cfg={"actor": True, "discriminator": False}, strict=True, map_location="cuda:0")
    policy = runner.get_inference_policy(device="cuda:0")
    robot = env.scene["robot"]
    ids = [robot.joint_names.index(j) for j in JOINTS]

    for vx in cmds:
        tw.ranges.lin_vel_x = (vx, vx)
        q, alive = [], torch.ones(n, dtype=torch.bool, device="cuda:0")
        with torch.inference_mode():
            obs, _ = venv.reset()
            for t in range(steps):
                obs, _, dones, _ = venv.step(policy(obs))
                alive &= ~dones.bool()
                if t >= 100:
                    q.append(robot.data.joint_pos[:, ids].cpu().numpy())
        q = np.stack(q, 1)[alive.cpu().numpy()]  # (episodes, T, 6)
        std = q.std(1).mean(0)
        mean = q.mean(1).mean(0)

        def corr(a, b):
            return float(np.mean([np.corrcoef(x[:, a], x[:, b])[0, 1] for x in q]))

        print(
            f"ARM {tag} vx {vx:.2f} | swing std shoulder pitch R {std[0]:.3f} L {std[1]:.3f}, elbow R {std[2]:.3f} "
            f"L {std[3]:.3f} (hip pitch {std[4]:.3f} / {std[5]:.3f}) | corr shoulder vs opposite hip R {corr(0, 5):+.2f} "
            f"L {corr(1, 4):+.2f} | mean shoulder pitch {mean[0]:+.2f} / {mean[1]:+.2f}, elbow {mean[2]:+.2f} / "
            f"{mean[3]:+.2f} rad | episodes {len(q)}/{n}"
        )
    env.close()


if __name__ == "__main__":
    main()
