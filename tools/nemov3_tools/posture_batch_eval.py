"""Batched posture / swing evaluation (per episode, many environments), the batched version of posture_eval.py.

usage: posture_batch_eval.py <task> <checkpoint.pt> [envs=64] [cmds=0.3,0.6,1.0,1.3] [steps=600] [tag]

Loads the checkpoint like `play` (PyTorch policy, deterministic). Every robot = one episode, first 100 steps
dropped; statistics per episode, then averaged over the episodes that did not fall. Positions in the heading
(yaw-only) frame relative to the midpoint of the two wheel centres:
  wheel split std   std of x_R - x_L (cm): fore-aft leg swing amplitude at the wheels
  CoM dy std        std of the whole-body CoM lateral offset (cm): side-to-side weight shift while skating
  base dx / dy      mean root offset from the wheel midpoint (cm)
  torso pitch       mean pitch of link_torso from gravity (deg; NEGATIVE = leaning forward, e.g. Fast06 -10.8)
  speed             mean forward speed in the heading frame (m/s)
  yaw rate          std of the base yaw rate (deg/s): twisting / wandering in place (most telling at cmd 0)
  joint vel         RMS joint velocity over all position joints (rad/s): how much the body keeps moving
"""

import os
import sys
from dataclasses import asdict

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

import booster_mjlab.tasks.velocity.config.nemov3  # noqa: F401


def main():
    task, ckpt = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    cmds = [float(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "0.3,0.6,1.0,1.3").split(",")]
    steps = int(sys.argv[5]) if len(sys.argv) > 5 else 600
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
    runner.load(ckpt, load_cfg={"actor": True, "discriminator": False}, strict=True, map_location="cuda:0")
    policy = runner.get_inference_policy(device="cuda:0")
    robot = env.scene["robot"]
    term = env.command_manager.get_term("twist")
    mj = env.sim.mj_model
    mj_names = [mj.body(i).name for i in range(mj.nbody)]
    mass = torch.tensor(
        [mj.body_mass[next(i for i, nm in enumerate(mj_names) if nm == b or nm.endswith("/" + b))] for b in robot.body_names],
        device="cuda:0",
        dtype=torch.float32,
    )
    wr, wl = robot.body_names.index("right_wheel_link"), robot.body_names.index("left_wheel_link")
    torso = robot.body_names.index("link_torso")
    pos_ids = [i for i, nm in enumerate(robot.joint_names) if "wheel" not in nm]
    gravity_w = torch.tensor([0.0, 0.0, -1.0], device="cuda:0").expand(n, 3)

    for c in cmds:
        term.cfg.ranges.lin_vel_x = (c, c)
        rows = []
        with torch.inference_mode():
            obs, _ = venv.reset()
            alive = torch.ones(n, dtype=torch.bool, device="cuda:0")
            for t in range(steps):
                obs, _, dones, _ = venv.step(policy(obs))
                alive &= ~dones.bool()
                if t < 100:
                    continue
                d = robot.data
                heading = yaw_quat(d.root_link_quat_w)
                mid = 0.5 * (d.body_link_pos_w[:, wr] + d.body_link_pos_w[:, wl])
                base = quat_apply_inverse(heading, d.root_link_pos_w - mid)
                com_w = (d.body_com_pos_w * mass[None, :, None]).sum(1) / mass.sum()
                com = quat_apply_inverse(heading, com_w - mid)
                split = quat_apply_inverse(heading, d.body_link_pos_w[:, wr] - d.body_link_pos_w[:, wl])
                g = quat_apply_inverse(d.body_link_quat_w[:, torso], gravity_w)
                pitch = torch.rad2deg(torch.atan2(-g[:, 0], -g[:, 2]))
                vel = quat_apply_inverse(heading, d.root_link_lin_vel_w)
                yaw_rate = torch.rad2deg(d.root_link_ang_vel_w[:, 2])
                qd_rms = d.joint_vel[:, pos_ids].square().mean(1).sqrt()
                rows.append(
                    torch.stack([split[:, 0], com[:, 1], base[:, 0], base[:, 1], pitch, vel[:, 0], yaw_rate, qd_rms], 1).cpu().numpy()
                )
        x = np.stack(rows, 1)  # [env, time, quantity]
        ok = alive.cpu().numpy()
        xs = x[ok]
        print(
            f"PB {tag} cmd {c:.2f} (fell {int((~ok).sum())}/{n}) | wheel split std {xs[:, :, 0].std(1).mean() * 100:.1f} cm | "
            f"CoM dy std {xs[:, :, 1].std(1).mean() * 100:.2f} cm | base dx {xs[:, :, 2].mean() * 100:+.1f} dy {xs[:, :, 3].mean() * 100:+.1f} cm | "
            f"torso pitch {xs[:, :, 4].mean():+.1f} deg | speed {xs[:, :, 5].mean():.2f} m/s | "
            f"yaw rate std {xs[:, :, 6].std(1).mean():.1f} deg/s | joint vel rms {xs[:, :, 7].mean():.2f} rad/s",
            flush=True,
        )
    env.close()


if __name__ == "__main__":
    main()
