"""Equivariance check of the Nemov3 symmetry augmentation against the simulator.

For random states s: write s, recompute observations (``compute(update_history=True)``: plain ``compute()``
returns the cached buffer), then write the mirrored state M(s) (mirrored root pose / velocity, mirrored joints,
mirrored command, clock phase + pi, mirrored last action) and recompute. The augmentation is correct iff
mirror(obs(s)) == obs(M(s)) term by term. The robot is lifted off the ground so contact terms are zero in both,
and the encoder-bias randomisation is removed.
usage: TASK=<task> symmetry_check.py [n_states=8]
"""

import math
import os
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

import booster_mjlab.tasks.velocity.config.nemov3  # noqa: F401
from booster_mjlab.robots.nemov3.nemov3_constants import nemov3_mirror_maps
from booster_mjlab.tasks.velocity.mdp.nemov3_symmetry import mirror_actions, mirror_obs

n_states = int(sys.argv[1]) if len(sys.argv) > 1 else 8
cfg = load_env_cfg(os.environ.get("TASK", "Mjlab-Velocity-Flat-Amp-Clock-Arms-Booster-Nemov3"), play=True)
cfg.scene.num_envs = 1
cfg.events.pop("encoder_bias", None)
cfg.events.pop("push_robot", None)
for g in cfg.observations.values():
    g.enable_corruption = False
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
robot = env.scene["robot"]
twist = env.command_manager.get_term("twist")
clock = env.command_manager.get_term("gait_clock") if "gait_clock" in env.command_manager.active_terms else None
dev = "cuda:0"
J = list(robot.joint_names)  # all 32 joints, or fewer with fixed_joints
n_act = env.action_manager.total_action_dim
jidx, jsign = nemov3_mirror_maps(J)
jidx, jsign = torch.tensor(jidx, device=dev), torch.tensor(jsign, device=dev)
Mm = np.diag([1.0, -1.0, 1.0])
rng = np.random.default_rng(0)
env.reset()


def set_state(pos, rot, v, w, q, qd, cmd, phase, act):
    wxyz = np.r_[rot.as_quat()[3], rot.as_quat()[:3]]
    robot.write_root_link_pose_to_sim(torch.tensor(np.r_[pos, wxyz], dtype=torch.float32, device=dev)[None])
    robot.write_root_link_velocity_to_sim(torch.tensor(np.r_[v, w], dtype=torch.float32, device=dev)[None])
    robot.write_joint_state_to_sim(q, qd)
    env.sim.forward()
    env.scene.update(dt=env.physics_dt)
    twist.vel_command_b[:] = torch.tensor(cmd, dtype=torch.float32, device=dev)
    if clock is not None:
        clock.phase[:] = phase
        clock._refresh_command()
    env.action_manager._action[:] = act
    return env.observation_manager.compute(update_history=True)


worst = {}
for _ in range(n_states):
    pos = np.array([0.3, 0.2, 1.2])
    rot = Rotation.from_euler("xyz", rng.uniform(-0.4, 0.4, 3))
    v, w = rng.uniform(-1, 1, 3), rng.uniform(-1, 1, 3)
    lo, hi = robot.data.soft_joint_pos_limits[0, :, 0].cpu().numpy(), robot.data.soft_joint_pos_limits[0, :, 1].cpu().numpy()
    lo, hi = np.clip(lo, -1.0, 1.0), np.clip(hi, -1.0, 1.0)
    q = torch.tensor(rng.uniform(lo, hi), dtype=torch.float32, device=dev)[None]
    qd = torch.tensor(rng.uniform(-2, 2, len(J)), dtype=torch.float32, device=dev)[None]
    cmd = rng.uniform(-1, 1, 3)
    phase = float(rng.uniform(0, 2 * math.pi))
    act = torch.tensor(rng.uniform(-1, 1, n_act), dtype=torch.float32, device=dev)[None]
    o1 = {g: x.clone() for g, x in set_state(pos, rot, v, w, q, qd, cmd, phase, act).items()}
    rot_m = Rotation.from_matrix(Mm @ rot.as_matrix() @ Mm)
    o2 = set_state(
        Mm @ pos, rot_m, Mm @ v, np.array([-w[0], w[1], -w[2]]),
        jsign * q[:, jidx], jsign * qd[:, jidx], [cmd[0], -cmd[1], -cmd[2]],
        (phase + math.pi) % (2 * math.pi), mirror_actions(act, env),
    )
    for g in ("actor", "critic"):
        pred = mirror_obs(env, g, o1[g])
        names = env.observation_manager._group_obs_term_names[g]
        dims = [int(np.prod(d)) for d in env.observation_manager._group_obs_term_dim[g]]
        off = 0
        for n, d in zip(names, dims):
            e = (pred[:, off:off + d] - o2[g][:, off:off + d]).abs().max().item()
            scale = o2[g][:, off:off + d].abs().max().item()
            key = f"{g}/{n}"
            worst[key] = max(worst.get(key, (0.0, 0.0)), (e, scale))
            off += d
print("SC max |mirror(obs(s)) - obs(M(s))| per term (and the term's magnitude):")
for k, (e, s) in worst.items():
    print(f"SC   {k:28s} err {e:.2e}   |obs| {s:.2f}   {'OK' if e < 1e-3 * max(1.0, s) else 'MISMATCH'}")
env.close()
