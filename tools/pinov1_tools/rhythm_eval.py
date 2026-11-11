"""Left/right symmetry of the alternation rhythm of a Pino v1 policy (and of the training clips).

usage: rhythm_eval.py <task> <checkpoint.pt> [speeds=0.3,0.6,0.9]

Per commanded speed, 32 robots, 10 s each (first 3 s dropped). From d = left hip pitch - right hip pitch (band-passed
0.2-3 Hz) per robot:
* time share: the fraction of the cycle with d > 0 ("left leg forward"); 0.5 = symmetric timing;
* half-cycle durations L->R / R->L (s) and their ratio;
* amplitude ratio: left / right hip-pitch swing (std); 1 = symmetric;
* offset: mean of the unfiltered d (rad) and of the rear wheels' fore-aft distance (cm, + = left ahead) -- a leg that
  stays ahead on average;
* stride frequency (Hz).
Medians over the robots, with the worst robot's |time share - 0.5|. The clips' values are printed first as the reference.
"""

import pickle
import sys
from dataclasses import asdict

import numpy as np
import torch
from scipy import signal

N = 32


def rhythm(dq: np.ndarray, lr: np.ndarray, fs: float):
    """dq: (T,) left - right hip pitch; lr: (T, 2) left / right hip pitch. Returns per-robot stats or None."""
    sos = signal.butter(2, [0.2, 3.0], btype="band", fs=fs, output="sos")
    x = signal.sosfiltfilt(sos, dq)
    if x.std() < 0.03:  # not alternating
        return None
    sign = x > 0
    edges = np.flatnonzero(np.diff(sign.astype(int)))
    if len(edges) < 3:
        return None
    halves = np.diff(edges) / fs
    first_pos = sign[edges[0] + 1]
    pos = halves[0::2] if first_pos else halves[1::2]
    neg = halves[1::2] if first_pos else halves[0::2]
    f = signal.sosfiltfilt(sos, lr, axis=0)
    return {
        "share": sign.mean(),
        "pos": pos.mean() if len(pos) else np.nan,
        "neg": neg.mean() if len(neg) else np.nan,
        "amp": f[:, 0].std() / max(f[:, 1].std(), 1e-6),
        "offset": dq.mean(),
        "freq": 1.0 / (2 * halves.mean()),
    }


def summarize(name: str, stats: list, foot_dx: float) -> str:
    stats = [s for s in stats if s is not None]
    if not stats:
        return f"{name}: no alternation"
    g = lambda k: np.nanmedian([s[k] for s in stats])
    worst = max(abs(s["share"] - 0.5) for s in stats)
    return (f"{name}: alternating {len(stats)} | left-forward share {g('share'):.2f} (worst |x-0.5| {worst:.2f}) | "
            f"half cycles L->R {g('pos'):.2f} s / R->L {g('neg'):.2f} s (ratio {g('pos') / g('neg'):.2f}) | "
            f"amp L/R {g('amp'):.2f} | offset {g('offset'):+.3f} rad, feet {100 * foot_dx:+.1f} cm | {g('freq'):.2f} Hz")


def reference():
    from booster_mjlab.robots.pinov1.pinov1_constants import PINOV1_JOINT_ORDER as J
    from scipy.spatial.transform import Rotation as R

    for name, lo in (("skating02", 0), ("skating01", 0), ("skatingfast06", 90)):
        d = pickle.load(open(f"assets/amp/pinov1_skating_straight/{name}.pkl", "rb"))
        q = np.asarray(d["dof_pos"])[lo:]
        lr = q[:, [J.index("left_hip_pitch_joint"), J.index("right_hip_pitch_joint")]]
        names = list(d["link_body_list"])
        rot = R.from_quat(np.asarray(d["root_rot"])[lo:])
        w = np.einsum("tij,tbj->tbi", rot.as_matrix(), np.asarray(d["local_body_pos"])[lo:])
        yaw = rot.as_euler("ZYX")[:, 0]
        dxy = w[:, names.index("left_rear_wheel_link"), :2] - w[:, names.index("right_rear_wheel_link"), :2]
        fdx = (dxy[:, 0] * np.cos(yaw) + dxy[:, 1] * np.sin(yaw)).mean()
        reps = 6 if len(lr) < 60 else 1  # one-cycle clips: tile so the filter sees several cycles
        lr = np.tile(lr, (reps, 1))
        print("RH " + summarize(f"REF {name}", [rhythm(lr[:, 0] - lr[:, 1], lr, float(d["fps"]))], fdx))


def policy(task: str, ckpt: str, speeds: list[float]):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    import booster_mjlab.tasks.velocity.config.pinov1  # noqa: F401

    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = N
    tw = cfg.commands["twist"]
    tw.rel_standing_envs = tw.rel_forward_envs = 0.0
    tw.resampling_time_range = (1e9, 1e9)
    tw.ranges.lin_vel_y = tw.ranges.ang_vel_z = (0.0, 0.0)
    agent = load_rl_cfg(task)
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    venv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = load_runner_cls(task)(venv, asdict(agent), device="cuda:0")
    runner.load(ckpt, load_cfg={"actor": True, "discriminator": False}, strict=True, map_location="cuda:0")
    pol = runner.get_inference_policy(device="cuda:0")
    robot = env.scene["robot"]
    term = env.command_manager.get_term("twist")
    hp = [robot.joint_names.index(n) for n in ("left_hip_pitch_joint", "right_hip_pitch_joint")]
    rw = [robot.body_names.index(n) for n in ("left_rear_wheel_link", "right_rear_wheel_link")]
    with torch.inference_mode():
        for vx in speeds:
            tw.ranges.lin_vel_x = (vx, vx)
            obs, _ = venv.reset()
            term.is_standing_env[:] = False
            Q, DX = [], []
            for t in range(500):
                term.vel_command_b[:, 0] = vx
                obs, *_ = venv.step(pol(obs))
                if t >= 150:
                    Q.append(robot.data.joint_pos[:, hp].clone())
                    qw = robot.data.root_link_quat_w
                    yaw = torch.atan2(2 * (qw[:, 0] * qw[:, 3] + qw[:, 1] * qw[:, 2]), 1 - 2 * (qw[:, 2] ** 2 + qw[:, 3] ** 2))
                    d = robot.data.body_link_pos_w[:, rw[0], :2] - robot.data.body_link_pos_w[:, rw[1], :2]
                    DX.append(d[:, 0] * torch.cos(yaw) + d[:, 1] * torch.sin(yaw))
            Q = torch.stack(Q).cpu().numpy()
            DX = torch.stack(DX).cpu().numpy()
            stats = [rhythm(Q[:, e, 0] - Q[:, e, 1], Q[:, e], 1.0 / env.step_dt) for e in range(N)]
            print("RH " + summarize(f"POL vx {vx}", stats, float(np.median(DX.mean(0)))))
    env.close()


if __name__ == "__main__":
    reference()
    if len(sys.argv) > 2:
        speeds = [float(v) for v in (sys.argv[3] if len(sys.argv) > 3 else "0.3,0.6,0.9").split(",")]
        policy(sys.argv[1], sys.argv[2], speeds)
