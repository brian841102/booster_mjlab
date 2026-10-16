"""Body pose over the course of a lift, normalised to the lifted side, policy vs SkatingFast06.

usage: lift_pose.py ref
       lift_pose.py <task> <checkpoint.pt> [envs=32] [vx=1.0,1.3] [steps=600] [tag] [--rows]

A lift = a run of frames where one wheel is > 3 cm higher than the other, longer than 0.1 s. Every lift
is split into BINS equal phases (lift-off -> touchdown) and these are averaged over all lifts of both sides, with left
lifts mirrored so that "lifted" / "support" always name the same roles. Rows:
  behind      support wheel x - lifted wheel x (cm, heading frame of the root; > 0 = the lifted foot is the rear one)
  hand        lifted-side hand x - support-side hand x (cm, relative to the torso, heading frame of the root);
              > 0 = the hand on the side of the lifted leg is ahead
  hand_torso  the same in the torso's own heading (root yaw + pelvis yaw), i.e. the arm swing without the torso twist
  twist       pelvis yaw toward the support side (rad; > 0 = the upper body turns toward the support leg)
  calf        lifted-leg knee (calf joint, rad; smaller = straighter)
  hip         lifted-leg hip pitch (rad; the sign of the reference tells extension)
  sup_calf    support-leg knee (rad)

The CRIT line checks the user's criteria (2026-09-26 / 28) for a Fast06-like fast gait:
  rear     share of lift time with the lifted foot behind the support foot ("在前面那只脚应该是支撑脚 在后面的脚是伸直离地脚";
           Fast06 87 %)                                                               -- pass >= REAR_MIN
  stretch  mean behind over the first 60 % of the lift (cm; the rear leg stretched back -- Exp 54 4.7 cm "没有向后伸直",
           Exp 52 10.1 cm, Fast06 16.6 cm)                                            -- pass >= STRETCH_MIN
  hand     share of lift time with the lifted-side hand ahead in the torso's heading ("整个抬脚过程中，抬起那侧的手要在
           前面"; Exp 74's hand swings behind in the second half = 同手同脚)           -- pass >= HAND_MIN
"""

import os
import pickle
import sys

import numpy as np

SINGLE = 0.03
MIN_LIFT_S = 0.1
BINS = 5
REF = "assets/amp/nemov3_skating_straight/skatingfast06.pkl"
ROOT_BODY = "link_leg2pelvis"
BODIES = ("right_wheel_link", "left_wheel_link", "link_rWristPitch", "link_lWristPitch", "link_torso")
JOINTS = ("right_calf_joint", "left_calf_joint", "right_hip_pitch_joint", "left_hip_pitch_joint", "joint_pelvisYaw")
ROWS = ("behind", "hand", "hand_torso", "twist", "calf", "hip", "sup_calf")
REAR_MIN, STRETCH_MIN, HAND_MIN = 0.80, 8.0, 0.90


def _segments(mask):
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return list(zip(edges[::2], edges[1::2]))


def lift_profiles(pos, q, yaw, dt):
    """pos (T, 5, 3) root-relative world offsets of BODIES, q (T, 5) JOINTS, yaw (T,) root yaw -> per-lift (BINS, rows)."""
    def fwd(p, ang):
        return np.cos(ang)[:, None] * p[..., 0] + np.sin(ang)[:, None] * p[..., 1]

    x_root = fwd(pos, yaw)  # (T, 5) forward coordinate in the root heading
    x_torso = fwd(pos - pos[:, 4:5], yaw + q[:, 4])  # relative to the torso, in the torso heading
    h = pos[..., 2]
    out = []
    for side in (0, 1):  # 0: right leg lifted, 1: left leg lifted
        up, sup = side, 1 - side
        s = 1.0 if side == 0 else -1.0  # pelvis yaw > 0 turns left; toward the support side = left when right is lifted
        for a, b in _segments(h[:, up] - h[:, sup] > SINGLE):
            if (b - a) * dt < MIN_LIFT_S:
                continue
            sl = slice(a, b)
            rows = np.stack([
                (x_root[sl, sup] - x_root[sl, up]) * 100,
                ((x_root[sl, 2 + up] - x_root[sl, 4]) - (x_root[sl, 2 + sup] - x_root[sl, 4])) * 100,
                (x_torso[sl, 2 + up] - x_torso[sl, 2 + sup]) * 100,
                s * q[sl, 4],
                q[sl, up],
                q[sl, 2 + up],
                q[sl, sup],
            ], -1)
            idx = np.minimum((np.arange(b - a) / (b - a) * BINS).astype(int), BINS - 1)
            out.append((np.stack([rows[idx == k].mean(0) for k in range(BINS)]), rows))
    return out


def report(tag, label, prof, rows=True):
    if not prof:
        print(f"CRIT {tag} {label} | no lifts", flush=True)
        return
    p = np.mean([b for b, _ in prof], 0)
    frames = np.concatenate([f for _, f in prof])
    rear = np.mean(frames[:, 0] > 0)
    stretch = p[:3, 0].mean()
    hand = np.mean(frames[:, 2] > 0)
    ok = lambda c: "ok" if c else "FAIL"
    print(f"CRIT {tag} {label} | rear {rear * 100:3.0f}% {ok(rear >= REAR_MIN)} | stretch {stretch:4.1f} cm "
          f"{ok(stretch >= STRETCH_MIN)} | hand {hand * 100:3.0f}% {ok(hand >= HAND_MIN)} | "
          f"{'PASS' if rear >= REAR_MIN and stretch >= STRETCH_MIN and hand >= HAND_MIN else 'fail'} | {len(prof)} lifts",
          flush=True)
    if rows:
        for i, r in enumerate(ROWS):
            fmt = "{:+6.1f}" if i < 3 else "{:+6.3f}"
            print(f"    {r:10s} " + " ".join(fmt.format(v) for v in p[:, i]), flush=True)


def reference():
    from scipy.spatial.transform import Rotation as R

    from booster_mjlab.robots.nemov3.nemov3_constants import NEMOV3_JOINT_ORDER

    d = pickle.load(open(REF, "rb"))
    fps = d["fps"]
    names = list(d["link_body_list"])
    rot = R.from_quat(np.asarray(d["root_rot"]))
    world = np.einsum("tij,tbj->tbi", rot.as_matrix(), np.asarray(d["local_body_pos"]))
    pos = world[:, [names.index(b) for b in BODIES]] - world[:, [names.index(ROOT_BODY)]]
    order = list(NEMOV3_JOINT_ORDER)
    q = np.asarray(d["dof_pos"])[:, [order.index(j) for j in JOINTS]]
    v = np.linalg.norm(np.gradient(np.asarray(d["root_pos"])[:, :2], axis=0) * fps, axis=1)
    steady = np.flatnonzero(v > 0.8)
    sl = slice(steady[0], steady[-1] + 1)
    return lift_profiles(pos[sl], q[sl], rot.as_euler("ZYX")[sl, 0], 1.0 / fps)


def policy(task, ckpt, n, vxs, steps, tag, rows):
    from dataclasses import asdict

    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    import booster_mjlab.tasks.velocity.config.nemov3  # noqa: F401
    from booster_mjlab.robots.nemov3.nemov3_constants import HOME_KEYFRAME

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
    pol = runner.get_inference_policy(device="cuda:0")
    robot = env.scene["robot"]
    bids = [robot.body_names.index(b) for b in BODIES]
    rid = robot.body_names.index(ROOT_BODY)
    # A welded joint (fixed_joints, e.g. pelvis yaw) is not in the robot: it stays at its HOME_KEYFRAME angle.
    jids = [robot.joint_names.index(j) for j in JOINTS if j in robot.joint_names]
    welded = {i: HOME_KEYFRAME.joint_pos.get(j, 0.0) for i, j in enumerate(JOINTS) if j not in robot.joint_names}
    for vx in vxs:
        tw.ranges.lin_vel_x = (vx, vx)
        P, Q, Y = [], [], []
        with torch.inference_mode():
            obs, _ = venv.reset()
            alive = torch.ones(n, dtype=torch.bool, device="cuda:0")
            for t in range(steps):
                obs, _, dones, _ = venv.step(pol(obs))
                alive &= ~dones.bool()
                if t >= 100:
                    p = robot.data.body_link_pos_w
                    P.append((p[:, bids] - p[:, rid : rid + 1]).cpu().numpy())
                    q = robot.data.joint_pos[:, jids].cpu().numpy()
                    for i, v in welded.items():
                        q = np.insert(q, i, v, axis=1)
                    Q.append(q)
                    qq = robot.data.body_link_quat_w[:, rid]
                    Y.append(torch.atan2(2 * (qq[:, 0] * qq[:, 3] + qq[:, 1] * qq[:, 2]),
                                         1 - 2 * (qq[:, 2] ** 2 + qq[:, 3] ** 2)).cpu().numpy())
        P, Q, Y = np.stack(P, 1), np.stack(Q, 1), np.stack(Y, 1)
        prof = []
        for e in np.flatnonzero(alive.cpu().numpy()):
            prof += lift_profiles(P[e], Q[e], Y[e], env.step_dt)
        report(tag, f"vx {vx:.2f}", prof, rows)
    env.close()


def main():
    rows = "--rows" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--rows"]
    if args[0] == "ref":
        report("ref", "Fast06", reference(), True)
        return
    task, ckpt = args[0], args[1]
    n = int(args[2]) if len(args) > 2 else 32
    vxs = [float(v) for v in (args[3] if len(args) > 3 else "1.0,1.3").split(",")]
    steps = int(args[4]) if len(args) > 4 else 600
    tag = args[5] if len(args) > 5 else os.path.basename(os.path.dirname(ckpt))[-24:]
    policy(task, ckpt, n, vxs, steps, tag, rows)


if __name__ == "__main__":
    main()
