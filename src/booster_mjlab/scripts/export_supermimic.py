"""Export a SuperMimic checkpoint for the robot runner: ONNX + per-clip reference motions (supermimic_vibed's format).

    uv run export-supermimic Mjlab-SuperMimic-Flat-Sim2Real-Booster-Nemov3 --checkpoint <run>/model_11000.pt

writes into ``<run>/deploy`` (``--out-dir``):
  <name>.onnx   one input obs (1, hist * OBS_DIM + GOAL_DIM) = [observation history, oldest first, newest last | goal]
                -> actions (1, J); obs / goal normalizers inside, fp32 (training ran the MLPs under bf16 autocast)
  <name>.json   contract: joint order, default_q, action scales, PD gains, layouts, checkpoint / file SHA256
  supermimic_<robot>_<clip>.motion   text, 50 Hz: header, joint names, default_q, action_scale, velocity-joint mask,
                then per frame: reference q (J), reference root quat xyzw (4), goal (GOAL_DIM). The goal depends on the
                reference frame only (MotionLib.goal), so the runner looks it up.
``<name>`` defaults to ``supermimic_<robot>_<expN>@<iteration>``.

Joint order: by default the deployment order is the clip CSV order (``SuperMimicCfg.csv_joint_names``, = the joint
order of supermimic_vibed's exports, which the runner's ``nemov3_supermimic_1270`` layout reads); the ONNX maps it to
and from the training order (mjlab's) inside. ``--joint-order train`` exports in the training order instead.

Observation per 50 Hz step (TrackingMdp.obs without noise), joints in the deployment order:
  (q - default_q) * pos_mask, qd, root-frame angular velocity, projected gravity R^T (0,0,-1), previous raw action,
  mat_to_6d(R^T R_ref[f]) = [M00, M01, M10, M11, M20, M21] with R yaw-aligned to the reference at the start.
History starts as the first observation repeated. Targets: q = default_q + scale * a (position joints), wheel speed
= scale * a (velocity joints), clamped to ``wheel_speed_limit`` when the task has one. Frame f advances by one after
each policy step.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Annotated, Literal

import torch
import tyro

from mjlab import TYRO_FLAGS


@dataclass(frozen=True)
class ExportConfig:
    task: Annotated[str, tyro.conf.Positional]
    checkpoint: str
    """``model_<it>.pt`` of a SuperMimic run."""
    out_dir: str | None = None
    """Default: ``<run>/deploy``."""
    name: str | None = None
    """Default: ``supermimic_<robot>_<expN>@<iteration>``."""
    joint_order: Literal["csv", "train"] = "csv"
    device: str = "cuda:0"


def _sha256(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _row(xs) -> str:
    return " ".join(f"{float(x):.9g}" for x in xs)


def _match(table, name: str):
    return next((v for pattern, v in table if re.fullmatch(pattern, name)), None)


class _DeployPolicy(torch.nn.Module):
    """Flat deployment-order observation -> deployment-order action, around the training-order policy."""

    def __init__(self, pi, hist: int, obs_dim: int, goal_frames: int, J: int, to_train: torch.Tensor, to_deploy: torch.Tensor):
        super().__init__()
        self.pi, self.hist, self.obs_dim, self.K, self.J = pi, hist, obs_dim, goal_frames, J
        # index maps on the flat obs / goal vectors: train[i] = deploy[obs_idx[i]]
        jb = lambda start: [start + int(k) for k in to_train]  # noqa: E731  (a joint block starting at `start`)
        o = list(range(obs_dim))
        o[0:J], o[J : 2 * J], o[2 * J + 6 : 3 * J + 6] = jb(0), jb(J), jb(2 * J + 6)
        gf = 15 + 2 * J
        g = list(range(gf))
        g[15 : 15 + J], g[15 + J : gf] = jb(15), jb(15 + J)
        self.register_buffer("obs_idx", torch.tensor([t * obs_dim + i for t in range(hist) for i in o]))
        self.register_buffer("goal_idx", torch.tensor([k * gf + i for k in range(goal_frames) for i in g]))
        self.register_buffer("act_idx", to_deploy.clone())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        n = self.hist * self.obs_dim
        o, g = obs[:, :n][:, self.obs_idx], obs[:, n:][:, self.goal_idx]
        return self.pi(o.reshape(-1, self.hist, self.obs_dim), g)[:, self.act_idx]


def main():
    args = tyro.cli(ExportConfig, config=TYRO_FLAGS)
    import mjlab.tasks  # noqa: F401  (registers the tasks)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    import booster_mjlab.supermimic.modules.common as common
    from booster_mjlab.supermimic.runners.supermimic_runner import build_networks
    from booster_mjlab.supermimic.utils.rotations import mat_to_quat
    from booster_mjlab.tasks.supermimic.mdp import CONTROL_DT, goal_frame_dim
    from booster_mjlab.tasks.supermimic.supermimic_env import SuperMimicEnv

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = load_env_cfg(args.task, play=True)
    cfg.scene.num_envs = 1
    sim = SuperMimicEnv(ManagerBasedRlEnv(cfg, device=args.device), motion_dir=ck["motion_dir"])
    assert tuple(ck["joint_names"]) == sim.joint_names, "checkpoint and task joint orders differ"
    sm, mdp, motion, J = sim.sm, sim.mdp, sim.motion, sim.num_joints
    c = ck["cfg"]
    K, Hh = c["goal_frames"], c["hist"]
    GOAL_DIM = goal_frame_dim(J) * K
    IN_DIM = Hh * mdp.obs_dim + GOAL_DIM
    train_names = list(sim.joint_names)
    deploy_names = list(sm.csv_joint_names) if args.joint_order == "csv" else train_names
    to_deploy = torch.tensor([train_names.index(n) for n in deploy_names])  # deploy[k] = train[to_deploy[k]]
    to_train = torch.tensor([deploy_names.index(n) for n in train_names])  # train[j] = deploy[to_train[j]]

    common.AMP_DTYPE = None
    pi = build_networks(sim, c, args.device)[1]
    pi.load_state_dict(ck["pi"])
    pi = pi.cpu().eval()
    policy = _DeployPolicy(pi, Hh, mdp.obs_dim, K, J, to_train, to_deploy).eval()

    run_dir = os.path.dirname(os.path.realpath(args.checkpoint))
    it = ck["iteration"]
    exp = re.sub(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_?", "", os.path.basename(run_dir)).split("_")[0] or "run"
    robot = "nemov3" if "Nemov3" in args.task else args.task.split("-")[-1].lower()
    name = args.name or f"supermimic_{robot}_{exp}@{it}"
    out_dir = args.out_dir or os.path.join(run_dir, "deploy")
    os.makedirs(out_dir, exist_ok=True)
    onnx_path = os.path.join(out_dir, f"{name}.onnx")
    torch.onnx.export(policy, (torch.zeros(1, IN_DIM),), onnx_path, input_names=["obs"], output_names=["actions"],
                      opset_version=17, dynamo=False)  # fmt: skip

    def deploy_goal(goal: torch.Tensor) -> torch.Tensor:  # (B, GOAL_DIM) training order -> deployment order
        gf = goal_frame_dim(J)
        g = goal.reshape(-1, K, gf).clone()
        g[..., 15 : 15 + J] = g[..., 15 : 15 + J][..., to_deploy]
        g[..., 15 + J :] = g[..., 15 + J :][..., to_deploy]
        return g.reshape(-1, K * gf)

    vel = sim.vel.cpu()[to_deploy]
    # float32 -> the configured values (-0.26, not -0.25999999)
    default_q = sim.default_q.cpu()[to_deploy].double().round(decimals=6)
    scale = sim.alpha.cpu()[to_deploy].double().round(decimals=6)
    clips = {}
    with torch.no_grad():
        for ci, clip in enumerate(motion.names):
            f = torch.arange(int(motion.clip_start[ci]), int(motion.clip_end[ci]) + 1, device=sim.device)
            goal, q = deploy_goal(motion.goal(f, K)).cpu(), motion.q[f].cpu()[:, to_deploy]
            quat = mat_to_quat(motion.root_R[f]).cpu()  # wxyz
            out = os.path.join(out_dir, f"supermimic_{robot}_{clip}.motion")
            with open(out, "w") as fh:
                fh.write(f"SUPERMIMIC_V1 {len(f)} {round(1 / CONTROL_DT)} {J} {GOAL_DIM}\n")
                fh.write(" ".join(deploy_names) + "\n")
                fh.write(_row(default_q) + "\n" + _row(scale) + "\n")
                fh.write(" ".join("1" if v else "0" for v in vel) + "\n")
                for i in range(len(f)):
                    fh.write(_row(q[i]) + " " + _row(quat[i, [1, 2, 3, 0]]) + " " + _row(goal[i]) + "\n")
            clips[clip] = {"file": os.path.basename(out), "sha256": _sha256(out), "frames": len(f),
                           "duration_s": round((len(f) - 1) * CONTROL_DT, 2)}  # fmt: skip

    # PD gains / effort limits from the compiled (nominal) model: position kp = gainprm[0], kd = -biasprm[2];
    # velocity actuators (wheels) kv = gainprm[0]
    mj, robot_ent = sim.env.sim.mj_model, sim.robot
    jids = robot_ent.data.indexing.joint_ids.tolist()
    gains = {}
    for j, n in enumerate(train_names):
        a = next(a for a in range(mj.nu) if mj.actuator_trnid[a, 0] == jids[j])
        is_vel = bool(sim.vel[j])
        gains[n] = {"kp": 0.0 if is_vel else float(mj.actuator_gainprm[a, 0]), "kd": float(-mj.actuator_biasprm[a, 2]),
                    "effort": float(mj.actuator_forcerange[a, 1]), "velocity": _match(sm.velocity_limits, n)}  # fmt: skip
    s2r = sm.sim2real
    meta = {
        "robot": robot,
        "task": args.task,
        "checkpoint": os.path.relpath(os.path.realpath(args.checkpoint), os.getcwd()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "onnx": os.path.basename(onnx_path),
        "onnx_sha256": _sha256(onnx_path),
        "iteration": it,
        "policy_hz": round(1 / CONTROL_DT),
        "hist": Hh,
        "goal_frames": K,
        "obs_dim": mdp.obs_dim,
        "goal_dim": GOAL_DIM,
        "input_dim": IN_DIM,
        "obs_layout": [["joint_pos_rel_masked", J], ["joint_vel", J], ["base_ang_vel_b", 3], ["projected_gravity_b", 3],
                       ["last_action_raw", J], ["ref_root_ori_rel_6d", 6]],  # fmt: skip
        "goal_frame_layout": [["root_dpos_heading", 3], ["root_rot_heading_6d", 6], ["root_linvel_b", 3], ["root_angvel_b", 3],
                              ["ref_q_masked", J], ["ref_qd_x0.1", J]],  # fmt: skip
        "joint_names": deploy_names,
        "train_joint_names": train_names,
        "velocity_joints": [n for n in deploy_names if n in sm.velocity_joints],
        "pos_mask": (~vel).float().tolist(),
        "default_q": default_q.tolist(),
        "action_scale": scale.tolist(),
        "wheel_speed_limit": s2r.wheel_speed_limit if s2r else None,
        "gains": {n: gains[n] for n in deploy_names},
        "clips": clips,
    }
    json.dump(meta, open(os.path.join(out_dir, f"{name}.json"), "w"), indent=1)
    print(f"[export] {onnx_path} [1,{IN_DIM}] -> [1,{J}] ({args.joint_order} joint order); "
          f"clips: {', '.join(f'{k} ({v["frames"]} frames)' for k, v in clips.items())}")  # fmt: skip


if __name__ == "__main__":
    main()
