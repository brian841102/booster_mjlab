"""Which left/right asymmetry reproduces the hardware's "leans to one side standing, asymmetric when driving"?

usage: asym_probe.py <task> <ckpt> [envs=32] [cases=all|name,name,...] [stand_s=20] [drive_s=12] [drive_vx=1.0]
Builds one env per case with ALL startup randomization and pushes removed (so the only asymmetry is the case), applies the
case, then: stands 0 m/s for stand_s (first 5 s = settle, the rest is measured), then drives drive_vx for drive_s (first 4 s
settle). Per phase, averaged over non-fallen envs (body frame at the window start):
  dpsi = heading change (deg), dy = lateral drift (cm), roll = mean trunk roll (deg, + = leaning to the robot's left),
  wL/wR = mean wheel speed (rad/s), hrL/hrR = mean hip roll (rad), pitch pk-pk L/R (rad, drive only), falls.
A case is "reproduced" when stand shows a one-sided roll / drift / wheel creep, and drive shows a heading offset.
Case syntax in CASES below: wheel kd scale of the left wheel, left-wheel friction scale, joint encoder bias (rad, joint's
encoder reads true+bias -> the policy believes the joint is further along than it is), torso COM offset (m), IMU tilt (deg: the observed gravity is rotated by roll about x and pitch about y)."""

import math
import sys
from dataclasses import asdict

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity import mdp
from mjlab.utils.lab_api.math import quat_apply_inverse

import booster_mjlab.tasks.velocity.config.nemov3  # noqa: F401

TORSO = "link_torso"
LEFT_WHEEL_GEOM = "left_wheel_collision"
HIPS = ("hip_roll", "hip_yaw", "hip_pitch")
# name -> dict(kd=left wheel kd scale, fric=left wheel friction scale, bias={joint: rad}, com=(x, y, z) m)
CASES: dict[str, dict] = {"baseline": {}}
for s in (0.7, 1.3):
    CASES[f"wheelKdL x{s}"] = {"kd": s}
CASES["wheelFricL x0.7"] = {"fric": 0.7}
for kind in HIPS:
    for b in (0.03, 0.05):
        CASES[f"bias L {kind} {b:+.2f}"] = {"bias": {f"left_{kind}_joint": b}}
CASES["bias L calf +0.03"] = {"bias": {"left_calf_joint": 0.03}}
CASES["bias L+R hip_roll +0.03 (same sign)"] = {"bias": {"left_hip_roll_joint": 0.03, "right_hip_roll_joint": 0.03}}
for roll, pitch in ((2.0, 0.0), (4.0, 0.0), (0.0, 2.0)):
    CASES[f"IMU offset roll {roll:+.0f} pitch {pitch:+.0f} deg"] = {"imu": (roll, pitch)}
for y in (0.01, 0.03):
    CASES[f"torso COM y {y * 100:+.0f}cm"] = {"com": (0.0, y, 0.0)}
CASES["torso COM x +2cm"] = {"com": (0.02, 0.0, 0.0)}


def run_case(task, ckpt, name, case, n, stand_s, drive_s, vx):
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = n
    for key in [k for k, e in cfg.events.items() if e.mode == "startup" or k == "push_robot"]:
        cfg.events.pop(key)
    torso = SceneEntityCfg("robot", body_names=(TORSO,))
    if "com" in case:
        cfg.events["asym_com"] = EventTermCfg(func=mdp.dr.body_com_offset, mode="startup",
                                              params={"asset_cfg": torso, "ranges": {i: (c, c) for i, c in enumerate(case["com"])},
                                                      "operation": "add"})
    if "fric" in case:
        cfg.events["asym_fric"] = EventTermCfg(func=mdp.dr.geom_friction, mode="startup",
                                               params={"asset_cfg": SceneEntityCfg("robot", geom_names=(LEFT_WHEEL_GEOM,)),
                                                       "ranges": (case["fric"], case["fric"]), "operation": "scale"})
    if "imu" in case:
        roll, pitch = (math.radians(a) for a in case["imu"])
        rot = torch.tensor([[math.cos(pitch), math.sin(roll) * math.sin(pitch), math.cos(roll) * math.sin(pitch)],
                            [0.0, math.cos(roll), -math.sin(roll)],
                            [-math.sin(pitch), math.sin(roll) * math.cos(pitch), math.cos(roll) * math.cos(pitch)]], device="cuda:0")
        gravity = cfg.observations["actor"].terms["projected_gravity"]
        base_fn = gravity.func
        gravity.func = lambda env, **kw: base_fn(env, **kw) @ rot.T  # the IMU's tilt: the policy sees a rotated gravity
    tw = cfg.commands["twist"]
    for r in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        setattr(tw.ranges, r, (0.0, 0.0))
    tw.rel_standing_envs = tw.rel_forward_envs = 0.0
    agent = load_rl_cfg(task)
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    venv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = load_runner_cls(task)(venv, asdict(agent), device="cuda:0")
    runner.load(ckpt, load_cfg={"actor": True, "discriminator": False}, strict=True, map_location="cuda:0")
    pol = runner.get_inference_policy(device="cuda:0")
    robot = env.scene["robot"]
    term = env.command_manager.get_term("twist")
    if "kd" in case:
        for act in robot.actuators:
            if "left_wheel_joint" in act.target_names:
                ids = act.global_ctrl_ids
                env.sim.model.actuator_gainprm[:, ids, 0] *= case["kd"]
                env.sim.model.actuator_biasprm[:, ids, 2] *= case["kd"]
    ji = {j: robot.joint_names.index(j) for j in ("left_hip_roll_joint", "right_hip_roll_joint", "left_hip_pitch_joint",
                                                   "right_hip_pitch_joint", "left_wheel_joint", "right_wheel_joint")}
    fell = torch.zeros(n, dtype=torch.bool, device="cuda:0")
    out = {}
    with torch.inference_mode():
        obs, _ = venv.reset()
        for jn, b in case.get("bias", {}).items():
            robot.data.encoder_bias[:, robot.joint_names.index(jn)] = b
        obs = venv.get_observations() if hasattr(venv, "get_observations") else obs
        for phase, cmd, total, settle in (("stand", 0.0, stand_s, 5.0), ("drive", vx, drive_s, 4.0)):
            term.vel_command_b[:, 0] = cmd
            acc = {k: torch.zeros(n, device="cuda:0") for k in ("roll", "wL", "wR", "hrL", "hrR")}
            pmin = torch.full((n, 2), 1e9, device="cuda:0"); pmax = torch.full((n, 2), -1e9, device="cuda:0")
            cnt = 0
            p0 = q0 = None
            for t in range(int(round(total / env.step_dt))):
                term.vel_command_b[:, 0] = cmd
                obs, _, d, _ = venv.step(pol(obs))
                fell |= d.bool()
                if t * env.step_dt < settle:
                    continue
                if p0 is None:
                    p0 = robot.data.root_link_pos_w.clone(); q0 = robot.data.root_link_quat_w.clone()
                    yaw0 = torch.atan2(2 * (q0[:, 0] * q0[:, 3] + q0[:, 1] * q0[:, 2]), 1 - 2 * (q0[:, 2] ** 2 + q0[:, 3] ** 2))
                g = quat_apply_inverse(robot.data.root_link_quat_w, torch.tensor([0.0, 0.0, -1.0], device="cuda:0").expand(n, 3))
                acc["roll"] += torch.rad2deg(torch.asin(g[:, 1].clamp(-1, 1)))
                jv, jp = robot.data.joint_vel, robot.data.joint_pos
                acc["wL"] += jv[:, ji["left_wheel_joint"]]; acc["wR"] += jv[:, ji["right_wheel_joint"]]
                acc["hrL"] += jp[:, ji["left_hip_roll_joint"]]; acc["hrR"] += jp[:, ji["right_hip_roll_joint"]]
                pitch = jp[:, [ji["left_hip_pitch_joint"], ji["right_hip_pitch_joint"]]]
                pmin = torch.minimum(pmin, pitch); pmax = torch.maximum(pmax, pitch)
                cnt += 1
            q1 = robot.data.root_link_quat_w
            yaw1 = torch.atan2(2 * (q1[:, 0] * q1[:, 3] + q1[:, 1] * q1[:, 2]), 1 - 2 * (q1[:, 2] ** 2 + q1[:, 3] ** 2))
            dpsi = torch.rad2deg((yaw1 - yaw0 + torch.pi) % (2 * torch.pi) - torch.pi)
            disp = quat_apply_inverse(q0, robot.data.root_link_pos_w - p0)
            ok = ~fell
            m = lambda x: x[ok].mean().item() if ok.any() else float("nan")
            out[phase] = dict(dpsi=m(dpsi), dx=m(disp[:, 0]) * 100, dy=m(disp[:, 1]) * 100, roll=m(acc["roll"] / cnt),
                              wL=m(acc["wL"] / cnt), wR=m(acc["wR"] / cnt), hrL=m(acc["hrL"] / cnt), hrR=m(acc["hrR"] / cnt),
                              pL=m(pmax[:, 0] - pmin[:, 0]), pR=m(pmax[:, 1] - pmin[:, 1]), falls=int(fell.sum()))
    env.close()
    return out


def main():
    task, ckpt = sys.argv[1:3]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    sel = sys.argv[4] if len(sys.argv) > 4 else "all"
    stand_s = float(sys.argv[5]) if len(sys.argv) > 5 else 20.0
    drive_s = float(sys.argv[6]) if len(sys.argv) > 6 else 12.0
    vx = float(sys.argv[7]) if len(sys.argv) > 7 else 1.0
    names = list(CASES) if sel == "all" else sel.split(",")
    print(f"ckpt {ckpt}  envs {n}  stand {stand_s}s (measure last {stand_s - 5:.0f}s)  drive {vx} m/s {drive_s}s (measure last {drive_s - 4:.0f}s)")
    print(f"{'case':38s} | {'STAND dpsi dy(cm) roll wL wR hrL hrR':46s} | {'DRIVE dpsi dy(cm) roll wL-wR hrL hrR pkL pkR':50s} | falls")
    for name in names:
        r = run_case(task, ckpt, name, CASES[name], n, stand_s, drive_s, vx)
        s, d = r["stand"], r["drive"]
        print(f"{name:38s} | {s['dpsi']:+6.1f} {s['dy']:+6.1f} {s['roll']:+5.2f} {s['wL']:+6.2f} {s['wR']:+6.2f} {s['hrL']:+6.3f} {s['hrR']:+6.3f}"
              f" | {d['dpsi']:+6.1f} {d['dy']:+6.1f} {d['roll']:+5.2f} {d['wL'] - d['wR']:+6.2f} {d['hrL']:+6.3f} {d['hrR']:+6.3f} {d['pL']:5.2f} {d['pR']:5.2f}"
              f" | {d['falls']}", flush=True)


if __name__ == "__main__":
    main()
