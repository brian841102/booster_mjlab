"""Standing (cmd 0) fall diagnosis: direction and lead-up of falls."""
import sys
from dataclasses import asdict
import numpy as np, torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat
import booster_mjlab.tasks.velocity.config.nemov3  # noqa
task, ckpt, tag = sys.argv[1:4]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 256
cfg = load_env_cfg(task, play=True); cfg.scene.num_envs = n
tw = cfg.commands["twist"]; tw.ranges.lin_vel_x = (0.0, 0.0); tw.ranges.lin_vel_y = (0.0, 0.0); tw.ranges.ang_vel_z = (0.0, 0.0)
tw.rel_standing_envs = 0.0; tw.rel_forward_envs = 0.0
agent = load_rl_cfg(task)
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0"); venv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
runner = load_runner_cls(task)(venv, asdict(agent), device="cuda:0")
runner.load(ckpt, load_cfg={"actor": True, "discriminator": False}, strict=True, map_location="cuda:0")
pol = runner.get_inference_policy(device="cuda:0")
r = env.scene["robot"]; J = r.joint_names
torso = r.body_names.index("link_torso")
ids = {k: J.index(k) for k in ("right_hip_pitch_joint", "left_hip_pitch_joint", "right_calf_joint", "left_calf_joint", "right_wheel_joint", "left_wheel_joint", "right_hip_yaw_joint", "left_hip_yaw_joint")}
g_w = torch.tensor([0.0, 0.0, -1.0], device="cuda:0").expand(n, 3)
T = 1500
rec = np.zeros((T, n, 9), dtype=np.float32)
first = np.full(n, -1)
with torch.inference_mode():
    obs, _ = venv.reset()
    for t in range(T):
        d = r.data
        g = quat_apply_inverse(d.body_link_quat_w[:, torso], g_w)
        pitch = torch.rad2deg(torch.atan2(-g[:, 0], -g[:, 2])); roll = torch.rad2deg(torch.atan2(g[:, 1], -g[:, 2]))
        v = quat_apply_inverse(yaw_quat(d.root_link_quat_w), d.root_link_lin_vel_w)
        q, qd = d.joint_pos, d.joint_vel
        rec[t] = torch.stack([pitch, roll, v[:, 0], v[:, 1], 0.5 * (q[:, ids["right_hip_pitch_joint"]] + q[:, ids["left_hip_pitch_joint"]]),
                              q[:, ids["right_hip_pitch_joint"]] - q[:, ids["left_hip_pitch_joint"]],
                              0.5 * (qd[:, ids["right_wheel_joint"]] + qd[:, ids["left_wheel_joint"]]),
                              q[:, ids["right_hip_yaw_joint"]] - q[:, ids["left_hip_yaw_joint"]],
                              0.5 * (q[:, ids["right_calf_joint"]] + q[:, ids["left_calf_joint"]])], 1).cpu().numpy()
        obs, _, dn, _ = venv.step(pol(obs))
        dn = dn.cpu().numpy().astype(bool)
        first[dn & (first < 0)] = t
fell = first >= 0
print(f"{tag}: fell {fell.sum()}/{n}")
def stats(sel_t, envs, name):
    x = rec[sel_t][:, envs]  # [time, env, q]
    print(f"  {name}: torso pitch {x[..., 0].mean():+.1f} (std over time {x[..., 0].std(0).mean():.2f}) deg | roll std {x[..., 1].std(0).mean():.2f} | fwd vel std {x[..., 2].std(0).mean():.3f} m/s | "
          f"hip pitch mean {x[..., 4].mean():+.3f} | R-L hip pitch {np.abs(x[..., 5]).mean():.3f} | calf mean {x[..., 8].mean():+.3f} | R-L hip yaw {np.abs(x[..., 7]).mean():.3f} | wheel vel std {x[..., 6].std(0).mean():.2f}")
surv = np.flatnonzero(~fell)
stats(slice(100, 1500), surv, "survivors 2-30 s")
if fell.any():
    fe = np.flatnonzero(fell)
    at = first[fe]
    p_at = rec[at - 1, fe, 0]; r_at = rec[at - 1, fe, 1]
    print(f"  fall direction at last step: pitch>+20 (forward) {np.sum(p_at > 20)}, pitch<-20 (backward) {np.sum(p_at < -20)}, |roll|>20 (sideways) {np.sum(np.abs(r_at) > 20)}")
    # lead-up: stats in windows before the fall
    for lo, hi, nm in ((300, 150, "3-6 s before"), (150, 50, "1-3 s before"), (50, 1, "last 1 s")):
        rows = [rec[max(a - lo, 0):max(a - hi, 1), e] for a, e in zip(at, fe) if a - lo >= 100]
        if rows:
            x = np.stack([r_.mean(0) for r_ in rows]); s_ = np.stack([r_.std(0) for r_ in rows])
            print(f"  fallers {nm} (n={len(rows)}): pitch {x[:, 0].mean():+.1f} (std {s_[:, 0].mean():.2f}) | roll std {s_[:, 1].mean():.2f} | fwd vel std {s_[:, 2].mean():.3f} | "
                  f"R-L hip pitch {np.abs(x[:, 5]).mean():.3f} | wheel vel std {s_[:, 6].mean():.2f}")
env.close()
