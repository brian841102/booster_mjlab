"""Standing / skating posture: is the body straight when standing, and how much do the legs swing and the CoM
shift sideways while skating?

usage: TASK=<task> [EP_STEPS=500] posture_eval.py <onnx> [episodes=4] [cmds=0,0.6,1.0,1.3] [tag]

All positions in the heading (yaw-only) frame, relative to the midpoint of the two wheel centers; first 100 steps
of each episode dropped. Per command:
  base dx/dy       root link (link_leg2pelvis) offset from the wheel midpoint (mean, std)
  CoM dy           whole-body centre of mass lateral offset (mean, std = side-to-side weight shift)
  wheel split      x_R - x_L (mean, std = fore-aft leg swing amplitude) and mean |x_R - x_L|
  torso tilt       pitch / roll of link_torso from gravity (deg, mean and std)
  drift            mean |base xy velocity| (m/s; should be ~0 at cmd 0)
"""

import os
import sys

import numpy as np
import onnxruntime as ort
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

import booster_mjlab.tasks.velocity.config.nemov3  # noqa: F401


def main():
    onnx_path = sys.argv[1]
    n_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    cmds = [float(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "0,0.6,1.0,1.3").split(",")]
    tag = sys.argv[4] if len(sys.argv) > 4 else os.path.basename(onnx_path)
    ep_steps = int(os.environ.get("EP_STEPS", 500))

    cfg = load_env_cfg(os.environ.get("TASK", "Mjlab-Velocity-Flat-Amp-Clock-Arms-Booster-Nemov3"), play=True)
    cfg.scene.num_envs = 1
    tw = cfg.commands["twist"]
    tw.ranges.lin_vel_y = (0.0, 0.0)
    tw.ranges.ang_vel_z = (0.0, 0.0)
    tw.rel_standing_envs = 0.0
    tw.rel_forward_envs = 0.0
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    robot = env.scene["robot"]
    term = env.command_manager.get_term("twist")
    mj = env.sim.mj_model
    mj_names = [mj.body(i).name for i in range(mj.nbody)]
    mass = torch.tensor(
        [mj.body_mass[next(i for i, n in enumerate(mj_names) if n == b or n.endswith("/" + b))] for b in robot.body_names],
        device="cuda:0",
        dtype=torch.float32,
    )
    wr, wl = robot.body_names.index("right_wheel_link"), robot.body_names.index("left_wheel_link")
    torso = robot.body_names.index("link_torso")
    gravity_w = torch.tensor([[0.0, 0.0, -1.0]], device="cuda:0")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    nm = sess.get_inputs()[0].name

    for c in cmds:
        term.cfg.ranges.lin_vel_x = (c, c)
        rows = []
        for _ in range(n_ep):
            obs, _ = env.reset()
            for t in range(ep_steps):
                a = sess.run(None, {nm: obs["actor"].cpu().numpy().astype(np.float32)})[0]
                obs, *_ = env.step(torch.as_tensor(a, device="cuda:0"))
                if t < 100:
                    continue
                d = robot.data
                heading = yaw_quat(d.root_link_quat_w)
                mid = 0.5 * (d.body_link_pos_w[:, wr] + d.body_link_pos_w[:, wl])
                base = quat_apply_inverse(heading, d.root_link_pos_w - mid)[0]
                com_w = (d.body_com_pos_w[0] * mass[:, None]).sum(0, keepdim=True) / mass.sum()
                com = quat_apply_inverse(heading, com_w - mid)[0]
                split = quat_apply_inverse(heading, d.body_link_pos_w[:, wr] - d.body_link_pos_w[:, wl])[0]
                g = quat_apply_inverse(d.body_link_quat_w[:, torso], gravity_w)[0]
                pitch = np.degrees(np.arctan2(-g[0].item(), -g[2].item()))
                roll = np.degrees(np.arctan2(g[1].item(), -g[2].item()))
                rows.append([base[0].item(), base[1].item(), com[0].item(), com[1].item(), split[0].item(), pitch, roll,
                             torch.norm(d.root_link_lin_vel_w[0, :2]).item()])
        x = np.array(rows)
        print(
            f"POS {tag} cmd {c:.1f}: base dx {x[:,0].mean()*100:+.1f}+-{x[:,0].std()*100:.1f} dy {x[:,1].mean()*100:+.1f}+-{x[:,1].std()*100:.1f} cm | "
            f"CoM dx {x[:,2].mean()*100:+.1f} dy {x[:,3].mean()*100:+.1f}+-{x[:,3].std()*100:.1f} cm | "
            f"wheel split x_R-x_L {x[:,4].mean()*100:+.1f}+-{x[:,4].std()*100:.1f} cm (|.| {np.abs(x[:,4]).mean()*100:.1f}) | "
            f"torso pitch {x[:,5].mean():+.1f}+-{x[:,5].std():.1f} roll {x[:,6].mean():+.1f}+-{x[:,6].std():.1f} deg | speed {x[:,7].mean():.2f}",
            flush=True,
        )
    env.close()


if __name__ == "__main__":
    main()
