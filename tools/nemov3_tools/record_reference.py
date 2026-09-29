"""Render a reference skating clip (kinematic replay of the retargeted mocap) the way record_video.py renders a policy.

usage: record_reference.py <out.mp4> [clip=assets/amp/nemov3_skating_straight/skatingfast06.pkl] [start_s=2] [seconds=12]
                           [width=854] [height=480] [title=Fast06 参考]

Each output frame sets the robot's root pose and joint angles to the clip's (no physics), shifted in height so the lowest
wheel bottom of the clip touches the ground, and renders it with record_video.py's camera (left front, following the
heading) and caption (speed, each wheel's bottom height). 25 fps real time.
"""

import pickle
import sys

import imageio_ffmpeg
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation as R

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

import booster_mjlab.tasks.velocity.config.nemov3  # noqa: F401
from booster_mjlab.robots.nemov3.nemov3_constants import NEMOV3_JOINT_ORDER

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
R_WHEEL = 0.075
TASK = "Mjlab-Velocity-Flat-Amp-Clock-Arms-NarrowStance-Mirror-SymLoss-Lift-Staged-Booster-Nemov3"


def main():
    out = sys.argv[1]
    clip = sys.argv[2] if len(sys.argv) > 2 else "assets/amp/nemov3_skating_straight/skatingfast06.pkl"
    start = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
    seconds = float(sys.argv[4]) if len(sys.argv) > 4 else 12.0
    width = int(sys.argv[5]) if len(sys.argv) > 5 else 854
    height = int(sys.argv[6]) if len(sys.argv) > 6 else 480
    title = sys.argv[7] if len(sys.argv) > 7 else "Fast06 参考"

    d = pickle.load(open(clip, "rb"))
    fps = d["fps"]
    root_pos = np.asarray(d["root_pos"], dtype=np.float64)
    rot = R.from_quat(np.asarray(d["root_rot"]))
    names = list(d["link_body_list"])
    world = np.einsum("tij,tbj->tbi", rot.as_matrix(), np.asarray(d["local_body_pos"])) + root_pos[:, None]
    wheels = world[:, [names.index("right_wheel_link"), names.index("left_wheel_link")]]
    lo, hi = int(start * fps), min(int((start + seconds) * fps), len(root_pos))
    dz = R_WHEEL - wheels[lo:hi, :, 2].min()
    speed = np.linalg.norm(np.gradient(root_pos[:, :2], axis=0) * fps, axis=1)
    yaw = np.degrees(rot.as_euler("ZYX")[:, 0])
    quat_wxyz = np.asarray(d["root_rot"])[:, [3, 0, 1, 2]]
    dof = np.asarray(d["dof_pos"])

    env_cfg = load_env_cfg(TASK, play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.viewer.width, env_cfg.viewer.height = width, height
    env_cfg.viewer.distance, env_cfg.viewer.elevation = 1.9, -10.0
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0", render_mode="rgb_array")
    env.reset()
    robot = env.scene["robot"]
    order = list(NEMOV3_JOINT_ORDER)
    cols = [order.index(j) for j in robot.joint_names]
    cam = env._offline_renderer._cam
    font = ImageFont.truetype(FONT, max(14, height // 26))
    small = ImageFont.truetype(FONT, max(12, height // 32))
    writer = imageio_ffmpeg.write_frames(out, (width, height), fps=25, codec="libx264", pix_fmt_out="yuv420p",
                                         output_params=["-crf", "23"], macro_block_size=2)
    writer.send(None)
    cam_az = None
    for k in range(int((hi - lo) / fps * 25)):
        i = lo + int(round(k * fps / 25))
        if i >= hi:
            break
        pose = torch.tensor(np.r_[root_pos[i] + [0, 0, dz], quat_wxyz[i]], dtype=torch.float32, device="cuda:0")[None]
        robot.write_root_link_pose_to_sim(pose)
        robot.write_joint_position_to_sim(torch.tensor(dof[i, cols], dtype=torch.float32, device="cuda:0")[None])
        env.sim.forward()
        target = yaw[i] + 225.0
        cam_az = target if cam_az is None else cam_az + 0.05 * ((target - cam_az + 180.0) % 360.0 - 180.0)
        cam.azimuth = cam_az
        img = Image.fromarray(env.render())
        dr = ImageDraw.Draw(img)
        h = np.clip(wheels[i, :, 2] + dz - R_WHEEL, 0, None) * 100
        dr.rectangle([0, 0, width, int(height * 0.19)], fill=(0, 0, 0))
        dr.text((12, 6), f"{title}（动捕回放）  速度 {speed[i]:.2f} m/s", font=font, fill=(255, 255, 255))
        lift = lambda cm: (255, 200, 0) if cm > 2 else (200, 200, 200)
        y2 = 8 + font.size + 4
        dr.text((12, y2), f"右轮离地 {h[0]:4.1f} cm", font=small, fill=lift(h[0]))
        dr.text((12 + width // 4, y2), f"左轮离地 {h[1]:4.1f} cm", font=small, fill=lift(h[1]))
        writer.send(np.asarray(img.convert("RGB")))
    writer.close()
    env.close()
    print(f"VIDEO {out} {(hi - lo) / fps:.0f} s {width}x{height}")


if __name__ == "__main__":
    main()
