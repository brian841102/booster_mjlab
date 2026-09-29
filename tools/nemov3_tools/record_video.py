"""Render a policy to an mp4 offscreen (no viewer / browser needed), stepping through a list of speed commands.

usage: record_video.py <task> <checkpoint.pt> <out.mp4> [segments=0:3,0.3:4,0.6:5,0.8:5,1.0:6,1.3:6] [width=854]
                       [height=480] [title]

segments = comma-separated "forward speed m/s:seconds" (yaw rate 0). One robot, deterministic policy, pushes off,
tracking camera from the robot's left front. Every frame gets a caption: commanded / actual speed and each wheel's bottom
height above the ground (so lifts are visible even in a small video). 25 fps real time (every 2nd 50 Hz step).
"""

import sys
from dataclasses import asdict

import imageio_ffmpeg
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import booster_mjlab.tasks.velocity.config.nemov3  # noqa: F401

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
R_WHEEL = 0.075


def main():
    task, ckpt, out = sys.argv[1], sys.argv[2], sys.argv[3]
    segs = [tuple(float(v) for v in s.split(":")) for s in (sys.argv[4] if len(sys.argv) > 4 else "0:3,0.3:4,0.6:5,0.8:5,1.0:6,1.3:6").split(",")]
    width = int(sys.argv[5]) if len(sys.argv) > 5 else 854
    height = int(sys.argv[6]) if len(sys.argv) > 6 else 480
    title = sys.argv[7] if len(sys.argv) > 7 else ""

    env_cfg = load_env_cfg(task, play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.viewer.width, env_cfg.viewer.height = width, height
    env_cfg.viewer.distance, env_cfg.viewer.elevation = 1.9, -10.0  # azimuth follows the heading (below)
    tw = env_cfg.commands["twist"]
    tw.rel_standing_envs = tw.rel_forward_envs = 0.0
    tw.ranges.lin_vel_y = tw.ranges.ang_vel_z = (0.0, 0.0)
    tw.ranges.lin_vel_x = (segs[0][0], segs[0][0])
    tw.resampling_time_range = (1.0e9, 1.0e9)
    agent_cfg = load_rl_cfg(task)
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0", render_mode="rgb_array")
    venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = load_runner_cls(task)(venv, asdict(agent_cfg), device="cuda:0")
    runner.load(ckpt, load_cfg={"actor": True, "discriminator": False}, strict=True, map_location="cuda:0")
    policy = runner.get_inference_policy(device="cuda:0")
    robot = env.scene["robot"]
    term = env.command_manager.get_term("twist")
    ids = [robot.body_names.index("right_wheel_link"), robot.body_names.index("left_wheel_link")]
    font = ImageFont.truetype(FONT, max(14, height // 26))
    small = ImageFont.truetype(FONT, max(12, height // 32))

    writer = imageio_ffmpeg.write_frames(out, (width, height), fps=25, codec="libx264", pix_fmt_out="yuv420p",
                                         output_params=["-crf", "23"], macro_block_size=2)
    writer.send(None)
    all_ids = torch.arange(env.num_envs, device="cuda:0")
    cam = env._offline_renderer._cam
    cam_az = None

    def heading_deg():
        q = robot.data.root_link_quat_w[0]
        return float(torch.rad2deg(torch.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))))

    with torch.inference_mode():
        obs, _ = venv.reset()
        for vx, seconds in segs:
            tw.ranges.lin_vel_x = (vx, vx)
            term._resample_command(all_ids)
            term.is_standing_env[:] = False
            for t in range(int(seconds / env.step_dt)):
                obs, *_ = venv.step(policy(obs))
                if t % 2:
                    continue
                # Resets randomize the heading, so the camera follows it: azimuth = heading + 225 deg puts the camera
                # at the robot's left front. Smoothed (~0.8 s) so the skating weave does not shake the view.
                target = heading_deg() + 225.0
                cam_az = target if cam_az is None else cam_az + 0.05 * ((target - cam_az + 180.0) % 360.0 - 180.0)
                cam.azimuth = cam_az
                img = Image.fromarray(env.render())
                d = ImageDraw.Draw(img)
                h = (robot.data.body_link_pos_w[0, ids, 2] - R_WHEEL).clamp_min(0).cpu().numpy() * 100
                speed = robot.data.root_link_lin_vel_b[0, 0].item()
                d.rectangle([0, 0, width, int(height * 0.19)], fill=(0, 0, 0))
                d.text((12, 6), f"{title}  指令 {vx:.1f} m/s   实际 {speed:.2f} m/s", font=font, fill=(255, 255, 255))
                lift = lambda cm: (255, 200, 0) if cm > 2 else (200, 200, 200)
                y2 = 8 + font.size + 4
                d.text((12, y2), f"右轮离地 {h[0]:4.1f} cm", font=small, fill=lift(h[0]))
                d.text((12 + width // 4, y2), f"左轮离地 {h[1]:4.1f} cm", font=small, fill=lift(h[1]))
                writer.send(np.asarray(img.convert("RGB")))
    writer.close()
    env.close()
    print(f"VIDEO {out} {sum(s for _, s in segs):.0f} s {width}x{height}")


if __name__ == "__main__":
    main()
