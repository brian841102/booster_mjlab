"""Render a policy to an mp4 offscreen (no viewer / browser needed), stepping through a list of speed commands.

usage: record_video.py <task> <checkpoint.pt> <out.mp4> [segments=forward] [width=854] [height=480] [title]

segments = a preset name (PRESETS below: forward, forward-turn, stand, forward-backward-turn) or
 comma-separated "forward speed m/s:seconds" (yaw rate 0) or "forward speed:yaw rate rad/s:seconds" (turning;
the caption then shows commanded / actual yaw rate too); a segment ending in "j" (e.g. "1.0:3j") presses the jump
button at its start (jump tasks, mdp/jump.py; the caption then marks the jump window). One robot, deterministic policy, pushes off,
tracking camera from the robot's left front. Every frame gets a caption: commanded / actual speed and each wheel's bottom
height above the ground (so lifts are visible even in a small video). 25 fps real time (every 2nd 50 Hz step).
Pino v1 tasks ("Pinov1" in the id) caption all four wheels (rear / front of each foot).
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
import booster_mjlab.tasks.velocity.config.pinov1  # noqa: F401
from booster_mjlab.robots.pinov1.pinov1_constants import PINOV1_WHEEL_BODY_RADII

# Named command sequences ("speed:seconds" or "speed:yaw rate:seconds"; a segment is not preceded by a reset).
PRESETS = {
    # Slow to fast straight skating: the default of the foot-lift runs (Exp 49+).
    "forward": "0:3,0.3:4,0.6:5,0.8:5,1.0:6,1.3:6",
    # Straight, then the turning limits going forward: +-0.6 / +-1.3 rad/s at 1.0 m/s, +1.3 at 0.6 m/s, in place +-1.3, stand.
    "forward-turn": "0:3,0.6:5,1.0:5,1.3:5,1.0:0.6:5,1.0:-0.6:5,1.0:1.3:5,1.0:-1.3:5,0.6:1.3:5,0:1.3:5,0:-1.3:5,0:3",
    # 30 s standing still.
    "stand": "0:30",
    # Stand, forward, then backward to -0.5 m/s and the turning limits while reversing (+-1.3 rad/s, 5 s each), in place
    # +-1.3 rad/s (5 s each), stand (Exp 90+).
    "forward-backward-turn": "0:3,0.6:5,1.0:5,0:2,-0.3:5,-0.5:5,-0.5:1.3:5,-0.5:-1.3:5,0:1.3:5,0:-1.3:5,0:3",
    # Jump button while skating at 0.6 / 1.0 / 1.3 m/s, twice per speed (task nemov3_velocity_amp_jump).
    "jump": "0:2,0.6:3,0.6:3j,0.6:3j,1.0:3,1.0:3j,1.0:3j,1.3:3,1.3:3j,1.3:3j,0:3",
}
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
R_WHEEL = 0.075
# (caption label, wheel body, radius) per robot.
NEMOV3_WHEELS = (("右轮离地", "right_wheel_link", R_WHEEL), ("左轮离地", "left_wheel_link", R_WHEEL))
PINOV1_WHEELS = tuple(
    (label, name, PINOV1_WHEEL_BODY_RADII[name])
    for label, name in (("左后轮", "left_rear_wheel_link"), ("左前轮", "left_front_wheel_link"),
                        ("右后轮", "right_rear_wheel_link"), ("右前轮", "right_front_wheel_link"))
)


def main():
    task, ckpt, out = sys.argv[1], sys.argv[2], sys.argv[3]
    spec = sys.argv[4] if len(sys.argv) > 4 else "forward"
    raw = PRESETS.get(spec, spec).split(",")
    presses = [s.endswith("j") for s in raw]
    segs = [[float(v) for v in s.rstrip("j").split(":")] for s in raw]
    segs = [(s[0], 0.0, s[1]) if len(s) == 2 else tuple(s) for s in segs]
    turning = any(wz for _, wz, _ in segs)
    width = int(sys.argv[5]) if len(sys.argv) > 5 else 854
    height = int(sys.argv[6]) if len(sys.argv) > 6 else 480
    title = sys.argv[7] if len(sys.argv) > 7 else ""

    env_cfg = load_env_cfg(task, play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.viewer.width, env_cfg.viewer.height = width, height
    pinov1 = "Pinov1" in task
    # azimuth follows the heading (below); pinov1 is taller (~0.75 m)
    env_cfg.viewer.distance, env_cfg.viewer.elevation = (2.1 if pinov1 else 1.9), -10.0
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
    jump = env.command_manager.get_term("jump") if "jump" in env.command_manager.active_terms else None
    assert jump is not None or not any(presses), "segments press the jump button but the task has no jump command"
    wheels = PINOV1_WHEELS if pinov1 else NEMOV3_WHEELS
    ids = [robot.body_names.index(name) for _, name, _ in wheels]
    radii = torch.tensor([r for *_, r in wheels], device="cuda:0")
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
        for (vx, wz, seconds), press in zip(segs, presses):
            if press:
                jump.trigger()
                obs = venv.get_observations()
            tw.ranges.lin_vel_x = (vx, vx)
            tw.ranges.ang_vel_z = (wz, wz)
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
                h = (robot.data.body_link_pos_w[0, ids, 2] - radii).clamp_min(0).cpu().numpy() * 100
                speed = robot.data.root_link_lin_vel_b[0, 0].item()
                d.rectangle([0, 0, width, int(height * 0.19)], fill=(0, 0, 0))
                d.text((12, 6), f"{title}  指令 {vx:.1f} m/s   实际 {speed:.2f} m/s", font=font, fill=(255, 255, 255))
                lift = lambda cm: (255, 200, 0) if cm > 2 else (200, 200, 200)
                y2 = 8 + font.size + 4
                step = width // 4 if len(wheels) == 2 else width // 5
                for k, (label, *_) in enumerate(wheels):
                    d.text((12 + k * step, y2), f"{label} {h[k]:4.1f} cm", font=small, fill=lift(h[k]))
                if jump is not None and bool(jump.active[0]):
                    d.text((12 + 3 * width // 4, 6), "跳跃!", font=font, fill=(255, 80, 80))
                if turning:
                    yaw = robot.data.root_link_ang_vel_w[0, 2].item()
                    # Two wheel labels leave the right half free; pinov1's four fill the row, so turn goes below them.
                    xy = (12 + width // 2, y2) if len(wheels) == 2 else (12, y2 + small.size + 4)
                    d.text(xy, f"转弯 指令 {wz:+.1f} 实际 {yaw:+.2f} rad/s", font=small,
                           fill=(120, 220, 255) if wz else (200, 200, 200))
                writer.send(np.asarray(img.convert("RGB")))
    writer.close()
    env.close()
    print(f"VIDEO {out} {sum(s for *_, s in segs):.0f} s {width}x{height}")


if __name__ == "__main__":
    main()
