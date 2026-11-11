"""Replay a Pino v1 skating NPZ (from csv_to_npz.py) on pinov1.xml: kinematic playback, no physics.

Examples (run from the repository root):

    # interactive viewer, loops
    uv run python assets/amp_sources/pinov1_skating/data/replay_npz_mujoco.py \
        --motion_file assets/amp_sources/pinov1_skating/data/npz/skating_op_fps30_from1.2s.npz

    # viewer camera follows the robot (follow = position only, turn the view with the mouse; front / back / left /
    # right / left_front / ... also turn with the robot heading; mouse zoom / elevation stay usable in every mode)
    uv run python assets/amp_sources/pinov1_skating/data/replay_npz_mujoco.py \
        --motion_file .../skating_op_fps30_from1.2s.npz --camera follow

    # render a video (nemov3 record_reference.py look: left-front camera, caption = speed and each wheel's height)
    MUJOCO_GL=egl uv run python assets/amp_sources/pinov1_skating/data/replay_npz_mujoco.py \
        --motion_file .../skating_op_fps30_from1.2s.npz --video out.mp4 --speed 0.5
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

from booster_mjlab.robots.pinov1.pinov1_constants import PINOV1_WHEEL_BODY_RADII, PINOV1_XML

# Where the camera sits around the robot, as an azimuth offset from the robot heading (MuJoCo camera azimuth = the
# direction the camera looks along, so 0 = behind the robot looking forward, 180 = in front looking back at it).
CAMERA_AZIMUTH_OFFSETS = {"back": 0.0, "front": 180.0, "left": 270.0, "right": 90.0,
                          "left_front": 225.0, "right_front": 135.0, "left_back": 315.0, "right_back": 45.0}
CAMERA_CHOICES = ("free", "follow", *CAMERA_AZIMUTH_OFFSETS)
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
WHEEL_LABELS = (("左后轮", "left_rear_wheel_link"), ("左前轮", "left_front_wheel_link"),
                ("右后轮", "right_rear_wheel_link"), ("右前轮", "right_front_wheel_link"))


def load_motion(path: Path, model: mujoco.MjModel):
    """Frames as (root position, root quaternion wxyz, joint positions) and the qpos addresses of the joints."""
    with np.load(path) as npz:
        fps = float(np.asarray(npz["fps"]).reshape(-1)[0])
        joint_names = [str(n) for n in npz["joint_names"]]
        joint_pos = np.asarray(npz["joint_pos"], dtype=np.float64)
        root_pos = np.asarray(npz["body_pos_w"][:, 0], dtype=np.float64)
        root_quat = np.asarray(npz["body_quat_w"][:, 0], dtype=np.float64)
    adr = np.array([model.jnt_qposadr[model.joint(n).id] for n in joint_names])
    return fps, root_pos, root_quat / np.linalg.norm(root_quat, axis=1, keepdims=True), joint_pos, adr


def apply_frame(model, data, adr, root_pos, root_quat, joint_pos, frame: int) -> None:
    data.qpos[:] = 0.0
    data.qpos[:3] = root_pos[frame]
    data.qpos[3:7] = root_quat[frame]
    data.qpos[adr] = joint_pos[frame]
    mujoco.mj_forward(model, data)


def heading_deg(root_quat: np.ndarray) -> np.ndarray:
    w, x, y, z = root_quat.T
    return np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class FollowCamera:
    """Keeps a free camera on the robot; for a named view also turns it with the (smoothed) robot heading."""

    def __init__(self, camera: mujoco.MjvCamera, mode: str, yaw: np.ndarray):
        self.camera, self.yaw, self.azimuth = camera, yaw, None
        self.offset = CAMERA_AZIMUTH_OFFSETS.get(mode)

    def update(self, frame: int, root_pos: np.ndarray) -> None:
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = [root_pos[frame, 0], root_pos[frame, 1], 0.4]
        if self.offset is None:  # "follow": position only, the user turns the view with the mouse
            return
        target = self.yaw[frame] + self.offset  # smoothed so the camera does not swing with every push stroke
        self.azimuth = target if self.azimuth is None else self.azimuth + 0.05 * ((target - self.azimuth + 180.0) % 360.0 - 180.0)
        self.camera.azimuth = self.azimuth


def run_viewer(model, data, motion, speed: float, loop: bool, camera_mode: str) -> None:
    import mujoco.viewer

    fps, root_pos, root_quat, joint_pos, adr = motion
    with mujoco.viewer.launch_passive(model, data) as viewer:
        follow = None
        if camera_mode != "free":
            follow = FollowCamera(viewer.cam, camera_mode, heading_deg(root_quat))
            viewer.cam.distance, viewer.cam.elevation = 2.1, -10.0  # set once; mouse zoom / tilt keep working
        frame = 0
        while viewer.is_running():
            apply_frame(model, data, adr, root_pos, root_quat, joint_pos, frame)
            if follow is not None:
                follow.update(frame, root_pos)
            viewer.sync()
            time.sleep(1.0 / fps / speed)
            frame += 1
            if frame == len(joint_pos):
                if not loop:
                    break
                frame = 0


def render_video(model, data, motion, path: Path, speed: float, width: int, height: int, title: str,
                 camera_mode: str = "left_front") -> None:
    """Same look as tools/nemov3_tools/record_reference.py: camera at the robot's left front following the heading,
    black caption band with the speed and each wheel's bottom height (yellow above 2 cm), H.264 at 25 fps."""
    import imageio_ffmpeg
    from PIL import Image, ImageDraw, ImageFont

    fps, root_pos, root_quat, joint_pos, adr = motion
    out_fps = 25.0
    model.vis.global_.offwidth, model.vis.global_.offheight = width, height
    renderer = mujoco.Renderer(model, height, width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance, camera.elevation = 2.1, -10.0  # pinov1 is ~0.75 m tall
    speed_mps = np.linalg.norm(np.gradient(root_pos[:, :2], 1.0 / fps, axis=0), axis=1)
    follow = FollowCamera(camera, camera_mode, heading_deg(root_quat))
    wheels = [(label, model.body(name).id, PINOV1_WHEEL_BODY_RADII[name]) for label, name in WHEEL_LABELS]
    font = ImageFont.truetype(FONT, max(14, height // 26))
    small = ImageFont.truetype(FONT, max(12, height // 32))
    writer = imageio_ffmpeg.write_frames(str(path), (width, height), fps=out_fps, codec="libx264",
                                         pix_fmt_out="yuv420p", output_params=["-crf", "18"], macro_block_size=2)
    writer.send(None)
    for k in range(int(len(joint_pos) / fps / speed * out_fps)):
        frame = min(int(round(k * speed * fps / out_fps)), len(joint_pos) - 1)
        apply_frame(model, data, adr, root_pos, root_quat, joint_pos, frame)
        follow.update(frame, root_pos)
        renderer.update_scene(data, camera)
        image = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(image)
        draw.rectangle([0, 0, width, int(height * 0.19)], fill=(0, 0, 0))
        draw.text((12, 6), f"{title}（动捕回放）  {frame / fps:4.2f} s  速度 {speed_mps[frame]:.2f} m/s", font=font,
                  fill=(255, 255, 255))
        y2 = 8 + font.size + 4
        for i, (label, body, radius) in enumerate(wheels):
            cm = max(0.0, 100 * (data.xpos[body][2] - radius))
            draw.text((12 + i * width // 4, y2), f"{label} {cm:4.1f} cm", font=small,
                      fill=(255, 200, 0) if cm > 2 else (200, 200, 200))
        writer.send(np.asarray(image))
    writer.close()
    print(f"VIDEO {path} {len(joint_pos) / fps / speed:.1f} s {width}x{height}")


def add_scene(spec: mujoco.MjSpec) -> None:
    """mjlab's training scene look (terrains/terrain_entity.py): checker ground with edge marks, sun light, sky."""
    spec.add_texture(name="skybox", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.3, 0.5, 0.7], rgb2=[0.0, 0.0, 0.0], width=512, height=512)
    spec.add_texture(name="groundplane", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     mark=mujoco.mjtMark.mjMARK_EDGE, rgb1=[0.2, 0.3, 0.4], rgb2=[0.1, 0.2, 0.3],
                     markrgb=[0.8, 0.8, 0.8], width=300, height=300)
    material = spec.add_material(name="groundplane", texuniform=True, texrepeat=[4.0, 4.0], reflectance=0.2)
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "groundplane"
    spec.worldbody.add_geom(name="terrain", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material="groundplane")
    spec.worldbody.add_light(name="sun", pos=[0, 0, 1.5], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                             castshadow=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--motion_file", type=Path, required=True)
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video", type=Path, default=None, help="Render to this mp4 instead of opening the viewer.")
    parser.add_argument("--camera", choices=CAMERA_CHOICES, default=None,
                        help="Camera: free (fixed, mouse only), follow (follow position, mouse turns the view), or a view that follows position "
                             "and heading: " + ", ".join(CAMERA_AZIMUTH_OFFSETS) + ". Default: free in the viewer, "
                             "left_front for --video.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    # mjlab adds the ground in training; add the same one here so the replay shows where the wheels meet it.
    spec = mujoco.MjSpec.from_file(str(PINOV1_XML))
    add_scene(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    motion = load_motion(args.motion_file, model)
    if args.video is None:
        run_viewer(model, data, motion, args.speed, args.loop, args.camera or "free")
    else:
        if args.camera == "free":
            parser.error("--video needs a following camera (follow or a named view)")
        render_video(model, data, motion, args.video, args.speed, args.width, args.height, args.motion_file.stem,
                     args.camera or "left_front")


if __name__ == "__main__":
    main()
