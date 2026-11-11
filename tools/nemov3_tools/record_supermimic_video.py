"""Render SuperMimic evaluation rollouts (``model_<it>_traj.npz`` from ``evaluate-supermimic``) to mp4, one per clip.

usage: record_supermimic_video.py <task> <model_<it>_traj.npz> [out_dir=<run>/videos] [width=854] [height=480]
       [--ghost]  (also draw the reference motion as a translucent blue ghost)

Same output as record_video.py (854x480, 25 fps = every 2nd 50 Hz step, libx264, caption bar on top), but offline from
the recorded states, no policy or physics. Camera: starts at the robot's left front (azimuth = initial heading + 225 deg),
5 deg down, distance such that the robot fills ~60% of the frame height; it follows the robot's position, not its
heading. Writes <out_dir>/<expN>@<it>_<clip>.mp4.

Rendering uses its own EGL context on the default display (mujoco.Renderer's per-device EGL path fails on some hosts).
"""

import ctypes
import json
import math
import os
import re
import sys

os.environ["PYOPENGL_PLATFORM"] = "egl"

import imageio_ffmpeg  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from OpenGL import EGL  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
ROBOT_HEIGHT = 0.82  # m, top of the head in the HOME pose
FILL = 0.6  # fraction of the frame height the robot takes
ELEVATION = -5.0  # deg
LOOKAT_HEIGHT = 0.62  # x ROBOT_HEIGHT: the camera aims above the robot's middle, so it sits lower in the frame


def egl_context(w: int, h: int) -> None:
    dpy = EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY)
    EGL.eglInitialize(dpy, None, None)
    attrs = (EGL.EGLint * 13)(EGL.EGL_SURFACE_TYPE, EGL.EGL_PBUFFER_BIT, EGL.EGL_RED_SIZE, 8, EGL.EGL_GREEN_SIZE, 8,
                              EGL.EGL_BLUE_SIZE, 8, EGL.EGL_DEPTH_SIZE, 24, EGL.EGL_RENDERABLE_TYPE, EGL.EGL_OPENGL_BIT,
                              EGL.EGL_NONE)  # fmt: skip
    cfg, n = EGL.EGLConfig(), EGL.EGLint()
    EGL.eglChooseConfig(dpy, attrs, ctypes.pointer(cfg), 1, ctypes.pointer(n))
    assert n.value > 0, "no EGL config"
    surf = EGL.eglCreatePbufferSurface(dpy, cfg, (EGL.EGLint * 5)(EGL.EGL_WIDTH, w, EGL.EGL_HEIGHT, h, EGL.EGL_NONE))
    EGL.eglBindAPI(EGL.EGL_OPENGL_API)
    EGL.eglMakeCurrent(dpy, surf, surf, EGL.eglCreateContext(dpy, cfg, EGL.EGL_NO_CONTEXT, None))


def scene_model(robot_spec: mujoco.MjSpec, width: int, height: int, ghost: bool) -> mujoco.MjModel:
    """Robot (and its reference ghost) on a checker floor."""
    spec = mujoco.MjSpec()
    spec.visual.global_.offwidth, spec.visual.global_.offheight = width, height
    # lighting of mjlab's scene, as record_video.py's AMP videos: headlight + a directional sun with shadows
    spec.visual.headlight.diffuse, spec.visual.headlight.ambient, spec.visual.headlight.specular = [0.6] * 3, [0.3] * 3, [0.0] * 3
    spec.visual.rgba.haze = [0.15, 0.25, 0.35, 1.0]
    spec.visual.quality.shadowsize = 8192
    # MuJoCo's classic blue checker floor and gradient sky
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     mark=mujoco.mjtMark.mjMARK_EDGE, rgb1=[0.2, 0.3, 0.4], rgb2=[0.1, 0.2, 0.3], markrgb=[0.8, 0.8, 0.8],
                     width=300, height=300)  # fmt: skip
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.3, 0.5, 0.7], rgb2=[0.0, 0.0, 0.0], width=512, height=3072)  # fmt: skip
    mat = spec.add_material(name="grid", texrepeat=[4, 4], texuniform=True, reflectance=0.2)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material="grid", contype=0, conaffinity=0)
    light = spec.worldbody.add_light(pos=[0, 0, 1.5], dir=[0, 0, -1], diffuse=[0.7, 0.7, 0.7], castshadow=True)
    light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    for name in ("robot", "ghost") if ghost else ("robot",):
        frame = spec.worldbody.add_frame()
        spec.attach(robot_spec.copy() if name == "ghost" else robot_spec, prefix=f"{name}/", frame=frame)
    m = spec.compile()
    m.stat.extent = 4.0  # mjlab's offscreen renderer: a local extent keeps the shadow map sharp
    if ghost:
        for g in range(m.ngeom):
            if (m.geom(g).name or "").startswith("ghost/") or m.body(m.geom_bodyid[g]).name.startswith("ghost/"):
                m.geom_rgba[g] = [0.35, 0.65, 1.0, 0.35]
    return m


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ghost = "--ghost" in sys.argv
    task, traj_path = args[0], args[1]
    run_dir = os.path.dirname(os.path.realpath(traj_path))
    out_dir = args[2] if len(args) > 2 else os.path.join(run_dir, "videos")
    width = int(args[3]) if len(args) > 3 else 854
    height = int(args[4]) if len(args) > 4 else 480
    egl_context(width, height)

    import mjlab.tasks  # noqa: F401  (registers the tasks)
    from mjlab.tasks.registry import load_env_cfg

    robot_spec = load_env_cfg(task, play=True).scene.entities["robot"].spec_fn()
    m = scene_model(robot_spec, width, height, ghost)
    d = mujoco.MjData(m)
    z = np.load(traj_path)
    names = [str(n) for n in z["joint_names"]]
    qadr = {p: np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{p}/{n}")] for n in names])
            for p in ("robot", "ghost") if ghost or p == "robot"}  # fmt: skip
    root = {p: m.jnt_qposadr[next(j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                                  and m.joint(j).name.startswith(f"{p}/"))] for p in qadr}  # fmt: skip
    it = int(re.search(r"model_(\d+)_traj", os.path.basename(traj_path)).group(1))
    exp = re.sub(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_?", "", os.path.basename(run_dir)).split("_")[0] or "run"
    ev_path = traj_path.replace("_traj.npz", "_eval.json")
    per_clip = json.load(open(ev_path))["per_clip"] if os.path.exists(ev_path) else {}
    os.makedirs(out_dir, exist_ok=True)

    ctx = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150)
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, ctx)
    scn, opt, cam = mujoco.MjvScene(m, 10000), mujoco.MjvOption(), mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.elevation = ELEVATION
    cam.distance = ROBOT_HEIGHT / (FILL * 2 * math.tan(math.radians(m.vis.global_.fovy) / 2))
    vp = mujoco.MjrRect(0, 0, width, height)
    font = ImageFont.truetype(FONT, max(14, height // 26))
    img = np.zeros((height, width, 3), np.uint8)
    for c, clip in enumerate(str(x) for x in z["names"]):
        n = int(z["alive"][:, c].sum())
        q0 = z["root_quat"][0, c]
        heading = math.degrees(math.atan2(2 * (q0[0] * q0[3] + q0[1] * q0[2]), 1 - 2 * (q0[2] ** 2 + q0[3] ** 2)))
        cam.azimuth = heading + 225.0  # left front of the robot at the start, then fixed
        out = os.path.join(out_dir, f"{exp}@{it}_{clip}.mp4")
        writer = imageio_ffmpeg.write_frames(out, (width, height), fps=25, codec="libx264", pix_fmt_out="yuv420p",
                                             output_params=["-crf", "23"], macro_block_size=2)  # fmt: skip
        writer.send(None)
        ret = f"回报 {per_clip[clip]:.3f}" if clip in per_clip else ""
        for t in range(0, n, 2):
            d.qpos[root["robot"] : root["robot"] + 3] = z["root_p"][t, c]
            d.qpos[root["robot"] + 3 : root["robot"] + 7] = z["root_quat"][t, c]
            d.qpos[qadr["robot"]] = z["q"][t, c]
            if ghost:
                f = int(z["f"][t, c])
                d.qpos[root["ghost"] : root["ghost"] + 3] = z["ref_root_p"][f]
                d.qpos[root["ghost"]] = z["root_p"][t, c][0]  # anchored at the robot's xy, as the tracking reward
                d.qpos[root["ghost"] + 1] = z["root_p"][t, c][1]
                quat = np.zeros(4)
                mujoco.mju_mat2Quat(quat, z["ref_root_R"][f].reshape(-1).astype(np.float64))
                d.qpos[root["ghost"] + 3 : root["ghost"] + 7] = quat
                d.qpos[qadr["ghost"]] = z["ref_q"][f]
            mujoco.mj_kinematics(m, d)
            mujoco.mj_camlight(m, d)  # light positions / directions (mj_kinematics alone leaves them at zero: unlit)
            cam.lookat[:] = [z["root_p"][t, c][0], z["root_p"][t, c][1], ROBOT_HEIGHT * LOOKAT_HEIGHT]  # follows the position
            mujoco.mjv_updateScene(m, d, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
            mujoco.mjr_render(vp, scn, ctx)
            mujoco.mjr_readPixels(img, None, vp, ctx)
            frame = Image.fromarray(np.flipud(img))
            draw = ImageDraw.Draw(frame)
            draw.rectangle([0, 0, width, int(height * 0.09)], fill=(0, 0, 0))
            draw.text((12, 4), f"{exp}@{it}  {clip}  {t * 0.02:5.1f} s  {ret}", font=font, fill=(255, 255, 255))
            writer.send(np.asarray(frame.convert("RGB")))
        writer.close()
        print(f"VIDEO {out} {(n - 1) * 0.02:.1f} s {width}x{height}", flush=True)


if __name__ == "__main__":
    main()
