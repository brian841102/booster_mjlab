"""Browser playback of SuperMimic evaluation rollouts (``model_<it>_traj.npz`` written by ``evaluate-supermimic``).

Offline: the recorded simulator states are posed with MuJoCo FK, no policy or physics runs (for live inference use
``uv run play``). The simulated robot is drawn solid, the reference motion as a translucent blue ghost.

    uv run replay-supermimic Mjlab-SuperMimic-Flat-Booster-Nemov3 --run-dir logs/rsl_rl/nemov3_supermimic/latest
    uv run replay-supermimic Mjlab-SuperMimic-Flat-Booster-Nemov3 --traj <run>/model_10000_traj.npz

``--run-dir`` adds a checkpoint dropdown (latest + history, refreshed while training goes on).
"""

from __future__ import annotations

import glob
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Annotated

import mujoco
import numpy as np
import tyro
import viser

from mjlab import TYRO_FLAGS

DT = 0.02  # 50 Hz control


@dataclass(frozen=True)
class ReplayConfig:
    task: Annotated[str, tyro.conf.Positional]
    """SuperMimic task, for the robot model."""
    traj: str | None = None
    """One ``model_<it>_traj.npz``."""
    run_dir: str | None = None
    """Run directory: checkpoint dropdown over its ``model_*_traj.npz``."""
    port: int = 8080
    host: str = "0.0.0.0"


def visual_geoms(model: mujoco.MjModel) -> list[int]:
    """Mesh geoms to draw: the URDF visuals (non-colliding) if present, else every mesh geom."""
    mesh = [g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
    vis = [g for g in mesh if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0]
    return vis or mesh


def mesh_arrays(model: mujoco.MjModel, mesh_id: int):
    va, vn = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
    fa, fn = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
    return model.mesh_vert[va : va + vn].copy(), model.mesh_face[fa : fa + fn].copy()


class Robot:
    """One robot in the viser scene: a mesh handle per visual geom, posed from qpos via MuJoCo FK."""

    def __init__(self, server: viser.ViserServer, model: mujoco.MjModel, name: str, color, opacity: float | None):
        self.model, self.data = model, mujoco.MjData(model)
        self.geoms = visual_geoms(model)
        self.handles = []
        for g in self.geoms:
            v, f = mesh_arrays(model, model.geom_dataid[g])
            self.handles.append(
                server.scene.add_mesh_simple(f"/{name}/geom_{g}", vertices=v, faces=f, color=color, opacity=opacity)
            )

    def set_qpos(self, qpos: np.ndarray):
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self.data)
        q = np.empty(4)
        for g, h in zip(self.geoms, self.handles):
            mujoco.mju_mat2Quat(q, self.data.geom_xmat[g])
            h.position = self.data.geom_xpos[g].copy()
            h.wxyz = q.copy()

    def set_visible(self, flag: bool):
        for h in self.handles:
            h.visible = flag


def main():
    a = tyro.cli(ReplayConfig, config=TYRO_FLAGS)
    import mjlab.tasks  # noqa: F401  (registers the tasks)
    from mjlab.tasks.registry import load_env_cfg

    env_cfg = load_env_cfg(a.task, play=True)
    model = env_cfg.scene.entities["robot"].spec_fn().compile()  # the robot alone
    length_scale = env_cfg.supermimic.length_scale
    run_dir = os.path.realpath(a.run_dir) if a.run_dir else None

    def iteration(path: str) -> int:
        return int(re.fullmatch(r"model_(\d+)_traj\.npz", os.path.basename(path)).group(1))

    def ckpt_files():
        return sorted(glob.glob(os.path.join(run_dir, "model_*_traj.npz")), key=iteration) if run_dir else []

    def load_ckpt(traj_path: str):
        ev = traj_path.replace("_traj.npz", "_eval.json")
        return dict(np.load(traj_path)), (json.load(open(ev))["per_clip"] if os.path.exists(ev) else {})

    if run_dir:
        assert ckpt_files(), f"no model_*_traj.npz in {run_dir} (run evaluate-supermimic first)"
        d, per_clip = load_ckpt(ckpt_files()[-1])
    else:
        assert a.traj, "give --traj or --run-dir"
        d, per_clip = load_ckpt(a.traj)
    names = [str(n) for n in d["names"]]
    joint_names = [str(n) for n in d["joint_names"]]
    # q columns are in the robot's joint order at training time, matched to the model by name
    qadr = np.array([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in joint_names])
    assert model.nq == 7 + len(joint_names) and d["q"].shape[-1] == len(joint_names), (model.nq, d["q"].shape)

    server = viser.ViserServer(host=a.host, port=a.port)
    server.scene.set_up_direction("+z")
    world = server.scene.add_frame("/world", show_axes=False)
    grid = server.scene.add_grid("/world/ground", width=40.0, height=40.0, cell_size=0.5, section_size=2.0)
    robot = Robot(server, model, "world/robot", color=(110, 112, 122), opacity=None)
    ghost = Robot(server, model, "world/reference", color=(90, 165, 255), opacity=0.35)

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        k = length_scale  # robot size relative to G1
        client.camera.position = (3.0 * k, -3.0 * k, 1.6 * k)
        client.camera.look_at = (0.0, 0.0, 0.8 * k)

    def ckpt_label(path: str) -> str:
        ev = path.replace("_traj.npz", "_eval.json")
        ret = f" · {json.load(open(ev))['norm_return']:.3f}" if os.path.exists(ev) else ""
        return f"it {iteration(path)}{ret}"

    def ckpt_options() -> dict[str, str]:  # label -> traj path, "latest" first, then history newest to oldest
        files = ckpt_files()[::-1]
        return {f"latest ({ckpt_label(files[0])})": files[0], **{ckpt_label(f): f for f in files}}

    with server.gui.add_folder("Playback"):
        if run_dir:
            opts = ckpt_options()
            ckpt_dd = server.gui.add_dropdown("Checkpoint", options=tuple(opts), initial_value=next(iter(opts)))
        clip_dd = server.gui.add_dropdown("Clip", options=tuple(names), initial_value=names[0])
        playing = server.gui.add_checkbox("Playing", initial_value=True)
        speed = server.gui.add_slider("Speed", min=0.1, max=3.0, step=0.1, initial_value=1.0)
        t_slider = server.gui.add_slider("Time [s]", min=0.0, max=1.0, step=DT, initial_value=0.0)
        loop = server.gui.add_checkbox("Loop", initial_value=True)
        prev_btn = server.gui.add_button("Previous clip")
        next_btn = server.gui.add_button("Next clip")
    with server.gui.add_folder("Display"):
        show_ghost = server.gui.add_checkbox("Show reference (blue ghost)", initial_value=True)
        anchor = server.gui.add_checkbox("Anchor reference to robot xy", initial_value=True)
        follow = server.gui.add_checkbox("Keep robot centered", initial_value=True)
        show_trail = server.gui.add_checkbox("Show robot ground path (orange)", initial_value=True)
        show_ref_trail = server.gui.add_checkbox("Show reference ground path (blue)", initial_value=False)
        trail_s = server.gui.add_slider("Path length [s]", min=0.5, max=10.0, step=0.5, initial_value=5.0)
    info = server.gui.add_markdown("")

    T = d["root_p"].shape[0]
    state = {"c": 0, "steps": 1, "fell_at": None}

    def select(name: str):
        c = names.index(name)
        alive = d["alive"][:, c]
        n_alive = int(alive.sum())
        fell = n_alive < T and n_alive < int(d["clip_len"][c]) - 1
        state.update(c=c, steps=max(n_alive, 1), fell_at=n_alive if fell else None)
        t_slider.max = round((state["steps"] - 1) * DT, 2)
        t_slider.value = 0.0
        ret = f"{per_clip[name]:.3f}" if name in per_clip else "n/a"
        status = "**terminated (fell)**" if fell else "completed"
        info.content = (
            f"**{name}**  \nlength {state['steps'] * DT:.1f} s · {status} · normalized return {ret}  \n"
            f"grey: simulated policy (mjlab, MuJoCo-Warp) · blue ghost: reference motion"
        )

    @clip_dd.on_update
    def _(_):
        select(clip_dd.value)

    @prev_btn.on_click
    def _(_):
        clip_dd.value = names[(names.index(clip_dd.value) - 1) % len(names)]

    @next_btn.on_click
    def _(_):
        clip_dd.value = names[(names.index(clip_dd.value) + 1) % len(names)]

    if run_dir:
        shown = {"opts": opts, "path": next(iter(opts.values())), "latest": True}

        def load_selected():
            nonlocal d, per_clip
            d, per_clip = load_ckpt(shown["path"])
            select(clip_dd.value)

        @ckpt_dd.on_update
        def _(_):
            shown["latest"] = ckpt_dd.value.startswith("latest")
            shown["path"] = shown["opts"][ckpt_dd.value]
            load_selected()

    @show_ghost.on_update
    def _(_):
        ghost.set_visible(show_ghost.value)

    select(names[0])
    print(f"viser replay of {len(names)} clips: open http://localhost:{a.port}", flush=True)
    qpos_r, qpos_g = model.qpos0.copy(), model.qpos0.copy()
    rquat = np.empty(4)
    last = time.time()
    t_play, t_shown = 0.0, 0.0  # playback time (float): rounding it to the 50 Hz grid every loop made speeds < ~1.3 play alike
    trails, trail_t = {}, 0.0

    def path(name, xy, color):
        """Ground polyline through xy (n, 2) under /world (it moves with the world when the robot is kept centred)."""
        if len(xy) < 2:
            if name in trails:
                trails.pop(name).remove()
            return
        pts = np.zeros((len(xy) - 1, 2, 3), np.float32)
        pts[:, 0, :2], pts[:, 1, :2], pts[..., 2] = xy[:-1], xy[1:], 0.005
        trails[name] = server.scene.add_line_segments(f"/world/{name}", points=pts, colors=color, line_width=3.0)
    next_scan = 0.0
    while True:
        now = time.time()
        if run_dir and now > next_scan:  # new checkpoints appear while training goes on
            next_scan = now + 10.0
            opts_new = ckpt_options()
            if list(opts_new) != list(shown["opts"]):
                shown["opts"] = opts_new
                keep = next(iter(opts_new)) if shown["latest"] else next((k for k, v in opts_new.items() if v == shown["path"] and not k.startswith("latest")), None)
                ckpt_dd.options = tuple(opts_new)
                ckpt_dd.value = keep or next(iter(opts_new))
        if abs(t_slider.value - t_shown) > 1e-6:  # the user dragged the slider, or a clip was selected
            t_play = t_slider.value
        if playing.value:
            t_play += (now - last) * speed.value
            if t_play > t_slider.max:
                t_play = 0.0 if loop.value else t_slider.max
            t_shown = round(min(t_play, t_slider.max) / DT) * DT
            t_slider.value = t_shown
        last = now
        c = state["c"]
        tt = min(int(round(t_slider.value / DT)), state["steps"] - 1)
        f = int(d["f"][tt, c])
        qpos_r[0:3], qpos_r[3:7] = d["root_p"][tt, c], d["root_quat"][tt, c]
        qpos_r[qadr] = d["q"][tt, c]
        qpos_g[0:3] = d["ref_root_p"][f]
        if anchor.value:  # as the tracking reward does: reference anchored at the robot's xy
            qpos_g[0:2] = qpos_r[0:2]
        mujoco.mju_mat2Quat(rquat, d["ref_root_R"][f].reshape(-1).astype(np.float64))
        qpos_g[3:7] = rquat
        qpos_g[qadr] = d["ref_q"][f]
        if now - trail_t > 0.1:  # ground paths of the last trail_s seconds, 10 Hz
            trail_t = now
            a0 = max(tt - int(trail_s.value / DT), 0)
            if show_trail.value:
                path("trail_robot", d["root_p"][a0 : tt + 1, c, :2], (255, 120, 40))
            elif "trail_robot" in trails:
                trails.pop("trail_robot").remove()
            if show_ref_trail.value:
                fs = d["f"][a0 : tt + 1, c].astype(int)
                ref_xy = d["ref_root_p"][fs, :2]
                if anchor.value:  # through the robot, like the anchored ghost
                    ref_xy = ref_xy - ref_xy[-1] + qpos_r[0:2]
                path("trail_reference", ref_xy, (90, 165, 255))
            elif "trail_reference" in trails:
                trails.pop("trail_reference").remove()
        with server.atomic():
            robot.set_qpos(qpos_r)
            if show_ghost.value:
                ghost.set_qpos(qpos_g)
            # shift the world instead of the camera, so mouse orbit/zoom stays with the user;
            # the grid moves in whole sections under the robot, so it still shows the motion
            xy = qpos_r[0:2] if follow.value else np.zeros(2)
            world.position = (-xy[0], -xy[1], 0.0)
            grid.position = (round(xy[0] / 2.0) * 2.0, round(xy[1] / 2.0) * 2.0, 0.0)
        time.sleep(1.0 / 50)


if __name__ == "__main__":
    main()
