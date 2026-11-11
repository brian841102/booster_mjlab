"""Show Pino v1's default poses side by side in the MuJoCo viewer (static, no physics).

usage: python tools/pinov1_tools/show_default_pose.py [names...]
names: mean (HOME_KEYFRAME_NEMO_CLIP_MEAN, the default of Exp 17+), b (HOME_KEYFRAME, the user's pose, Exp 1-16),
ref (HOME_KEYFRAME_REFERENCE_MEAN). Default: mean b. Robots stand 0.8 m apart along y, left to right in the given order;
the viewer only draws the poses (time is not advanced), so they never fall.
"""

import re
import sys

import mujoco
import mujoco.viewer

from booster_mjlab.robots.pinov1.pinov1_constants import (
    HOME_KEYFRAME,
    HOME_KEYFRAME_NEMO_CLIP_MEAN,
    HOME_KEYFRAME_REFERENCE_MEAN,
    get_spec,
)

POSES = {"mean": HOME_KEYFRAME_NEMO_CLIP_MEAN, "b": HOME_KEYFRAME, "ref": HOME_KEYFRAME_REFERENCE_MEAN}
SPACING = 0.8


def main() -> None:
    names = sys.argv[1:] or ["mean", "b"]
    world = mujoco.MjSpec()
    world.worldbody.add_light(pos=(0, 0, 3), dir=(0, 0, -1), diffuse=(0.8, 0.8, 0.8))
    world.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=(5, 5, 0.1), rgba=(0.3, 0.4, 0.5, 1))
    for k, name in enumerate(names):
        frame = world.worldbody.add_frame(pos=(0, SPACING * (k - (len(names) - 1) / 2), 0))
        world.attach(get_spec(), prefix=f"{name}/", frame=frame)
    model = world.compile()
    data = mujoco.MjData(model)
    for k, name in enumerate(names):
        pose = POSES[name]
        for j in range(model.njnt):
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if not jname.startswith(f"{name}/"):
                continue
            adr = model.jnt_qposadr[j]
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                data.qpos[adr : adr + 3] = (pose.pos[0], pose.pos[1] + SPACING * (k - (len(names) - 1) / 2), pose.pos[2])
                data.qpos[adr + 3 : adr + 7] = (1, 0, 0, 0)
                continue
            local = jname[len(name) + 1 :]
            for pattern, value in pose.joint_pos.items():
                if re.fullmatch(pattern, local):
                    data.qpos[adr] = value
                    break
        print(f"{name}: y = {SPACING * (k - (len(names) - 1) / 2):+.1f} m, base height {pose.pos[2]:.3f} m")
    mujoco.mj_forward(model, data)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_forward(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
