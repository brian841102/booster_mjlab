"""Pino v1 (XDream X1) constants.

Pino v1 is a wheeled humanoid: each foot carries two in-line wheels like a skate -- a driven rear wheel
(``*_rear_wheel_joint``, velocity control) and a free-spinning front wheel (``*_front_wheel_joint``, no actuator).
The other 21 joints (legs incl. ankle pitch, torso yaw, arms) are position controlled. As for nemov3, pinov1.xml
carries the actuators itself (gains from the XDream X1 IsaacLab config, see the header of pinov1.xml) and this
module wraps them with ``XmlActuatorCfg``.
"""

from __future__ import annotations

import functools
import re
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityCfg
from mjlab.entity.entity import EntityArticulationInfoCfg

from booster_mjlab import BOOSTER_MJLAB_SRC_PATH

if TYPE_CHECKING:
    import torch

##
# MJCF and assets.
##

PINOV1_XML: Path = BOOSTER_MJLAB_SRC_PATH / "robots" / "pinov1" / "xmls" / "pinov1.xml"
assert PINOV1_XML.exists()


def get_spec(pd_gains: tuple[tuple[str, float, float], ...] = (), wheel_kv: float | None = None) -> mujoco.MjSpec:
    """Load pinov1.xml. ``pd_gains`` = (joint regex, kp, kd) entries overriding the XML gains of the matching position
    actuators; ``wheel_kv`` overrides the rear wheels' velocity gain (as nemov3_constants.get_spec)."""
    spec = mujoco.MjSpec.from_file(str(PINOV1_XML))
    for pattern, kp, kd in pd_gains:
        matched = [a for a in spec.actuators if a.name.endswith("_act") and re.fullmatch(pattern, a.target)]
        assert matched, f"pd_gains: no position actuator for '{pattern}'"
        for actuator in matched:
            # <position>: force = kp * (target - q) - kd * qdot, i.e. gainprm[0] = kp, biasprm[1:3] = (-kp, -kd).
            actuator.gainprm[0] = kp
            actuator.biasprm[1] = -kp
            actuator.biasprm[2] = -kd
    if wheel_kv is not None:
        for actuator in [a for a in spec.actuators if a.name.endswith("_vel")]:
            # <velocity>: force = kv * (target - qdot), i.e. gainprm[0] = kv, biasprm[2] = -kv.
            actuator.gainprm[0] = wheel_kv
            actuator.biasprm[2] = -wheel_kv
    return spec


# nemov3 Exp 33's gains (nemov3.xml: every position actuator kp 50 / kv 0.5, wheels kv 0.5), for the PD comparison
# (Exp 18, user 2026-11-04: "把PD改成和nemov3一样做对照实验"); pinov1's own force limits stay.
NEMOV3_EXP33_PD_GAINS: tuple[tuple[str, float, float], ...] = ((r".*", 50.0, 0.5),)
NEMOV3_EXP33_WHEEL_KV = 0.5

# Deployment PD (user 2026-11-05: "肩 pitch / roll 肩 yaw、肘、腕 统一换成18/0.8"): nemov3 Exp 33's gains with every arm joint
# at the nemov3 hardware's shoulder gains (kp 18 / kd 0.8). Later entries override earlier ones.
PINOV1_DEPLOY_PD_GAINS: tuple[tuple[str, float, float], ...] = NEMOV3_EXP33_PD_GAINS + (
    (r".*_(shoulder_pitch|shoulder_roll|shoulder_yaw|elbow|wrist)_joint", 18.0, 0.8),
)

PINOV1_ARM_JOINT_PATTERN = r".*_(shoulder_pitch|shoulder_roll|shoulder_yaw|elbow|wrist)_joint"

# Rear-wheel (ENCOS EC-A5016-P2-17H) no-load top speed at the robot's 48 V bus: 34.8 rad/s at the datasheet's 72 V,
# scaled by 48 / 72 (docs/pinov1/pinov1_spec.md). Exp 32 reached p99 24.6 rad/s at 0.9 m/s (26.4 turning).
PINOV1_WHEEL_SPEED_LIMIT_48V = 23.0


##
# Joint order.
##

# XML joint declaration order (= motion .pkl ``dof_pos`` columns = ``robot.data.default_joint_pos``), wheels included.
PINOV1_JOINT_ORDER: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_rear_wheel_joint",
    "left_front_wheel_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_rear_wheel_joint",
    "right_front_wheel_joint",
    "torso_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_joint",
)

# Driven (velocity-controlled) rear wheels and free front wheels; everything else is position controlled.
PINOV1_WHEEL_JOINT_NAMES: tuple[str, ...] = ("left_rear_wheel_joint", "right_rear_wheel_joint")
PINOV1_FRONT_WHEEL_JOINT_NAMES: tuple[str, ...] = ("left_front_wheel_joint", "right_front_wheel_joint")
PINOV1_POSITION_JOINT_NAMES: tuple[str, ...] = tuple(
    n for n in PINOV1_JOINT_ORDER if n not in PINOV1_WHEEL_JOINT_NAMES + PINOV1_FRONT_WHEEL_JOINT_NAMES
)
# The 21 joints of the retargeting model (pino_joint21_nowheel_retarget.xml), in its order: the column order of
# the skating CSVs in assets/amp_sources/pinov1_skating.
PINOV1_RETARGET_JOINT_ORDER: tuple[str, ...] = PINOV1_POSITION_JOINT_NAMES

# Wheel bodies and outer radii (collision mesh extents). Both wheels of a foot touch the ground together: the front
# wheel sits 12.5 mm lower in the ankle frame and is 12.4 mm smaller.
PINOV1_WHEEL_BODY_RADII: dict[str, float] = {
    "left_rear_wheel_link": 0.0499,
    "left_front_wheel_link": 0.0375,
    "right_rear_wheel_link": 0.0499,
    "right_front_wheel_link": 0.0375,
}

##
# Keyframe config.
##

# Default pose from the XDream X1 IsaacLab config (user, 2026-11-03: start from this pose, "B"; switch to the
# reference's mean crouch, "A", if the legs cannot reach the skating stance -- the nemov3 lesson, see HOME_KEYFRAME in
# nemov3_constants.py). Root height 0.39 m puts the lowest wheel ~2 mm above the ground (forward kinematics: 0.388).
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0, 0, 0.39),
    joint_pos={
        ".*_hip_pitch_joint": -0.1,
        ".*_knee_joint": 0.2,
        ".*_ankle_pitch_joint": -0.13,
        ".*_shoulder_pitch_joint": 0.15,
        "left_shoulder_roll_joint": 0.12,
        "right_shoulder_roll_joint": -0.12,
        ".*_elbow_joint": 1.2,
    },
    joint_vel={".*": 0.0},
)

# Default pose "A" (user 2026-11-03: "先用B 后面有问题 再改成A"): the skating clip's mean crouch, as nemov3's HOME_KEYFRAME.
# With "B" the trained policies stood and skated around hip pitch -0.65 / -0.47, knee 0.86 / 0.78 -- the clip's crouch
# (mean -0.54 / 0.82) -- while every standing-pose term pulled them to B's -0.1 / 0.2, i.e. standing and skating were two
# different postures; nemov3 stands in its skating crouch (tools/pinov1_finetune.md, "关键差异"). Hip pitch, knee,
# shoulder pitch and elbow are the left/right-symmetric clip means; the ankle is the angle that puts both wheels of a
# foot on the ground at that hip / knee (the clip mean -0.09 would leave the rear wheel 2 cm up); the rest stays "B".
# Root height 0.365 m puts the wheels ~2 mm above the ground.
HOME_KEYFRAME_REFERENCE_MEAN = EntityCfg.InitialStateCfg(
    pos=(0, 0, 0.365),
    joint_pos={
        ".*_hip_pitch_joint": -0.54,
        ".*_knee_joint": 0.82,
        ".*_ankle_pitch_joint": -0.30,
        ".*_shoulder_pitch_joint": 0.23,
        "left_shoulder_roll_joint": 0.12,
        "right_shoulder_roll_joint": -0.12,
        ".*_elbow_joint": 0.97,
    },
    joint_vel={".*": 0.0},
)

# Default pose for the nemov3 clip set retargeted to pinov1 (assets/amp/pinov1_skating_straight; user 2026-11-04: train
# the nemov3 way, warm-up pose = the clips' mean, left/right mirrored so it is symmetric): the clip-equal mean of
# Skating01 / Skating02 / SkatingFast06, averaged with its mirror image (nemov3's HOME_KEYFRAME + SYMMETRIC_HIP_YAW_DEFAULT).
# Both wheels of a foot touch the ground in it (front 2.5 mm higher); root height 0.387 m leaves the wheels ~2 mm up.
HOME_KEYFRAME_NEMO_CLIP_MEAN = EntityCfg.InitialStateCfg(
    pos=(0, 0, 0.387),
    joint_pos={
        ".*_hip_pitch_joint": -0.20,
        "left_hip_roll_joint": 0.032,
        "right_hip_roll_joint": -0.032,
        "left_hip_yaw_joint": -0.078,
        "right_hip_yaw_joint": 0.078,
        ".*_knee_joint": 0.35,
        ".*_ankle_pitch_joint": -0.18,
        ".*_shoulder_pitch_joint": 0.088,
        "left_shoulder_roll_joint": 0.456,
        "right_shoulder_roll_joint": -0.456,
        "left_shoulder_yaw_joint": -0.10,
        "right_shoulder_yaw_joint": 0.10,
        ".*_elbow_joint": 1.17,
        "left_wrist_joint": 0.023,
        "right_wrist_joint": -0.023,
    },
    joint_vel={".*": 0.0},
)

##
# Left/right mirror.
##

# Every left/right joint pair has the same axis on both sides (pinov1.xml at the zero pose; the hip-pitch axes tilt
# by +-15 deg about x and map onto each other), so under the y -> -y mirror rotations about x / z (roll, yaw, torso
# yaw, wrist) flip sign and rotations about y (pitch, knee, ankle, elbow, wheels) keep it.
_MIRROR_SIGN_BY_KIND = {
    "hip_pitch_joint": 1.0, "hip_roll_joint": -1.0, "hip_yaw_joint": -1.0, "knee_joint": 1.0,
    "ankle_pitch_joint": 1.0, "wheel_joint": 1.0, "torso_joint": -1.0,
    "shoulder_pitch_joint": 1.0, "shoulder_roll_joint": -1.0, "shoulder_yaw_joint": -1.0,
    "elbow_joint": 1.0, "wrist_joint": -1.0,
}  # fmt: skip


def pinov1_mirror_partner(name: str) -> str:
    """Left/right counterpart of a pinov1 joint (the torso maps to itself)."""
    if name.startswith("left_"):
        return "right_" + name[len("left_") :]
    if name.startswith("right_"):
        return "left_" + name[len("right_") :]
    return name


def pinov1_mirror_sign(name: str) -> float:
    for kind, sign in _MIRROR_SIGN_BY_KIND.items():
        if name.endswith(kind):
            return sign
    raise KeyError(f"no mirror sign for joint {name}")


def pinov1_mirror_maps(joint_names) -> tuple[list[int], list[float]]:
    """``(index_map, sign_map)`` such that ``mirrored[i] = sign_map[i] * original[index_map[i]]``."""
    names = list(joint_names)
    return (
        [names.index(pinov1_mirror_partner(n)) for n in names],
        [pinov1_mirror_sign(n) for n in names],
    )


##
# Action scale.
##

# Same heuristic as nemov3 / K1: 0.25 * forcerange / kp per position joint -> 0.26 rad (legs, torso, ankles) and
# 0.20 rad (arms). Per-joint overrides for the skating stride belong in the task config (as for nemov3).
_ACTION_SCALE_FACTOR = 0.25

# Rear wheels: 1.3 m/s on a 0.0499 m wheel is 26 rad/s (velocity limit 34.77 rad/s in xdream.py).
PINOV1_WHEEL_ACTION_SCALE = 30.0


def _joint_names_in_actuator_order(spec: mujoco.MjSpec) -> list[str]:
    """Joint names in actuator declaration order (= ``Entity.actuator_names`` order; see nemov3_constants.py)."""
    model = spec.compile()
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, model.actuator_trnid[i, 0]) for i in range(model.nu)]


def _build_action_scale(spec: mujoco.MjSpec, factor: float = _ACTION_SCALE_FACTOR) -> dict[str, float]:
    model = spec.compile()
    out: dict[str, float] = {}
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name is None or not name.endswith("_act"):
            continue
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, model.actuator_trnid[i, 0])
        out[joint_name] = factor * float(model.actuator_forcerange[i, 1]) / float(model.actuator_gainprm[i, 0])
    return out


PINOV1_ACTION_SCALE: dict[str, float] = _build_action_scale(get_spec())

##
# Articulation.
##

# One XmlActuatorCfg group per actuator (same reason as nemov3: domain-randomization events index Entity.actuators
# by per-actuator id). The front wheels have no actuator.
PINOV1_ARTICULATION = EntityArticulationInfoCfg(
    actuators=tuple(XmlActuatorCfg(target_names_expr=(n,)) for n in _joint_names_in_actuator_order(get_spec())),
    soft_joint_pos_limit_factor=0.95,
)


# Actuator command delay groups, one per motor bus (user's colleague, 2026-11-06: left arm, right arm, waist, left leg,
# right leg; each rear wheel goes with its leg): one lag per group, drawn at reset and kept for the episode by
# mdp/motor_dynamics.py:sample_actuator_delay (Exp 48).
PINOV1_DELAY_GROUPS: tuple[str, ...] = (
    r"left_(shoulder_pitch|shoulder_roll|shoulder_yaw|elbow|wrist)_joint",  # left arm
    r"right_(shoulder_pitch|shoulder_roll|shoulder_yaw|elbow|wrist)_joint",  # right arm
    r"torso_joint",  # waist
    r"left_(hip_pitch|hip_roll|hip_yaw|knee|ankle_pitch|rear_wheel)_joint",  # left leg + rear wheel
    r"right_(hip_pitch|hip_roll|hip_yaw|knee|ankle_pitch|rear_wheel)_joint",  # right leg + rear wheel
)


def get_pinov1_robot_cfg(
    default_keyframe: EntityCfg.InitialStateCfg = HOME_KEYFRAME,
    pd_gains: tuple[tuple[str, float, float], ...] = (),
    wheel_kv: float | None = None,
    actuator_delay_steps: int = 0,
) -> EntityCfg:
    """A fresh Pino v1 robot configuration instance; ``pd_gains`` / ``wheel_kv`` override the XML gains (get_spec);
    ``actuator_delay_steps`` > 0 gives every actuator a command delay buffer of up to N physics steps whose lag the env
    sets at reset (PINOV1_DELAY_GROUPS; the buffer never resamples on its own, as nemov3)."""
    spec_fn = functools.partial(get_spec, tuple(pd_gains), wheel_kv) if (pd_gains or wheel_kv is not None) else get_spec
    articulation = PINOV1_ARTICULATION
    if actuator_delay_steps:
        delay = {"delay_min_lag": 0, "delay_max_lag": actuator_delay_steps, "delay_hold_prob": 1.0}
        articulation = EntityArticulationInfoCfg(
            actuators=tuple(XmlActuatorCfg(target_names_expr=(n,), **delay) for n in _joint_names_in_actuator_order(get_spec())),
            soft_joint_pos_limit_factor=PINOV1_ARTICULATION.soft_joint_pos_limit_factor,
        )
    return EntityCfg(init_state=default_keyframe, spec_fn=spec_fn, articulation=articulation)


def get_pinov1_default_joint_pos(keyframe: EntityCfg.InitialStateCfg = HOME_KEYFRAME) -> "torch.Tensor":
    """Default joint positions in PINOV1_JOINT_ORDER (regex keys resolved like mjlab does)."""
    import torch

    def value(name: str) -> float:
        return next((v for k, v in keyframe.joint_pos.items() if re.fullmatch(k, name)), 0.0)

    return torch.tensor([value(n) for n in PINOV1_JOINT_ORDER], dtype=torch.float32)


if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity

    robot = Entity(get_pinov1_robot_cfg())
    viewer.launch(robot.spec.compile())
