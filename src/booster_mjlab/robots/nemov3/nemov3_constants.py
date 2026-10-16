"""Nemov3 (wheeled-legged) constants.

Nemov3 is a wheeled-legged humanoid: the legs end in driven wheels
(``left_wheel_joint`` / ``right_wheel_joint``) instead of feet, so it has no
biped swing/stance gait. The MJCF's own ``<actuator>`` block already carries
working PD gains for every joint (``<position>`` actuators for everything
except the wheels, which use ``<velocity>`` actuators) straight from the
source project, so this module wraps those XML-native actuators with
``XmlActuatorCfg`` instead of re-deriving per-motor torque/speed curves the
way ``booster_k1/k1_constants.py`` does for K1's own motors.
"""

from __future__ import annotations

import dataclasses
import functools
import re
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityCfg
from mjlab.entity.entity import EntityArticulationInfoCfg

from booster_mjlab import BOOSTER_MJLAB_SRC_PATH

if TYPE_CHECKING:
    import torch

##
# MJCF and assets.
##

NEMOV3_XML: Path = (
    BOOSTER_MJLAB_SRC_PATH / "robots" / "nemov3" / "xmls" / "nemov3.xml"
)
assert NEMOV3_XML.exists()


def get_spec(
    fixed_joints: tuple[str, ...] = (), pd_gains: tuple[tuple[str, float, float], ...] = ()
) -> mujoco.MjSpec:
    """Load nemov3.xml. ``fixed_joints`` are welded: the joint and its actuator are removed and the child body is
    fixed at the joint's HOME_KEYFRAME angle (see NEMOV3_FIXED_HEAD_WAIST_JOINTS). ``pd_gains`` = (joint regex, kp,
    kd) entries that override the XML gains of the matching joints' position actuators (see NEMOV3_DEPLOY_PD_GAINS)."""
    spec = mujoco.MjSpec.from_file(str(NEMOV3_XML))
    for geom in spec.geoms:
        if geom.name and "_collision" in geom.name.lower():
            geom.group = 3
    for name in fixed_joints:
        _weld_joint(spec, name, HOME_KEYFRAME.joint_pos.get(name, 0.0))
    for pattern, kp, kd in pd_gains:
        matched = [a for a in spec.actuators if re.fullmatch(pattern, a.target)]
        assert matched, f"pd_gains: no actuator for '{pattern}'"
        for actuator in matched:
            # <position>: force = kp * (target - q) - kd * qdot, i.e. gainprm[0] = kp, biasprm[1:3] = (-kp, -kd).
            assert actuator.biastype == mujoco.mjtBias.mjBIAS_AFFINE, f"{actuator.name} is not a position actuator"
            actuator.gainprm[0] = kp
            actuator.biasprm[1] = -kp
            actuator.biasprm[2] = -kd
    return spec


def _weld_joint(spec: mujoco.MjSpec, name: str, q: float) -> None:
    """Remove joint ``name`` (and the actuators driving it), baking the joint value ``q`` into its body's frame."""
    joint = spec.joint(name)
    body = joint.parent
    if q != 0.0:
        axis = np.asarray(joint.axis, dtype=np.float64) / np.linalg.norm(joint.axis)
        body_quat = np.asarray(body.quat, dtype=np.float64)
        if joint.type == mujoco.mjtJoint.mjJNT_HINGE:
            # Rotation by q about the joint axis through the joint anchor (body frame).
            dq = np.zeros(4)
            mujoco.mju_axisAngle2Quat(dq, axis, q)
            anchor = np.asarray(joint.pos, dtype=np.float64)
            rotated = np.zeros(3)
            mujoco.mju_rotVecQuat(rotated, anchor, dq)
            local_shift = anchor - rotated
            new_quat = np.zeros(4)
            mujoco.mju_mulQuat(new_quat, body_quat, dq)
            body.quat = new_quat
        elif joint.type == mujoco.mjtJoint.mjJNT_SLIDE:
            local_shift = axis * q
        else:
            raise ValueError(f"Cannot weld joint '{name}' of type {joint.type}")
        shift = np.zeros(3)
        mujoco.mju_rotVecQuat(shift, local_shift, body_quat)
        body.pos = np.asarray(body.pos, dtype=np.float64) + shift
    for actuator in [a for a in spec.actuators if a.target == name]:
        spec.delete(actuator)
    spec.delete(joint)


##
# Joint order.
##

# Order matches the XML joint declaration order (= the order used in motion
# .pkl `dof_pos` arrays and in `robot.data.default_joint_pos` at runtime).
# Includes the wheel joints; callers that need to exclude them (e.g. the AMP
# discriminator feature set) should filter with a regex that skips
# `.*_wheel_joint`.
NEMOV3_JOINT_ORDER: tuple[str, ...] = (
    "joint_pelvisYaw",
    "joint_pelvisPitch",
    "joint_pelvisRoll",
    "joint_rShoulderPitch",
    "joint_rShoulderRoll",
    "joint_rShoulderYaw",
    "joint_rElbow",
    "joint_rWristRoll",
    "joint_rWristPitch",
    "joint_lShoulderPitch",
    "joint_lShoulderRoll",
    "joint_lShoulderYaw",
    "joint_lElbow",
    "joint_lWristRoll",
    "joint_lWristPitch",
    "head_roll_joint",
    "head_pitch_joint",
    "head_yaw_joint",
    "joint_rLinearOut",
    "joint_rLinear",
    "joint_lLinearOut",
    "joint_lLinear",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_calf_joint",
    "right_wheel_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_calf_joint",
    "left_wheel_joint",
)

# The source skating NPZs store joint angles in the original IsaacLab
# articulation's sign convention; two joints have their axis flipped relative
# to nemov3.xml. Multiply source values by these before using them as MJCF
# joint positions (mirrors NEMOV3_MUJOCO_JOINT_SIGNS in
# assets/amp_sources/nemov3_skating/data/replay_npz_mujoco.py, the reference
# replay for this data). Joints not listed keep their sign.
NEMOV3_NPZ_TO_MJCF_JOINT_SIGNS: dict[str, float] = {
    "joint_lShoulderRoll": -1.0,
    "joint_rShoulderRoll": -1.0,
}

# Actuators driving the wheels are `<velocity>` type in the XML (continuous
# rolling), everything else is `<position>`. `find_joints_by_actuator_names`
# (used by both `JointPositionActionCfg.actuator_names` and
# `XmlActuatorCfg.target_names_expr`) matches against the *joint* names each
# actuator drives, not the XML actuator element's own name.
NEMOV3_WHEEL_JOINT_NAMES: tuple[str, ...] = ("left_wheel_joint", "right_wheel_joint")
NEMOV3_POSITION_JOINT_NAMES: tuple[str, ...] = tuple(
    name for name in NEMOV3_JOINT_ORDER if name not in NEMOV3_WHEEL_JOINT_NAMES
)

# `SceneEntityCfg(actuator_names=...)` (used by e.g. the `pd_gains` domain
# randomization event) resolves against `Entity.actuator_names`, which -
# unlike `find_joints_by_actuator_names` above - is the literal XML
# `<actuator name=...>` string, not the joint it drives. All position
# actuators in nemov3.xml are named `<joint>_act`; the wheels are
# `<joint>_vel`.
NEMOV3_POSITION_ACTUATOR_NAME_EXPR: tuple[str, ...] = (r".*_act$",)


##
# Keyframe config.
##

# The reference skating clips are performed in a crouched stance, not standing
# upright: averaged over the three training clips (equal weight per clip, since
# Fast06 has ~14x the frames of Skating01/02) the legs sit at hip pitch ~-0.26,
# calf ~+0.43, hip roll ~+0.08, hip yaw ~-0.06/-0.09 rad. The AMP discriminator
# scores joint positions *relative to this default*, so an upright default made
# the policy's "resting" features differ from the expert's by a large static
# offset (~0.7 rad on the calf) that alone let the discriminator separate them
# ~100% of the time -- and that offset was far outside what exploration noise
# could reach given the action scale. Pitch/calf/roll are left-right averaged
# to keep the default symmetric; yaw keeps the per-side means (its axes are
# mirrored, and the data is mildly asymmetric). Root height is the value that
# puts the wheel-ground sites exactly on z=0 for this pose (forward
# kinematics). Values are in MJCF sign convention (source NPZ values with
# NEMOV3_NPZ_TO_MJCF_JOINT_SIGNS applied): shoulder roll +1.22 = arms down at
# the sides (the unflipped -1.22 put the arms up in the air).
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0, 0, 0.50),
    joint_pos={
        "left_hip_pitch_joint": -0.26,
        "right_hip_pitch_joint": -0.26,
        "left_calf_joint": 0.43,
        "right_calf_joint": 0.43,
        "left_hip_roll_joint": 0.08,
        "right_hip_roll_joint": 0.08,
        "left_hip_yaw_joint": -0.09,
        "right_hip_yaw_joint": -0.06,
        "joint_pelvisPitch": 0.01,
        "joint_lShoulderRoll": 1.22,
        "joint_rShoulderRoll": 1.22,
    },
    joint_vel={".*": 0.0},
)


##
# Left/right mirror (used by the AMP motion mirror augmentation and the PPO
# symmetry augmentation).
##

# Sign a joint angle takes under a left/right (y -> -y) mirror; a left joint maps to
# the same-kind right joint with this sign. Derived from the joint axes of nemov3.xml
# at the zero pose (hinge axes are pseudovectors, sliders polar vectors; every joint pair
# aligned to |cos| = 1.00): rotations about z / x flip (yaw, roll of the pelvis / head /
# shoulder-yaw / wrist-roll), rotations about y do not, and the hip roll / hip yaw of the
# two legs have mirrored axes so both stay positive.
_MIRROR_SIGN_BY_KIND = {
    "joint_pelvisYaw": -1.0, "joint_pelvisPitch": 1.0, "joint_pelvisRoll": -1.0,
    "ShoulderPitch": 1.0, "ShoulderRoll": 1.0, "ShoulderYaw": -1.0, "Elbow": 1.0,
    "WristRoll": -1.0, "WristPitch": 1.0,
    "head_roll_joint": -1.0, "head_pitch_joint": 1.0, "head_yaw_joint": -1.0,
    "LinearOut": -1.0, "Linear": 1.0,
    "hip_pitch_joint": 1.0, "hip_roll_joint": 1.0, "hip_yaw_joint": 1.0,
    "calf_joint": 1.0, "wheel_joint": 1.0,
}  # fmt: skip


def nemov3_mirror_partner(name: str) -> str:
    """Left/right counterpart of a nemov3 joint (central joints map to themselves)."""
    for a, b in (("right_", "left_"), ("_rS", "_lS"), ("_rE", "_lE"), ("_rW", "_lW"), ("_rL", "_lL")):
        if a in name:
            return name.replace(a, b)
        if b in name:
            return name.replace(b, a)
    return name


def nemov3_mirror_sign(name: str) -> float:
    for kind, sign in _MIRROR_SIGN_BY_KIND.items():
        if name == kind or name.endswith(kind):
            return sign
    raise KeyError(f"no mirror sign for joint {name}")


def nemov3_mirror_maps(joint_names) -> tuple[list[int], list[float]]:
    """``(index_map, sign_map)`` such that ``mirrored[i] = sign_map[i] * original[index_map[i]]``."""
    names = list(joint_names)
    return (
        [names.index(nemov3_mirror_partner(n)) for n in names],
        [nemov3_mirror_sign(n) for n in names],
    )


##
# Action scale.
##

# Same heuristic as K1 (`booster_k1/k1_constants.py::_ACTION_SCALE_FACTOR`):
# 0.25 * forcerange / kp per position-controlled joint, read directly from
# the XML's own actuator gains rather than re-derived motor curves. This gives
# nemov3 only ~0.09-0.1 rad/unit (K1's motors are far stronger, so its hip
# pitch is 0.212, knee 0.35), which made the reference skating stride look
# unreachable by exploration (|a| p95 ~7-9 units vs. noise ~0.5). On its own,
# enlarging it did NOT help, and a global factor of 1.0 made task performance
# worse (Exp 5/6 in tools/nemov3_finetune.md), so the global factor stays 0.25.
# Hip pitch / calf (and, for the arms variant, shoulder pitch / elbow) get a
# per-joint override in tasks/velocity/config/nemov3/env_cfgs.py, which did
# turn out to be needed for the skating gait once the discriminator setup was
# fixed (Exp 11+).
_ACTION_SCALE_FACTOR = 0.25

# Target forward speed range we're training for (0-1.3 m/s straight line).
# Wheel radius measured from the wheel mesh AABB (~0.075 m outer radius), so
# 1.3 m/s needs ~17.3 rad/s of wheel spin; scale gives headroom above that.
NEMOV3_WHEEL_RADIUS_M = 0.075
NEMOV3_WHEEL_ACTION_SCALE = 20.0


def _joint_names_in_actuator_order(spec: mujoco.MjSpec) -> list[str]:
    """Joint names in the compiled model's *actuator* declaration order.

    This is not the same as ``NEMOV3_JOINT_ORDER`` (worldbody/joint
    declaration order) -- the XML's ``<actuator>`` block lists actuators in
    its own order. `Entity.actuator_names` follows this order, so
    `NEMOV3_ARTICULATION` below must declare its per-joint groups in the same
    order too, or actuator-id-based lookups (e.g. domain randomization
    events matching a subset of actuators) silently pair the wrong group
    with the wrong resolved id.
    """
    model = spec.compile()
    names = []
    for i in range(model.nu):
        joint_id = model.actuator_trnid[i, 0]
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        assert joint_name is not None
        names.append(joint_name)
    return names


def _build_action_scale(spec: mujoco.MjSpec, factor: float = _ACTION_SCALE_FACTOR) -> dict[str, float]:
    """Build joint-name -> scale dict from the XML's own position actuators."""
    model = spec.compile()
    out: dict[str, float] = {}
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name is None or not name.endswith("_act"):
            continue
        joint_id = model.actuator_trnid[i, 0]
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        kp = float(model.actuator_gainprm[i, 0])
        force_limit = float(model.actuator_forcerange[i, 1])
        if joint_name is not None and kp > 0:
            out[joint_name] = factor * force_limit / kp
    return out


NEMOV3_ACTION_SCALE: dict[str, float] = _build_action_scale(get_spec())


##
# Articulation.
##

# mjlab only recognizes actuators declared through `EntityArticulationInfoCfg`
# (raw XML `<actuator>` elements aren't picked up on their own -- unlike K1's
# xml, which has no `<actuator>` block at all and relies entirely on
# `MotorPositionActuatorCfg`). `XmlActuatorCfg` wraps the actuators already
# authored in nemov3.xml (PD gains, force limits) without re-declaring them;
# it auto-detects `<position>` vs `<velocity>` per group.
#
# One group per joint (not one blanket group for all positions + one for
# both wheels): `Entity.actuators` is indexed per *declared group*, but
# domain-randomization events like `pd_gains` that only match a subset of
# actuators (e.g. "positions but not wheels") resolve `actuator_ids` in the
# flat per-actuator-name space and index `Entity.actuators` with them
# directly -- only safe when every actuator has its own group (a *full*
# wildcard match, like K1's blanket ".*", takes a different, group-count-
# agnostic code path, which is why K1 doesn't need this).
NEMOV3_ARTICULATION = EntityArticulationInfoCfg(
    actuators=tuple(
        XmlActuatorCfg(target_names_expr=(joint_name,))
        for joint_name in _joint_names_in_actuator_order(get_spec())
    ),
    soft_joint_pos_limit_factor=0.9,
)


##
# Fixed (welded) joints.
##

# Joints the current hardware does not actuate yet (user, 2026-09-29: head and waist mechanisms not designed): head
# roll / pitch / yaw, the waist (pelvis yaw / pitch / roll) and its two push-rod linkages (LinearOut hinge + Linear
# slide per side, hanging from link_pelvisYaw), and the distal wrist joint (wrist pitch; wrist roll stays). Welding them
# in the model (get_spec(fixed_joints)) makes them rigid like the locked hardware -- holding them with their PD
# actuators instead would leave the waist compliant (kp 50 Nm/rad, 20 Nm). Tasks pick the set per robot build, so the
# joints come back by leaving them out of the set.
NEMOV3_FIXED_HEAD_WAIST_JOINTS: tuple[str, ...] = (
    "head_roll_joint",
    "head_pitch_joint",
    "head_yaw_joint",
    "joint_pelvisYaw",
    "joint_pelvisPitch",
    "joint_pelvisRoll",
    "joint_rLinearOut",
    "joint_rLinear",
    "joint_lLinearOut",
    "joint_lLinear",
    "joint_rWristPitch",
    "joint_lWristPitch",
)


def nemov3_active_joint_names(fixed_joints: tuple[str, ...] = ()) -> tuple[str, ...]:
    """NEMOV3_JOINT_ORDER without ``fixed_joints`` (= the robot's joint order when they are welded)."""
    return tuple(name for name in NEMOV3_JOINT_ORDER if name not in fixed_joints)


def drop_fixed_head_waist_joints(motion_file):
    """``dataset_transform`` for a robot built with NEMOV3_FIXED_HEAD_WAIST_JOINTS: drops their columns from the clips'
    ``dof_pos`` (NEMOV3_JOINT_ORDER, 32 columns) so the clips match the robot's joints. Runs after the augmentations
    (the mirror needs all 32 columns)."""
    keep = [NEMOV3_JOINT_ORDER.index(name) for name in nemov3_active_joint_names(NEMOV3_FIXED_HEAD_WAIST_JOINTS)]
    return dataclasses.replace(motion_file, dof_pos=np.asarray(motion_file.dof_pos)[:, keep])


##
# Hardware actuation.
##

# The PD gains the hardware runs (user, 2026-10-12: "上身电机训练PD 过大 之后请按照实际部署PD 来训练"); the XML has kp 50 /
# kd 0.5 on every position joint. The legs (50 / 0.5) and the wheels (velocity control, kd 0.5) already match.
# (joint regex, kp, kd).
NEMOV3_DEPLOY_PD_GAINS: tuple[tuple[str, float, float], ...] = (
    (r"joint_[rl]Shoulder(Pitch|Roll)", 18.0, 0.8),
    (r"joint_[rl](ShoulderYaw|Elbow|WristRoll)", 1.5, 0.1),
)

# Command latency of the hardware's motor bus (flamingo_rl_train's DelayedPD actuators: 0-4 physics steps, one lag per
# actuator group, drawn at reset and kept for the episode). The buffers never resample on their own (hold_prob 1); the
# env's reset event mdp/motor_dynamics.py:sample_actuator_delay sets the lags per group.
NEMOV3_DELAY_GROUPS: tuple[str, ...] = (
    r"joint_[rl](Shoulder.*|Elbow|Wrist.*)",  # arms
    r".*_(hip_.*|calf)_joint",  # legs
    r".*_wheel_joint",  # wheels
    r"joint_pelvis.*",  # waist (welded on the current hardware)
    r"head_.*_joint|joint_[rl]Linear.*",  # head, waist push rods (welded)
)


def get_nemov3_robot_cfg(
    default_keyframe: EntityCfg.InitialStateCfg = HOME_KEYFRAME,
    fixed_joints: tuple[str, ...] = (),
    pd_gains: tuple[tuple[str, float, float], ...] = (),
    actuator_delay_steps: int = 0,
) -> EntityCfg:
    """Get a fresh Nemov3 robot configuration instance.

    Returns a new EntityCfg instance each time to avoid mutation issues when
    the config is shared across multiple places. ``fixed_joints`` are welded and ``pd_gains`` override the XML gains
    (see get_spec); ``actuator_delay_steps`` > 0 gives every actuator a command delay buffer of up to N physics steps
    whose lag the env sets at reset (NEMOV3_DELAY_GROUPS).
    """
    if not (fixed_joints or pd_gains or actuator_delay_steps):
        return EntityCfg(
            init_state=default_keyframe,
            spec_fn=get_spec,
            articulation=NEMOV3_ARTICULATION,
        )
    fixed_joints, pd_gains = tuple(fixed_joints), tuple(pd_gains)
    delay = (
        {
            "delay_min_lag": 0,
            "delay_max_lag": actuator_delay_steps,
            "delay_hold_prob": 1.0,  # never resampled by the buffer; set per group at reset
        }
        if actuator_delay_steps
        else {}
    )
    return EntityCfg(
        init_state=dataclasses.replace(
            default_keyframe,
            joint_pos={k: v for k, v in default_keyframe.joint_pos.items() if k not in fixed_joints},
        ),
        spec_fn=functools.partial(get_spec, fixed_joints, *((pd_gains,) if pd_gains else ())),
        articulation=EntityArticulationInfoCfg(
            actuators=tuple(
                XmlActuatorCfg(target_names_expr=(joint_name,), **delay)
                for joint_name in _joint_names_in_actuator_order(get_spec(fixed_joints))
            ),
            soft_joint_pos_limit_factor=0.9,
        ),
    )


def get_nemov3_default_joint_pos(
    keyframe: EntityCfg.InitialStateCfg = HOME_KEYFRAME,
) -> "torch.Tensor":
    """Return the Nemov3 default joint positions as a 1D tensor.

    Order matches `NEMOV3_JOINT_ORDER` (= the XML joint order = the order
    used in motion .pkl `dof_pos` arrays).
    """
    import torch

    return torch.tensor(
        [keyframe.joint_pos.get(name, 0.0) for name in NEMOV3_JOINT_ORDER],
        dtype=torch.float32,
    )


if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity

    robot = Entity(get_nemov3_robot_cfg())
    viewer.launch(robot.spec.compile())
