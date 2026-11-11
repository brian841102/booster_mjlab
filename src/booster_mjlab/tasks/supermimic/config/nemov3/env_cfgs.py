"""Nemov3 SuperMimic environment: the 20-joint hardware build tracking retargeted dance clips.

The robot is booster_mjlab's nemov3 with the head / waist / wrist pitch welded (NEMOV3_FIXED_HEAD_WAIST_JOINTS) and the
hardware PD gains (NEMOV3_DEPLOY_PD_GAINS), plus three changes that the SuperMimic model (supermimic_vibed,
``nemov3_sim_hw.xml``) had and that the reference training run (exp5, 3 dance clips) used:
- elbow range (-1.57, 0.1): the XML's +1.57 lets the elbows fold backwards, which a policy did in the HOME pose;
- arm effort limits from the URDF (shoulder pitch / roll 5.6 Nm, shoulder yaw / elbow / wrist roll 1.27 Nm; the XML
  has 20 Nm);
- no self-collision (robot geoms collide with the ground only).

``sim2real=True`` (the ``-Sim2Real-`` task) adds the randomization of the deployed AMP turning policy (Exp 90, see
``tools/nemov3_finetune.md``) and a wheel speed clamp; every drawn quantity goes into the world model's sim context xi
(``tools/nemov3_finetune_supermimic.md``, exp2).
"""

from __future__ import annotations

import dataclasses
import glob
import os
import re

import mujoco
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from mjlab.managers import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from booster_mjlab import BOOSTER_MJLAB_SRC_PATH
from booster_mjlab.mdp.terrain import randomize_terrain_contact
from booster_mjlab.robots.nemov3.nemov3_constants import (
    NEMOV3_DELAY_GROUPS,
    NEMOV3_DEPLOY_PD_GAINS,
    NEMOV3_FIXED_HEAD_WAIST_JOINTS,
    NEMOV3_POSITION_ACTUATOR_NAME_EXPR,
    NEMOV3_POSITION_JOINT_NAMES,
    NEMOV3_WHEEL_JOINT_NAMES,
    get_nemov3_robot_cfg,
    get_spec,
)
from booster_mjlab.tasks.supermimic.supermimic_env_cfg import (
    SuperMimicCfg,
    SuperMimicEnvCfg,
    SuperMimicSim2RealCfg,
    make_supermimic_env_cfg,
)
from booster_mjlab.tasks.velocity.mdp.motor_dynamics import ActuatorDelayCfg

# The three dance clips most experiments were trained on (roller32, roller_house, three_turn); clip CSVs are made by
# the retarget submodule (scripts/make_supermimic_clip.py) and are not committed.
NEMOV3_DANCE3_MOTION_DIR = "submodules/retarget/data/supermimic_motions/nemov3_dance3"

# Joint columns of a SuperMimic clip CSV (= retarget's CSV_JOINT_NAMES, Isaac Lab's articulation order).
NEMOV3_CSV_JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_calf_joint", "left_wheel_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_calf_joint", "right_wheel_joint",
    "joint_lShoulderPitch", "joint_lShoulderRoll", "joint_lShoulderYaw", "joint_lElbow", "joint_lWristRoll",
    "joint_rShoulderPitch", "joint_rShoulderRoll", "joint_rShoulderYaw", "joint_rElbow", "joint_rWristRoll",
)  # fmt: skip

# Wheel links spin, so their orientation is not tracked; the calf links carry the wheel's heading and camber.
NEMOV3_TRACKED_BODIES = (
    "link_leg2pelvis", "left_hip_roll_link", "left_calf_link", "right_hip_roll_link", "right_calf_link",
    "link_lShoulderRoll", "link_lElbow", "link_lWristRoll", "link_rShoulderRoll", "link_rElbow", "link_rWristRoll",
)  # fmt: skip

# Wheel centres in their calf frames (wheel link pos in nemov3.xml): the lift reward points.
NEMOV3_WHEEL_CENTRES = (("left_calf_link", (0.0, -0.043, -0.175)), ("right_calf_link", (0.0, 0.043, -0.175)))

NEMOV3_ELBOW_RANGE = (-1.57, 0.1)
# (joint regex, effort limit Nm) from the URDF (train_v2.urdf)
NEMOV3_URDF_ARM_EFFORT_LIMITS = (
    (r"joint_[rl]Shoulder(Pitch|Roll)", 5.6),
    (r"joint_[rl](ShoulderYaw|Elbow|WristRoll)", 1.27),
)
# (joint regex, velocity limit rad/s) from the URDF: not simulated (MuJoCo has no joint speed limit), written into the
# deployment contract (scripts/export_supermimic.py) as supermimic_vibed did
NEMOV3_URDF_VELOCITY_LIMITS = (
    (r".*_(hip_(pitch|roll|yaw)|calf)_joint", 10.0),
    (r"joint_[rl]Shoulder(Pitch|Roll)", 23.87),
    (r"joint_[rl](ShoulderYaw|Elbow|WristRoll)", 40.2),
    (r".*_wheel_joint", 25.0),
)


# Sim-to-real (AMP Exp 90's settings, user 2026-11-02: wheel friction / pseudo-inertia mass scale / encoder offsets as
# Exp 90, but the CoM shift kept at SuperMimic's +-(1.25, 2.5, 2.5) cm and the gentler pushes of supermimic_vibed exp6).
# = NEMOV3_MERGED_DELAY of the velocity task: per group (arms / legs / wheels) a lag of 0-4 physics steps + a command lag
# round(U(0, 30 ms) / 5 ms) shared by the non-wheel and by the wheel joints: 0-50 ms in total.
NEMOV3_SUPERMIMIC_DELAY = ActuatorDelayCfg(
    groups=NEMOV3_DELAY_GROUPS,
    max_lag=4,
    cmd_groups=(r"(?!.*_wheel_joint$).*", r".*_wheel_joint"),
    cmd_delay_range_s=(0.0, 0.03),
)
NEMOV3_WHEEL_COLLISION_GEOMS = ("left_wheel_collision", "right_wheel_collision")
NEMOV3_WHEEL_SPEED_LIMIT = 25.0  # rad/s, the hardware runner's clamp of the wheel speed command


def nemov3_supermimic_spec() -> mujoco.MjSpec:
    spec = get_spec(NEMOV3_FIXED_HEAD_WAIST_JOINTS, NEMOV3_DEPLOY_PD_GAINS)
    for name in ("joint_lElbow", "joint_rElbow"):
        spec.joint(name).range = list(NEMOV3_ELBOW_RANGE)
    for pattern, effort in NEMOV3_URDF_ARM_EFFORT_LIMITS:
        for actuator in [a for a in spec.actuators if re.fullmatch(pattern, a.target)]:
            actuator.forcerange = [-effort, effort]
    for geom in spec.geoms:
        if geom.contype or geom.conaffinity:
            geom.contype, geom.conaffinity = 0, 1  # touches the ground (contype 1), not the robot
    return spec


def nemov3_supermimic_env_cfg(
    play: bool = False, motion_dir: str = NEMOV3_DANCE3_MOTION_DIR, sim2real: bool = False
) -> SuperMimicEnvCfg:
    cfg = make_supermimic_env_cfg(play=play)
    delay_steps = NEMOV3_SUPERMIMIC_DELAY.buffer_steps(cfg.sim.mujoco.timestep) if sim2real else 0
    robot = get_nemov3_robot_cfg(
        fixed_joints=NEMOV3_FIXED_HEAD_WAIST_JOINTS, pd_gains=NEMOV3_DEPLOY_PD_GAINS, actuator_delay_steps=delay_steps
    )
    cfg.scene.entities = {"robot": dataclasses.replace(robot, spec_fn=nemov3_supermimic_spec)}
    com_range = (0.0125, 0.025, 0.025)
    pos_scale, vel_scale = 0.5, 10.0
    # Used by `uv run play` (env.step); training sets the same targets in SuperMimicEnv.step.
    cfg.actions = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=tuple(n for n in NEMOV3_POSITION_JOINT_NAMES if n not in NEMOV3_FIXED_HEAD_WAIST_JOINTS),
            scale=pos_scale,
            use_default_offset=True,
            preserve_order=True,
        ),
        "wheel_vel": JointVelocityActionCfg(
            entity_name="robot", actuator_names=NEMOV3_WHEEL_JOINT_NAMES, scale=vel_scale, use_default_offset=False
        ),
    }
    if play:  # one env per clip
        clips = glob.glob(os.path.join(BOOSTER_MJLAB_SRC_PATH.parents[1], motion_dir, "*.csv"))
        cfg.scene.num_envs = max(1, len(clips))
    cfg.supermimic = SuperMimicCfg(
        motion_dir=motion_dir,
        csv_joint_names=NEMOV3_CSV_JOINT_NAMES,
        tracked_bodies=NEMOV3_TRACKED_BODIES,
        velocity_joints=NEMOV3_WHEEL_JOINT_NAMES,
        action_scale=pos_scale,  # rad per action unit (G1's BeyondMimic scales are ~0.3-0.55)
        velocity_action_scale=vel_scale,  # rad/s per action unit (0.75 m/s at r = 0.075 m)
        lift_points=NEMOV3_WHEEL_CENTRES,
        length_scale=0.5,
        term_height=0.15,
        reset_noise=not play,
        com_body="link_torso",
        com_range=com_range,
        velocity_limits=NEMOV3_URDF_VELOCITY_LIMITS,
        slip_points=tuple((body, centre, (0.0, 1.0, 0.0)) for body, centre in NEMOV3_WHEEL_CENTRES),  # axle = calf y
    )
    if "torso_com" in cfg.events:
        params = cfg.events["torso_com"].params
        params["asset_cfg"] = dataclasses.replace(params["asset_cfg"], body_names=("link_torso",))
        params["ranges"] = {i: (-r, r) for i, r in enumerate(com_range)}
    if sim2real:
        _add_sim2real(cfg)
    return cfg


def _add_sim2real(cfg: SuperMimicEnvCfg) -> None:
    sm = cfg.supermimic
    sm.sim2real = SuperMimicSim2RealCfg(
        actuator_delay=NEMOV3_SUPERMIMIC_DELAY,
        wheel_speed_limit=NEMOV3_WHEEL_SPEED_LIMIT,
        push_interval_s=(5.0, 10.0),
        push_vel_xy=0.2,
    )
    sm.friction_geoms, sm.friction_range = NEMOV3_WHEEL_COLLISION_GEOMS, (0.75, 1.25)
    sm.joint_offset_range = 0.015  # Exp 90's encoder bias
    sm.joint_vel_noise = 1.5  # Exp 90's joint velocity observation noise
    if not cfg.events:  # play / evaluation: no randomization
        return
    torso = SceneEntityCfg("robot", body_names=(sm.com_body,))
    cfg.events["robot_friction"].params["asset_cfg"] = SceneEntityCfg("robot", geom_names=sm.friction_geoms)
    cfg.events["robot_friction"].params["ranges"] = sm.friction_range
    # the torso's mass and inertia scale by e^(2 alpha) (COM unchanged), then the CoM offset, then +-1 kg payload
    cfg.events = {
        "robot_friction": cfg.events["robot_friction"],
        "trunk_inertia": EventTermCfg(
            mode="startup", func=dr.pseudo_inertia, params={"asset_cfg": torso, "alpha_range": (-0.05, 0.05)}
        ),
        "torso_com": cfg.events["torso_com"],
        "trunk_mass": EventTermCfg(
            mode="startup",
            func=dr.body_mass,
            params={"asset_cfg": torso, "ranges": sm.sim2real.mass_range, "operation": "add"},
        ),
        "pd_gains": EventTermCfg(
            mode="startup",
            func=dr.pd_gains,
            params={
                "asset_cfg": SceneEntityCfg("robot", actuator_names=NEMOV3_POSITION_ACTUATOR_NAME_EXPR),
                "operation": "scale",
                "kp_range": sm.sim2real.kp_range,
                "kd_range": sm.sim2real.kd_range,
            },
        ),
        "terrain_contact": EventTermCfg(
            mode="startup",
            func=randomize_terrain_contact,
            params={
                "asset_cfg": SceneEntityCfg("terrain"),
                "solref_ranges": dict(enumerate(sm.sim2real.solref_ranges)),
                "solimp_ranges": dict(enumerate(sm.sim2real.solimp_ranges)),
                "shared_random": True,
            },
        ),
    }
