"""Pino v1 (wheeled humanoid) velocity tracking environment configuration.

Follows the nemov3 recipe that produced the first working skating gait (nemov3 Exp 33: phase clock, legs + arms in the
AMP features, narrow stance, mirrored clips + PPO mirror loss), on flat ground and straight forward 0-1.3 m/s (user,
2026-11-03: straight first). See docs/pinov1_amp_skating_plan.md and tools/pinov1_finetune.md.

Differences from nemov3 (config/nemov3/env_cfgs.py):

* Each foot has two in-line wheels: a driven rear wheel (velocity action) and a free front wheel (no actuator, not
  observed by the actor -- the hardware has no encoder to rely on there).
* Optional no-lift constraint (``no_lift=True``): the only reference clip lifts a wheel > 1 cm in ~half of its frames,
  and the first goal is skating without lifting, so every wheel above NO_LIFT_HEIGHT_M can be penalized.
* The hip roll axes are not mirrored (the right hip spreads with a negative roll), so the narrow-stance penalty gets an
  outward sign per hip.
"""

from __future__ import annotations

import re

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp

from booster_mjlab.robots import PINOV1_ACTION_SCALE, PINOV1_WHEEL_ACTION_SCALE, get_pinov1_robot_cfg
from booster_mjlab.robots.pinov1.pinov1_constants import (
    HOME_KEYFRAME,
    HOME_KEYFRAME_REFERENCE_MEAN,
    NEMOV3_EXP33_PD_GAINS,
    NEMOV3_EXP33_WHEEL_KV,
    PINOV1_ARM_JOINT_PATTERN,
    PINOV1_DELAY_GROUPS,
    PINOV1_DEPLOY_PD_GAINS,
    PINOV1_FRONT_WHEEL_JOINT_NAMES,
    PINOV1_JOINT_ORDER,
    PINOV1_POSITION_JOINT_NAMES,
    PINOV1_WHEEL_BODY_RADII,
    PINOV1_WHEEL_JOINT_NAMES,
)
from booster_mjlab.tasks.velocity.mdp.gait_clock import GaitClockCommandCfg
from booster_mjlab.tasks.velocity.mdp.play_controls import PlayControlsCommandCfg, push_if_enabled
from booster_mjlab.tasks.velocity.mdp.rest_standing_command import RestStandingVelocityCommandCfg
from booster_mjlab.tasks.velocity.mdp.turn_in_place_command import TurnInPlaceVelocityCommandCfg
from booster_mjlab.tasks.velocity.mdp.rewards import (
    joint_target_limit_excess,
    standing_joint_deviation_excess,
    standing_base_vel_l2,
    standing_joint_vel_l2,
)
from booster_mjlab.mdp.events import velocity_gain_scale
from booster_mjlab.tasks.velocity.mdp.motor_dynamics import sample_actuator_delay
from booster_mjlab.tasks.velocity.mdp.stance import (
    hip_roll_excess,
    hip_roll_inward_backward,
    backward_hip_pitch_stagger,
    in_place_hip_yaw_min_toe_in,
    in_place_hip_yaw_toe_in,
    turning_hip_yaw_deviation,
)
from booster_mjlab.tasks.velocity.mdp.wheel_feet import (
    reward_weight_schedule,
    standing_wheel_height_excess,
    wheel_height_excess,
)
from booster_mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

TRUNK_BODY = "torso_link"
WHEEL_LINK_NAMES = tuple(PINOV1_WHEEL_BODY_RADII)
WHEEL_COLLISION_GEOMS = tuple(name.replace("_link", "_collision") for name in WHEEL_LINK_NAMES)
FOOT_LINK_NAMES = ("left_ankle_pitch_link", "right_ankle_pitch_link")

# Gait clock (observation only): the reference clip pushes with each leg about once a second, a left-right cycle of
# ~2 s (nemov3: 0.74 Hz).
GAIT_FREQUENCY_HZ = 0.5

# Larger action scale for the joints the discriminator scores (0.25 * forcerange / kp gives 0.26 rad legs / 0.20 arms).
# Default pose "B" (user 2026-11-03) is upright: the reference's mean crouch is hip pitch -0.54 / knee 0.82 against the
# default -0.1 / 0.2, i.e. ~1.3 / 1.5 action units with these scales (nemov3: the stride ~1-3 units once enlarged).
LEG_SWING_ACTION_SCALE = {".*_hip_pitch_joint": 0.35, ".*_knee_joint": 0.4}
ARM_SWING_ACTION_SCALE = {".*_shoulder_pitch_joint": 0.25, ".*_elbow_joint": 0.25}

# Narrow stance: squared hinge on the outward hip roll above this. In the reference, a hip whose wheels are both on the
# ground spreads <= 0.14 (right, p95) / sits at the inner limit (left); the lifted leg spreads up to 0.46-0.64 rad.
# 0.15 = the grounded p95 (nemov3 used 0.14).
HIP_ROLL_LIMIT_RAD = 0.15
HIP_ROLL_SPREAD_WEIGHT = -100.0

# No-lift constraint: sum over the four wheels of max(bottom height - NO_LIFT_HEIGHT_M, 0) / NO_LIFT_SCALE_M, i.e. 1 per
# cm above 1 cm per wheel; x NO_LIFT_WEIGHT, and x (1 - style weight) = 0.3 in the AMP mix, a 5 cm lift of one foot
# costs ~2.4 / step, above the style reward's ~1-2.
NO_LIFT_HEIGHT_M = 0.01
NO_LIFT_SCALE_M = 0.01
NO_LIFT_WEIGHT = -2.0

# Front wheels down (front_down=True; user 2026-11-03: lifting the heel is fine -- the passive front wheel still rolls
# on the ground -- but the leading leg must not raise its toe): the no-lift penalty on the two front wheels only. Exp 2
# raised the toe in 11-33 % of the time while moving, the clip in 0-8 %.
FRONT_WHEEL_LINK_NAMES = ("left_front_wheel_link", "right_front_wheel_link")

# Stable standing in the default pose (stand_still=True): the base task's standing_pose_l1 (-0.5, mean |q - default| in
# rad, so a 0.2 rad deviation costs 0.1) is far too weak next to the style / tracking terms -- Exp 1-3 stepped in place
# when told to stand. The standing pose weight goes up and a joint-velocity penalty (also only when commanded to stand)
# is added; more of the environments stand.
STANDING_POSE_WEIGHT = -4.0
STANDING_JOINT_VEL_WEIGHT = -1.0
STANDING_ENV_FRACTION = 0.2
# Exp 7 standing at 0 m/s rolled away at -0.5 m/s (the tracking reward is flat there): penalize the base velocity of a
# standing robot (``standing_base_vel`` weight, ramped like the other standing terms).
STANDING_BASE_VEL_WEIGHT = -2.0
# The base yaw-rate tracking tolerance (std sqrt(0.5) = 0.7 rad/s = 40 deg/s) is far too loose for a straight-line task:
# Exp 7 wobbled at 60-70 deg/s while moving (Exp 4: 30).
YAW_TRACKING_STD = 0.3

# Staged shaping (staged=True; Exp 6, from scratch): with every penalty on from iteration 0 Exp 5 never learned to
# track a speed (track_linear_velocity 0.04 for 1200 iterations vs 0.85 at 800 in Exp 1) -- it settled into the
# stand-still optimum. nemov3's staged recipe (config/nemov3/env_cfgs.py, LIFT_WEIGHT_SCHEDULES): learn the skating first,
# shape later -- each shaping term's weight moves linearly from its start value to its full weight between two
# iterations (start, end, first iteration, last iteration; 24 env steps per iteration).
STEPS_PER_ITERATION = 24
STAGED_SCHEDULES = {
    "hip_roll_spread": (0.0, HIP_ROLL_SPREAD_WEIGHT, 400, 800),
    "front_wheel_lift_penalty": (0.0, NO_LIFT_WEIGHT, 400, 1000),
    "standing_pose_l1": (-0.5, STANDING_POSE_WEIGHT, 400, 1000),
    "standing_joint_vel_l2": (0.0, STANDING_JOINT_VEL_WEIGHT, 400, 1000),
}

# Survival (survival=True; Exp 11): with all shaping terms at full weight the task reward per step went negative
# (Exp 10 @1060: +1.46 tracking / upright vs -2.43 penalties, action rate alone -1.2), and a standing robot gets no style
# reward -- so ending the episode early pays, and Exp 6 / 8 / 9 / 10 collapsed (episode length 480 -> 230) as soon as
# the penalties outgrew the rewards. A per-step alive bonus keeps living worth it and a one-off termination penalty
# (falls / illegal contact; time-outs are not penalized) makes falling expensive.
ALIVE_WEIGHT = 2.0
TERMINATION_WEIGHT = -100.0

# Training episode length (nemov3 trained with 10 s since Exp 22).
PINOV1_TRAIN_EPISODE_LENGTH_S = 10.0


# Turning (turn=True, Exp 28, user 2026-11-05: "加入转弯的训练按exp25配方 从头训练 并模仿nemov3 首次训练出好的转弯步态的
# 方式"): nemov3 Exp 41 trained its straight recipe from scratch with the yaw-rate command uniform in +-1.3 rad/s, no
# turning reference clips (the mirror loss maps left onto right turns).
TURN_YAW_RATE_MAX = 1.3


def pinov1_flat_env_cfg(
    play: bool = False,
    no_lift: bool = False,
    narrow_stance: bool = False,
    front_down: bool = False,
    stand_still: bool = False,
    staged: bool = False,
    standing_weights: tuple[float, float] = (STANDING_POSE_WEIGHT, STANDING_JOINT_VEL_WEIGHT),
    standing_base_vel: float = 0.0,
    yaw_tracking_std: float | None = None,
    rest_standing: bool = False,
    foot_frame_contact: bool = False,
    survival: bool = False,
    track_weight: float | None = None,
    standing_clock_off: bool = False,
    standing_wheels_down: float = 0.0,
    reference_default_pose: bool = False,
    default_keyframe: EntityCfg.InitialStateCfg | None = None,
    max_speed: float = 1.3,
    turn: bool = False,
    turn_yaw_rate_max: float = TURN_YAW_RATE_MAX,
    backward: float = 0.0,
    gait_frequency_hz: float = GAIT_FREQUENCY_HZ,
    hip_roll_limit: float = HIP_ROLL_LIMIT_RAD,
    nemov3_pd: bool = False,
    deploy_pd: bool = False,
    arm_pd: tuple[float, float] | None = None,
    wheel_speed_limit: float | None = None,
    wheel_kv: float | None = None,
    wheel_kv_randomization: bool = False,
    backward_clock_off: bool = False,
    in_place_clock_off: bool = False,
    target_limit_weight: float = 0.0,
    standing_hip_yaw: tuple[float, float] | None = None,
    standing_pose_hold: tuple[float, float, tuple[str, ...]] | None = None,
    backward_hip_roll_inward: tuple[float, float] | None = None,
    hip_roll_inward_turning: bool = False,
    hip_roll_inward_standing: bool = False,
    turn_in_place_envs: float = 0.0,
    termination_penalty: float | None = None,
    yaw_track_weight: float | None = None,
    actuator_delay_steps: int = 0,
    actuator_delay_groups: tuple[str, ...] = PINOV1_DELAY_GROUPS,
    trunk_mass_add: tuple[float, float] | None = None,
    in_place_toe_in: tuple[float, float, float] | None = None,
    in_place_min_toe_in: tuple[float, float] | None = None,
    backward_stagger: tuple[float, float] | None = None,
    turning_hip_yaw: tuple[float, float] | None = None,
) -> ManagerBasedRlEnvCfg:
    """Pino v1 flat-terrain straight-line skating: AMP style, gait clock, forward command 0-1.3 m/s. ``no_lift`` /
    ``narrow_stance`` add the no-lift penalty / the hip-roll spread penalty (off in Exp 1, user 2026-11-03: see the gait
    without constraints first); ``front_down`` the no-lift penalty on the front wheels only (heel lifts stay free);
    ``stand_still`` strengthens the standing pose and adds the standing joint-velocity penalty (STANDING_*); ``staged``
    ramps those shaping terms in over training (STAGED_SCHEDULES); ``standing_weights`` = (standing pose, standing joint
    velocity) weights of ``stand_still``; ``standing_base_vel`` (< 0) adds the standing base-velocity penalty with that
    weight, ``yaw_tracking_std`` overrides the yaw-rate tracking tolerance; ``rest_standing`` starts the standing environments at
    rest in the default pose (mdp/rest_standing_command.py); ``foot_frame_contact`` counts the ankle link (the foot frame
    between the two wheels, 1.4-2.7 cm above the ground when both wheels touch) as part of the foot, not an illegal
    ground contact; ``survival`` adds the alive bonus and the termination penalty (ALIVE_WEIGHT, TERMINATION_WEIGHT); ``track_weight``
    overrides the linear-velocity tracking weight (base 2.25); ``standing_clock_off`` zeroes the gait clock observation
    while commanded to stand; ``standing_wheels_down`` (< 0) penalizes any wheel above 1 cm while commanded to stand
    with that weight (ramped like the other standing terms); ``reference_default_pose`` uses default pose "A"
    (HOME_KEYFRAME_REFERENCE_MEAN, the clip's mean crouch) instead of the user's "B", ``default_keyframe`` any other
    default pose; ``max_speed`` is the top of the forward command, ``turn`` adds yaw-rate
    commands in +-``turn_yaw_rate_max`` rad/s (default TURN_YAW_RATE_MAX), ``backward=V`` forward commands down to -V m/s,
    ``backward_hip_roll_inward`` = (weight, limit rad) penalizes hip roll inward of the limit while driving backward, ``gait_frequency_hz`` the phase clock's frequency and
    ``hip_roll_limit`` the narrow-stance limit; ``deploy_pd`` additionally sets every arm joint to kp 18 / kd 0.8, ``arm_pd`` = (kp, kd) to any value instead;
    ``wheel_speed_limit`` clips the rear-wheel speed target [rad/s]; ``wheel_kv`` overrides the rear wheels' velocity gain,
    ``wheel_kv_randomization`` scales it x0.8-1.2 like the joints' kd; ``backward_clock_off`` (with ``standing_clock_off``) also
    zeroes the gait clock below -0.05 m/s; ``nemov3_pd`` swaps the xdream.py gains for nemov3 Exp 33's (kp 50 / kd 0.5,
    wheels kv 0.5; the action scales stay pinov1's); ``target_limit_weight`` (< 0) penalizes position targets beyond the
    soft joint limits (no leaning on a mechanical stop); ``standing_hip_yaw`` = (weight, tolerance rad) holds the hip yaws
    at the default while commanded to stand; ``standing_pose_hold`` = (weight, tolerance rad, joint patterns) does the same
    for other joints; ``actuator_delay_steps`` = N delays every actuator command by 0..N physics steps, one lag per
    joint group (``actuator_delay_groups``, default PINOV1_DELAY_GROUPS = one per motor bus; (".*",) = one lag for the
    whole body) drawn per episode (nemov3 Exp 82); ``trunk_mass_add`` = (lo, hi) kg adds a point
    mass at the torso COM per environment at startup (nemov3 Exp 88; it overwrites trunk_inertia's mass draw, whose
    COM / inertia draws stay); ``in_place_toe_in`` = (weight, asymmetry tolerance, minimum toe-in) penalizes asymmetric or
    too little toe-in of the hip yaws while turning in place (mdp/stance.py:in_place_hip_yaw_toe_in, Exp 49a);
    ``in_place_min_toe_in`` = (weight, minimum toe-in) holds each foot at least that toe-in while turning in place (linear
    hinge, mdp/stance.py:in_place_hip_yaw_min_toe_in, Exp 49a3); ``backward_stagger`` = (weight, tolerance rad) penalizes the
    hip pitch difference L - R beyond the tolerance while driving straight backward (mdp/stance.py:backward_hip_pitch_stagger,
    Exp 56); ``turning_hip_yaw`` = (weight, tolerance rad) penalizes the hip yaws away from the default pose while turning
    in place or backward (off while driving forward; mdp/stance.py:turning_hip_yaw_deviation, Exp 57)."""
    cfg = make_velocity_env_cfg()
    keyframe = default_keyframe or (HOME_KEYFRAME_REFERENCE_MEAN if reference_default_pose else None)
    gains = {"pd_gains": NEMOV3_EXP33_PD_GAINS, "wheel_kv": NEMOV3_EXP33_WHEEL_KV} if nemov3_pd else {}
    if deploy_pd:  # nemov3_pd with the arms at kp 18 / kd 0.8 (PINOV1_DEPLOY_PD_GAINS)
        assert nemov3_pd, "deploy_pd builds on nemov3_pd"
        gains = {"pd_gains": PINOV1_DEPLOY_PD_GAINS, "wheel_kv": NEMOV3_EXP33_WHEEL_KV}
    if arm_pd is not None:  # nemov3_pd with every arm joint at (kp, kd) (Exp 33: 25 / 1.0)
        assert nemov3_pd and not deploy_pd, "arm_pd builds on nemov3_pd and replaces deploy_pd"
        gains = {"pd_gains": NEMOV3_EXP33_PD_GAINS + ((PINOV1_ARM_JOINT_PATTERN, *arm_pd),), "wheel_kv": NEMOV3_EXP33_WHEEL_KV}
    if wheel_kv is not None:  # rear-wheel velocity gain override (Exp 34: 1.0)
        assert gains, "wheel_kv builds on nemov3_pd"
        gains["wheel_kv"] = wheel_kv
    cfg.scene.entities = {"robot": get_pinov1_robot_cfg(keyframe or HOME_KEYFRAME, **gains, actuator_delay_steps=actuator_delay_steps)}

    # Flat terrain; buffer sizes as nemov3.
    cfg.sim.njmax = 300
    cfg.sim.mujoco.ccd_iterations = 60
    cfg.sim.contact_sensor_maxmatch = 64
    cfg.sim.nconmax = 50
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # Sensors (the generic names the base mdp terms look for). One contact slot per foot: both of its wheels.
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="subtree", pattern=r"^(left|right)_ankle_pitch_link$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    non_foot_ground_cfg = ContactSensorCfg(
        name="non_foot_ground_contact",
        primary=ContactMatch(
            mode="body",
            entity="robot",
            pattern=r".*",
            exclude=WHEEL_LINK_NAMES + (FOOT_LINK_NAMES if foot_frame_contact else ()),
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        # torso_link's subtree = torso + arms (the legs hang from the pelvis).
        primary=ContactMatch(mode="subtree", pattern=TRUNK_BODY, entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern=TRUNK_BODY, entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    cfg.scene.sensors = (feet_ground_cfg, non_foot_ground_cfg, self_collision_cfg)

    # Actions: position targets for the 21 joints, velocity targets for the two rear wheels.
    cfg.actions["joint_pos"] = JointPositionActionCfg(
        entity_name="robot",
        actuator_names=PINOV1_POSITION_JOINT_NAMES,
        scale=_action_scale(),
        use_default_offset=True,
    )
    cfg.actions["wheel_vel"] = JointVelocityActionCfg(
        entity_name="robot",
        actuator_names=PINOV1_WHEEL_JOINT_NAMES,
        scale=PINOV1_WHEEL_ACTION_SCALE,
        use_default_offset=True,
        # wheel_speed_limit (Exp 33): clip the rear-wheel speed target to the motor's top speed [rad/s].
        clip=None if wheel_speed_limit is None else {".*": (-wheel_speed_limit, wheel_speed_limit)},
    )
    cfg.viewer.body_name = TRUNK_BODY

    # Command: straight forward 0-1.3 m/s, 10 % standing; no lateral / yaw, no command curriculum.
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, mdp.UniformVelocityCommandCfg)
    twist_cmd.heading_command = False
    twist_cmd.rel_standing_envs = 0.1
    # backward=V (Exp 30, nemov3 Exp 90's way): forward command in [-V, max_speed] instead of [0, max_speed], no new
    # reward (nemov3 Exp 88 had never seen a negative command and spread its legs when driven backward).
    twist_cmd.ranges.lin_vel_x = (0.0 - backward, max_speed)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    # turn=True (Exp 28, nemov3 Exp 41's way): yaw rate uniform in [-turn_yaw_rate_max, turn_yaw_rate_max]; the base
    # config's rel_forward_envs (10 %) keep going straight so the straight gait stays trained.
    twist_cmd.ranges.ang_vel_z = (-turn_yaw_rate_max, turn_yaw_rate_max) if turn else (0.0, 0.0)
    twist_cmd.ranges.heading = None

    # Events: point the robot-specific ones at pinov1's bodies / geoms. limb_inertia is dropped as for nemov3
    # (pseudo-inertia perturbations NaN'd thin inertia tensors there).
    cfg.events["pd_gains"].params["asset_cfg"] = SceneEntityCfg("robot", actuator_names=(r".*_act$",))
    if wheel_kv_randomization:  # the rear wheels' velocity gain gets the same x0.8-1.2 as the joints' kd (Exp 34)
        cfg.events["wheel_kv"] = EventTermCfg(
            func=velocity_gain_scale,
            mode="startup",
            params={"kd_range": cfg.events["pd_gains"].params["kd_range"],
                    "asset_cfg": SceneEntityCfg("robot", actuator_names=(r".*_vel$",))},
        )
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = WHEEL_COLLISION_GEOMS
    cfg.events["trunk_inertia"].params["asset_cfg"].body_names = (TRUNK_BODY,)
    cfg.events.pop("limb_inertia", None)
    if trunk_mass_add is not None:  # after trunk_inertia: "add" starts from the XML mass, so it replaces that mass draw
        cfg.events["trunk_mass"] = EventTermCfg(
            func=mdp.dr.body_mass,
            mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names=(TRUNK_BODY,)), "ranges": trunk_mass_add, "operation": "add"},
        )
    if actuator_delay_steps:
        cfg.events["actuator_delay"] = EventTermCfg(
            func=sample_actuator_delay,
            mode="reset",
            params={"groups": actuator_delay_groups, "max_lag": actuator_delay_steps},
        )

    # Rewards.
    cfg.rewards["upright"].params["asset_cfg"].body_names = (TRUNK_BODY,)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (TRUNK_BODY,)
    # Upper-body joints the discriminator does not score stay near the default (shoulder pitch / elbow are AMP's).
    posture = cfg.rewards["upper_body_posture"].params
    posture_joints = (r"torso_joint", r".*_shoulder_roll_joint", r".*_shoulder_yaw_joint", r".*_wrist_joint")
    posture["asset_cfg"].joint_names = posture_joints
    posture["std_standing"] = {j: 0.05 for j in posture_joints}
    posture["std_walking"] = {j: 0.15 for j in posture_joints}
    posture["std_running"] = {j: 0.2 for j in posture_joints}
    # The standing pose penalty over the joints with a pose (a wheel angle grows with distance).
    cfg.rewards["standing_pose_l1"].params["asset_cfg"].joint_names = PINOV1_POSITION_JOINT_NAMES
    if narrow_stance:
        cfg.rewards["hip_roll_spread"] = RewardTermCfg(
            func=hip_roll_excess,
            weight=HIP_ROLL_SPREAD_WEIGHT,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=("left_hip_roll_joint", "right_hip_roll_joint"), preserve_order=True
                ),
                "limit": hip_roll_limit,
                "outward_signs": (1.0, -1.0),
            },
        )
    if no_lift:
        cfg.rewards["wheel_lift_penalty"] = RewardTermCfg(
            func=wheel_height_excess,
            weight=NO_LIFT_WEIGHT,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=WHEEL_LINK_NAMES, preserve_order=True),
                "limit": NO_LIFT_HEIGHT_M,
                "scale": NO_LIFT_SCALE_M,
                "wheel_radius": tuple(PINOV1_WHEEL_BODY_RADII.values()),
            },
        )
    if stand_still:
        cfg.rewards["standing_pose_l1"].weight = standing_weights[0]
        cfg.rewards["standing_joint_vel_l2"] = RewardTermCfg(
            func=standing_joint_vel_l2,
            weight=standing_weights[1],
            params={
                "command_name": "twist",
                "asset_cfg": SceneEntityCfg("robot", joint_names=PINOV1_POSITION_JOINT_NAMES),
            },
        )
        twist_cmd.rel_standing_envs = STANDING_ENV_FRACTION
        if standing_base_vel:
            cfg.rewards["standing_base_vel_l2"] = RewardTermCfg(
                func=standing_base_vel_l2, weight=standing_base_vel, params={"command_name": "twist"}
            )
    if yaw_tracking_std is not None:
        cfg.rewards["track_angular_velocity"].params["std"] = yaw_tracking_std
    if standing_wheels_down:
        cfg.rewards["standing_wheel_lift_penalty"] = RewardTermCfg(
            func=standing_wheel_height_excess,
            weight=standing_wheels_down,
            params={
                "command_name": "twist",
                "asset_cfg": SceneEntityCfg("robot", body_names=WHEEL_LINK_NAMES, preserve_order=True),
                "limit": NO_LIFT_HEIGHT_M,
                "scale": NO_LIFT_SCALE_M,
                "wheel_radius": tuple(PINOV1_WHEEL_BODY_RADII.values()),
            },
        )
    if standing_hip_yaw is not None:
        cfg.rewards["standing_hip_yaw"] = RewardTermCfg(
            func=standing_joint_deviation_excess,
            weight=standing_hip_yaw[0],
            params={
                "command_name": "twist",
                "tolerance": standing_hip_yaw[1],
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*_hip_yaw_joint",)),
            },
        )
    if in_place_toe_in is not None:
        cfg.rewards["in_place_toe_in"] = RewardTermCfg(
            func=in_place_hip_yaw_toe_in,
            weight=in_place_toe_in[0],
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=("left_hip_yaw_joint", "right_hip_yaw_joint"), preserve_order=True),
                "asym_tolerance": in_place_toe_in[1],
                "min_toe_in": in_place_toe_in[2],
            },
        )
    if in_place_min_toe_in is not None:
        cfg.rewards["in_place_min_toe_in"] = RewardTermCfg(
            func=in_place_hip_yaw_min_toe_in,
            weight=in_place_min_toe_in[0],
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=("left_hip_yaw_joint", "right_hip_yaw_joint"), preserve_order=True),
                "min_toe_in": in_place_min_toe_in[1],
            },
        )
    if backward_stagger is not None:
        cfg.rewards["backward_stagger"] = RewardTermCfg(
            func=backward_hip_pitch_stagger,
            weight=backward_stagger[0],
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=("left_hip_pitch_joint", "right_hip_pitch_joint"), preserve_order=True),
                "tolerance": backward_stagger[1],
            },
        )
    if turning_hip_yaw is not None:
        cfg.rewards["turning_hip_yaw"] = RewardTermCfg(
            func=turning_hip_yaw_deviation,
            weight=turning_hip_yaw[0],
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r".*_hip_yaw_joint",)), "tolerance": turning_hip_yaw[1]},
        )
    if standing_pose_hold is not None:
        cfg.rewards["standing_pose_hold"] = RewardTermCfg(
            func=standing_joint_deviation_excess,
            weight=standing_pose_hold[0],
            params={
                "command_name": "twist",
                "tolerance": standing_pose_hold[1],
                "asset_cfg": SceneEntityCfg("robot", joint_names=standing_pose_hold[2]),
            },
        )
    if backward_hip_roll_inward is not None:
        # (weight, limit rad): hips may not roll inward of ``limit`` while driving backward (Exp 31).
        cfg.rewards["backward_hip_roll_inward"] = RewardTermCfg(
            func=hip_roll_inward_backward,
            weight=backward_hip_roll_inward[0],
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=("left_hip_roll_joint", "right_hip_roll_joint"), preserve_order=True
                ),
                "limit": backward_hip_roll_inward[1],
                "outward_signs": (1.0, -1.0),
                # hip_roll_inward_turning (Exp 35): also while turning, full from |yaw-rate command| 0.1 rad/s.
                "turn_full_at": 0.1 if hip_roll_inward_turning else None,
                "standing": hip_roll_inward_standing,
            },
        )
    if target_limit_weight:
        cfg.rewards["joint_target_limit"] = RewardTermCfg(func=joint_target_limit_excess, weight=target_limit_weight)
    if track_weight is not None:
        cfg.rewards["track_linear_velocity"].weight = track_weight
    if survival:
        cfg.rewards["alive"] = RewardTermCfg(func=mdp.is_alive, weight=ALIVE_WEIGHT)
        cfg.rewards["termination"] = RewardTermCfg(func=mdp.is_terminated, weight=TERMINATION_WEIGHT)
    if termination_penalty is not None:  # one-off penalty on falls only, no alive bonus (autonomous loop Exp 44)
        assert not survival
        cfg.rewards["termination"] = RewardTermCfg(func=mdp.is_terminated, weight=termination_penalty)
    if yaw_track_weight is not None:  # yaw-rate tracking weight (default 2.0; Exp 44: 3.0)
        cfg.rewards["track_angular_velocity"].weight = yaw_track_weight
    if rest_standing:
        cfg.commands["twist"] = RestStandingVelocityCommandCfg.from_cfg(twist_cmd)
        twist_cmd = cfg.commands["twist"]
    if turn_in_place_envs > 0.0:  # extra in-place turning commands (autonomous loop, 2026-11-06)
        assert not rest_standing, "turn_in_place_envs and rest_standing are separate command classes"
        cfg.commands["twist"] = TurnInPlaceVelocityCommandCfg.from_cfg(twist_cmd, turn_in_place_envs)
        twist_cmd = cfg.commands["twist"]
    if front_down:
        cfg.rewards["front_wheel_lift_penalty"] = RewardTermCfg(
            func=wheel_height_excess,
            weight=NO_LIFT_WEIGHT,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=FRONT_WHEEL_LINK_NAMES, preserve_order=True),
                "limit": NO_LIFT_HEIGHT_M,
                "scale": NO_LIFT_SCALE_M,
                "wheel_radius": tuple(PINOV1_WHEEL_BODY_RADII[n] for n in FRONT_WHEEL_LINK_NAMES),
            },
        )
    if staged:
        schedules = {**STAGED_SCHEDULES, "standing_pose_l1": (-0.5, standing_weights[0], 400, 1000),
                     "standing_joint_vel_l2": (0.0, standing_weights[1], 400, 1000),
                     "standing_base_vel_l2": (0.0, standing_base_vel, 400, 1000),
                     "standing_wheel_lift_penalty": (0.0, standing_wheels_down, 400, 1000)}
        for name, (w0, w1, it0, it1) in schedules.items():
            if name not in cfg.rewards:
                continue
            cfg.rewards[name].weight = w0
            cfg.curriculum[f"{name}_weight"] = CurriculumTermCfg(
                func=reward_weight_schedule,
                params={
                    "term_name": name,
                    "start_weight": w0,
                    "end_weight": w1,
                    "start_step": it0 * STEPS_PER_ITERATION,
                    "end_step": it1 * STEPS_PER_ITERATION,
                },
            )
    # Biped swing / landing terms do not apply to wheels (as nemov3).
    for name in ("air_time", "foot_clearance", "foot_swing_height", "foot_slip", "soft_landing"):
        cfg.rewards.pop(name, None)
    for group in cfg.observations.values():
        group.terms.pop("foot_height", None)
        group.terms.pop("foot_air_time", None)
    for name in ("terrain_levels", "soft_landing_weight", "command_vel"):
        cfg.curriculum.pop(name, None)

    # Observations: no wheel angles (they grow with distance travelled -- nemov3 Exp 18/19); the actor does not see
    # the free front wheels' speed either.
    for group_name in ("actor", "critic"):
        cfg.observations[group_name].terms["joint_pos"].params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=PINOV1_POSITION_JOINT_NAMES
        )
    cfg.observations["actor"].terms["joint_vel"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=tuple(n for n in PINOV1_JOINT_ORDER if n not in PINOV1_FRONT_WHEEL_JOINT_NAMES)
    )

    cfg.commands["gait_clock"] = GaitClockCommandCfg(
        resampling_time_range=(1.0e9, 1.0e9),
        frequency_hz=gait_frequency_hz,
        standing_command="twist" if standing_clock_off else None,
        backward_speed=0.05 if backward_clock_off else None,
        linear_off=in_place_clock_off,
    )
    for group in cfg.observations.values():
        group.terms["gait_clock"] = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "gait_clock"})

    cfg.episode_length_s = PINOV1_TRAIN_EPISODE_LENGTH_S
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.terminations.pop("illegal_contact", None)
        # Pushes stay available behind the viewer's "Push disturbance" checkbox (as nemov3).
        cfg.commands["play_controls"] = PlayControlsCommandCfg(resampling_time_range=(1.0e9, 1.0e9))
        push_robot = cfg.events["push_robot"]
        push_robot.func = push_if_enabled
        push_robot.params["command_name"] = "play_controls"
        # viser's joystick GUI needs max > 0 on every axis.
        twist_cmd.ranges.lin_vel_y = (0.0, 0.1)
        if not turn:
            twist_cmd.ranges.ang_vel_z = (0.0, 0.1)
    return cfg


def _action_scale() -> dict[str, float]:
    """PINOV1_ACTION_SCALE with the swing overrides applied (one entry per joint: mjlab rejects a joint matched by both
    its name and a pattern)."""
    overrides = {**LEG_SWING_ACTION_SCALE, **ARM_SWING_ACTION_SCALE}
    return {
        name: next((v for pattern, v in overrides.items() if re.fullmatch(pattern, name)), scale)
        for name, scale in PINOV1_ACTION_SCALE.items()
    }
