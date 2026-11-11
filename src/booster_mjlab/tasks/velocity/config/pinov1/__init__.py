from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import register_mjlab_task

from booster_mjlab.tasks.velocity.config import with_amp_obs_group
from booster_mjlab.tasks.velocity.rl.runner import VelocityAmpOnPolicyRunner

from .env_cfgs import pinov1_flat_env_cfg
from booster_mjlab.robots.pinov1.pinov1_constants import HOME_KEYFRAME_NEMO_CLIP_MEAN, PINOV1_WHEEL_SPEED_LIMIT_48V

from booster_mjlab.tasks.velocity.mdp.fast_style_label import command_speed_fraction, fast_command_label

from .rl_cfg import PINOV1_AMP_DATASET_ROOT_STRAIGHT, pinov1_amp_ppo_runner_cfg

# Speed-conditioned style (Exp 26): the discriminator also sees a label, 1 on SkatingFast06 / while the forward command
# is >= 0.6 m/s, so fast commands must look like Fast06 (big arm swing) and slow ones like Skating01 / 02.
PINOV1_FAST_LABEL_EXPERT_FN = "booster_mjlab.tasks.velocity.mdp.fast_style_label:fast_clip_label_from_motion"

# Joints the AMP discriminator scores: the leg swing (hip pitch, knee) and the arm swing (shoulder pitch, elbow), as
# nemov3's arms variant. Hip roll / yaw stay out (nemov3 Exp 7: the reference holds them still while exploration noise
# jitters them, an easy cue for the discriminator -- and pinov1's hip roll sits clipped at its limit in ~3/4 of the
# frames), the ankles too (clipped at both limits in the retargeting). Joint positions only, no joint velocities or
# base state (nemov3 Exp 13 / 14).
PINOV1_AMP_JOINT_NAMES = (
    r".*_hip_pitch_joint",
    r".*_knee_joint",
    r".*_shoulder_pitch_joint",
    r".*_elbow_joint",
)

# + ankle pitch (Exp 3, user 2026-11-03: the policy lifts the front wheel -- "翘脚尖" -- 3-4x as often as the clip; the
# ankle was free). Its reference sits clamped at the soft limits in ~1/4 of the frames.
PINOV1_AMP_ANKLE_JOINT_NAMES = PINOV1_AMP_JOINT_NAMES + (r".*_ankle_pitch_joint",)

# Ground-contact sites of the four wheels: the motion pose-pool reset rejects poses with a wheel below the ground.
PINOV1_WHEEL_GROUND_SITES = (
    "left_rear_wheel_ground",
    "left_front_wheel_ground",
    "right_rear_wheel_ground",
    "right_front_wheel_ground",
)


def _with_amp(env_cfg, runner_cfg, amp_joints=PINOV1_AMP_JOINT_NAMES, rest_reset_fraction=0.0):
    """AMP observation group and pose-pool reset; with ``runner_cfg.amp_extra_obs_fn`` = PINOV1_FAST_LABEL_EXPERT_FN the
    policy-side fast-command label is appended as the last AMP term."""
    cfg = with_amp_obs_group(
        env_cfg,
        dataset_root=runner_cfg.dataset_root,
        speed_factor=runner_cfg.speed_factor,
        augmentations=runner_cfg.dataset_augmentations,
    )
    # The runner reads the style weight from this curriculum term (the wrapper pins it to 0.3).
    style = cfg.curriculum["amp_style_weight"].params
    style["start_weight"] = style["end_weight"] = runner_cfg.style_reward_weight
    amp = cfg.observations["amp"]
    amp.terms["joint_pos"].params["asset_cfg"] = SceneEntityCfg("robot", joint_names=amp_joints)
    for name in ("joint_vel", "projected_gravity"):
        amp.terms.pop(name)
    if runner_cfg.amp_extra_obs_fn == PINOV1_FAST_LABEL_EXPERT_FN and runner_cfg.amp_soft_condition:
        max_speed = env_cfg.commands["twist"].ranges.lin_vel_x[1]
        amp.terms["command_speed_fraction"] = ObservationTermCfg(
            func=command_speed_fraction, params={"max_speed": max_speed}
        )
    elif runner_cfg.amp_extra_obs_fn == PINOV1_FAST_LABEL_EXPERT_FN:
        amp.terms["fast_command_label"] = ObservationTermCfg(func=fast_command_label)
    else:
        assert runner_cfg.amp_extra_obs_fn is None, "no policy-side term for this expert feature"
    reset = cfg.events["reset_robot_from_motion"].params
    reset["foot_sites"] = PINOV1_WHEEL_GROUND_SITES
    reset["default_pose_fraction"] = rest_reset_fraction
    return cfg


def _register(name, env_kwargs=None, rl_kwargs=None, amp_joints=PINOV1_AMP_JOINT_NAMES, rest_reset_fraction=0.0):
    """Register ``Mjlab-Velocity-Flat-Amp{name}[-Muon]-Booster-Pinov1``."""
    env_kwargs = env_kwargs or {}
    for use_muon in (False, True):
        rl_cfg = pinov1_amp_ppo_runner_cfg(use_muon=use_muon, **(rl_kwargs or {}))
        register_mjlab_task(
            task_id=f"Mjlab-Velocity-Flat-Amp{name}{'-Muon' if use_muon else ''}-Booster-Pinov1",
            env_cfg=_with_amp(pinov1_flat_env_cfg(**env_kwargs), rl_cfg, amp_joints, rest_reset_fraction),
            play_env_cfg=_with_amp(pinov1_flat_env_cfg(play=True, **env_kwargs), rl_cfg, amp_joints, rest_reset_fraction),
            rl_cfg=rl_cfg,
            runner_cls=VelocityAmpOnPolicyRunner,
        )


# Straight-line skating, nemov3 Exp 33's recipe without its narrow-stance penalty (Exp 1, user 2026-11-03: "不抬脚 窄站姿
# 都别放 我先看看不加上约束会训练出啥样子的步态").
_register("")
# + the hip-roll spread penalty only (Exp 2, user 2026-11-03: "把窄站姿加回来"; Exp 1 spread the hips to 0.74-0.76 rad).
_register("-NarrowStance", {"narrow_stance": True})
# Exp 2 + no style reward while standing + ankle pitch in the discriminator (Exp 3, user 2026-11-03; its discriminator
# input differs from Exp 1/2: resume with --agent.reset-discriminator True).
_register(
    "-NarrowStance-StandMask-AnkleAmp",
    {"narrow_stance": True},
    {"style_standing_mask": True},
    PINOV1_AMP_ANKLE_JOINT_NAMES,
)
# Exp 2 + the front wheels kept on the ground (Exp 4, user 2026-11-03: no toe raising; the ankle stays out of the
# discriminator and standing keeps its style reward, as Exp 2). Same discriminator input as Exp 1/2.
_register("-NarrowStance-FrontDown", {"narrow_stance": True, "front_down": True})
# Autonomous loop (user 2026-11-03: no toe raising, alternating gait, stable default-pose standing; from scratch):
# Exp 2 + front wheels down + stronger standing pose / standing joint-velocity penalty and no style reward while
# standing (Exp 5).
_register(
    "-NarrowStance-FrontDown-StandStill",
    {"narrow_stance": True, "front_down": True, "stand_still": True},
    {"style_standing_mask": True},
)
# Exp 5 with the shaping terms ramped in over iterations 400-1000 (Exp 6): Exp 5 never learned to track a speed.
_register(
    "-NarrowStance-FrontDown-StandStill-Staged",
    {"narrow_stance": True, "front_down": True, "stand_still": True, "staged": True},
    {"style_standing_mask": True},
)
# Exp 6 + 30 % of the resets start at rest in the default pose, softer standing weights (Exp 7): Exp 6's standing
# robots fell within 1-3 s -- a standing command started from a reference frame at ~1 m/s, and zero actions topple the
# default pose backwards (PD sag), so the stand has to be learned from rest as well.
_register(
    "-NarrowStance-FrontDown-StandStill-Staged-RestReset",
    {"narrow_stance": True, "front_down": True, "stand_still": True, "staged": True, "standing_weights": (-2.0, -0.5)},
    {"style_standing_mask": True},
    rest_reset_fraction=0.3,
)
# Exp 7 + standing base-velocity penalty, tighter yaw-rate tracking, standing pose -3 (Exp 8): Exp 7 stood without
# falling but rolled backward at 0.5 m/s and wobbled 60-70 deg/s in yaw.
_register(
    "-NarrowStance-FrontDown-StandStill-Staged-RestReset-StandVel",
    {"narrow_stance": True, "front_down": True, "stand_still": True, "staged": True, "standing_weights": (-3.0, -0.5),
     "standing_base_vel": -2.0, "yaw_tracking_std": 0.3},
    {"style_standing_mask": True},
    rest_reset_fraction=0.3,
)
# Exp 8 collapsed again once the standing terms ramped in (episode length 490 -> 255; a standing command started from a
# ~1 m/s reference frame): every standing environment now starts at rest in the default pose (mdp/rest_standing_command.py),
# yaw-rate tracking tolerance 0.5 instead of 0.3 (the reward was flat at 0.3 for a robot wobbling at 70 deg/s) (Exp 9).
_register(
    "-NarrowStance-FrontDown-StandStill-Staged-RestStand",
    {"narrow_stance": True, "front_down": True, "stand_still": True, "staged": True, "standing_weights": (-3.0, -0.5),
     "standing_base_vel": -2.0, "yaw_tracking_std": 0.5, "rest_standing": True},
    {"style_standing_mask": True},
    rest_reset_fraction=0.2,
)
# Exp 9 + the ankle link (foot frame between the wheels) no longer an illegal ground contact (Exp 10): Exp 9's standing
# robots, started at rest in the default pose, all ended within 0.2-0.26 s by illegal_contact at ~0-5 deg pitch -- the
# foot frame touching the ground, not a fall.
_register(
    "-NarrowStance-FrontDown-StandStill-Staged-RestStand-FootFrame",
    {"narrow_stance": True, "front_down": True, "stand_still": True, "staged": True, "standing_weights": (-3.0, -0.5),
     "standing_base_vel": -2.0, "yaw_tracking_std": 0.5, "rest_standing": True, "foot_frame_contact": True},
    {"style_standing_mask": True},
    rest_reset_fraction=0.2,
)
# Exp 10 + alive bonus and termination penalty (Exp 11): Exp 10 collapsed at ~1080 -- the task reward per step had gone
# negative, so falling early paid.
_register(
    "-NarrowStance-FrontDown-StandStill-Staged-RestStand-FootFrame-Survival",
    {"narrow_stance": True, "front_down": True, "stand_still": True, "staged": True, "standing_weights": (-3.0, -0.5),
     "standing_base_vel": -2.0, "yaw_tracking_std": 0.5, "rest_standing": True, "foot_frame_contact": True,
     "survival": True},
    {"style_standing_mask": True},
    rest_reset_fraction=0.2,
)
# Exp 11 + entropy coefficient 0.001 and speed tracking weight 4 (Exp 12): Exp 11 no longer collapsed, but its action
# std kept climbing (0.43 -> 0.61) and it ran slow (1.3 m/s -> 0.74).
_register(
    "-NarrowStance-FrontDown-StandStill-Staged-RestStand-FootFrame-Survival-LowEnt",
    {"narrow_stance": True, "front_down": True, "stand_still": True, "staged": True, "standing_weights": (-3.0, -0.5),
     "standing_base_vel": -2.0, "yaw_tracking_std": 0.5, "rest_standing": True, "foot_frame_contact": True,
     "survival": True, "track_weight": 4.0},
    {"style_standing_mask": True, "entropy_coef": 0.001},
    rest_reset_fraction=0.2,
)
# Exp 12 + no gait clock while standing + stronger standing stillness (joint velocity -0.5 -> -3, base velocity -2 -> -4)
# (Exp 13): Exp 12 tracked speed and kept the toes down, but still fidgeted when told to stand (joint velocity 0.44 rad/s,
# a heel up 15 % of the time).
_register(
    "-NarrowStance-FrontDown-StandStill-Staged-RestStand-FootFrame-Survival-LowEnt-StandClockOff",
    {"narrow_stance": True, "front_down": True, "stand_still": True, "staged": True, "standing_weights": (-3.0, -3.0),
     "standing_base_vel": -4.0, "yaw_tracking_std": 0.5, "rest_standing": True, "foot_frame_contact": True,
     "survival": True, "track_weight": 4.0, "standing_clock_off": True},
    {"style_standing_mask": True, "entropy_coef": 0.001},
    rest_reset_fraction=0.2,
)
# Exp 13 + all four wheels down while standing (Exp 14): Exp 13 stood still-ish (joint velocity 0.14-0.18 rad/s) but on one
# raised heel, i.e. on a passive front wheel that cannot brake, and rolled backward at ~0.4 m/s.
_register(
    "-NarrowStance-FrontDown-StandStill-Staged-RestStand-FootFrame-Survival-LowEnt-StandClockOff-StandWheelsDown",
    {"narrow_stance": True, "front_down": True, "stand_still": True, "staged": True, "standing_weights": (-3.0, -3.0),
     "standing_base_vel": -4.0, "yaw_tracking_std": 0.5, "rest_standing": True, "foot_frame_contact": True,
     "survival": True, "track_weight": 4.0, "standing_clock_off": True, "standing_wheels_down": -4.0},
    {"style_standing_mask": True, "entropy_coef": 0.001},
    rest_reset_fraction=0.2,
)
# Exp 4's recipe (nemov3 Exp 33 + narrow stance + front wheels down) with default pose "A", the clip's mean crouch,
# from scratch (Exp 15): the key difference to nemov3 -- its default pose is its clip's mean crouch, pinov1's "B" is
# 0.4-0.6 rad more upright than how its policies stand and skate.
_register("-NarrowStance-FrontDown-PoseA", {"narrow_stance": True, "front_down": True, "reference_default_pose": True})
# Exp 15 + the foot frame as part of the foot + all four wheels down while standing (Exp 16): with pose "A" the gait
# matched nemov3's (yaw wobble 8 deg/s, clean alternation) but rode on the passive front wheels with both heels up all
# the time, standing included (and so could not stand still), and the foot frame ended ~1 episode in 8 by illegal contact.
_register(
    "-NarrowStance-FrontDown-PoseA-FootFrame-StandWheelsDown",
    {"narrow_stance": True, "front_down": True, "reference_default_pose": True, "foot_frame_contact": True,
     "standing_wheels_down": -4.0},
)
# nemov3 Exp 33 the nemov3 way (Exp 17, user 2026-11-04: "全面专换用新csv……按照成功过exp33使过的片段……全面按照nemov3之前的
# 流程来训练"): Skating01 / Skating02 / trimmed SkatingFast06 retargeted to pinov1, default pose = their mirror-symmetric
# mean, Exp 33's recipe (phase clock 0.74 Hz = the clips' 1.35 s stride, legs + arms in the AMP features, narrow stance at
# the Skating01 / 02 maximum outward hip roll -- 0.065 here, nemov3 0.14 --, mirrored clips + PPO mirror loss), forward
# command 0-0.9 m/s (nemov3 1.3 = 1.2x its clips' top speed 1.1; these top out at 0.72). The foot frame counts as foot
# (a collision-model fix, not part of the recipe).
_register(
    "-Straight-Exp33",
    {"narrow_stance": True, "foot_frame_contact": True, "default_keyframe": HOME_KEYFRAME_NEMO_CLIP_MEAN,
     "max_speed": 0.9, "gait_frequency_hz": 0.74, "hip_roll_limit": 0.065},
    {"dataset_root": PINOV1_AMP_DATASET_ROOT_STRAIGHT},
)
# Exp 17 with nemov3 Exp 33's PD gains (Exp 18, user 2026-11-04: "把PD改成和nemov3一样做对照实验"): same data / recipe /
# action scales, only kp 40-15 / kd 3-2 -> kp 50 / kd 0.5 and wheel kv 4 -> 0.5.
_register(
    "-Straight-Exp33-Nemov3PD",
    {"narrow_stance": True, "foot_frame_contact": True, "default_keyframe": HOME_KEYFRAME_NEMO_CLIP_MEAN,
     "max_speed": 0.9, "gait_frequency_hz": 0.74, "hip_roll_limit": 0.065, "nemov3_pd": True},
    {"dataset_root": PINOV1_AMP_DATASET_ROOT_STRAIGHT},
)
# Exp 18 + no leaning on a mechanical stop (Exp 19, user 2026-11-04: "先训练一个保守点 前后交替但是不抬脚的"): Exp 18
# held both hip rolls on their inner stops (targets 0.3 rad past them, 12-15 Nm) to stand stiff on the soft PD.
_register(
    "-Straight-Exp33-Nemov3PD-NoStop",
    {"narrow_stance": True, "foot_frame_contact": True, "default_keyframe": HOME_KEYFRAME_NEMO_CLIP_MEAN,
     "max_speed": 0.9, "gait_frequency_hz": 0.74, "hip_roll_limit": 0.065, "nemov3_pd": True, "target_limit_weight": -10.0},
    {"dataset_root": PINOV1_AMP_DATASET_ROOT_STRAIGHT},
)
# Exp 19 + front wheels down (Exp 20, user 2026-11-04: "从Exp 19续1000轮加前轮贴地惩罚"): Exp 19 raised the front wheel
# (toe) 10-22 % of the time at 0.6-0.9 m/s. Same discriminator input as Exp 19 -- resume it.
_register(
    "-Straight-Exp33-Nemov3PD-NoStop-FrontDown",
    {"narrow_stance": True, "foot_frame_contact": True, "default_keyframe": HOME_KEYFRAME_NEMO_CLIP_MEAN,
     "max_speed": 0.9, "gait_frequency_hz": 0.74, "hip_roll_limit": 0.065, "nemov3_pd": True, "target_limit_weight": -10.0,
     "front_down": True},
    {"dataset_root": PINOV1_AMP_DATASET_ROOT_STRAIGHT},
)
# Exp 20 + the hip yaws held at the default while standing, 0.1 rad tolerance (Exp 21, user 2026-11-04: "选1 从Exp
# 19@1200续训800轮 容忍0.1rad"): Exp 20 stood with its hip yaws spread +-0.4-0.5 rad, different per robot.
_register(
    "-Straight-Exp33-Nemov3PD-NoStop-FrontDown-StandYaw",
    {"narrow_stance": True, "foot_frame_contact": True, "default_keyframe": HOME_KEYFRAME_NEMO_CLIP_MEAN,
     "max_speed": 0.9, "gait_frequency_hz": 0.74, "hip_roll_limit": 0.065, "nemov3_pd": True, "target_limit_weight": -10.0,
     "front_down": True, "standing_hip_yaw": (-20.0, 0.1)},
    {"dataset_root": PINOV1_AMP_DATASET_ROOT_STRAIGHT},
)
# Exp 21 + no gait clock while standing (Exp 22, user 2026-11-04: "选1 从Exp 19@1200续训800轮"): Exp 21's standing legs
# kept a slow fore-aft alternation at the clock's 0.75 Hz.
_register(
    "-Straight-Exp33-Nemov3PD-NoStop-FrontDown-StandYaw-StandClockOff",
    {"narrow_stance": True, "foot_frame_contact": True, "default_keyframe": HOME_KEYFRAME_NEMO_CLIP_MEAN,
     "max_speed": 0.9, "gait_frequency_hz": 0.74, "hip_roll_limit": 0.065, "nemov3_pd": True, "target_limit_weight": -10.0,
     "front_down": True, "standing_hip_yaw": (-20.0, 0.1), "standing_clock_off": True},
    {"dataset_root": PINOV1_AMP_DATASET_ROOT_STRAIGHT},
)
# Exp 22 with the standing hip-yaw hold tightened: tolerance 0.1 -> 0.05 rad, weight -20 -> -40 (Exp 23, user 2026-11-05:
# "静止时 hip yaw 超过0.05就惩罚 惩罚力度增大两倍").
_register(
    "-Straight-Exp33-Nemov3PD-NoStop-FrontDown-StandYaw005-StandClockOff",
    {"narrow_stance": True, "foot_frame_contact": True, "default_keyframe": HOME_KEYFRAME_NEMO_CLIP_MEAN,
     "max_speed": 0.9, "gait_frequency_hz": 0.74, "hip_roll_limit": 0.065, "nemov3_pd": True, "target_limit_weight": -10.0,
     "front_down": True, "standing_hip_yaw": (-40.0, 0.05), "standing_clock_off": True},
    {"dataset_root": PINOV1_AMP_DATASET_ROOT_STRAIGHT},
)
# Exp 23 with the forward command capped at 0.7 m/s (Exp 24, user 2026-11-05: "c训练到0.7就好"): within the clips'
# speed (top 0.72 m/s); Exp 22's gait at 0.9 m/s was bistable (one leg leading).
_register(
    "-Straight-Exp33-Nemov3PD-NoStop-FrontDown-StandYaw005-StandClockOff-Max07",
    {"narrow_stance": True, "foot_frame_contact": True, "default_keyframe": HOME_KEYFRAME_NEMO_CLIP_MEAN,
     "max_speed": 0.7, "gait_frequency_hz": 0.74, "hip_roll_limit": 0.065, "nemov3_pd": True, "target_limit_weight": -10.0,
     "front_down": True, "standing_hip_yaw": (-40.0, 0.05), "standing_clock_off": True},
    {"dataset_root": PINOV1_AMP_DATASET_ROOT_STRAIGHT},
)
# Exp 23 + the default pose held while standing for the joints behind its 0.126 rad pose deviation -- hip pitch, ankle
# pitch, knee, shoulder pitch, elbow; tolerance 0.05 rad, weight -40 as the hip yaws (Exp 25, user 2026-11-05: "选1 从
# Exp 19@1200续训800轮 容忍度仿照之前用0.05rad"). Exp 23 stood in a random fore-aft split (|hip / ankle / shoulder pitch
# - default| 0.28) with the elbows 0.23 more and the knees 0.11 less bent.
_register(
    "-Straight-Exp33-Nemov3PD-NoStop-FrontDown-StandYaw005-StandClockOff-StandPose005",
    {"narrow_stance": True, "foot_frame_contact": True, "default_keyframe": HOME_KEYFRAME_NEMO_CLIP_MEAN,
     "max_speed": 0.9, "gait_frequency_hz": 0.74, "hip_roll_limit": 0.065, "nemov3_pd": True, "target_limit_weight": -10.0,
     "front_down": True, "standing_hip_yaw": (-40.0, 0.05), "standing_clock_off": True,
     "standing_pose_hold": (-40.0, 0.05, (r".*_hip_pitch_joint", r".*_ankle_pitch_joint", r".*_knee_joint",
                                          r".*_shoulder_pitch_joint", r".*_elbow_joint"))},
    {"dataset_root": PINOV1_AMP_DATASET_ROOT_STRAIGHT},
)
# Exp 25 + the speed-conditioned discriminator (fast-clip label; Exp 26, user 2026-11-05: "选1 从Exp 25@2000续训1000轮
# 目标为强化高速时的大摆手风格"). Exp 25 at 0.9 m/s: ~31 % of the robots swung the arms like Fast06, the rest like
# Skating01 / 02. The discriminator input grows by one per frame: resume with --agent.reset-discriminator True.
_register(
    "-Straight-Exp33-Nemov3PD-NoStop-FrontDown-StandYaw005-StandClockOff-StandPose005-FastLabel",
    {"narrow_stance": True, "foot_frame_contact": True, "default_keyframe": HOME_KEYFRAME_NEMO_CLIP_MEAN,
     "max_speed": 0.9, "gait_frequency_hz": 0.74, "hip_roll_limit": 0.065, "nemov3_pd": True, "target_limit_weight": -10.0,
     "front_down": True, "standing_hip_yaw": (-40.0, 0.05), "standing_clock_off": True,
     "standing_pose_hold": (-40.0, 0.05, (r".*_hip_pitch_joint", r".*_ankle_pitch_joint", r".*_knee_joint",
                                          r".*_shoulder_pitch_joint", r".*_elbow_joint"))},
    {"dataset_root": PINOV1_AMP_DATASET_ROOT_STRAIGHT, "amp_extra_obs_fn": PINOV1_FAST_LABEL_EXPERT_FN},
)
# Exp 26 with a continuous speed condition instead of the 0 / 1 label at 0.6 m/s (Exp 27, user 2026-11-05: "试试看 不要
# 影响之前的exp 进行exp27 训练方式 和exp26相同"): v = forward command / 0.9 m/s; each expert sample takes v from a
# policy sample and comes from Fast06 with probability v, from Skating01 / 02 otherwise (arXiv 2605.18611). Exp 26 had
# a fixed stagger at 0.5-0.6 m/s, right below its switch point.
_register(
    "-Straight-Exp33-Nemov3PD-NoStop-FrontDown-StandYaw005-StandClockOff-StandPose005-SoftSpeed",
    {"narrow_stance": True, "foot_frame_contact": True, "default_keyframe": HOME_KEYFRAME_NEMO_CLIP_MEAN,
     "max_speed": 0.9, "gait_frequency_hz": 0.74, "hip_roll_limit": 0.065, "nemov3_pd": True, "target_limit_weight": -10.0,
     "front_down": True, "standing_hip_yaw": (-40.0, 0.05), "standing_clock_off": True,
     "standing_pose_hold": (-40.0, 0.05, (r".*_hip_pitch_joint", r".*_ankle_pitch_joint", r".*_knee_joint",
                                          r".*_shoulder_pitch_joint", r".*_elbow_joint"))},
    {"dataset_root": PINOV1_AMP_DATASET_ROOT_STRAIGHT, "amp_extra_obs_fn": PINOV1_FAST_LABEL_EXPERT_FN,
     "amp_soft_condition": True},
)
# Exp 25's recipe + turning, from scratch (Exp 28, user 2026-11-05: "下一步我们加入转弯的训练按exp25配方 从头训练 并
# 模仿nemov3 首次训练出好的转弯步态的方式进行新训练exp28"): nemov3 Exp 41 = its straight recipe + yaw rate +-1.3 rad/s.
_register(
    "-Straight-Exp33-Nemov3PD-NoStop-FrontDown-StandYaw005-StandClockOff-StandPose005-Turn",
    {"narrow_stance": True, "foot_frame_contact": True, "default_keyframe": HOME_KEYFRAME_NEMO_CLIP_MEAN,
     "max_speed": 0.9, "gait_frequency_hz": 0.74, "hip_roll_limit": 0.065, "nemov3_pd": True, "target_limit_weight": -10.0,
     "front_down": True, "standing_hip_yaw": (-40.0, 0.05), "standing_clock_off": True,
     "standing_pose_hold": (-40.0, 0.05, (r".*_hip_pitch_joint", r".*_ankle_pitch_joint", r".*_knee_joint",
                                          r".*_shoulder_pitch_joint", r".*_elbow_joint")),
     "turn": True},
    {"dataset_root": PINOV1_AMP_DATASET_ROOT_STRAIGHT},
)
# Exp 28 with the mirror loss 10 -> 50 (nemov3 Exp 42) and the yaw-rate command capped at +-0.9 rad/s, from scratch
# (Exp 29, user 2026-11-05: "镜像损失系数从 10 提到 50 从头训练一次 这次转弯指令最大给到0.9就好"). Exp 28: left / right
# turns not mirror images, 0.3 m/s rhythm skewed, sharp turns at low speed fell.
_register(
    "-Straight-Exp33-Nemov3PD-NoStop-FrontDown-StandYaw005-StandClockOff-StandPose005-Turn09-Mirror50",
    {"narrow_stance": True, "foot_frame_contact": True, "default_keyframe": HOME_KEYFRAME_NEMO_CLIP_MEAN,
     "max_speed": 0.9, "gait_frequency_hz": 0.74, "hip_roll_limit": 0.065, "nemov3_pd": True, "target_limit_weight": -10.0,
     "front_down": True, "standing_hip_yaw": (-40.0, 0.05), "standing_clock_off": True,
     "standing_pose_hold": (-40.0, 0.05, (r".*_hip_pitch_joint", r".*_ankle_pitch_joint", r".*_knee_joint",
                                          r".*_shoulder_pitch_joint", r".*_elbow_joint")),
     "turn": True, "turn_yaw_rate_max": 0.9},
    {"dataset_root": PINOV1_AMP_DATASET_ROOT_STRAIGHT, "mirror_loss_coeff": 50.0},
)
# Exp 29 + backward commands [-0.5, 0] m/s, no new reward, resumed from Exp 29@2000 (Exp 30, user 2026-11-05: "仿照nemov3
# 把后退指令加进来"; "用-0.5"): nemov3 Exp 90 = Exp 88 + vx in [-0.5, 1.3], resumed for 1000 iterations.
_register(
    "-Straight-Exp33-Nemov3PD-NoStop-FrontDown-StandYaw005-StandClockOff-StandPose005-Turn09-Mirror50-Backward",
    {"narrow_stance": True, "foot_frame_contact": True, "default_keyframe": HOME_KEYFRAME_NEMO_CLIP_MEAN,
     "max_speed": 0.9, "gait_frequency_hz": 0.74, "hip_roll_limit": 0.065, "nemov3_pd": True, "target_limit_weight": -10.0,
     "front_down": True, "standing_hip_yaw": (-40.0, 0.05), "standing_clock_off": True,
     "standing_pose_hold": (-40.0, 0.05, (r".*_hip_pitch_joint", r".*_ankle_pitch_joint", r".*_knee_joint",
                                          r".*_shoulder_pitch_joint", r".*_elbow_joint")),
     "turn": True, "turn_yaw_rate_max": 0.9, "backward": 0.5},
    {"dataset_root": PINOV1_AMP_DATASET_ROOT_STRAIGHT, "mirror_loss_coeff": 50.0},
)
# Exp 30 + a hip-roll adduction penalty only while driving backward (weight -100 as the spread penalty, limit 0 rad),
# resumed from Exp 29@2000 like Exp 30 (Exp 31, user 2026-11-05: "后退时脚微微开不要夹顶到限位 摆动可以先保留"; "只在
# 后退时生效"). Exp 30 at -0.5 m/s held both hips at the inward stop (-0.08 rad), rear wheels 14 cm apart.
_register(
    "-Straight-Exp33-Nemov3PD-NoStop-FrontDown-StandYaw005-StandClockOff-StandPose005-Turn09-Mirror50-Backward-BackRollIn",
    {"narrow_stance": True, "foot_frame_contact": True, "default_keyframe": HOME_KEYFRAME_NEMO_CLIP_MEAN,
     "max_speed": 0.9, "gait_frequency_hz": 0.74, "hip_roll_limit": 0.065, "nemov3_pd": True, "target_limit_weight": -10.0,
     "front_down": True, "standing_hip_yaw": (-40.0, 0.05), "standing_clock_off": True,
     "standing_pose_hold": (-40.0, 0.05, (r".*_hip_pitch_joint", r".*_ankle_pitch_joint", r".*_knee_joint",
                                          r".*_shoulder_pitch_joint", r".*_elbow_joint")),
     "turn": True, "turn_yaw_rate_max": 0.9, "backward": 0.5, "backward_hip_roll_inward": (-100.0, 0.0)},
    {"dataset_root": PINOV1_AMP_DATASET_ROOT_STRAIGHT, "mirror_loss_coeff": 50.0},
)
# Exp 31 with the deployment PD: every arm joint kp 18 / kd 0.8 (legs, ankles, torso 50 / 0.5, rear wheels kd 0.5 as
# before) (Exp 32, user 2026-11-05: "先换部署 PD"; "肩 pitch / roll 肩 yaw、肘、腕 统一换成18/0.8").
_register(
    "-Straight-Exp33-Nemov3PD-NoStop-FrontDown-StandYaw005-StandClockOff-StandPose005-Turn09-Mirror50-Backward-BackRollIn-DeployPD",
    {"narrow_stance": True, "foot_frame_contact": True, "default_keyframe": HOME_KEYFRAME_NEMO_CLIP_MEAN,
     "max_speed": 0.9, "gait_frequency_hz": 0.74, "hip_roll_limit": 0.065, "nemov3_pd": True, "deploy_pd": True,
     "target_limit_weight": -10.0, "front_down": True, "standing_hip_yaw": (-40.0, 0.05), "standing_clock_off": True,
     "standing_pose_hold": (-40.0, 0.05, (r".*_hip_pitch_joint", r".*_ankle_pitch_joint", r".*_knee_joint",
                                          r".*_shoulder_pitch_joint", r".*_elbow_joint")),
     "turn": True, "turn_yaw_rate_max": 0.9, "backward": 0.5, "backward_hip_roll_inward": (-100.0, 0.0)},
    {"dataset_root": PINOV1_AMP_DATASET_ROOT_STRAIGHT, "mirror_loss_coeff": 50.0},
)
# Exp 32 with every arm joint at kp 25 / kd 1.0 (shoulders ~6 Hz, damping ratio ~0.75) and the rear-wheel speed target
# clipped to 23 rad/s (5016 at the 48 V bus), from scratch (Exp 33, user 2026-11-05: "从头使用25/1 转速限制也加上去就用
# 23rad"). Exp 32's arms (18 / 0.8) looked too soft; its rear wheels reached p99 ~25 rad/s at 0.9 m/s (docs/pinov1/pinov1_spec.md).
_register(
    "-Straight-Exp33-Nemov3PD-NoStop-FrontDown-StandYaw005-StandClockOff-StandPose005-Turn09-Mirror50-Backward-BackRollIn-Arm25-Wheel23",
    {"narrow_stance": True, "foot_frame_contact": True, "default_keyframe": HOME_KEYFRAME_NEMO_CLIP_MEAN,
     "max_speed": 0.9, "gait_frequency_hz": 0.74, "hip_roll_limit": 0.065, "nemov3_pd": True, "arm_pd": (25.0, 1.0),
     "wheel_speed_limit": PINOV1_WHEEL_SPEED_LIMIT_48V,
     "target_limit_weight": -10.0, "front_down": True, "standing_hip_yaw": (-40.0, 0.05), "standing_clock_off": True,
     "standing_pose_hold": (-40.0, 0.05, (r".*_hip_pitch_joint", r".*_ankle_pitch_joint", r".*_knee_joint",
                                          r".*_shoulder_pitch_joint", r".*_elbow_joint")),
     "turn": True, "turn_yaw_rate_max": 0.9, "backward": 0.5, "backward_hip_roll_inward": (-100.0, 0.0)},
    {"dataset_root": PINOV1_AMP_DATASET_ROOT_STRAIGHT, "mirror_loss_coeff": 50.0},
)
# Exp 33 + gait clock off while driving backward + rear-wheel velocity gain 0.5 -> 1.0, randomized x0.8-1.2 like the
# joints' kd; resumed from Exp 33@3000 (Exp 34, user 2026-11-06: "后退关掉时钟 轮子kd…"; "a. 另外随机化也加在轮子上 kd先用
# 1.0 试试看"). Exp 33 swung its legs backward like forward (clock-driven) and did not turn in place (no wheel differential).
_register(
    "-Straight-Exp33-Nemov3PD-NoStop-FrontDown-StandYaw005-StandClockOff-StandPose005-Turn09-Mirror50-Backward-BackRollIn-Arm25-Wheel23-BackClockOff-WheelKv1",
    {"narrow_stance": True, "foot_frame_contact": True, "default_keyframe": HOME_KEYFRAME_NEMO_CLIP_MEAN,
     "max_speed": 0.9, "gait_frequency_hz": 0.74, "hip_roll_limit": 0.065, "nemov3_pd": True, "arm_pd": (25.0, 1.0),
     "wheel_speed_limit": PINOV1_WHEEL_SPEED_LIMIT_48V, "wheel_kv": 1.0, "wheel_kv_randomization": True,
     "backward_clock_off": True,
     "target_limit_weight": -10.0, "front_down": True, "standing_hip_yaw": (-40.0, 0.05), "standing_clock_off": True,
     "standing_pose_hold": (-40.0, 0.05, (r".*_hip_pitch_joint", r".*_ankle_pitch_joint", r".*_knee_joint",
                                          r".*_shoulder_pitch_joint", r".*_elbow_joint")),
     "turn": True, "turn_yaw_rate_max": 0.9, "backward": 0.5, "backward_hip_roll_inward": (-100.0, 0.0)},
    {"dataset_root": PINOV1_AMP_DATASET_ROOT_STRAIGHT, "mirror_loss_coeff": 50.0},
)
# Exp 34 + the hip-roll adduction penalty also while turning (weight -100, limit 0 rad as before), resumed from Exp 34@4000
# (Exp 35, user 2026-11-06: "exp34配方＋做1 产生exp35"). Exp 34 rolled the hips in when backing up with a turn and in place.
_register(
    "-Straight-Exp33-Nemov3PD-NoStop-FrontDown-StandYaw005-StandClockOff-StandPose005-Turn09-Mirror50-Backward-BackRollIn-Arm25-Wheel23-BackClockOff-WheelKv1-TurnRollIn",
    {"narrow_stance": True, "foot_frame_contact": True, "default_keyframe": HOME_KEYFRAME_NEMO_CLIP_MEAN,
     "max_speed": 0.9, "gait_frequency_hz": 0.74, "hip_roll_limit": 0.065, "nemov3_pd": True, "arm_pd": (25.0, 1.0),
     "wheel_speed_limit": PINOV1_WHEEL_SPEED_LIMIT_48V, "wheel_kv": 1.0, "wheel_kv_randomization": True,
     "backward_clock_off": True, "hip_roll_inward_turning": True,
     "target_limit_weight": -10.0, "front_down": True, "standing_hip_yaw": (-40.0, 0.05), "standing_clock_off": True,
     "standing_pose_hold": (-40.0, 0.05, (r".*_hip_pitch_joint", r".*_ankle_pitch_joint", r".*_knee_joint",
                                          r".*_shoulder_pitch_joint", r".*_elbow_joint")),
     "turn": True, "turn_yaw_rate_max": 0.9, "backward": 0.5, "backward_hip_roll_inward": (-100.0, 0.0)},
    {"dataset_root": PINOV1_AMP_DATASET_ROOT_STRAIGHT, "mirror_loss_coeff": 50.0},
)
# Exp 32's recipe with the arms at kp 25 / kd 1.0, the rear-wheel speed target clipped to 25 rad/s and the rear-wheel
# velocity gain randomized x0.8-1.2 (gain itself kept at 0.5), resumed from Exp 32@2000 (Exp 36, user 2026-11-06: "照你建议
# 训练exp36 续训1000轮 轮速上限 设定为25 rad/s"). Exp 32 still turned in place (0.48-0.53 rad/s at +-0.9) and did not swing
# its legs backward; Exp 33 (arms 25 / 1.0 + 23 rad/s clip, from scratch) lost both -- this separates the two changes from
# a fresh start.
_register(
    "-Straight-Exp33-Nemov3PD-NoStop-FrontDown-StandYaw005-StandClockOff-StandPose005-Turn09-Mirror50-Backward-BackRollIn-Arm25-Wheel25-WheelKvRand",
    {"narrow_stance": True, "foot_frame_contact": True, "default_keyframe": HOME_KEYFRAME_NEMO_CLIP_MEAN,
     "max_speed": 0.9, "gait_frequency_hz": 0.74, "hip_roll_limit": 0.065, "nemov3_pd": True, "arm_pd": (25.0, 1.0),
     "wheel_speed_limit": 25.0, "wheel_kv_randomization": True,
     "target_limit_weight": -10.0, "front_down": True, "standing_hip_yaw": (-40.0, 0.05), "standing_clock_off": True,
     "standing_pose_hold": (-40.0, 0.05, (r".*_hip_pitch_joint", r".*_ankle_pitch_joint", r".*_knee_joint",
                                          r".*_shoulder_pitch_joint", r".*_elbow_joint")),
     "turn": True, "turn_yaw_rate_max": 0.9, "backward": 0.5, "backward_hip_roll_inward": (-100.0, 0.0)},
    {"dataset_root": PINOV1_AMP_DATASET_ROOT_STRAIGHT, "mirror_loss_coeff": 50.0},
)
# Exp 36 / 37's task without the rear-wheel velocity-gain randomization (Exp 38, user 2026-11-06: "把轮子的pd 随机化拿掉
# 产生 exp38 然后对比exp31 37 38"), trained from scratch for 3000 iterations like Exp 37.
_register(
    "-Straight-Exp33-Nemov3PD-NoStop-FrontDown-StandYaw005-StandClockOff-StandPose005-Turn09-Mirror50-Backward-BackRollIn-Arm25-Wheel25",
    {"narrow_stance": True, "foot_frame_contact": True, "default_keyframe": HOME_KEYFRAME_NEMO_CLIP_MEAN,
     "max_speed": 0.9, "gait_frequency_hz": 0.74, "hip_roll_limit": 0.065, "nemov3_pd": True, "arm_pd": (25.0, 1.0),
     "wheel_speed_limit": 25.0,
     "target_limit_weight": -10.0, "front_down": True, "standing_hip_yaw": (-40.0, 0.05), "standing_clock_off": True,
     "standing_pose_hold": (-40.0, 0.05, (r".*_hip_pitch_joint", r".*_ankle_pitch_joint", r".*_knee_joint",
                                          r".*_shoulder_pitch_joint", r".*_elbow_joint")),
     "turn": True, "turn_yaw_rate_max": 0.9, "backward": 0.5, "backward_hip_roll_inward": (-100.0, 0.0)},
    {"dataset_root": PINOV1_AMP_DATASET_ROOT_STRAIGHT, "mirror_loss_coeff": 50.0},
)
# Autonomous loop (user 2026-11-06 night: match / beat Exp 31 with the arms at 25 / 1.0). Stage 1 of a two-stage run like
# Exp 29 -> 31: Exp 29's recipe (turning +-0.9, mirror loss 50, no backward) with the arms 25 / 1.0, the 25 rad/s wheel clip
# and the wheel-gain randomization; stage 2 resumes it on the Exp 36 / 37 task (backward + backward roll-in penalty).
_register(
    "-Straight-Exp33-Nemov3PD-NoStop-FrontDown-StandYaw005-StandClockOff-StandPose005-Turn09-Mirror50-Arm25-Wheel25-WheelKvRand",
    {"narrow_stance": True, "foot_frame_contact": True, "default_keyframe": HOME_KEYFRAME_NEMO_CLIP_MEAN,
     "max_speed": 0.9, "gait_frequency_hz": 0.74, "hip_roll_limit": 0.065, "nemov3_pd": True, "arm_pd": (25.0, 1.0),
     "wheel_speed_limit": 25.0, "wheel_kv_randomization": True,
     "target_limit_weight": -10.0, "front_down": True, "standing_hip_yaw": (-40.0, 0.05), "standing_clock_off": True,
     "standing_pose_hold": (-40.0, 0.05, (r".*_hip_pitch_joint", r".*_ankle_pitch_joint", r".*_knee_joint",
                                          r".*_shoulder_pitch_joint", r".*_elbow_joint")),
     "turn": True, "turn_yaw_rate_max": 0.9},
    {"dataset_root": PINOV1_AMP_DATASET_ROOT_STRAIGHT, "mirror_loss_coeff": 50.0},
)
# Exp 36's task + Exp 35's fixes (gait clock off while backing up, hip roll-in penalty also while turning).
_register(
    "-Straight-Exp33-Nemov3PD-NoStop-FrontDown-StandYaw005-StandClockOff-StandPose005-Turn09-Mirror50-Backward-BackRollIn-Arm25-Wheel25-WheelKvRand-BackClockOff-TurnRollIn",
    {"narrow_stance": True, "foot_frame_contact": True, "default_keyframe": HOME_KEYFRAME_NEMO_CLIP_MEAN,
     "max_speed": 0.9, "gait_frequency_hz": 0.74, "hip_roll_limit": 0.065, "nemov3_pd": True, "arm_pd": (25.0, 1.0),
     "wheel_speed_limit": 25.0, "wheel_kv_randomization": True, "backward_clock_off": True, "hip_roll_inward_turning": True,
     "target_limit_weight": -10.0, "front_down": True, "standing_hip_yaw": (-40.0, 0.05), "standing_clock_off": True,
     "standing_pose_hold": (-40.0, 0.05, (r".*_hip_pitch_joint", r".*_ankle_pitch_joint", r".*_knee_joint",
                                          r".*_shoulder_pitch_joint", r".*_elbow_joint")),
     "turn": True, "turn_yaw_rate_max": 0.9, "backward": 0.5, "backward_hip_roll_inward": (-100.0, 0.0)},
    {"dataset_root": PINOV1_AMP_DATASET_ROOT_STRAIGHT, "mirror_loss_coeff": 50.0},
)
# Exp 35 + 15 % of the environments turning in place (forward 0, yaw rate +-0.9), resumed from Exp 35@5000 (Exp 41,
# autonomous loop attempt 2). Exp 35 met every goal except turning in place (0.21 / 0.37 rad/s vs Exp 31's 0.31 / 0.52).
_register(
    "-Straight-Exp33-Nemov3PD-NoStop-FrontDown-StandYaw005-StandClockOff-StandPose005-Turn09-Mirror50-Backward-BackRollIn-Arm25-Wheel23-BackClockOff-WheelKv1-TurnRollIn-InPlace15",
    {"narrow_stance": True, "foot_frame_contact": True, "default_keyframe": HOME_KEYFRAME_NEMO_CLIP_MEAN,
     "max_speed": 0.9, "gait_frequency_hz": 0.74, "hip_roll_limit": 0.065, "nemov3_pd": True, "arm_pd": (25.0, 1.0),
     "wheel_speed_limit": PINOV1_WHEEL_SPEED_LIMIT_48V, "wheel_kv": 1.0, "wheel_kv_randomization": True,
     "backward_clock_off": True, "hip_roll_inward_turning": True, "turn_in_place_envs": 0.15,
     "target_limit_weight": -10.0, "front_down": True, "standing_hip_yaw": (-40.0, 0.05), "standing_clock_off": True,
     "standing_pose_hold": (-40.0, 0.05, (r".*_hip_pitch_joint", r".*_ankle_pitch_joint", r".*_knee_joint",
                                          r".*_shoulder_pitch_joint", r".*_elbow_joint")),
     "turn": True, "turn_yaw_rate_max": 0.9, "backward": 0.5, "backward_hip_roll_inward": (-100.0, 0.0)},
    {"dataset_root": PINOV1_AMP_DATASET_ROOT_STRAIGHT, "mirror_loss_coeff": 50.0},
)
# Exp 36 + Exp 35's fixes (gait clock off while backing up, hip roll-in penalty also while turning) + the roll-in penalty
# also while standing; resumed from Exp 36@3000 (Exp 42, autonomous loop attempt 3). Exp 36 turns in place like Exp 31
# (symmetric wheel differential, toes out) but swung its legs backing up and rolled its hips in when standing / turning.
_register(
    "-Straight-Exp33-Nemov3PD-NoStop-FrontDown-StandYaw005-StandClockOff-StandPose005-Turn09-Mirror50-Backward-BackRollIn-Arm25-Wheel25-WheelKvRand-BackClockOff-TurnRollIn-StandRollIn",
    {"narrow_stance": True, "foot_frame_contact": True, "default_keyframe": HOME_KEYFRAME_NEMO_CLIP_MEAN,
     "max_speed": 0.9, "gait_frequency_hz": 0.74, "hip_roll_limit": 0.065, "nemov3_pd": True, "arm_pd": (25.0, 1.0),
     "wheel_speed_limit": 25.0, "wheel_kv_randomization": True, "backward_clock_off": True, "hip_roll_inward_turning": True,
     "hip_roll_inward_standing": True,
     "target_limit_weight": -10.0, "front_down": True, "standing_hip_yaw": (-40.0, 0.05), "standing_clock_off": True,
     "standing_pose_hold": (-40.0, 0.05, (r".*_hip_pitch_joint", r".*_ankle_pitch_joint", r".*_knee_joint",
                                          r".*_shoulder_pitch_joint", r".*_elbow_joint")),
     "turn": True, "turn_yaw_rate_max": 0.9, "backward": 0.5, "backward_hip_roll_inward": (-100.0, 0.0)},
    {"dataset_root": PINOV1_AMP_DATASET_ROOT_STRAIGHT, "mirror_loss_coeff": 50.0},
)
# Exp 42 + a fall (termination) penalty -100 and the yaw-rate tracking weight 2.0 -> 3.0, resumed from Exp 42@4000
# (Exp 44, autonomous loop attempt 5). Exp 42@4000 fell now and then (4 / 832 in the turn grid); training it further
# without changes (Exp 43) removed the falls but turned in place ever slower (0.60 -> 0.39 rad/s at +-0.9).
_register(
    "-Straight-Exp33-Nemov3PD-NoStop-FrontDown-StandYaw005-StandClockOff-StandPose005-Turn09-Mirror50-Backward-BackRollIn-Arm25-Wheel25-WheelKvRand-BackClockOff-TurnRollIn-StandRollIn-FallPen-Yaw3",
    {"narrow_stance": True, "foot_frame_contact": True, "default_keyframe": HOME_KEYFRAME_NEMO_CLIP_MEAN,
     "max_speed": 0.9, "gait_frequency_hz": 0.74, "hip_roll_limit": 0.065, "nemov3_pd": True, "arm_pd": (25.0, 1.0),
     "wheel_speed_limit": 25.0, "wheel_kv_randomization": True, "backward_clock_off": True, "hip_roll_inward_turning": True,
     "hip_roll_inward_standing": True, "termination_penalty": -100.0, "yaw_track_weight": 3.0,
     "target_limit_weight": -10.0, "front_down": True, "standing_hip_yaw": (-40.0, 0.05), "standing_clock_off": True,
     "standing_pose_hold": (-40.0, 0.05, (r".*_hip_pitch_joint", r".*_ankle_pitch_joint", r".*_knee_joint",
                                          r".*_shoulder_pitch_joint", r".*_elbow_joint")),
     "turn": True, "turn_yaw_rate_max": 0.9, "backward": 0.5, "backward_hip_roll_inward": (-100.0, 0.0)},
    {"dataset_root": PINOV1_AMP_DATASET_ROOT_STRAIGHT, "mirror_loss_coeff": 50.0},
)
# Exp 31's recipe + arms 25 / 1.0, the 25 rad/s wheel clip, the wheel-gain randomization, the gait clock off while backing
# up and while turning in place, the hip roll-in penalty also while turning / standing; trained from scratch for 5000
# iterations (Exp 45, user 2026-11-06: "相比exp31 只采用 #1 #2 #3 #4(不只后退原地转也关闭) #5 。#6 #7 好像没啥区别 不用。从头
# 训练5000轮"). No fall penalty, yaw tracking weight 2.0 as Exp 31.
_register(
    "-Straight-Exp33-Nemov3PD-NoStop-FrontDown-StandYaw005-StandClockOff-StandPose005-Turn09-Mirror50-Backward-BackRollIn-Arm25-Wheel25-WheelKvRand-BackClockOff-InPlaceClockOff-TurnRollIn-StandRollIn",
    {"narrow_stance": True, "foot_frame_contact": True, "default_keyframe": HOME_KEYFRAME_NEMO_CLIP_MEAN,
     "max_speed": 0.9, "gait_frequency_hz": 0.74, "hip_roll_limit": 0.065, "nemov3_pd": True, "arm_pd": (25.0, 1.0),
     "wheel_speed_limit": 25.0, "wheel_kv_randomization": True, "backward_clock_off": True, "in_place_clock_off": True,
     "hip_roll_inward_turning": True, "hip_roll_inward_standing": True,
     "target_limit_weight": -10.0, "front_down": True, "standing_hip_yaw": (-40.0, 0.05), "standing_clock_off": True,
     "standing_pose_hold": (-40.0, 0.05, (r".*_hip_pitch_joint", r".*_ankle_pitch_joint", r".*_knee_joint",
                                          r".*_shoulder_pitch_joint", r".*_elbow_joint")),
     "turn": True, "turn_yaw_rate_max": 0.9, "backward": 0.5, "backward_hip_roll_inward": (-100.0, 0.0)},
    {"dataset_root": PINOV1_AMP_DATASET_ROOT_STRAIGHT, "mirror_loss_coeff": 50.0},
)
# Mean-reward ablation (Exp 46a, 46b, ..., user 2026-11-06: "查看啥原因导致exp45 mean reward 成长很慢 exp29 exp32都是从头训练成长
# 很快 ... 训练500轮就可以查看"): Exp 32's task (fast: mean reward 4.6 at iteration 300) + one of Exp 45's changes each.
# Exp 46a: + arms 25 / 1.0 (instead of 18 / 0.8).
_register(
    "-Straight-Exp33-Nemov3PD-NoStop-FrontDown-StandYaw005-StandClockOff-StandPose005-Turn09-Mirror50-Backward-BackRollIn-Arm25",
    {"narrow_stance": True, "foot_frame_contact": True, "default_keyframe": HOME_KEYFRAME_NEMO_CLIP_MEAN,
     "max_speed": 0.9, "gait_frequency_hz": 0.74, "hip_roll_limit": 0.065, "nemov3_pd": True, "arm_pd": (25.0, 1.0),
     "target_limit_weight": -10.0, "front_down": True, "standing_hip_yaw": (-40.0, 0.05), "standing_clock_off": True,
     "standing_pose_hold": (-40.0, 0.05, (r".*_hip_pitch_joint", r".*_ankle_pitch_joint", r".*_knee_joint",
                                          r".*_shoulder_pitch_joint", r".*_elbow_joint")),
     "turn": True, "turn_yaw_rate_max": 0.9, "backward": 0.5, "backward_hip_roll_inward": (-100.0, 0.0)},
    {"dataset_root": PINOV1_AMP_DATASET_ROOT_STRAIGHT, "mirror_loss_coeff": 50.0},
)
# Exp 46b: + the 25 rad/s rear-wheel speed clip (arms 18 / 0.8 as Exp 32).
_register(
    "-Straight-Exp33-Nemov3PD-NoStop-FrontDown-StandYaw005-StandClockOff-StandPose005-Turn09-Mirror50-Backward-BackRollIn-DeployPD-Wheel25",
    {"narrow_stance": True, "foot_frame_contact": True, "default_keyframe": HOME_KEYFRAME_NEMO_CLIP_MEAN,
     "max_speed": 0.9, "gait_frequency_hz": 0.74, "hip_roll_limit": 0.065, "nemov3_pd": True, "deploy_pd": True, "wheel_speed_limit": 25.0,
     "target_limit_weight": -10.0, "front_down": True, "standing_hip_yaw": (-40.0, 0.05), "standing_clock_off": True,
     "standing_pose_hold": (-40.0, 0.05, (r".*_hip_pitch_joint", r".*_ankle_pitch_joint", r".*_knee_joint",
                                          r".*_shoulder_pitch_joint", r".*_elbow_joint")),
     "turn": True, "turn_yaw_rate_max": 0.9, "backward": 0.5, "backward_hip_roll_inward": (-100.0, 0.0)},
    {"dataset_root": PINOV1_AMP_DATASET_ROOT_STRAIGHT, "mirror_loss_coeff": 50.0},
)
# Exp 46c: + the rear-wheel gain randomization.
_register(
    "-Straight-Exp33-Nemov3PD-NoStop-FrontDown-StandYaw005-StandClockOff-StandPose005-Turn09-Mirror50-Backward-BackRollIn-DeployPD-WheelKvRand",
    {"narrow_stance": True, "foot_frame_contact": True, "default_keyframe": HOME_KEYFRAME_NEMO_CLIP_MEAN,
     "max_speed": 0.9, "gait_frequency_hz": 0.74, "hip_roll_limit": 0.065, "nemov3_pd": True, "deploy_pd": True, "wheel_kv_randomization": True,
     "target_limit_weight": -10.0, "front_down": True, "standing_hip_yaw": (-40.0, 0.05), "standing_clock_off": True,
     "standing_pose_hold": (-40.0, 0.05, (r".*_hip_pitch_joint", r".*_ankle_pitch_joint", r".*_knee_joint",
                                          r".*_shoulder_pitch_joint", r".*_elbow_joint")),
     "turn": True, "turn_yaw_rate_max": 0.9, "backward": 0.5, "backward_hip_roll_inward": (-100.0, 0.0)},
    {"dataset_root": PINOV1_AMP_DATASET_ROOT_STRAIGHT, "mirror_loss_coeff": 50.0},
)
# Exp 46d: + the gait clock off while backing up / turning in place.
_register(
    "-Straight-Exp33-Nemov3PD-NoStop-FrontDown-StandYaw005-StandClockOff-StandPose005-Turn09-Mirror50-Backward-BackRollIn-DeployPD-BackClockOff-InPlaceClockOff",
    {"narrow_stance": True, "foot_frame_contact": True, "default_keyframe": HOME_KEYFRAME_NEMO_CLIP_MEAN,
     "max_speed": 0.9, "gait_frequency_hz": 0.74, "hip_roll_limit": 0.065, "nemov3_pd": True, "deploy_pd": True, "backward_clock_off": True, "in_place_clock_off": True,
     "target_limit_weight": -10.0, "front_down": True, "standing_hip_yaw": (-40.0, 0.05), "standing_clock_off": True,
     "standing_pose_hold": (-40.0, 0.05, (r".*_hip_pitch_joint", r".*_ankle_pitch_joint", r".*_knee_joint",
                                          r".*_shoulder_pitch_joint", r".*_elbow_joint")),
     "turn": True, "turn_yaw_rate_max": 0.9, "backward": 0.5, "backward_hip_roll_inward": (-100.0, 0.0)},
    {"dataset_root": PINOV1_AMP_DATASET_ROOT_STRAIGHT, "mirror_loss_coeff": 50.0},
)
# Exp 46e: + the hip roll-in penalty also while turning / standing.
_register(
    "-Straight-Exp33-Nemov3PD-NoStop-FrontDown-StandYaw005-StandClockOff-StandPose005-Turn09-Mirror50-Backward-BackRollIn-DeployPD-TurnRollIn-StandRollIn",
    {"narrow_stance": True, "foot_frame_contact": True, "default_keyframe": HOME_KEYFRAME_NEMO_CLIP_MEAN,
     "max_speed": 0.9, "gait_frequency_hz": 0.74, "hip_roll_limit": 0.065, "nemov3_pd": True, "deploy_pd": True, "hip_roll_inward_turning": True, "hip_roll_inward_standing": True,
     "target_limit_weight": -10.0, "front_down": True, "standing_hip_yaw": (-40.0, 0.05), "standing_clock_off": True,
     "standing_pose_hold": (-40.0, 0.05, (r".*_hip_pitch_joint", r".*_ankle_pitch_joint", r".*_knee_joint",
                                          r".*_shoulder_pitch_joint", r".*_elbow_joint")),
     "turn": True, "turn_yaw_rate_max": 0.9, "backward": 0.5, "backward_hip_roll_inward": (-100.0, 0.0)},
    {"dataset_root": PINOV1_AMP_DATASET_ROOT_STRAIGHT, "mirror_loss_coeff": 50.0},
)
# Exp 45 without the rear-wheel speed clip (Exp 47, user 2026-11-06: "轮子别加转速限制 沿用exp45 配方 … 从头训练exp47 3000轮").
# Ablation Exp 46b: the 25 rad/s clip alone slowed the mean reward from 4.6 to 0.6 at iteration 300.
_register(
    "-Straight-Exp33-Nemov3PD-NoStop-FrontDown-StandYaw005-StandClockOff-StandPose005-Turn09-Mirror50-Backward-BackRollIn-Arm25-WheelKvRand-BackClockOff-InPlaceClockOff-TurnRollIn-StandRollIn",
    {"narrow_stance": True, "foot_frame_contact": True, "default_keyframe": HOME_KEYFRAME_NEMO_CLIP_MEAN,
     "max_speed": 0.9, "gait_frequency_hz": 0.74, "hip_roll_limit": 0.065, "nemov3_pd": True, "arm_pd": (25.0, 1.0),
     "wheel_kv_randomization": True, "backward_clock_off": True, "in_place_clock_off": True,
     "hip_roll_inward_turning": True, "hip_roll_inward_standing": True,
     "target_limit_weight": -10.0, "front_down": True, "standing_hip_yaw": (-40.0, 0.05), "standing_clock_off": True,
     "standing_pose_hold": (-40.0, 0.05, (r".*_hip_pitch_joint", r".*_ankle_pitch_joint", r".*_knee_joint",
                                          r".*_shoulder_pitch_joint", r".*_elbow_joint")),
     "turn": True, "turn_yaw_rate_max": 0.9, "backward": 0.5, "backward_hip_roll_inward": (-100.0, 0.0)},
    {"dataset_root": PINOV1_AMP_DATASET_ROOT_STRAIGHT, "mirror_loss_coeff": 50.0},
)
# Exp 47 + actuator delay 0-15 ms (0-3 physics steps, one lag per motor bus per episode: left arm, right arm, waist,
# left leg + wheel, right leg + wheel) and the torso point mass -1..+1 kg (nemov3 Exp 88), resumed from Exp 47@2000
# (Exp 48b; user 2026-11-06: "左手是一个 右手是一组 腰一组 左腿一驵 右腿一组 按照这样分驵 执行器延迟（0~15ms）",
# "再弄个有分组的exp48b 一样从exp47 续训1000轮").
_register(
    "-Straight-Exp33-Nemov3PD-NoStop-FrontDown-StandYaw005-StandClockOff-StandPose005-Turn09-Mirror50-Backward-BackRollIn-Arm25-WheelKvRand-BackClockOff-InPlaceClockOff-TurnRollIn-StandRollIn-BusDelay15-TrunkMass",
    {"narrow_stance": True, "foot_frame_contact": True, "default_keyframe": HOME_KEYFRAME_NEMO_CLIP_MEAN,
     "max_speed": 0.9, "gait_frequency_hz": 0.74, "hip_roll_limit": 0.065, "nemov3_pd": True, "arm_pd": (25.0, 1.0),
     "wheel_kv_randomization": True, "backward_clock_off": True, "in_place_clock_off": True,
     "hip_roll_inward_turning": True, "hip_roll_inward_standing": True,
     "target_limit_weight": -10.0, "front_down": True, "standing_hip_yaw": (-40.0, 0.05), "standing_clock_off": True,
     "standing_pose_hold": (-40.0, 0.05, (r".*_hip_pitch_joint", r".*_ankle_pitch_joint", r".*_knee_joint",
                                          r".*_shoulder_pitch_joint", r".*_elbow_joint")),
     "turn": True, "turn_yaw_rate_max": 0.9, "backward": 0.5, "backward_hip_roll_inward": (-100.0, 0.0),
     "actuator_delay_steps": 3, "trunk_mass_add": (-1.0, 1.0)},
    {"dataset_root": PINOV1_AMP_DATASET_ROOT_STRAIGHT, "mirror_loss_coeff": 50.0},
)
# Exp 48b with one lag for the whole body (Exp 48a; user 2026-11-06: "改成全身共用延迟 不要分组 延迟0~15ms 重新训练exp48a").
_register(
    "-Straight-Exp33-Nemov3PD-NoStop-FrontDown-StandYaw005-StandClockOff-StandPose005-Turn09-Mirror50-Backward-BackRollIn-Arm25-WheelKvRand-BackClockOff-InPlaceClockOff-TurnRollIn-StandRollIn-WholeDelay15-TrunkMass",
    {"narrow_stance": True, "foot_frame_contact": True, "default_keyframe": HOME_KEYFRAME_NEMO_CLIP_MEAN,
     "max_speed": 0.9, "gait_frequency_hz": 0.74, "hip_roll_limit": 0.065, "nemov3_pd": True, "arm_pd": (25.0, 1.0),
     "wheel_kv_randomization": True, "backward_clock_off": True, "in_place_clock_off": True,
     "hip_roll_inward_turning": True, "hip_roll_inward_standing": True,
     "target_limit_weight": -10.0, "front_down": True, "standing_hip_yaw": (-40.0, 0.05), "standing_clock_off": True,
     "standing_pose_hold": (-40.0, 0.05, (r".*_hip_pitch_joint", r".*_ankle_pitch_joint", r".*_knee_joint",
                                          r".*_shoulder_pitch_joint", r".*_elbow_joint")),
     "turn": True, "turn_yaw_rate_max": 0.9, "backward": 0.5, "backward_hip_roll_inward": (-100.0, 0.0),
     "actuator_delay_steps": 3, "actuator_delay_groups": (r".*",), "trunk_mass_add": (-1.0, 1.0)},
    {"dataset_root": PINOV1_AMP_DATASET_ROOT_STRAIGHT, "mirror_loss_coeff": 50.0},
)
# Exp 48a + symmetric toe-in of the hip yaws while turning in place (penalty -40 x [max(0, |q_L + q_R| - 0.05)^2 +
# max(0, 0.15 - (q_R - q_L) / 2)^2]), resumed from Exp 48a@2999 (Exp 49a; user 2026-11-06: "对称而且要内八", "维持原来设计
# 利用48a续训1000轮49a看看").
_register(
    "-Straight-Exp33-Nemov3PD-NoStop-FrontDown-StandYaw005-StandClockOff-StandPose005-Turn09-Mirror50-Backward-BackRollIn-Arm25-WheelKvRand-BackClockOff-InPlaceClockOff-TurnRollIn-StandRollIn-WholeDelay15-TrunkMass-InPlaceToeIn",
    {"narrow_stance": True, "foot_frame_contact": True, "default_keyframe": HOME_KEYFRAME_NEMO_CLIP_MEAN,
     "max_speed": 0.9, "gait_frequency_hz": 0.74, "hip_roll_limit": 0.065, "nemov3_pd": True, "arm_pd": (25.0, 1.0),
     "wheel_kv_randomization": True, "backward_clock_off": True, "in_place_clock_off": True,
     "hip_roll_inward_turning": True, "hip_roll_inward_standing": True,
     "target_limit_weight": -10.0, "front_down": True, "standing_hip_yaw": (-40.0, 0.05), "standing_clock_off": True,
     "standing_pose_hold": (-40.0, 0.05, (r".*_hip_pitch_joint", r".*_ankle_pitch_joint", r".*_knee_joint",
                                          r".*_shoulder_pitch_joint", r".*_elbow_joint")),
     "turn": True, "turn_yaw_rate_max": 0.9, "backward": 0.5, "backward_hip_roll_inward": (-100.0, 0.0),
     "actuator_delay_steps": 3, "actuator_delay_groups": (r".*",), "in_place_toe_in": (-40.0, 0.05, 0.15), "trunk_mass_add": (-1.0, 1.0)},
    {"dataset_root": PINOV1_AMP_DATASET_ROOT_STRAIGHT, "mirror_loss_coeff": 50.0},
)
# Exp 49a with the in-place toe-in penalty at -100 instead of -40, resumed from Exp 49a@3998 (Exp 49a2; user 2026-11-06:
# "从49a续训1000轮 做2 变成exp49a2" -- Exp 49a left |q_L + q_R| at 0.30, above the 0.05 tolerance).
_register(
    "-Straight-Exp33-Nemov3PD-NoStop-FrontDown-StandYaw005-StandClockOff-StandPose005-Turn09-Mirror50-Backward-BackRollIn-Arm25-WheelKvRand-BackClockOff-InPlaceClockOff-TurnRollIn-StandRollIn-WholeDelay15-TrunkMass-InPlaceToeIn100",
    {"narrow_stance": True, "foot_frame_contact": True, "default_keyframe": HOME_KEYFRAME_NEMO_CLIP_MEAN,
     "max_speed": 0.9, "gait_frequency_hz": 0.74, "hip_roll_limit": 0.065, "nemov3_pd": True, "arm_pd": (25.0, 1.0),
     "wheel_kv_randomization": True, "backward_clock_off": True, "in_place_clock_off": True,
     "hip_roll_inward_turning": True, "hip_roll_inward_standing": True,
     "target_limit_weight": -10.0, "front_down": True, "standing_hip_yaw": (-40.0, 0.05), "standing_clock_off": True,
     "standing_pose_hold": (-40.0, 0.05, (r".*_hip_pitch_joint", r".*_ankle_pitch_joint", r".*_knee_joint",
                                          r".*_shoulder_pitch_joint", r".*_elbow_joint")),
     "turn": True, "turn_yaw_rate_max": 0.9, "backward": 0.5, "backward_hip_roll_inward": (-100.0, 0.0),
     "actuator_delay_steps": 3, "actuator_delay_groups": (r".*",), "in_place_toe_in": (-100.0, 0.05, 0.15), "trunk_mass_add": (-1.0, 1.0)},
    {"dataset_root": PINOV1_AMP_DATASET_ROOT_STRAIGHT, "mirror_loss_coeff": 50.0},
)
# Exp 48a + each foot held at >= 0.22 rad toe-in (Exp 31's in-place level, L -0.23 / R +0.22) while turning in place,
# linear hinge x -50; Exp 49a's symmetric toe-in reward dropped. Resumed from Exp 48a@2999 (Exp 49a3; user 2026-11-06:
# "设计有问题 做49a3 强置把脚约束到内八至少预设位置和exp31相同程度 刚刚的奖励放弃 重新写一个奖励函数").
_register(
    "-Straight-Exp33-Nemov3PD-NoStop-FrontDown-StandYaw005-StandClockOff-StandPose005-Turn09-Mirror50-Backward-BackRollIn-Arm25-WheelKvRand-BackClockOff-InPlaceClockOff-TurnRollIn-StandRollIn-WholeDelay15-TrunkMass-InPlaceMinToeIn",
    {"narrow_stance": True, "foot_frame_contact": True, "default_keyframe": HOME_KEYFRAME_NEMO_CLIP_MEAN,
     "max_speed": 0.9, "gait_frequency_hz": 0.74, "hip_roll_limit": 0.065, "nemov3_pd": True, "arm_pd": (25.0, 1.0),
     "wheel_kv_randomization": True, "backward_clock_off": True, "in_place_clock_off": True,
     "hip_roll_inward_turning": True, "hip_roll_inward_standing": True,
     "target_limit_weight": -10.0, "front_down": True, "standing_hip_yaw": (-40.0, 0.05), "standing_clock_off": True,
     "standing_pose_hold": (-40.0, 0.05, (r".*_hip_pitch_joint", r".*_ankle_pitch_joint", r".*_knee_joint",
                                          r".*_shoulder_pitch_joint", r".*_elbow_joint")),
     "turn": True, "turn_yaw_rate_max": 0.9, "backward": 0.5, "backward_hip_roll_inward": (-100.0, 0.0),
     "actuator_delay_steps": 3, "actuator_delay_groups": (r".*",), "in_place_min_toe_in": (-50.0, 0.22), "trunk_mass_add": (-1.0, 1.0)},
    {"dataset_root": PINOV1_AMP_DATASET_ROOT_STRAIGHT, "mirror_loss_coeff": 50.0},
)
# Exp 48a with the gait clock kept on while turning in place (in_place_clock_off dropped; backward still zeroes it),
# resumed from Exp 48a@2999 (Exp 49a4; user 2026-11-07: "从48a续训1000轮 打开原地转时钟 做exp49a4" -- Exp 31 / 42 / 44,
# trained with the clock on in place, turn in place at 0.5-0.6 rad/s with alternating legs; 45 / 47 / 48 / 49a* without at 0.2-0.37).
_register(
    "-Straight-Exp33-Nemov3PD-NoStop-FrontDown-StandYaw005-StandClockOff-StandPose005-Turn09-Mirror50-Backward-BackRollIn-Arm25-WheelKvRand-BackClockOff-TurnRollIn-StandRollIn-WholeDelay15-TrunkMass",
    {"narrow_stance": True, "foot_frame_contact": True, "default_keyframe": HOME_KEYFRAME_NEMO_CLIP_MEAN,
     "max_speed": 0.9, "gait_frequency_hz": 0.74, "hip_roll_limit": 0.065, "nemov3_pd": True, "arm_pd": (25.0, 1.0),
     "wheel_kv_randomization": True, "backward_clock_off": True, "in_place_clock_off": False,
     "hip_roll_inward_turning": True, "hip_roll_inward_standing": True,
     "target_limit_weight": -10.0, "front_down": True, "standing_hip_yaw": (-40.0, 0.05), "standing_clock_off": True,
     "standing_pose_hold": (-40.0, 0.05, (r".*_hip_pitch_joint", r".*_ankle_pitch_joint", r".*_knee_joint",
                                          r".*_shoulder_pitch_joint", r".*_elbow_joint")),
     "turn": True, "turn_yaw_rate_max": 0.9, "backward": 0.5, "backward_hip_roll_inward": (-100.0, 0.0),
     "actuator_delay_steps": 3, "actuator_delay_groups": (r".*",), "trunk_mass_add": (-1.0, 1.0)},
    {"dataset_root": PINOV1_AMP_DATASET_ROOT_STRAIGHT, "mirror_loss_coeff": 50.0},
)
# Exp 50a with the backward hip roll-in penalty x3 (-100 -> -300; the same term also covers turning and standing),
# resumed from Exp 50a@4999 (Exp 51a; user 2026-11-08: "以 50a 为底，先只处理后退加转弯时的髋内收，把后退髋内收惩罚加重，
# 续训 1000 轮" -- Exp 50a rolled the hips in 21 % of the time backing up with a turn).
_register(
    "-Straight-Exp33-Nemov3PD-NoStop-FrontDown-StandYaw005-StandClockOff-StandPose005-Turn09-Mirror50-Backward-BackRollIn-Arm25-WheelKvRand-BackClockOff-TurnRollIn-StandRollIn-WholeDelay15-TrunkMass-BackRollIn300",
    {"narrow_stance": True, "foot_frame_contact": True, "default_keyframe": HOME_KEYFRAME_NEMO_CLIP_MEAN,
     "max_speed": 0.9, "gait_frequency_hz": 0.74, "hip_roll_limit": 0.065, "nemov3_pd": True, "arm_pd": (25.0, 1.0),
     "wheel_kv_randomization": True, "backward_clock_off": True, "in_place_clock_off": False,
     "hip_roll_inward_turning": True, "hip_roll_inward_standing": True,
     "target_limit_weight": -10.0, "front_down": True, "standing_hip_yaw": (-40.0, 0.05), "standing_clock_off": True,
     "standing_pose_hold": (-40.0, 0.05, (r".*_hip_pitch_joint", r".*_ankle_pitch_joint", r".*_knee_joint",
                                          r".*_shoulder_pitch_joint", r".*_elbow_joint")),
     "turn": True, "turn_yaw_rate_max": 0.9, "backward": 0.5, "backward_hip_roll_inward": (-300.0, 0.0),
     "actuator_delay_steps": 3, "actuator_delay_groups": (r".*",), "trunk_mass_add": (-1.0, 1.0)},
    {"dataset_root": PINOV1_AMP_DATASET_ROOT_STRAIGHT, "mirror_loss_coeff": 50.0},
)
# Exp 44 + the actuator delay 0-15 ms shared by the whole body (as Exp 50a), resumed from Exp 44@5000 (Exp 52; user
# 2026-11-09: "从exp44续训1000轮 加上执行器延迟0~15ms 做exp52" -- Exp 44 drifts least on a pure forward command).
_register(
    "-Straight-Exp33-Nemov3PD-NoStop-FrontDown-StandYaw005-StandClockOff-StandPose005-Turn09-Mirror50-Backward-BackRollIn-Arm25-Wheel25-WheelKvRand-BackClockOff-TurnRollIn-StandRollIn-FallPen-Yaw3-WholeDelay15",
    {"narrow_stance": True, "foot_frame_contact": True, "default_keyframe": HOME_KEYFRAME_NEMO_CLIP_MEAN,
     "max_speed": 0.9, "gait_frequency_hz": 0.74, "hip_roll_limit": 0.065, "nemov3_pd": True, "arm_pd": (25.0, 1.0),
     "wheel_speed_limit": 25.0, "wheel_kv_randomization": True, "backward_clock_off": True, "hip_roll_inward_turning": True,
     "hip_roll_inward_standing": True, "termination_penalty": -100.0, "yaw_track_weight": 3.0,
     "target_limit_weight": -10.0, "front_down": True, "standing_hip_yaw": (-40.0, 0.05), "standing_clock_off": True,
     "standing_pose_hold": (-40.0, 0.05, (r".*_hip_pitch_joint", r".*_ankle_pitch_joint", r".*_knee_joint",
                                          r".*_shoulder_pitch_joint", r".*_elbow_joint")),
     "turn": True, "turn_yaw_rate_max": 0.9, "backward": 0.5, "backward_hip_roll_inward": (-100.0, 0.0),
     "actuator_delay_steps": 3, "actuator_delay_groups": (r".*",)},
    {"dataset_root": PINOV1_AMP_DATASET_ROOT_STRAIGHT, "mirror_loss_coeff": 50.0},
)
# Exp 44 + the torso point mass -1..+1 kg only (no delay), resumed from Exp 44@5000 (Exp 53; user 2026-11-09:
# "似乎加上延迟就是偏转主因 先不加 这次做个实验 exp53只增加质量随机化").
_register(
    "-Straight-Exp33-Nemov3PD-NoStop-FrontDown-StandYaw005-StandClockOff-StandPose005-Turn09-Mirror50-Backward-BackRollIn-Arm25-Wheel25-WheelKvRand-BackClockOff-TurnRollIn-StandRollIn-FallPen-Yaw3-TrunkMass",
    {"narrow_stance": True, "foot_frame_contact": True, "default_keyframe": HOME_KEYFRAME_NEMO_CLIP_MEAN,
     "max_speed": 0.9, "gait_frequency_hz": 0.74, "hip_roll_limit": 0.065, "nemov3_pd": True, "arm_pd": (25.0, 1.0),
     "wheel_speed_limit": 25.0, "wheel_kv_randomization": True, "backward_clock_off": True, "hip_roll_inward_turning": True,
     "hip_roll_inward_standing": True, "termination_penalty": -100.0, "yaw_track_weight": 3.0,
     "target_limit_weight": -10.0, "front_down": True, "standing_hip_yaw": (-40.0, 0.05), "standing_clock_off": True,
     "standing_pose_hold": (-40.0, 0.05, (r".*_hip_pitch_joint", r".*_ankle_pitch_joint", r".*_knee_joint",
                                          r".*_shoulder_pitch_joint", r".*_elbow_joint")),
     "turn": True, "turn_yaw_rate_max": 0.9, "backward": 0.5, "backward_hip_roll_inward": (-100.0, 0.0),
     "trunk_mass_add": (-1.0, 1.0)},
    {"dataset_root": PINOV1_AMP_DATASET_ROOT_STRAIGHT, "mirror_loss_coeff": 50.0},
)
# Exp 44's config without the rear-wheel speed clip and without the fall penalty, trained from scratch for 3000 its
# (Exp 54; user 2026-11-09: "把 Exp 44 的配置一模一样从头训练3000 取消轮速度25限制 取消摔倒惩罚").
_register(
    "-Straight-Exp33-Nemov3PD-NoStop-FrontDown-StandYaw005-StandClockOff-StandPose005-Turn09-Mirror50-Backward-BackRollIn-Arm25-WheelKvRand-BackClockOff-TurnRollIn-StandRollIn-Yaw3",
    {"narrow_stance": True, "foot_frame_contact": True, "default_keyframe": HOME_KEYFRAME_NEMO_CLIP_MEAN,
     "max_speed": 0.9, "gait_frequency_hz": 0.74, "hip_roll_limit": 0.065, "nemov3_pd": True, "arm_pd": (25.0, 1.0),
     "wheel_kv_randomization": True, "backward_clock_off": True, "hip_roll_inward_turning": True,
     "hip_roll_inward_standing": True, "yaw_track_weight": 3.0,
     "target_limit_weight": -10.0, "front_down": True, "standing_hip_yaw": (-40.0, 0.05), "standing_clock_off": True,
     "standing_pose_hold": (-40.0, 0.05, (r".*_hip_pitch_joint", r".*_ankle_pitch_joint", r".*_knee_joint",
                                          r".*_shoulder_pitch_joint", r".*_elbow_joint")),
     "turn": True, "turn_yaw_rate_max": 0.9, "backward": 0.5, "backward_hip_roll_inward": (-100.0, 0.0)},
    {"dataset_root": PINOV1_AMP_DATASET_ROOT_STRAIGHT, "mirror_loss_coeff": 50.0},
)
# Exp 54 with the yaw tracking weight back at 2.0, from scratch to 2000 its (Exp 55; user 2026-11-09: "Exp 54 配置改回转向权重
# 2 从头训练到 2000 轮" -- does the yaw weight 3 keep the forward leg swing small? Exp 54: hip pitch std 0.09 @ 2000).
_register(
    "-Straight-Exp33-Nemov3PD-NoStop-FrontDown-StandYaw005-StandClockOff-StandPose005-Turn09-Mirror50-Backward-BackRollIn-Arm25-WheelKvRand-BackClockOff-TurnRollIn-StandRollIn",
    {"narrow_stance": True, "foot_frame_contact": True, "default_keyframe": HOME_KEYFRAME_NEMO_CLIP_MEAN,
     "max_speed": 0.9, "gait_frequency_hz": 0.74, "hip_roll_limit": 0.065, "nemov3_pd": True, "arm_pd": (25.0, 1.0),
     "wheel_kv_randomization": True, "backward_clock_off": True, "hip_roll_inward_turning": True,
     "hip_roll_inward_standing": True,
     "target_limit_weight": -10.0, "front_down": True, "standing_hip_yaw": (-40.0, 0.05), "standing_clock_off": True,
     "standing_pose_hold": (-40.0, 0.05, (r".*_hip_pitch_joint", r".*_ankle_pitch_joint", r".*_knee_joint",
                                          r".*_shoulder_pitch_joint", r".*_elbow_joint")),
     "turn": True, "turn_yaw_rate_max": 0.9, "backward": 0.5, "backward_hip_roll_inward": (-100.0, 0.0)},
    {"dataset_root": PINOV1_AMP_DATASET_ROOT_STRAIGHT, "mirror_loss_coeff": 50.0},
)
# Exp 55 + a fore-aft stagger penalty while driving straight backward (|hip pitch L - R| beyond 0.05 rad, squared, x -40),
# resumed from Exp 55@1999 (Exp 56; user 2026-11-10: "Exp 55@2000 续训exp56 直线后退（不包括转弯）时约束一下脚错开的姿势").
_register(
    "-Straight-Exp33-Nemov3PD-NoStop-FrontDown-StandYaw005-StandClockOff-StandPose005-Turn09-Mirror50-Backward-BackRollIn-Arm25-WheelKvRand-BackClockOff-TurnRollIn-StandRollIn-BackStagger",
    {"narrow_stance": True, "foot_frame_contact": True, "default_keyframe": HOME_KEYFRAME_NEMO_CLIP_MEAN,
     "max_speed": 0.9, "gait_frequency_hz": 0.74, "hip_roll_limit": 0.065, "nemov3_pd": True, "arm_pd": (25.0, 1.0),
     "wheel_kv_randomization": True, "backward_clock_off": True, "hip_roll_inward_turning": True,
     "hip_roll_inward_standing": True,
     "target_limit_weight": -10.0, "front_down": True, "standing_hip_yaw": (-40.0, 0.05), "standing_clock_off": True,
     "standing_pose_hold": (-40.0, 0.05, (r".*_hip_pitch_joint", r".*_ankle_pitch_joint", r".*_knee_joint",
                                          r".*_shoulder_pitch_joint", r".*_elbow_joint")),
     "turn": True, "turn_yaw_rate_max": 0.9, "backward": 0.5, "backward_hip_roll_inward": (-100.0, 0.0),
     "backward_stagger": (-40.0, 0.05)},
    {"dataset_root": PINOV1_AMP_DATASET_ROOT_STRAIGHT, "mirror_loss_coeff": 50.0},
)
# Exp 56 + the hip yaws held near the default while turning in place or backing up with a turn (beyond 0.1 rad, squared,
# x -40; off while driving forward), resumed from Exp 56@2998 (Exp 57; user 2026-11-10: "从exp56续训1000轮 转弯时罚hip yaw
# 偏离默认 做exp57" -> "只在原地转和后退转弯生效").
_register(
    "-Straight-Exp33-Nemov3PD-NoStop-FrontDown-StandYaw005-StandClockOff-StandPose005-Turn09-Mirror50-Backward-BackRollIn-Arm25-WheelKvRand-BackClockOff-TurnRollIn-StandRollIn-BackStagger-TurnYaw",
    {"narrow_stance": True, "foot_frame_contact": True, "default_keyframe": HOME_KEYFRAME_NEMO_CLIP_MEAN,
     "max_speed": 0.9, "gait_frequency_hz": 0.74, "hip_roll_limit": 0.065, "nemov3_pd": True, "arm_pd": (25.0, 1.0),
     "wheel_kv_randomization": True, "backward_clock_off": True, "hip_roll_inward_turning": True,
     "hip_roll_inward_standing": True,
     "target_limit_weight": -10.0, "front_down": True, "standing_hip_yaw": (-40.0, 0.05), "standing_clock_off": True,
     "standing_pose_hold": (-40.0, 0.05, (r".*_hip_pitch_joint", r".*_ankle_pitch_joint", r".*_knee_joint",
                                          r".*_shoulder_pitch_joint", r".*_elbow_joint")),
     "turn": True, "turn_yaw_rate_max": 0.9, "backward": 0.5, "backward_hip_roll_inward": (-100.0, 0.0),
     "backward_stagger": (-40.0, 0.05),
     "turning_hip_yaw": (-40.0, 0.1)},
    {"dataset_root": PINOV1_AMP_DATASET_ROOT_STRAIGHT, "mirror_loss_coeff": 50.0},
)
# + the no-lift penalty and the hip-roll spread penalty (skating without lifting a wheel).
_register("-NoLift", {"no_lift": True, "narrow_stance": True})
