import re

from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import register_mjlab_task

from booster_mjlab.robots.nemov3.nemov3_constants import (
    NEMOV3_DELAY_GROUPS,
    NEMOV3_FIXED_HEAD_WAIST_JOINTS,
    nemov3_active_joint_names,
)
from booster_mjlab.tasks.velocity.config import with_amp_obs_group
from booster_mjlab.tasks.velocity.mdp.motor_dynamics import ActuatorDelayCfg, MotorTargetDynamicsCfg
from booster_mjlab.tasks.velocity.mdp.wheel_feet import (
    GATED_AMP_JOINTS,
    base_forward_speed,
    gated_base_pitch,
    gated_joint_pos,
    wheel_height_diff,
)
from booster_mjlab.tasks.velocity.rl.runner import VelocityAmpOnPolicyRunner

from .env_cfgs import FOOT_LIFT_HIP_ROLL_LIMIT_RAD, UPRIGHT_STD_MOVING_RELAXED, nemov3_flat_env_cfg
from .rl_cfg import (
    NEMOV3_AMP_DATASET_ROOT_SYMMETRIC,
    NEMOV3_MIRROR_LOSS_COEFF,
    NEMOV3_STYLE_REWARD_WEIGHT,
    nemov3_amp_ppo_runner_cfg,
)

# Joints the AMP discriminator scores: hip pitch + calf only. Excludes the
# wheels (every source clip has the wheel joints frozen at 0 rad, so scoring
# wheel pos/vel would fight the wheel-velocity task reward), and -- since
# Exp 7 -- also hip roll/yaw: the reference keeps those nearly still while the
# policy's exploration noise makes them jitter, and a permutation-importance
# analysis of the trained discriminators showed hip roll/yaw position+velocity
# make up ~80% of what the discriminator can exploit, while the leg-swing
# features (hip pitch/calf) barely register. See tools/nemov3_finetune.md
# (排查 3). AMP obs is 4-dim per step since Exp 14 (4 joint pos; 8-dim with joint vel in
# Exp 13, 14-dim with base lin vel (3) + projected gravity (3) before that).
NEMOV3_AMP_JOINT_NAMES = (
    r".*_hip_pitch_joint",
    r".*_calf_joint",
)

# Arm-swing variant: the discriminator also scores shoulder pitch + elbow (the
# reference arms swing +-0.17-0.32 rad in shoulder pitch and, in Fast06, +-0.38
# rad in elbow). Shoulder roll/yaw stay under the `upper_body_posture` penalty.
NEMOV3_AMP_ARM_JOINT_NAMES = NEMOV3_AMP_JOINT_NAMES + (
    r"joint_[rl]ShoulderPitch",
    r"joint_[rl]Elbow",
)

# Foot-lift variant ("Twist"): also score pelvis yaw + hip roll + hip yaw. In
# SkatingFast06 (the only clip that lifts a foot, by 8.5 cm) the foot-height
# difference correlates most with joint_pelvisYaw (|corr| 0.98, std 0.124 rad),
# then the elbows / shoulder pitch, right_hip_roll (0.77), left/right_hip_yaw
# (0.71 / 0.65) -- hip pitch and calf are not even in the top 10 -- and none of
# them was in the AMP features (tools/nemov3_finetune.md, Exp 27).
NEMOV3_AMP_TWIST_JOINT_NAMES = NEMOV3_AMP_ARM_JOINT_NAMES + (
    r"joint_pelvisYaw",
    r".*_hip_roll_joint",
    r".*_hip_yaw_joint",
)

# Foot-lift skating (Exp 50+): also score shoulder yaw + pelvis yaw -- the diagonal arm swing (the arm opposite the
# leading foot swings forward and across, mostly shoulder yaw) and the upper-body twist of Fast06. Needs env
# arm_yaw=True (frees shoulder yaw from the posture penalty).
NEMOV3_AMP_SKATE_JOINT_NAMES = NEMOV3_AMP_ARM_JOINT_NAMES + (
    r"joint_[rl]ShoulderYaw",
    r"joint_pelvisYaw",
)

# Foot-lift skating: the discriminator also sees the right-minus-left wheel height (joint angles alone cannot tell a
# lifted wheel from a crouch) and the base forward speed (the style is judged per speed, so above ~0.8 m/s only the
# lifting Fast06 counts as real); the expert side is appended by the motion loader (amp_extra_obs_fn). Exp 49 used the
# reference lifts as they are (Fast06 peaks at 8.5 cm), Exp 50+ scaled to ~6 cm (user: "离地六公分感觉就差不多了").
NEMOV3_LIFT_SPEED_EXPERT_FN = "booster_mjlab.tasks.velocity.mdp.wheel_feet:lift_features_from_motion"
NEMOV3_LIFT_6CM_SPEED_EXPERT_FN = "booster_mjlab.tasks.velocity.mdp.wheel_feet:lift_features_6cm_from_motion"
# Speed-gated feature set (Exp 57 / 59): the joints above scored at all speeds, hip roll / yaw, shoulder yaw, pelvis yaw
# and the root pitch only at speed (mdp.wheel_feet.AMP_SPEED_GATE), with instance noise on the channels Skating01/02
# did not capture (dataset transform).
NEMOV3_GATED_LIFT_EXPERT_FN = "booster_mjlab.tasks.velocity.mdp.wheel_feet:gated_lift_features_from_motion"
NEMOV3_NOISE_UNCAPTURED_TRANSFORM = "booster_mjlab.tasks.velocity.mdp.wheel_feet:noise_uncaptured_joints"
# Clips for a robot with NEMOV3_FIXED_HEAD_WAIST_JOINTS welded: their dof_pos columns dropped.
NEMOV3_DROP_FIXED_HEAD_WAIST_TRANSFORM = (
    "booster_mjlab.robots.nemov3.nemov3_constants:drop_fixed_head_waist_joints"
)


def with_nemov3_lift_features(cfg, gated=False):
    """Append [(gated joints,) wheel height difference, base forward speed(, gated root pitch)] as the last AMP terms, the
    order of the expert-side function."""
    terms = cfg.observations["amp"].terms
    if gated:
        terms["gated_joint_pos"] = ObservationTermCfg(
            func=gated_joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=GATED_AMP_JOINTS, preserve_order=True)},
        )
    terms["wheel_height_diff"] = ObservationTermCfg(
        func=wheel_height_diff,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=("right_wheel_link", "left_wheel_link"), preserve_order=True
            )
        },
    )
    terms["base_forward_speed"] = ObservationTermCfg(func=base_forward_speed)
    if gated:
        terms["gated_base_pitch"] = ObservationTermCfg(func=gated_base_pitch)
    return cfg

# Whether the discriminator also sees the torso state (base linear velocity +
# projected gravity). Off since Exp 13: the mocap reference has an extremely
# tight torso orientation (projected gravity y std 0.006, ~0.3 deg) that the
# simulated policy can never match under exploration noise/pushes, so those
# features are a cheap cue that dominates the discriminator's exploitable gap
# (permutation analysis on Exp 12: fixing gravity alone lifts D by +0.30 once
# the legs match) without saying anything about the leg gait. The velocity
# task reward already covers speed. See tools/nemov3_finetune.md (Exp 13).
NEMOV3_AMP_INCLUDE_BASE_STATE = False

# Whether the discriminator sees joint velocities. Off since Exp 14: the
# policy's exploration noise (iid position-target noise every 20 ms) alone
# produces joint-velocity std ~1.5-2.2 rad/s (measured; the deterministic policy
# has 0.2-0.45), which matches the reference (1.85 rad/s hip pitch) without any
# coherent swing, so the velocity marginal is spoofed for free. Joint position
# over the K-frame window carries the same motion information (finite
# differences) but can't be faked by per-step noise. See tools/nemov3_finetune.md.
NEMOV3_AMP_INCLUDE_JOINT_VEL = False

# Ground-contact sites used by the AMP motion-pool reset to avoid spawning
# with a wheel clipped through the floor (see nemov3.xml).
NEMOV3_WHEEL_GROUND_SITES = ("left_wheel_ground", "right_wheel_ground")


def with_nemov3_amp_joints(cfg, joint_names=NEMOV3_AMP_JOINT_NAMES):
    """Narrow the AMP observation group to Nemov3's leg DOFs (no wheels)."""
    amp_group = cfg.observations["amp"]
    for term in amp_group.terms.values():
        if "asset_cfg" in term.params:
            term.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=joint_names)
    if not NEMOV3_AMP_INCLUDE_JOINT_VEL:
        amp_group.terms.pop("joint_vel", None)
    if not NEMOV3_AMP_INCLUDE_BASE_STATE:
        for name in ("base_lin_vel", "projected_gravity"):
            amp_group.terms.pop(name, None)
    reset_evt = cfg.events.get("reset_robot_from_motion")
    if reset_evt is not None:
        reset_evt.params["foot_sites"] = NEMOV3_WHEEL_GROUND_SITES
    return cfg


def _with_amp_reset_cfg(
    env_cfg,
    runner_cfg,
    joint_names=NEMOV3_AMP_JOINT_NAMES,
    lift_features=None,
    gated=False,
    style_weight=NEMOV3_STYLE_REWARD_WEIGHT,
    reset_transform=False,
):
    cfg = with_amp_obs_group(
        env_cfg,
        dataset_root=runner_cfg.dataset_root,
        speed_factor=runner_cfg.speed_factor,
        dataset_weights=runner_cfg.dataset_weights,
        augmentations=runner_cfg.dataset_augmentations,
        include_base_lin_vel=True,
    )
    # The shared wrapper pins this curriculum to 0.3 -> 0.3; the runner reads
    # the weight from here, not from the runner cfg's style_reward_weight.
    style_params = cfg.curriculum["amp_style_weight"].params
    style_params["start_weight"] = style_weight
    style_params["end_weight"] = style_weight
    if reset_transform:
        # The reset poses come from the same clips: a robot with welded joints needs their columns dropped there too.
        cfg.events["reset_robot_from_motion"].params["dataset_transform"] = runner_cfg.dataset_transform
    cfg = with_nemov3_amp_joints(cfg, joint_names)
    return with_nemov3_lift_features(cfg, gated) if lift_features else cfg


def _register(
    name,
    env_kwargs=None,
    rl_kwargs=None,
    amp_joints=NEMOV3_AMP_JOINT_NAMES,
    lift_features=None,
    gated=False,
    style_weight=NEMOV3_STYLE_REWARD_WEIGHT,
):
    """Register ``Mjlab-Velocity-Flat-Amp{name}[-Muon]-Booster-Nemov3``.

    ``lift_features`` = the expert-side function of the discriminator's foot-lift features (NEMOV3_*_EXPERT_FN), with
    ``gated`` the speed-gated set.
    """
    env_kwargs = env_kwargs or {}
    rl_kwargs = {**(rl_kwargs or {}), **({"amp_extra_obs_fn": lift_features} if lift_features else {})}
    fixed = bool(env_kwargs.get("fixed_joints"))
    if fixed:
        # The clip transform drops exactly NEMOV3_FIXED_HEAD_WAIST_JOINTS; another set needs its own transform.
        assert tuple(env_kwargs["fixed_joints"]) == NEMOV3_FIXED_HEAD_WAIST_JOINTS, "no clip transform for this set"
        assert "dataset_transform" not in rl_kwargs, "fixed_joints tasks use the column-dropping clip transform"
        rl_kwargs["dataset_transform"] = NEMOV3_DROP_FIXED_HEAD_WAIST_TRANSFORM
        # The discriminator cannot score welded joints (e.g. pelvis yaw in NEMOV3_AMP_SKATE_JOINT_NAMES).
        active = nemov3_active_joint_names(NEMOV3_FIXED_HEAD_WAIST_JOINTS)
        amp_joints = tuple(p for p in amp_joints if any(re.fullmatch(p, name) for name in active))
    amp_args = (amp_joints, lift_features, gated, style_weight, fixed)
    for use_muon in (False, True):
        rl_cfg = nemov3_amp_ppo_runner_cfg(use_muon=use_muon, **rl_kwargs)
        register_mjlab_task(
            task_id=f"Mjlab-Velocity-Flat-Amp{name}{'-Muon' if use_muon else ''}-Booster-Nemov3",
            env_cfg=_with_amp_reset_cfg(nemov3_flat_env_cfg(**env_kwargs), rl_cfg, *amp_args),
            play_env_cfg=_with_amp_reset_cfg(nemov3_flat_env_cfg(play=True, **env_kwargs), rl_cfg, *amp_args),
            rl_cfg=rl_cfg,
            runner_cls=VelocityAmpOnPolicyRunner,
        )


_CLOCK_ARMS = {"clock": True, "arms": True}
_NARROW = {**_CLOCK_ARMS, "narrow_stance": True}

_register("")
# Free-running 0.74 Hz phase clock as an observation only (no gait reward), Exp 17+.
_register("-Clock", {"clock": True})
# + arms in the AMP features (Exp 20).
_register("-Clock-Arms", _CLOCK_ARMS, amp_joints=NEMOV3_AMP_ARM_JOINT_NAMES)
# Foot-lift rewards: K1-style (Exp 25), shaped (Exp 26/27), shaped + AMP on pelvis yaw /
# hip roll / hip yaw (Exp 28; its discriminator input differs, resume with
# --agent.reset-discriminator True).
_register("-Clock-Arms-FootLift", {**_CLOCK_ARMS, "foot_lift": True}, amp_joints=NEMOV3_AMP_ARM_JOINT_NAMES)
_register("-Clock-Arms-FootLiftShaped", {**_CLOCK_ARMS, "foot_lift": "shaped"}, amp_joints=NEMOV3_AMP_ARM_JOINT_NAMES)
_register("-Clock-Arms-FootLiftTwist", {**_CLOCK_ARMS, "foot_lift": "shaped"}, amp_joints=NEMOV3_AMP_TWIST_JOINT_NAMES)
# Hip-roll upper-limit penalty, feet not too wide (Exp 29). NOTE: remove/relax the hip-roll
# penalty before training foot lifts again (see mdp/stance.py).
_register("-Clock-Arms-NarrowStance", _NARROW, amp_joints=NEMOV3_AMP_ARM_JOINT_NAMES)
# + left/right mirrored reference clips (Exp 30).
_register("-Clock-Arms-NarrowStance-Mirror", _NARROW, {"mirror": True}, NEMOV3_AMP_ARM_JOINT_NAMES)
# + PPO mirror loss with the mirror-symmetric default hip yaw it needs (Exp 33+): stable when resuming an
# asymmetric checkpoint. (PPO symmetry data augmentation, rl_cfg `symmetry=True`, diverged when resuming, Exp 32.)
_register(
    "-Clock-Arms-NarrowStance-Mirror-SymLoss",
    {**_NARROW, "symmetric_default": True},
    {"mirror": True, "mirror_loss_coeff": NEMOV3_MIRROR_LOSS_COEFF},
    NEMOV3_AMP_ARM_JOINT_NAMES,
)
# Baseline (Exp 33) config + left/right turning: yaw-rate command in +-TURN_YAW_RATE_MAX (1.3 rad/s, Exp 41+). The reference
# clips are straight-line only; the command mirror (vx, -vy, -wz) makes the mirror loss map left turns to right turns.
_register(
    "-Clock-Arms-NarrowStance-Mirror-SymLoss-Turn",
    {**_NARROW, "symmetric_default": True, "turn": True},
    {"mirror": True, "mirror_loss_coeff": NEMOV3_MIRROR_LOSS_COEFF},
    NEMOV3_AMP_ARM_JOINT_NAMES,
)
# Exp 78 (user, 2026-09-29: deploy the turning policy Exp 41 on the hardware, whose head and waist cannot move yet):
# the same task with NEMOV3_FIXED_HEAD_WAIST_JOINTS welded -- head roll / pitch / yaw, waist yaw / pitch / roll, the
# waist push-rod linkages and the wrist pitch joints; 20 joints remain (18 position + 2 wheels).
_register(
    "-Clock-Arms-NarrowStance-Mirror-SymLoss-Turn-FixHeadWaist",
    {**_NARROW, "symmetric_default": True, "turn": True, "fixed_joints": NEMOV3_FIXED_HEAD_WAIST_JOINTS},
    {"mirror": True, "mirror_loss_coeff": NEMOV3_MIRROR_LOSS_COEFF},
    NEMOV3_AMP_ARM_JOINT_NAMES,
)
# Exp 79: Exp 78 with the left/right symmetric reference clips (Exp 36's data; it evened out the elbow swing).
_register(
    "-Clock-Arms-NarrowStance-SymClips-SymLoss-Turn-FixHeadWaist",
    {**_NARROW, "symmetric_default": True, "turn": True, "fixed_joints": NEMOV3_FIXED_HEAD_WAIST_JOINTS},
    {
        "mirror": True,
        "mirror_loss_coeff": NEMOV3_MIRROR_LOSS_COEFF,
        "dataset_root": NEMOV3_AMP_DATASET_ROOT_SYMMETRIC,
    },
    NEMOV3_AMP_ARM_JOINT_NAMES,
)
# Sim-to-real for the turning policy (user, 2026-10-12, after Exp 78 vibrated violently on the hardware; the
# timing model is flamingo_rl_train's, see tools/nemov3_finetune.md). Each step adds one item to the one before:
# Exp 81 the hardware's PD gains (NEMOV3_DEPLOY_PD_GAINS); Exp 82 + a per-episode actuator command delay of 0-20 ms;
# Exp 83 + the motor-target delay 0-30 ms (motor rate fixed at the hardware's 200 Hz, no low-pass). (Exp 84, + a
# low-pass tau 0-30 ms, was strange and asymmetric on the hardware; its task was removed, see tools/nemov3_finetune.md.)
NEMOV3_ACTUATOR_DELAY_STEPS = 4
NEMOV3_MOTOR_DELAY = MotorTargetDynamicsCfg(control_hz_choices=(200.0,), delay_range_s=(0.0, 0.03))
_TURN_FIXED = {**_NARROW, "symmetric_default": True, "turn": True, "fixed_joints": NEMOV3_FIXED_HEAD_WAIST_JOINTS}
_DEPLOY_PD = {**_TURN_FIXED, "deploy_pd": True}
_ACT_DELAY = {**_DEPLOY_PD, "actuator_delay_steps": NEMOV3_ACTUATOR_DELAY_STEPS}
for _suffix, _env in (
    ("-DeployPD", _DEPLOY_PD),
    ("-DeployPD-ActDelay", _ACT_DELAY),
    ("-DeployPD-ActDelay-CmdDelay", {**_ACT_DELAY, "motor_dynamics": NEMOV3_MOTOR_DELAY}),
):
    _register(
        "-Clock-Arms-NarrowStance-Mirror-SymLoss-Turn-FixHeadWaist" + _suffix,
        _env,
        {"mirror": True, "mirror_loss_coeff": NEMOV3_MIRROR_LOSS_COEFF},
        NEMOV3_AMP_ARM_JOINT_NAMES,
    )
# Exp 85 (user, after the hardware tests -- Exp 83 best): Exp 83's two delays in series merged into one buffer. Two
# pure delays add, so per joint group (arms / legs / wheels) the lag = an actuator lag uniform 0-4 steps + a
# motor-target lag round(uniform(0, 30 ms) / 5 ms) shared per action term (arms + legs, wheels): same distribution
# (0-50 ms, peak 25 ms) and grouping as Exp 83, so the same training conditions without MotorTargetDynamics.
# (Exp 86, the simplified version with one uniform 0-50 ms lag for the whole body, was shakier on the hardware; its task
# was removed, see tools/nemov3_finetune.md.)
NEMOV3_MERGED_DELAY = ActuatorDelayCfg(
    groups=NEMOV3_DELAY_GROUPS,
    max_lag=NEMOV3_ACTUATOR_DELAY_STEPS,
    cmd_groups=(r"(?!.*_wheel_joint$).*", r".*_wheel_joint"),
    cmd_delay_range_s=NEMOV3_MOTOR_DELAY.delay_range_s,
)
_register(
    "-Clock-Arms-NarrowStance-Mirror-SymLoss-Turn-FixHeadWaist-DeployPD-MergedDelay",
    {**_DEPLOY_PD, "actuator_delay": NEMOV3_MERGED_DELAY},
    {"mirror": True, "mirror_loss_coeff": NEMOV3_MIRROR_LOSS_COEFF},
    NEMOV3_AMP_ARM_JOINT_NAMES,
)
# Exp 88 (user, 2026-10-14: Exp 85 falls after standing still long on the hardware): Exp 85 + the torso mass
# randomization (-1..+1 kg; the COM draw of Exp 85's trunk_inertia event stays, +-5 cm per axis), flamingo_rl_train's old
# recipe. (The variant that replaced the COM draw by flamingo's +-3 cm was never trained and its task was removed.)
_register(
    "-Clock-Arms-NarrowStance-Mirror-SymLoss-Turn-FixHeadWaist-DeployPD-MergedDelay-TrunkMass",
    {**_DEPLOY_PD, "actuator_delay": NEMOV3_MERGED_DELAY, "trunk_mass_com": "mass"},
    {"mirror": True, "mirror_loss_coeff": NEMOV3_MIRROR_LOSS_COEFF},
    NEMOV3_AMP_ARM_JOINT_NAMES,
)
# Foot-lift skating (docs/nemov3_foot_lift_plan.md; tools/nemov3_finetune.md, section 抬脚). Only the runs the log keeps
# as milestones are registered here; the other experiments are described in the log.
# Exp 46: single-wheel feasibility probe -- ride on the right wheel with the left one held up, no style reward, hip-roll
# limit relaxed to 0.2 rad, 0-1.2 m/s. Deliberately one-sided, so no mirror loss.
_register(
    "-Clock-Arms-SingleWheel",
    {**_NARROW, "single_wheel": True, "hip_roll_limit": FOOT_LIFT_HIP_ROLL_LIMIT_RAD},
    amp_joints=NEMOV3_AMP_ARM_JOINT_NAMES,
    style_weight=0.0,
)
# The lift runs share the Exp 33 base (straight, all clips + mirror, mirror loss), the hip-roll limit 0.2 rad (user),
# [B]-style contact-timing rewards above 0.6 m/s and a discriminator that also sees the wheel height difference and the
# forward speed; Fast06 weighted 2x in the expert samples / reset poses.
_LIFT = {**_NARROW, "symmetric_default": True, "hip_roll_limit": FOOT_LIFT_HIP_ROLL_LIMIT_RAD}
_LIFT_RL = {"mirror": True, "mirror_loss_coeff": NEMOV3_MIRROR_LOSS_COEFF, "dataset_weights": [1.0, 1.0, 2.0]}
_NATURAL = {**_LIFT, "lean": True, "arm_yaw": True}
_GATED_RL = {**_LIFT_RL, "dataset_transform": NEMOV3_NOISE_UNCAPTURED_TRANSFORM}
# Exp 49: lift pay strong from the start (lift="early") -- the first lifts: alternating rear-wheel lifts, but too fast
# and high (stamping in place).
_register(
    "-Clock-Arms-NarrowStance-Mirror-SymLoss-Lift-SpeedDisc-Early",
    {**_LIFT, "lift": "early"},
    _LIFT_RL,
    NEMOV3_AMP_ARM_JOINT_NAMES,
    NEMOV3_LIFT_SPEED_EXPERT_FN,
)
# Exp 50 / 51 (after the user watched Exp 49): the natural targets at once (lift="natural": slower rhythm, ~6 cm
# reference lifts + a bound above 8 cm), forward lean, diagonal arm swing / upper-body twist (shoulder + pelvis yaw in
# the discriminator, shoulder yaw free) -- no lift from scratch (50), lifts dropped when resumed from Exp 49 (51).
_register(
    "-Clock-Arms-NarrowStance-Mirror-SymLoss-Lift-Natural",
    {**_NATURAL, "lift": "natural"},
    _LIFT_RL,
    NEMOV3_AMP_SKATE_JOINT_NAMES,
    NEMOV3_LIFT_6CM_SPEED_EXPERT_FN,
)
# Exp 52 (rerun as Exp 76): learn first, then shape (lift="staged": Exp 49's recipe for 400 iterations, then a move to
# the Exp 50 targets over 400-1200). The fast posture the user approved -- the rear wheel lifted and stretched back, the
# lifted-side hand ahead -- at ~1.0 s per stride cycle; but no slow alternating stride.
_register(
    "-Clock-Arms-NarrowStance-Mirror-SymLoss-Lift-Staged",
    {**_NATURAL, "lift": "staged"},
    _LIFT_RL,
    NEMOV3_AMP_SKATE_JOINT_NAMES,
    NEMOV3_LIFT_6CM_SPEED_EXPERT_FN,
)
# Exp 57: speed-gated discriminator (Exp 52's recipe without the lean reward) -- the slow alternating stride came back
# with the lifts, but the lifted foot swings forward (the front one is lifted) and the lean was lost (~2 deg).
_register(
    "-Clock-Arms-NarrowStance-Mirror-SymLoss-Lift-Staged-Gated",
    {**_LIFT, "lift": "staged", "arm_yaw": True},
    _GATED_RL,
    NEMOV3_AMP_ARM_JOINT_NAMES,
    NEMOV3_GATED_LIFT_EXPERT_FN,
    gated=True,
)
# Exp 59a / 59b: Exp 57 @1800 resumed with the staged end values and the upright tilt tolerance relaxed while moving;
# "-Lean" (59b) adds the forward-lean reward back (user: allowed if needed to look good): lean 8-9 deg.
for _suffix, _lean in (("", False), ("-Lean", True)):
    _register(
        f"-Clock-Arms-NarrowStance-Mirror-SymLoss-Lift-StagedEnd-Gated-Upright1{_suffix}",
        {
            **_LIFT,
            "lift": "staged_end",
            "arm_yaw": True,
            "lean": _lean,
            "upright_std_moving": UPRIGHT_STD_MOVING_RELAXED,
        },
        _GATED_RL,
        NEMOV3_AMP_ARM_JOINT_NAMES,
        NEMOV3_GATED_LIFT_EXPERT_FN,
        gated=True,
    )
# Exp 77a / 77a2 (user: "符合Fast06的节奏"): Exp 52 + lifts timed by the gait clock. 77a (lift_clock=True) dropped its
# lifts while the window was ramped in; 77a2 (lift_clock="compensated") -- Fast06's tempo (1.34 s per stride cycle) and
# the first natural foot-lift gait (user, 2026-09-29: "77a2 很好 姿态有自然").
for _suffix, _clock in (("-ClockWindow", True), ("-ClockWindow2", "compensated")):
    _register(
        "-Clock-Arms-NarrowStance-Mirror-SymLoss-Lift-Staged" + _suffix,
        {**_NATURAL, "lift": "staged", "lift_clock": _clock},
        _LIFT_RL,
        NEMOV3_AMP_SKATE_JOINT_NAMES,
        NEMOV3_LIFT_6CM_SPEED_EXPERT_FN,
    )
# Exp 80 (user, 2026-09-30: a foot-lift model for the hardware with the head and waist fixed): Exp 77a2 with the same
# 12 joints welded as Exp 78. The waist twist goes with pelvis yaw (the discriminator no longer scores it); the
# diagonal arm swing stays on shoulder yaw.
_register(
    "-Clock-Arms-NarrowStance-Mirror-SymLoss-Lift-Staged-ClockWindow2-FixHeadWaist",
    {**_NATURAL, "lift": "staged", "lift_clock": "compensated", "fixed_joints": NEMOV3_FIXED_HEAD_WAIST_JOINTS},
    _LIFT_RL,
    NEMOV3_AMP_SKATE_JOINT_NAMES,
    NEMOV3_LIFT_6CM_SPEED_EXPERT_FN,
)
# Exp 87 (user, 2026-10-14): the foot-lift skating of Exp 80 with Exp 85's sim-to-real setup (the hardware's PD gains +
# the merged actuator delay) -- a deployable foot-lift model.
_register(
    "-Clock-Arms-NarrowStance-Mirror-SymLoss-Lift-Staged-ClockWindow2-FixHeadWaist-DeployPD-MergedDelay",
    {
        **_NATURAL,
        "lift": "staged",
        "lift_clock": "compensated",
        "fixed_joints": NEMOV3_FIXED_HEAD_WAIST_JOINTS,
        "deploy_pd": True,
        "actuator_delay": NEMOV3_MERGED_DELAY,
    },
    _LIFT_RL,
    NEMOV3_AMP_SKATE_JOINT_NAMES,
    NEMOV3_LIFT_6CM_SPEED_EXPERT_FN,
)
# + reference clips made left/right symmetric one by one (Exp 36+).
_register(
    "-Clock-Arms-NarrowStance-SymClips-SymLoss",
    {**_NARROW, "symmetric_default": True},
    {
        "mirror": True,
        "mirror_loss_coeff": NEMOV3_MIRROR_LOSS_COEFF,
        "dataset_root": NEMOV3_AMP_DATASET_ROOT_SYMMETRIC,
    },
    NEMOV3_AMP_ARM_JOINT_NAMES,
)
