from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import register_mjlab_task

from booster_mjlab.tasks.velocity.config import with_amp_obs_group
from booster_mjlab.tasks.velocity.rl.runner import VelocityAmpOnPolicyRunner

from .env_cfgs import nemov3_flat_env_cfg
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


def _with_amp_reset_cfg(env_cfg, runner_cfg, joint_names=NEMOV3_AMP_JOINT_NAMES):
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
    style_params["start_weight"] = NEMOV3_STYLE_REWARD_WEIGHT
    style_params["end_weight"] = NEMOV3_STYLE_REWARD_WEIGHT
    return with_nemov3_amp_joints(cfg, joint_names)


def _register(name, env_kwargs=None, rl_kwargs=None, amp_joints=NEMOV3_AMP_JOINT_NAMES):
    """Register ``Mjlab-Velocity-Flat-Amp{name}[-Muon]-Booster-Nemov3``."""
    env_kwargs = env_kwargs or {}
    for use_muon in (False, True):
        rl_cfg = nemov3_amp_ppo_runner_cfg(use_muon=use_muon, **(rl_kwargs or {}))
        register_mjlab_task(
            task_id=f"Mjlab-Velocity-Flat-Amp{name}{'-Muon' if use_muon else ''}-Booster-Nemov3",
            env_cfg=_with_amp_reset_cfg(nemov3_flat_env_cfg(**env_kwargs), rl_cfg, amp_joints),
            play_env_cfg=_with_amp_reset_cfg(
                nemov3_flat_env_cfg(play=True, **env_kwargs), rl_cfg, amp_joints
            ),
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
