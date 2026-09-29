"""RL configuration for the Nemov3 velocity task.

Only an AMP runner is provided (no plain-PPO variant): the point of this
task is training against the skating reference clips.
"""

from mjlab.rl import RslRlModelCfg

from booster_mjlab.rl.config import RslRlSymmetryCfg
from booster_mjlab.amp.runners import (
    AmpDiscriminatorCfg,
    AmpOnPolicyRunnerCfg,
    AmpRslRlMuonPpoAlgorithmCfg,
    AmpRslRlPpoAlgorithmCfg,
)

# Local dataset directory produced by
# `uv run python -m booster_mjlab.scripts.nemov3_skating_to_pkl`: Skating01
# (fast/large-stride), Skating02 (slow/small-stride), and the trimmed
# SkatingFast06 (static lead-in removed). All three are straight-line
# clips (no real turning in the source data -- see the analysis in
# assets/amp_sources/nemov3_skating), spanning roughly 0.3-1.3 m/s.
NEMOV3_AMP_DATASET_ROOT = "assets/amp/nemov3_skating_straight"

# The three clips, each projected onto a left/right symmetric stride, q_sym(t) = 0.5 (q(t) + M q(t - P/2))
# (src/booster_mjlab/scripts/nemov3_symmetrize_clips.py). Each source clip is lopsided on its own (Skating01
# swings the right calf / elbow 1.59 / 2.3x more; Skating02's R->L / L->R timing is uneven), which the
# discriminator then accepts per episode; mirror augmentation only balances the set as a whole.
NEMOV3_AMP_DATASET_ROOT_SYMMETRIC = "assets/amp/nemov3_skating_symmetric"

# Weight of the AMP style reward vs. the task reward (task gets 1 - weight).
# K1 uses 0.3, and can, because a biped's task reward already forces
# alternating steps (plus explicit air-time/clearance/swing-height terms), so
# the style reward only picks among feasible gaits. A wheeled robot meets the
# velocity task with frozen legs, so the style reward has to carry the whole
# burden of producing the gait -- at 0.3 it was ~10% of the total return and
# the legs never moved (tools/nemov3_finetune.md, Exp 4-7). NOTE: the value the
# runner actually uses is set by the "amp_style_weight" curriculum term (see
# config/__init__.py), which overrides `style_reward_weight` below.
NEMOV3_STYLE_REWARD_WEIGHT = 0.7

# Number of consecutive AMP frames per discriminator sample. K1's default is 10
# (0.2 s at 50 Hz). Exp 10 tried 40 (0.8 s) while the legs were still frozen: the
# discriminator got stronger but there was nothing to climb, so it was reverted.
# Exp 15 revisits a long window (30 = 0.6 s, ~55% of the 1.05-1.33 s stride)
# now that the legs can move and the AMP features are joint positions only: a
# static or slowly-drifting pose is then plainly different from a sweeping one
# (tools/nemov3_finetune.md).
NEMOV3_AMP_HISTORY_STEPS = 30

# Weight of the discriminator's R1-style gradient penalty (expert samples).
# K1 uses 10. With 10 the nemov3 discriminator is extremely flat (logit
# gradient norm ~0.09 at equilibrium: policy_pred 0.445 vs expert_pred 0.555),
# so the style reward barely differs between policy samples and gives PPO little
# to climb (tools/nemov3_finetune.md, Exp 11 -> Exp 12).
NEMOV3_GRAD_PENALTY_LAMBDA = 1.0

# Weight of the PPO mirror loss (mean-action equivariance MSE) for the -SymLoss task. At the Exp 23 @1600
# start the loss is ~0.13 (per-dim RMS asymmetry ~0.36 rad-units); 10 makes it dominate the first updates so
# the policy is symmetrised quickly, while the adaptive (KL-based) learning rate bounds the step size.
NEMOV3_MIRROR_LOSS_COEFF = 10.0


def nemov3_amp_ppo_runner_cfg(
    use_muon: bool = False,
    dataset_root: str = NEMOV3_AMP_DATASET_ROOT,
    mirror: bool = False,
    symmetry: bool = False,
    mirror_loss_coeff: float = 0.0,
    amp_extra_obs_fn: str | None = None,
    dataset_weights: list[float] | None = None,
    dataset_transform: str | None = None,
) -> AmpOnPolicyRunnerCfg:
    """Create AMP runner configuration for the Nemov3 velocity task.

    ``amp_extra_obs_fn`` computes extra per-frame expert features for the discriminator (``MotionData -> (T, E)``,
    matching the last terms of the "amp" observation group). ``dataset_weights`` follow the sorted clip files
    (skating01, skating02, skatingfast06); each clip's weight is its share of the expert samples and of the reset
    poses, independent of the clip length. ``dataset_transform`` is applied to every loaded clip (e.g. instance noise on
    channels a clip did not capture).
    """
    algorithm_cls = AmpRslRlMuonPpoAlgorithmCfg if use_muon else AmpRslRlPpoAlgorithmCfg
    cfg = _nemov3_amp_ppo_runner_cfg(algorithm_cls, dataset_root, mirror, use_muon)
    cfg.amp_extra_obs_fn = amp_extra_obs_fn
    cfg.dataset_weights = dataset_weights
    cfg.dataset_transform = dataset_transform
    if symmetry or mirror_loss_coeff > 0.0:
        # Left/right symmetry through the PPO symmetry extension (same mechanism as K1's "Flat-Amp-DA" task,
        # with a Nemov3-specific mirror function; the clock mirrors to phase + pi). Needs a mirror-symmetric
        # default pose (env symmetric_default=True).
        # symmetry=True: data augmentation. It copies the old log-probs of each sample to its mirrored copy,
        # which is only valid for an already-symmetric policy: resumed from an asymmetric checkpoint the mirrored
        # importance ratios are ~exp(-13), so the mirrored half contributes no gradient and just dilutes the
        # surrogate against the entropy bonus (Exp 32: std 0.40 -> 0.49, symmetry loss 0.13 -> 0.33).
        # mirror_loss_coeff > 0: MSE between the mean action on the mirrored observation and the mirrored mean
        # action (no importance ratios), stable when resuming an asymmetric policy (Exp 33+).
        cfg.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=symmetry,
            use_mirror_loss=mirror_loss_coeff > 0.0,
            mirror_loss_coeff=mirror_loss_coeff,
            data_augmentation_func="booster_mjlab.tasks.velocity.mdp.nemov3_symmetry:augment_symmetries",
        )
    return cfg


def _nemov3_amp_ppo_runner_cfg(algorithm_cls, dataset_root, mirror, use_muon) -> AmpOnPolicyRunnerCfg:
    return AmpOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "rsl_rl.modules.distribution:GaussianDistribution",
                "init_std": 1.0,
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        discriminator=AmpDiscriminatorCfg(
            hidden_layer_sizes=(256, 128),
            reward_scale=1.0,
            reward_clamp_epsilon=1.0e-4,
            loss_type="bce",
            loss_fn_kwargs={},
        ),
        algorithm=algorithm_cls(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            # Halved from K1's 0.01: Nemov3's action space is 32-dim (30
            # position + 2 wheel velocity) vs. K1's 22-dim, and a diagonal
            # Gaussian's entropy is additive across dims, so the same
            # coefficient exerts a stronger absolute pull toward higher
            # exploration here. Confirmed via a first training run
            # (nemov3_velocity_amp_run1): Policy/mean_std and Loss/entropy
            # bottomed out around iteration 200 then climbed for the rest of
            # the run (0.49->0.72 by it. 500) instead of continuing to
            # converge the way K1's did (monotonic 0.99->0.43 over 2000
            # iterations) -- everything downstream (action_rate_l2,
            # dof_pos_limits, AMP task_reward) degraded in lockstep with
            # that entropy climb. KL divergence and value loss stayed normal
            # throughout, ruling out a step-size/PPO-update issue.
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            grad_penalty_lambda=NEMOV3_GRAD_PENALTY_LAMBDA,
        ),
        experiment_name="nemov3_velocity_amp" + ("_muon" if use_muon else ""),
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=30_000,
        speed_factor=1.0,
        dataset_root=dataset_root,
        # No "mirror" entry: MirrorAugmentation hard-codes K1's 22-DOF joint
        # layout (see motion/motion_augmentations.py) and would raise on
        # Nemov3's 32-DOF clips. The reference set is only 3 short clips, so
        # speed augmentation carries more of the diversity burden here than
        # it does for K1 -- widened to +/-30% accordingly.
        dataset_augmentations=[
            # Left/right mirrored copies of every clip (x every speed variant), so the
            # discriminator sees both "right leg leads" and "left leg leads": Skating01 has
            # R/L hip-pitch std 0.35/0.27 and calf 0.20/0.13, Skating02 an uneven R->L / L->R
            # timing (0.75 / 0.60 s). Left out by default (Exp <= 29 trained without it).
            *([{"name": "mirror", "robot": "nemov3"}] if mirror else []),
            {"name": "speed", "percent": 10.0},
            {"name": "speed", "percent": -10.0},
            {"name": "speed", "percent": 20.0},
            {"name": "speed", "percent": -20.0},
            {"name": "speed", "percent": 30.0},
            {"name": "speed", "percent": -30.0},
        ],
        style_reward_weight=NEMOV3_STYLE_REWARD_WEIGHT,
        num_amp_obs_steps=NEMOV3_AMP_HISTORY_STEPS,
    )
