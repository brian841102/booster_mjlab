"""RL configuration for the Pino v1 velocity task: the nemov3 AMP runner settings (config/nemov3/rl_cfg.py, where each
value's history is documented) with pinov1's clip, mirror table and symmetry function."""

from booster_mjlab.amp.runners import AmpOnPolicyRunnerCfg
from booster_mjlab.rl.config import RslRlSymmetryCfg
from booster_mjlab.tasks.velocity.config.nemov3.rl_cfg import nemov3_amp_ppo_runner_cfg

# Produced by `uv run python -m booster_mjlab.scripts.pinov1_skating_to_pkl` (one clip, 4.07 s at 0.54-1.37 m/s).
PINOV1_AMP_DATASET_ROOT = "assets/amp/pinov1_skating"

# The nemov3 Exp 33 clip set (Skating01 / 02 / trimmed SkatingFast06) retargeted to pinov1 (user 2026-11-04).
PINOV1_AMP_DATASET_ROOT_STRAIGHT = "assets/amp/pinov1_skating_straight"

PINOV1_MIRROR_LOSS_COEFF = 10.0


def pinov1_amp_ppo_runner_cfg(
    use_muon: bool = False,
    style_standing_mask: bool = False,
    entropy_coef: float | None = None,
    dataset_root: str = PINOV1_AMP_DATASET_ROOT,
    amp_extra_obs_fn: str | None = None,
    amp_soft_condition: bool = False,
    mirror_loss_coeff: float = PINOV1_MIRROR_LOSS_COEFF,
) -> AmpOnPolicyRunnerCfg:
    """nemov3's runner (K = 30 frames, gradient penalty 1, style weight 0.7, speed augmentation +-10/20/30 %) with the
    pinov1 clip mirrored by pinov1's joint table and the PPO mirror loss (nemov3 Exp 33). ``style_standing_mask``: no
    style reward while commanded to stand (the clip starts at 0.54 m/s, it has no standing still). ``entropy_coef``
    overrides nemov3's 0.005 (Exp 12: with the standing terms the action std climbed 0.43 -> 0.61 instead of settling at
    ~0.34 as in Exp 1 / 4). ``amp_extra_obs_fn``: expert side of extra discriminator features (Exp 26: the fast-clip
    label, mdp.fast_style_label). ``amp_soft_condition``: continuous speed condition instead (Exp 27). ``mirror_loss_coeff``: PPO mirror
    loss weight (Exp 29: 50, as nemov3 Exp 42)."""
    cfg = nemov3_amp_ppo_runner_cfg(
        use_muon=use_muon, dataset_root=dataset_root, mirror=True, amp_extra_obs_fn=amp_extra_obs_fn
    )
    cfg.experiment_name = "pinov1_velocity_amp" + ("_muon" if use_muon else "")
    cfg.amp_soft_condition = amp_soft_condition
    if style_standing_mask:
        cfg.style_mask_command = "twist"
    if entropy_coef is not None:
        cfg.algorithm.entropy_coef = entropy_coef
    cfg.dataset_augmentations = [
        {"name": "mirror", "robot": "pinov1"} if aug.get("name") == "mirror" else aug for aug in cfg.dataset_augmentations
    ]
    cfg.algorithm.symmetry_cfg = RslRlSymmetryCfg(
        use_data_augmentation=False,
        use_mirror_loss=True,
        mirror_loss_coeff=mirror_loss_coeff,
        data_augmentation_func="booster_mjlab.tasks.velocity.mdp.pinov1_symmetry:augment_symmetries",
    )
    return cfg
