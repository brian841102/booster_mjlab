"""SuperMimic runner configuration (paper Table A.3 hyperparameters)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mjlab.rl import RslRlBaseRunnerCfg


@dataclass
class SuperMimicRunnerCfg(RslRlBaseRunnerCfg):
    """Collect -> fit the world model -> BPTT policy update through it + critic (Algorithm A.1).

    One iteration collects ``1 / utd`` vectorized simulator steps (on average), then runs ``wm_batches`` world-model
    updates and one policy + critic update. ``num_steps_per_env``, ``obs_groups`` and ``clip_actions`` of the base
    config are unused.
    """

    class_name: str = "SuperMimicRunner"
    experiment_name: str = "supermimic"
    logger: Literal["wandb", "tensorboard"] = "tensorboard"
    """Scalars also go to ``train_log.csv`` in the run directory."""
    max_iterations: int = 12_000
    save_interval: int = 1000
    """Checkpoint every ``save_interval`` iterations up to ``save_late_from``, then every ``save_interval_late``."""
    save_interval_late: int = 500
    save_late_from: int = 3000
    log_interval: int = 50

    buffer_steps: int = 1000
    """Replay buffer length in vectorized steps (x num_envs transitions)."""
    utd: float = 0.4
    """Updates per vectorized simulator step."""
    horizon: int = 32
    """Imagination horizon of the policy update."""
    goal_frames: int = 10
    """Future reference frames in the goal."""
    hist: int = 10
    """Observation history length of the policy."""
    wm_window: int = 8
    """World-model multi-step rollout length."""
    wm_batches: int = 10
    wm_batch: int = 2048
    pi_batch: int = 4096
    lr: float = 2e-4
    lr_decay_iters: int = 250_000
    """Cosine decay to 3% of ``lr`` over this many iterations."""
    gamma: float = 0.99
    lam: float = 0.95
    action_noise: float = 0.1
    """Gaussian exploration noise on the action during collection."""
    critic_epochs: int = 5
    critic_minibatches: int = 4
    use_critic: bool = True
    warmup_steps: int = 64
    """Simulator steps before any update."""
    wm_only_iters: int = 500
    """World-model-only iterations at the start."""
    wm_action_hist: int = 0
    """Previous actions (a_{t-1}, ..., a_{t-H}) the world model also sees. With actuator delays the executed command
    is an older action still in the delay buffer; without them the world model cannot know it (exp2 / 排查 1 in
    tools/nemov3_finetune_supermimic.md). Not in the paper."""
    amp_dtype: Literal["bf16", "none"] = "bf16"
    """Autocast dtype for the MLP matmuls."""
