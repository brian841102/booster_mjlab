"""Speed-conditioned AMP style: one extra discriminator feature telling which clip style is expected.

Pino v1 (Exp 26, user 2026-11-05: "选1 ... 目标为强化高速时的大摆手风格"): Exp 25 at 0.9 m/s split into two styles per
robot -- ~31 % swung the arms like SkatingFast06 (elbow std 0.31, shoulder pitch std 0.35 in the clip), the rest kept the
small swing of Skating01 / 02 -- because the discriminator does not know the speed, so both count as real. With this
label the expert side is 1 on SkatingFast06 (every mirror / speed variant) and 0 on the other clips; the policy side is
1 while the forward command is >= ``threshold``. Fast commands must then look like Fast06, slow ones like Skating01 / 02.
"""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv

FAST_STYLE_CLIP = "skatingfast06"
FAST_STYLE_COMMAND_THRESHOLD = 0.6


def fast_command_label(
    env: ManagerBasedRlEnv, command_name: str = "twist", threshold: float = FAST_STYLE_COMMAND_THRESHOLD
) -> torch.Tensor:
    """Policy side: 1 while the forward command is >= ``threshold`` [m/s], else 0; shape (N, 1)."""
    return (env.command_manager.get_command(command_name)[:, 0:1] >= threshold).float()


def fast_clip_label_from_motion(data) -> torch.Tensor:
    """Expert side (``amp_extra_obs_fn``): 1 on every frame of SkatingFast06, 0 on the other clips; shape (T, 1)."""
    assert data.clip_name is not None, "the motion loader sets clip_name for local clip files"
    value = 1.0 if data.clip_name == FAST_STYLE_CLIP else 0.0
    return torch.full((data.num_frames, 1), value, device=data.joint_positions.device)


def command_speed_fraction(
    env: ManagerBasedRlEnv, command_name: str = "twist", max_speed: float = 0.9
) -> torch.Tensor:
    """Policy side of the continuous condition (Exp 27, arXiv 2605.18611): v = clamp(forward command / ``max_speed``,
    0, 1), shape (N, 1). The expert side keeps :func:`fast_clip_label_from_motion`; the runner's ``amp_soft_condition``
    samples each expert history from Fast06 with probability v of a policy sample and writes v into it."""
    return (env.command_manager.get_command(command_name)[:, 0:1] / max_speed).clamp(0.0, 1.0)
