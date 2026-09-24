"""Stance-width constraint via the hip-roll joints.

NOTE (future foot-lift training): this penalty limits how far the hips may abduct
(feet spread). Fast06's foot-lifting reaches hip roll ~0.25 rad, i.e. beyond the
limit used here, so it may block the freedom a lifting gait needs -- remove or
relax it (``limit``) when training foot lifts again.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def hip_roll_excess(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, limit: float
) -> torch.Tensor:
    """Squared hinge on the (outward-positive) hip roll angles above ``limit`` [rad], summed over both hips.

    For both hips a positive roll widens the stance (measured: wheel track 0.196 / 0.226 / 0.258 m at hip
    roll ~0.035 / 0.08 / 0.13 rad in the Exp 23 policy at 0.6 / 1.0 / 1.3 m/s).
    """
    robot = env.scene[asset_cfg.name]
    q = robot.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.sum((q - limit).clamp_min(0.0).square(), dim=1)

