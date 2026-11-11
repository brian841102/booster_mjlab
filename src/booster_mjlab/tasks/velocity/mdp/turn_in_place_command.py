"""Velocity command with an extra share of in-place turning (forward and lateral 0, yaw rate from the usual range).

Pino v1 autonomous loop (2026-11-06): with the forward command uniform in [-0.5, 0.9] m/s only ~14 % of the
environments get |vx| < 0.1 together with a yaw-rate command, and Exp 33-35 (arms 25 / 1.0) turned in place at only
0.2-0.37 rad/s for a 0.6 / 0.9 rad/s command (Exp 31: 0.31 / 0.52).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import torch
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class TurnInPlaceVelocityCommand(UniformVelocityCommand):
    cfg: TurnInPlaceVelocityCommandCfg

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        r = torch.empty(len(env_ids), device=self.device)
        pick = (r.uniform_(0.0, 1.0) < self.cfg.rel_turn_in_place_envs) & ~self.is_standing_env[env_ids] & ~self.is_forward_env[env_ids]
        ids = env_ids[pick]
        if len(ids) > 0:
            self.vel_command_b[ids, 0] = 0.0
            self.vel_command_b[ids, 1] = 0.0
            self.vel_command_w[ids] = self.vel_command_b[ids]


@dataclass(kw_only=True)
class TurnInPlaceVelocityCommandCfg(UniformVelocityCommandCfg):
    rel_turn_in_place_envs: float = 0.0
    """Share of the (non-standing, non-forward-only) environments whose forward / lateral command is set to 0, keeping
    the sampled yaw rate."""

    def build(self, env: ManagerBasedRlEnv) -> TurnInPlaceVelocityCommand:
        return TurnInPlaceVelocityCommand(self, env)

    @classmethod
    def from_cfg(cls, cfg: UniformVelocityCommandCfg, rel_turn_in_place_envs: float) -> TurnInPlaceVelocityCommandCfg:
        return cls(**{f.name: getattr(cfg, f.name) for f in fields(cfg) if f.init}, rel_turn_in_place_envs=rel_turn_in_place_envs)
