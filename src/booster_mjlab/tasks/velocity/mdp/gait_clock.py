"""Periodic gait clock for wheeled skating (observation only).

A biped can't move without stepping, so an AMP style reward alone is enough to
shape its gait. A wheeled robot can satisfy velocity tracking with the wheels
alone, and a memoryless MLP policy has no time signal to build a periodic gait
from. This command is a free-running phase the policy observes as (sin, cos);
there is no gait reward -- the style is left entirely to the AMP discriminator
(see tools/nemov3_finetune.md, Exp 17).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_TWO_PI = 2.0 * math.pi


class GaitClockCommand(CommandTerm):
    """Per-env phase that advances at a fixed frequency; observed as (sin, cos)."""

    cfg: GaitClockCommandCfg

    def __init__(self, cfg: GaitClockCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.phase = torch.rand(self.num_envs, device=self.device) * _TWO_PI
        self._command = torch.zeros(self.num_envs, 2, device=self.device)
        self._refresh_command()

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _refresh_command(self) -> None:
        self._command[:, 0] = torch.sin(self.phase)
        self._command[:, 1] = torch.cos(self.phase)

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        # Random starting phase per episode so the policy can't rely on a fixed one.
        self.phase[env_ids] = torch.rand(len(env_ids), device=self.device) * _TWO_PI
        self._refresh_command()

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        # Advance only on the per-step path (env_ids is None); the reset path
        # passes the reset env ids and must not tick the clock.
        if env_ids is None:
            step_dt = self._env.step_dt
            self.phase = (self.phase + _TWO_PI * self.cfg.frequency_hz * step_dt) % _TWO_PI
        self._refresh_command()


@dataclass(kw_only=True)
class GaitClockCommandCfg(CommandTermCfg):
    frequency_hz: float = 0.8
    """Gait frequency. The reference skating clips have a stride period of
    ~1.05-1.33 s, i.e. 0.75-0.95 Hz, with left/right ~180 degrees apart."""

    def build(self, env: ManagerBasedRlEnv) -> GaitClockCommand:
        return GaitClockCommand(self, env)

