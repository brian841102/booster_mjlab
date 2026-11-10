"""Per-environment random gain on a joint-velocity action (Exp 91).

The wheel loop of the Exp 90 policy is stable only for a narrow wheel-action gain (about x0.7..x1.2 in simulation); the
hardware sits at the edge of it and rocks at ~5.8 Hz when standing. Drawing a gain per environment at every reset makes the
policy see a range of plant gains in training, so it needs no compromise between standing (low gain) and speed (full gain).
The gain multiplies the processed target only: the observation still carries the raw policy action, as on the hardware where
``onnx_wheel_action_scale`` scales the output after the policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.actions.actions import JointVelocityAction, JointVelocityActionCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class RandomGainJointVelocityActionCfg(JointVelocityActionCfg):
    gain_range: tuple[float, float] = (1.0, 1.0)
    """Uniform range of the per-episode gain; (1, 1) = no randomization (the play / evaluation setting)."""

    def build(self, env: ManagerBasedRlEnv) -> RandomGainJointVelocityAction:
        return RandomGainJointVelocityAction(self, env)


class RandomGainJointVelocityAction(JointVelocityAction):
    cfg: RandomGainJointVelocityActionCfg

    def __init__(self, cfg: RandomGainJointVelocityActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)
        self._gain = torch.ones(self.num_envs, 1, device=self.device)
        self._resample(torch.arange(self.num_envs, device=self.device))

    @property
    def gain(self) -> torch.Tensor:
        return self._gain

    def _resample(self, env_ids: torch.Tensor) -> None:
        lo, hi = self.cfg.gain_range
        self._gain[env_ids] = torch.empty(len(env_ids), 1, device=self.device).uniform_(float(lo), float(hi))

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        self._processed_actions = self._processed_actions * self._gain

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        ids = torch.arange(self.num_envs, device=self.device)
        if env_ids is not None and not isinstance(env_ids, slice):
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        elif isinstance(env_ids, slice):
            ids = ids[env_ids]
        self._resample(ids)
