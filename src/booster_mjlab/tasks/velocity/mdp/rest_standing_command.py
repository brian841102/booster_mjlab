"""Velocity command whose standing environments start at rest in the default pose.

The AMP pose-pool reset (mdp/events.py:reset_from_pose_pool) starts every environment from a reference frame, root
velocity included (~1 m/s for the skating clips). A robot commanded to stand then has to brake from that speed before
it can stand at all, and learns to step in place instead (Pino v1 Exp 1-6, tools/pinov1_finetune.md). The reset events
run before the command resample, so they cannot know which environments will stand; this command term does it after
its own resample: the environments that drew "standing" get the default pose, no velocity and zero yaw.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import torch
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class RestStandingVelocityCommand(UniformVelocityCommand):
    def reset(self, env_ids):
        extras = super().reset(env_ids)
        assert isinstance(env_ids, torch.Tensor)
        stand_ids = env_ids[self.is_standing_env[env_ids]]
        if len(stand_ids) > 0:
            robot = self.robot
            root = robot.data.default_root_state[stand_ids].clone()
            root[:, 0:3] += self._env.scene.env_origins[stand_ids]
            robot.write_root_link_pose_to_sim(root[:, 0:7], env_ids=stand_ids)
            robot.write_root_link_velocity_to_sim(torch.zeros(len(stand_ids), 6, device=self.device), env_ids=stand_ids)
            joint_pos = robot.data.default_joint_pos[stand_ids].clone()
            robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=stand_ids)
        return extras


@dataclass(kw_only=True)
class RestStandingVelocityCommandCfg(UniformVelocityCommandCfg):
    def build(self, env: ManagerBasedRlEnv) -> RestStandingVelocityCommand:
        return RestStandingVelocityCommand(self, env)

    @classmethod
    def from_cfg(cls, cfg: UniformVelocityCommandCfg) -> RestStandingVelocityCommandCfg:
        """The same command configuration, built into the rest-standing term."""
        return cls(**{f.name: getattr(cfg, f.name) for f in fields(cfg) if f.init})
