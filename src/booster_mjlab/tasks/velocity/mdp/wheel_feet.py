"""K1-style swing-height / air-time rewards adapted to a wheeled foot.

K1's foot rewards measure height with a terrain height scan and treat "foot
speed" as swing speed. A wheel keeps rolling at ~the robot speed while it is
planted, so those terms are re-expressed here with (a) the wheel bottom height
``z_center - wheel_radius`` (independent of foot tilt) and (b) the wheel
contact sensor to tell swing (air) from stance, so a planted rolling wheel is
never penalised. All terms fade in with the commanded speed (``ramp_lo`` ->
``ramp_hi``) instead of switching on at one threshold, to avoid a sudden
style change at some speed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def _speed_ramp(
    env: ManagerBasedRlEnv, command_name: str, lo: float, hi: float
) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    speed = torch.norm(cmd[:, :2], dim=1)
    return ((speed - lo) / max(hi - lo, 1e-6)).clamp(0.0, 1.0)


def _wheel_height(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, radius: float):
    robot = env.scene[asset_cfg.name]
    return (robot.data.body_link_pos_w[:, asset_cfg.body_ids, 2] - radius).clamp_min(0.0)


def wheel_air_time(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    ramp_lo: float,
    ramp_hi: float,
    threshold_min: float = 0.05,
    threshold_max: float = 0.7,
) -> torch.Tensor:
    """K1 ``feet_air_time`` with a smooth speed ramp: +1 per wheel whose current air time is in range."""
    sensor = env.scene[sensor_name]
    air = sensor.data.current_air_time
    in_range = ((air > threshold_min) & (air < threshold_max)).float().sum(dim=1)
    return in_range * _speed_ramp(env, command_name, ramp_lo, ramp_hi)


def wheel_clearance(
    env: ManagerBasedRlEnv,
    target_height: float,
    sensor_name: str,
    command_name: str,
    ramp_lo: float,
    ramp_hi: float,
    asset_cfg: SceneEntityCfg,
    wheel_radius: float = 0.075,
) -> torch.Tensor:
    """K1 ``feet_clearance`` restricted to swing wheels: ``|h - target| * horizontal speed`` while in the air."""
    sensor = env.scene[sensor_name]
    in_air = (sensor.data.found == 0).float()
    robot = env.scene[asset_cfg.name]
    h = _wheel_height(env, asset_cfg, wheel_radius)
    vel = torch.norm(robot.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1)
    cost = torch.sum(in_air * (h - target_height).abs() * vel, dim=1)
    return cost * _speed_ramp(env, command_name, ramp_lo, ramp_hi)


class wheel_swing_height:
    """K1 ``feet_swing_height``: penalise ``(peak_height / target - 1)^2`` of each swing wheel at landing."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        n = len(cfg.params["asset_cfg"].body_names)
        self.peak = torch.zeros((env.num_envs, n), device=env.device)
        self.step_dt = env.step_dt

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        target_height: float,
        command_name: str,
        ramp_lo: float,
        ramp_hi: float,
        asset_cfg: SceneEntityCfg,
        wheel_radius: float = 0.075,
    ) -> torch.Tensor:
        sensor = env.scene[sensor_name]
        h = _wheel_height(env, asset_cfg, wheel_radius)
        in_air = sensor.data.found == 0
        self.peak = torch.where(in_air, torch.maximum(self.peak, h), self.peak)
        first_contact = sensor.compute_first_contact(dt=self.step_dt)
        err = self.peak / target_height - 1.0
        cost = torch.sum(err.square() * first_contact.float(), dim=1)
        self.peak = torch.where(first_contact, torch.zeros_like(self.peak), self.peak)
        return cost * _speed_ramp(env, command_name, ramp_lo, ramp_hi)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.peak[env_ids] = 0.0


def wheel_lift(
    env: ManagerBasedRlEnv,
    target_height: float,
    sensor_name: str,
    command_name: str,
    ramp_lo: float,
    ramp_hi: float,
    asset_cfg: SceneEntityCfg,
    wheel_radius: float = 0.075,
    threshold_min: float = 0.02,
    threshold_max: float = 0.9,
    rear_scale: float = 0.05,
    height_power: float = 0.5,
) -> torch.Tensor:
    """Positive, barrier-free lift reward (replaces K1's penalty-only shaping, which never got the wheels off the ground).

    Per wheel: ``in_air(0.02 < air_time < 0.9 s) * min(h / target, 1)^p * rear_gate``, summed over the two wheels and
    faded in with the commanded speed. ``h`` is the wheel-bottom height, so any partial lift already pays and
    the reward saturates at ``target_height``; ``p < 1`` (0.5) makes the first centimetres pay disproportionately
    (Exp 26 with p = 1 and weight 1.0 never left 0). The air-time cap means holding a wheel up forever stops paying
    after 0.9 s, so the policy has to land and lift again (alternation). ``rear_gate`` is a soft indicator
    that this wheel is behind the other one along the robot's forward axis (Fast06 lifts the rear foot,
    ~0.21 m behind at the peak), so lifting the front foot pays much less.
    """
    assert len(asset_cfg.body_ids) == 2, "wheel_lift expects exactly two wheel bodies"
    robot = env.scene[asset_cfg.name]
    sensor = env.scene[sensor_name]
    air = sensor.data.current_air_time
    in_window = ((air > threshold_min) & (air < threshold_max)).float()
    h = _wheel_height(env, asset_cfg, wheel_radius)
    height_frac = (h / target_height).clamp(0.0, 1.0) ** height_power
    pos = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    d_base = quat_apply_inverse(robot.data.root_link_quat_w, pos[:, 0] - pos[:, 1])
    dx = d_base[:, 0]
    rear_gate = torch.stack(
        [torch.sigmoid(-dx / rear_scale), torch.sigmoid(dx / rear_scale)], dim=1
    )
    value = torch.sum(in_window * height_frac * rear_gate, dim=1)
    return value * _speed_ramp(env, command_name, ramp_lo, ramp_hi)
