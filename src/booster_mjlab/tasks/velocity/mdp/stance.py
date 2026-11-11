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
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, limit: float, outward_signs: tuple[float, ...] | None = None
) -> torch.Tensor:
    """Squared hinge on the (outward-positive) hip roll angles above ``limit`` [rad], summed over both hips.

    For both nemov3 hips a positive roll widens the stance (measured: wheel track 0.196 / 0.226 / 0.258 m at hip
    roll ~0.035 / 0.08 / 0.13 rad in the Exp 23 policy at 0.6 / 1.0 / 1.3 m/s). Robots whose hip roll axes are not
    mirrored (pinov1: the right hip spreads with a negative roll) give the outward sign per joint in ``outward_signs``
    (asset_cfg joint order, use preserve_order=True).
    """
    robot = env.scene[asset_cfg.name]
    q = robot.data.joint_pos[:, asset_cfg.joint_ids]
    if outward_signs is not None:
        q = q * torch.tensor(outward_signs, device=q.device, dtype=q.dtype)
    return torch.sum((q - limit).clamp_min(0.0).square(), dim=1)


def hip_roll_inward_backward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    limit: float,
    outward_signs: tuple[float, ...],
    command_name: str = "twist",
    full_at: float = 0.1,
    turn_full_at: float | None = None,
    standing: bool = False,
) -> torch.Tensor:
    """Squared hinge on the hip roll angles *inward* of ``limit`` [rad] (outward-positive, as :func:`hip_roll_excess`),
    summed over both hips, only while driving backward: scaled by clamp(-forward command / ``full_at``, 0, 1).

    Pino v1 Exp 31 (user 2026-11-05: "后退时脚微微开不要夹顶到限位"): Exp 30 driven at -0.5 m/s held both hips at the
    inward mechanical stop (-0.08 rad) 99 % of the time, the rear wheels 14 cm apart. ``turn_full_at`` [rad/s] extends it
    to turning (Exp 35).
    """
    robot = env.scene[asset_cfg.name]
    q = robot.data.joint_pos[:, asset_cfg.joint_ids] * torch.tensor(outward_signs, device=env.device)
    cmd = env.command_manager.get_command(command_name)
    gate = (-cmd[:, 0] / full_at).clamp(0.0, 1.0)
    if turn_full_at is not None:
        # Exp 35: also while turning (|yaw-rate command| ramps 0 -> turn_full_at), in place or while moving -- Exp 34 rolled
        # the hips in when backing up with a turn (90 % of the robots) and in place (100 %).
        gate = torch.maximum(gate, (cmd[:, 2].abs() / turn_full_at).clamp(0.0, 1.0))
    if standing:
        # Autonomous loop (Exp 42): also while commanded to stand (|v| + |wz| below 0.05, ramping to 0 at 0.05) -- Exp 36
        # stood with the hips rolled in (50 % of the time past the soft limit).
        speed = torch.norm(cmd[:, :2], dim=1) + cmd[:, 2].abs()
        gate = torch.maximum(gate, 1.0 - (speed / 0.05).clamp(max=1.0))
    return torch.sum((limit - q).clamp_min(0.0).square(), dim=1) * gate



def in_place_hip_yaw_toe_in(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    asym_tolerance: float = 0.05,
    min_toe_in: float = 0.15,
    command_name: str = "twist",
    standing_speed: float = 0.05,
    turn_full_at: float = 0.1,
) -> torch.Tensor:
    """Symmetric toe-in of both hip yaws while turning in place (Pino v1 Exp 49a, user 2026-11-06: "对称而且要内八").

    ``asset_cfg`` holds (left, right) hip yaw in that order (``preserve_order``). On Pino v1 left - / right + is toe-in, so
    asymmetry = |q_L + q_R| and toe-in = (q_R - q_L) / 2; the penalty is
    max(0, asymmetry - ``asym_tolerance``)^2 + max(0, ``min_toe_in`` - toe-in)^2, scaled by the in-place gate
    (1 - clamp(|v_xy command| / ``standing_speed``)) * clamp(|yaw-rate command| / ``turn_full_at``). Exp 31 turned in place
    at L -0.23 / R +0.22; Exp 47 / 48 at L -0.5 / R -0.3 (both feet to one side) and turned half as fast.
    """
    q = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    asym = (q[:, 0] + q[:, 1]).abs()
    toe_in = 0.5 * (q[:, 1] - q[:, 0])
    penalty = (asym - asym_tolerance).clamp_min(0.0).square() + (min_toe_in - toe_in).clamp_min(0.0).square()
    cmd = env.command_manager.get_command(command_name)
    gate = (1.0 - (torch.norm(cmd[:, :2], dim=1) / standing_speed).clamp(max=1.0)) * (cmd[:, 2].abs() / turn_full_at).clamp(0.0, 1.0)
    return penalty * gate


def in_place_hip_yaw_min_toe_in(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    min_toe_in: float = 0.22,
    command_name: str = "twist",
    standing_speed: float = 0.05,
    turn_full_at: float = 0.1,
) -> torch.Tensor:
    """Each foot at least ``min_toe_in`` [rad] toe-in while turning in place (Pino v1 Exp 49a3, user 2026-11-06: "强置把脚约束
    到内八至少预设位置和exp31相同程度", replacing in_place_hip_yaw_toe_in, whose asymmetry term let one foot make up for
    the other).

    ``asset_cfg`` holds (left, right) hip yaw in that order (``preserve_order``); toe-in is -q_L and +q_R. Linear hinge
    sum_i max(0, ``min_toe_in`` - toe-in_i) (a constant push all the way to the bound, not a squared one that fades near
    it), scaled by the in-place gate (1 - clamp(|v_xy command| / ``standing_speed``)) * clamp(|yaw-rate command| /
    ``turn_full_at``). Exp 31 turned in place at L -0.23 / R +0.22.
    """
    q = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    toe_in = torch.stack((-q[:, 0], q[:, 1]), dim=1)
    penalty = (min_toe_in - toe_in).clamp_min(0.0).sum(dim=1)
    cmd = env.command_manager.get_command(command_name)
    gate = (1.0 - (torch.norm(cmd[:, :2], dim=1) / standing_speed).clamp(max=1.0)) * (cmd[:, 2].abs() / turn_full_at).clamp(0.0, 1.0)
    return penalty * gate


def backward_hip_pitch_stagger(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    tolerance: float = 0.05,
    command_name: str = "twist",
    full_at: float = 0.1,
    turn_off_at: float = 0.1,
) -> torch.Tensor:
    """Fore-aft foot stagger while driving straight backward (Pino v1 Exp 56, user 2026-11-10: "直线后退（不包括转弯）时
    约束一下脚错开的姿势").

    Exp 44 / 54 / 55 back up with one foot 7-10 cm ahead of the other (hip pitch L - R 0.3-0.45 rad; knees within 0.05),
    Exp 31 with both feet side by side (0.03). ``asset_cfg`` holds (left, right) hip pitch. Squared hinge
    max(0, |q_L - q_R| - ``tolerance``)^2, scaled by the straight-backward gate clamp(-vx command / ``full_at``, 0, 1) *
    (1 - clamp(|yaw-rate command| / ``turn_off_at``)) -- off when turning.
    """
    q = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    penalty = ((q[:, 0] - q[:, 1]).abs() - tolerance).clamp_min(0.0).square()
    cmd = env.command_manager.get_command(command_name)
    gate = (-cmd[:, 0] / full_at).clamp(0.0, 1.0) * (1.0 - (cmd[:, 2].abs() / turn_off_at).clamp(max=1.0))
    return penalty * gate


def turning_hip_yaw_deviation(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    tolerance: float = 0.1,
    command_name: str = "twist",
    turn_full_at: float = 0.1,
    forward_off_at: float = 0.05,
) -> torch.Tensor:
    """Hip yaws away from the default pose while turning in place or backing up with a turn (Pino v1 Exp 57, user
    2026-11-10: "转弯时罚hip yaw偏离默认" -> "只在原地转和后退转弯生效").

    Exp 55 / 56 turned by yawing both feet to the same side (one toe-out ~0.35, the other toe-in ~0.4-0.5) and the toe-out
    foot's hip was pushed into roll-in by the ground although the policy commanded it outward; Exp 54 turned with the feet
    nearly straight (within ~0.16 of the default) on the rear-wheel differential, faster and without roll-in.
    Sum over the joints of max(0, |q - q_default| - ``tolerance``)^2, scaled by clamp(|yaw-rate command| / ``turn_full_at``)
    * (1 - clamp(forward command / ``forward_off_at``, 0, 1)) -- off while driving forward (forward turns showed no roll-in
    and skating forward may use the hip yaws), on in place and backward.
    """
    asset = env.scene[asset_cfg.name]
    deviation = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    penalty = (deviation.abs() - tolerance).clamp_min(0.0).square().sum(dim=1)
    cmd = env.command_manager.get_command(command_name)
    gate = (cmd[:, 2].abs() / turn_full_at).clamp(0.0, 1.0) * (1.0 - (cmd[:, 0] / forward_off_at).clamp(0.0, 1.0))
    return penalty * gate
