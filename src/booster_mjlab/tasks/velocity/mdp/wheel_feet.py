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

import math
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


def wheel_height_diff(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """AMP feature: right minus left wheel-center height [m], shape (N, 1).

    The discriminator otherwise only sees joint angles, which cannot tell a lifted wheel from a crouch with both
    wheels on the ground. The difference needs no ground reference (the reference clips float the support wheel by
    0.8-2 cm) and flips sign under the left/right mirror. ``asset_cfg`` must list the right wheel first
    (``preserve_order=True``). Expert side: :func:`wheel_height_diff_from_motion`.
    """
    z = env.scene[asset_cfg.name].data.body_link_pos_w[:, asset_cfg.body_ids, 2]
    return z[:, 0:1] - z[:, 1:2]


def wheel_height_diff_from_motion(data) -> torch.Tensor:
    """Expert side of :func:`wheel_height_diff` for the AMP motion loader (``amp_extra_obs_fn``), shape (T, 1).

    Uses the clip's per-frame body positions in the root frame (``local_body_pos``, mirrored / resampled by the
    augmentations like the joints) rotated by the root quaternion (scipy x, y, z, w). Matches the world-frame wheel
    height difference of the source npz exactly (checked on SkatingFast06: max error 0).
    """
    names = list(data.link_body_list)
    d = data.local_body_pos[:, names.index("right_wheel_link")] - data.local_body_pos[:, names.index("left_wheel_link")]
    q = data.base_quat
    qv, qw = q[:, :3], q[:, 3:4]
    t = 2.0 * torch.cross(qv, d, dim=-1)
    world = d + qw * t + torch.cross(qv, t, dim=-1)
    return world[:, 2:3]


def base_forward_speed(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """AMP feature: base forward speed in the base frame [m/s], shape (N, 1). With it the discriminator judges the
    style per speed (the clips cover ~0.2-0.4 / 0.4-0.8 / 0.7-1.3 m/s for Skating02 / 01 / Fast06 with the +-30%
    speed variants), so above ~0.8 m/s only the lifting Fast06 style counts as real."""
    return env.scene[asset_cfg.name].data.root_link_lin_vel_b[:, 0:1]


def lift_features_from_motion(data) -> torch.Tensor:
    """Expert side of [wheel_height_diff, base_forward_speed], shape (T, 2)."""
    return torch.cat([wheel_height_diff_from_motion(data), data.base_lin_velocities_local[:, 0:1]], dim=-1)


# User (2026-09-25, after watching Exp 49): "抬脚高度也可以低一点比较安全，离地六公分感觉就差不多了" -- the
# reference lifts (Fast06 peaks at 8.5 cm) are shown to the discriminator scaled to ~6 cm.
LIFT_REFERENCE_HEIGHT_SCALE = 6.0 / 8.5


def lift_features_6cm_from_motion(data) -> torch.Tensor:
    """:func:`lift_features_from_motion` with the reference wheel height difference scaled to ~6 cm lifts."""
    feats = lift_features_from_motion(data)
    return torch.cat([feats[:, :1] * LIFT_REFERENCE_HEIGHT_SCALE, feats[:, 1:]], dim=-1)


def base_pitch(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """AMP feature: forward pitch of the root link from gravity [rad] (+ = leaning forward), shape (N, 1)."""
    g = env.scene[asset_cfg.name].data.projected_gravity_b
    return torch.atan2(g[:, 0:1], -g[:, 2:3])


# Instance noise for channels a reference clip did not capture (held constant, e.g. Skating01/02 have no root
# orientation and no shoulder yaw): an exactly constant expert channel lets the discriminator separate the policy by its
# exploration jitter alone and stop scoring the leg swing (Exp 7 with hip roll/yaw; Exp 52 lost its slow-speed swing
# after shoulder / pelvis yaw entered the discriminator). Noise std roughly matches that jitter.
UNCAPTURED_STD_THRESHOLD = 0.005
UNCAPTURED_PITCH_NOISE_RAD = 0.026
UNCAPTURED_JOINT_NOISE_RAD = 0.05
UNCAPTURED_JOINT_NAMES = ("joint_lShoulderYaw", "joint_rShoulderYaw", "joint_pelvisYaw")


def _pitch_from_motion(data) -> torch.Tensor:
    q = data.base_quat  # scipy x, y, z, w
    qv, qw = q[:, :3], q[:, 3:4]
    down = torch.tensor([0.0, 0.0, -1.0]).to(q).expand_as(qv)
    t = 2.0 * torch.cross(-qv, down, dim=-1)  # rotate world gravity into the root frame (inverse rotation)
    g = down + qw * t + torch.cross(-qv, t, dim=-1)
    pitch = torch.atan2(g[:, 0:1], -g[:, 2:3])
    if pitch.std() < UNCAPTURED_STD_THRESHOLD:
        gen = torch.Generator().manual_seed(pitch.shape[0])
        pitch = pitch + UNCAPTURED_PITCH_NOISE_RAD * torch.randn(pitch.shape, generator=gen).to(pitch)
    return pitch


def noise_uncaptured_joints(motion_file):
    """``dataset_transform`` for the AMP expert data: jitter UNCAPTURED_JOINT_NAMES where a clip holds them constant."""
    from booster_mjlab.robots.nemov3.nemov3_constants import NEMOV3_JOINT_ORDER

    import dataclasses

    import numpy as np

    dof = np.array(motion_file.dof_pos, dtype=np.float32, copy=True)
    rng = np.random.default_rng(dof.shape[0])
    for name in UNCAPTURED_JOINT_NAMES:
        i = list(NEMOV3_JOINT_ORDER).index(name)
        if dof[:, i].std() < UNCAPTURED_STD_THRESHOLD:
            dof[:, i] += UNCAPTURED_JOINT_NOISE_RAD * rng.standard_normal(dof.shape[0]).astype(np.float32)
    return dataclasses.replace(motion_file, dof_pos=dof)


# Speed-gated AMP features (Exp 57+): the slow reference is only Skating01 / 02, one 1.34 s stride each -- too little to
# score 15 joints (Exp 53-56: the discriminator memorised the whole-body combination, so swinging the legs alone looked
# *less* real and the slow swing died) -- while Fast06 has 19 s at speed. These channels (the sideways push, diagonal
# arm swing, upper-body twist, forward lean) are multiplied by a gate of the *actual* forward speed, 0 below
# AMP_SPEED_GATE[0] and 1 above AMP_SPEED_GATE[1], on the policy and the reference side alike, so slow skating is judged
# on the Exp 49 features (leg pitch / calf / arm pitch / elbow) and fast skating on all of them.
AMP_SPEED_GATE = (0.6, 1.0)


GATED_AMP_JOINTS = (
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "joint_lShoulderYaw",
    "joint_rShoulderYaw",
    "joint_pelvisYaw",
)


def _speed_gate(v: torch.Tensor) -> torch.Tensor:
    lo, hi = AMP_SPEED_GATE
    return ((v - lo) / (hi - lo)).clamp(0.0, 1.0)


def gated_joint_pos(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """AMP feature: absolute joint angles of ``asset_cfg`` (GATED_AMP_JOINTS, preserve_order) x the speed gate."""
    robot = env.scene[asset_cfg.name]
    gate = _speed_gate(robot.data.root_link_lin_vel_b[:, 0:1])
    return robot.data.joint_pos[:, asset_cfg.joint_ids] * gate


def gated_base_pitch(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """AMP feature: :func:`base_pitch` x the speed gate (lean is scored only at speed; slow clips carry none)."""
    robot = env.scene[asset_cfg.name]
    return base_pitch(env, asset_cfg) * _speed_gate(robot.data.root_link_lin_vel_b[:, 0:1])


def gated_lift_features_from_motion(data) -> torch.Tensor:
    """Expert side of [gated_joint_pos (7), wheel_height_diff (~6 cm), base_forward_speed, gated_base_pitch]."""
    from booster_mjlab.robots.nemov3.nemov3_constants import NEMOV3_JOINT_ORDER

    order = list(NEMOV3_JOINT_ORDER)
    v = data.base_lin_velocities_local[:, 0:1]
    gate = _speed_gate(v)
    q = data.joint_positions[:, [order.index(j) for j in GATED_AMP_JOINTS]] * gate
    height = wheel_height_diff_from_motion(data) * LIFT_REFERENCE_HEIGHT_SCALE
    return torch.cat([q, height, v, _pitch_from_motion(data) * gate], dim=-1)


def wheel_height_excess(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, limit: float, scale: float, wheel_radius: float = 0.075
) -> torch.Tensor:
    """Safety bound on lift height: ``sum(max(h - limit, 0)) / scale`` over the wheels (0 below ``limit``)."""
    h = _wheel_height(env, asset_cfg, wheel_radius)
    return torch.sum((h - limit).clamp_min(0.0), dim=1) / scale


def wheel_jump(
    env: ManagerBasedRlEnv, sensor_name: str, command_name: str, ramp_lo: float, ramp_hi: float
) -> torch.Tensor:
    """[B]'s air-time penalty in our terms: 1 while both wheels are off the ground (a hop), faded in with speed."""
    found = env.scene[sensor_name].data.found
    return (found == 0).all(dim=1).float() * _speed_ramp(env, command_name, ramp_lo, ramp_hi)


def wheel_single_stance(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    ramp_lo: float,
    ramp_hi: float,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    max_time: float,
    wheel_radius: float = 0.075,
    height_power: float = 0.5,
    clock_window: float | None = None,
    clock_alpha: float = 1.0,
    clock_gain: float = 1.0,
) -> torch.Tensor:
    """[B]'s single_stance_time: pays while exactly one wheel is off the ground, for at most ``max_time`` per lift.

    Scaled by ``min(h / target, 1)^p`` of the lifted wheel so hovering a few millimetres (which the height feature
    of the discriminator cannot tell from rolling) earns little. The cap makes riding on one wheel for long stop
    paying, so the robot has to put the wheel down and lift again.
    With ``clock_window`` (Exp 77a2) the lifts are timed by the observed gait clock: the right wheel's lift pays in the
    first ``clock_window`` of each clock cycle (phase fraction [0, w)) and the left wheel's in [0.5, 0.5 + w) -- the
    mirror of "right at phase p" is "left at p + 1/2", so the mirror loss keeps it symmetric -- with both wheels down in
    between. Outside its window a lift earns ``1 - clock_alpha`` of the pay and inside it
    ``1 - clock_alpha + clock_alpha * clock_gain``; ``clock_alpha`` is ramped 0 -> 1 by a curriculum. With
    ``clock_gain = 1 / w`` a lift at a random clock phase keeps its expected pay while alpha ramps (without it, Exp 77a
    dropped every lift) and a lift timed to the clock earns up to 1 / w times more.
    Sensor and ``asset_cfg`` share the model order (right, left), as in :func:`wheel_lift`.
    """
    sensor = env.scene[sensor_name]
    up = sensor.data.found == 0
    single = up & ~up.flip(1)
    air = sensor.data.current_air_time
    pay = (single & (air < max_time)).float()
    if clock_window is not None and clock_alpha > 0.0:
        frac = (env.command_manager.get_term("gait_clock").phase / (2.0 * math.pi)) % 1.0
        window = torch.stack([frac < clock_window, (frac - 0.5) % 1.0 < clock_window], dim=1).float()
        pay = pay * ((1.0 - clock_alpha) + clock_alpha * clock_gain * window)
    h = _wheel_height(env, asset_cfg, wheel_radius)
    value = torch.sum(pay * (h / target_height).clamp(0.0, 1.0) ** height_power, dim=1)
    return value * _speed_ramp(env, command_name, ramp_lo, ramp_hi)


class wheel_support_switch:
    """[B]'s support_leg_switch: +1 when a wheel lands after a real lift and the previous real lift was the other wheel.

    A real lift has an air time in [``t_min``, ``t_max``] and a peak wheel-bottom height of at least ``min_height``,
    so alternating micro-hops do not pay. The first real lift of an episode counts as a switch.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self.peak = torch.zeros((env.num_envs, 2), device=env.device)
        self.last = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        self.wheel_ids = torch.arange(2, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        command_name: str,
        ramp_lo: float,
        ramp_hi: float,
        asset_cfg: SceneEntityCfg,
        min_height: float,
        t_min: float,
        t_max: float,
        wheel_radius: float = 0.075,
    ) -> torch.Tensor:
        sensor = env.scene[sensor_name]
        up = sensor.data.found == 0
        h = _wheel_height(env, asset_cfg, wheel_radius)
        self.peak = torch.where(up, torch.maximum(self.peak, h), self.peak)
        landed = sensor.compute_first_contact(env.step_dt)
        air = sensor.data.last_air_time
        real = landed & (air >= t_min) & (air <= t_max) & (self.peak >= min_height)
        switched = (real & (self.last.unsqueeze(1) != self.wheel_ids)).any(dim=1)
        self.last = torch.where(real[:, 0], 0, torch.where(real[:, 1], 1, self.last))
        self.peak = torch.where(landed, torch.zeros_like(self.peak), self.peak)
        return switched.float() * _speed_ramp(env, command_name, ramp_lo, ramp_hi)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.peak[env_ids] = 0.0
        self.last[env_ids] = -1


def reward_weight_schedule(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    term_name: str,
    start_weight: float,
    end_weight: float,
    start_step: int,
    end_step: int,
) -> torch.Tensor:
    """Curriculum term: move a reward term's weight linearly from ``start_weight`` to ``end_weight`` between two
    env steps (``env.common_step_counter``; one PPO iteration = ``num_steps_per_env`` = 24 steps). Used for the
    [B]-style contact-timing weights (the paper keeps the support-switch reward weak for the first 5000 steps and
    ramps it up by 32000, and relaxes the hop penalty from 1.25 to 0.9 over 0-22000)."""
    del env_ids
    step = env.common_step_counter
    t = min(max((step - start_step) / max(end_step - start_step, 1), 0.0), 1.0)
    weight = start_weight + t * (end_weight - start_weight)
    env.reward_manager.get_term_cfg(term_name).weight = weight
    return torch.tensor(weight)


def reward_param_schedule(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    term_name: str,
    param: str,
    start_value: float,
    end_value: float,
    start_step: int,
    end_step: int,
) -> torch.Tensor:
    """Curriculum term: move one parameter of a reward term linearly between two env steps (like
    :func:`reward_weight_schedule`), e.g. the minimum air time a lift needs before the support switch pays."""
    del env_ids
    step = env.common_step_counter
    t = min(max((step - start_step) / max(end_step - start_step, 1), 0.0), 1.0)
    value = start_value + t * (end_value - start_value)
    env.reward_manager.get_term_cfg(term_name).params[param] = value
    return torch.tensor(value)


def single_wheel_lift(
    env: ManagerBasedRlEnv,
    target_height: float,
    lifted_cfg: SceneEntityCfg,
    support_cfg: SceneEntityCfg,
    wheel_radius: float = 0.075,
    support_max_height: float = 0.02,
) -> torch.Tensor:
    """Single-wheel feasibility probe (foot-lift plan, phase 0): ride on one wheel, the other held up.

    ``min(h_lifted / target, 1)`` while the support wheel is on the ground (bottom below ``support_max_height``),
    so hopping with both wheels up does not pay. Contact is read from the wheel heights, not the contact sensor,
    to stay independent of the sensor's body order.
    """
    h_lift = _wheel_height(env, lifted_cfg, wheel_radius)[:, 0]
    h_support = _wheel_height(env, support_cfg, wheel_radius)[:, 0]
    return (h_lift / target_height).clamp(0.0, 1.0) * (h_support < support_max_height).float()
