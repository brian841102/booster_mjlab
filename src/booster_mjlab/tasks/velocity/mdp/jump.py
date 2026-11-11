"""Jump on a button while skating (task nemov3_velocity_amp_jump, tools/nemov3_finetune_jump.md).

The "button" is a command term: when it fires, the policy observes ``(flag, phase)`` = (1, t / duration) for
``duration`` seconds, then (0, 0). On the hardware the remote's button would start the same window. In training it fires
at random while the forward command is at least ``min_speed``; in play only through :meth:`JumpCommand.trigger`.

Rewards pay for both wheels being off the ground inside the window (scaled by the lower wheel's height up to the target
clearance), bound the height above it, shape the take-off with the trunk's upward velocity, and penalize hops outside
the window. The reference skating clips contain no jump, so the AMP style reward is masked from the trigger until the
discriminator's history window no longer holds a frame of the jump (``recovery`` seconds after the window): the runner
multiplies the style reward by ``env.amp_style_mask`` when the attribute exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

AMP_STYLE_MASK_ATTR = "amp_style_mask"
_NEVER = 1.0e6  # time since the last trigger of an env that has not jumped this episode


class JumpCommand(CommandTerm):
    """Per-env jump window; observed as (flag, phase)."""

    cfg: JumpCommandCfg

    def __init__(self, cfg: JumpCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.t_since = torch.full((self.num_envs,), _NEVER, device=self.device)
        self.jumps = torch.zeros(self.num_envs, device=self.device)
        self.z0 = torch.zeros(self.num_envs, device=self.device)  # trunk height at the press
        # Per-jump bookkeeping: airborne steps / peak clearance in the current window; finished jumps and successes.
        self._air_steps = torch.zeros(self.num_envs, device=self.device)
        self._peak = torch.zeros(self.num_envs, device=self.device)
        self.finished = 0.0
        self.succeeded = 0.0
        # Envs whose button can fire this episode (env_fraction): the rest skate with the full AMP style reward.
        self.enabled = torch.rand(self.num_envs, device=self.device) < self.cfg.env_fraction
        self._command = torch.zeros(self.num_envs, 2, device=self.device)
        self._refresh()

    @property
    def command(self) -> torch.Tensor:
        return self._command

    @property
    def active(self) -> torch.Tensor:
        return self.t_since < self.cfg.duration_s

    @property
    def phase(self) -> torch.Tensor:
        return torch.where(self.active, self.t_since / self.cfg.duration_s, torch.zeros_like(self.t_since))

    @property
    def style_mask(self) -> torch.Tensor:
        return (self.t_since >= self.cfg.duration_s + self.cfg.recovery_s).float()

    def trigger(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Press the button (play / evaluation)."""
        ids = slice(None) if env_ids is None else env_ids
        self.t_since[ids] = 0.0
        self.z0[ids] = self._env.scene["robot"].data.root_link_pos_w[ids, 2]
        self.jumps[ids] += 1
        self._refresh()

    def _track_success(self, dt: float) -> None:
        """Count a finished jump (the window just ended) as a success with >= 2 airborne steps and >= 2 cm."""
        active = self.active
        sensor = self._env.scene[self.cfg.sensor_name]
        robot = self._env.scene["robot"]
        air = (sensor.data.found == 0).all(dim=1) & active
        wheels = [robot.body_names.index(b) for b in self.cfg.wheel_body_names]
        h = (robot.data.body_link_pos_w[:, wheels, 2] - self.cfg.wheel_radius).clamp_min(0.0).min(dim=1).values
        self._air_steps += air.float()
        self._peak = torch.where(air, torch.maximum(self._peak, h), self._peak)
        ends = active & (self.t_since + dt >= self.cfg.duration_s)
        if ends.any():
            ok = ends & (self._air_steps >= 2) & (self._peak >= 0.02)
            self.finished += float(ends.sum())
            self.succeeded += float(ok.sum())
            self._air_steps[ends] = 0.0
            self._peak[ends] = 0.0

    def _refresh(self) -> None:
        self._command[:, 0] = self.active.float()
        self._command[:, 1] = self.phase
        setattr(self._env, AMP_STYLE_MASK_ATTR, self.style_mask)

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        # Reset path: no jump pending in a new episode.
        self.t_since[env_ids] = _NEVER
        self.jumps[env_ids] = 0.0
        self.enabled[env_ids] = torch.rand(len(env_ids), device=self.device) < self.cfg.env_fraction
        self._refresh()

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None:  # per-step path; the reset path passes the reset env ids
            dt = self._env.step_dt
            self._track_success(dt)
            self.t_since += dt
            if self.cfg.auto_trigger:
                vx = self._env.command_manager.get_command(self.cfg.twist_command_name)[:, 0]
                ready = self.t_since >= self.cfg.duration_s + self.cfg.cooldown_s
                ready &= (vx >= self.cfg.min_speed) & self.enabled
                fire = ready & (torch.rand(self.num_envs, device=self.device) < dt / self.cfg.mean_interval_s)
                self.t_since[fire] = 0.0
                self.z0[fire] = self._env.scene["robot"].data.root_link_pos_w[fire, 2]
                self.jumps[fire] += 1
        self._refresh()


@dataclass(kw_only=True)
class JumpCommandCfg(CommandTermCfg):
    duration_s: float = 0.6
    """Length of the jump window (flag = 1): crouch, push off, ~0.2 s of flight for 5 cm, land."""
    recovery_s: float = 0.6
    """AMP style reward stays masked this long after the window (the discriminator's 30-frame history at 50 Hz)."""
    cooldown_s: float = 1.0
    """No new trigger for this long after a window ends."""
    mean_interval_s: float = 2.5
    """Mean waiting time of a trigger once ready (training)."""
    min_speed: float = 0.3
    """Fires only while the forward command is at least this (jumping while skating forward)."""
    env_fraction: float = 1.0
    """Share of the envs (drawn per episode) whose button fires in training. The AMP style reward is masked around
    every jump; J04 (every env, a press every ~3.3 s of forward skating) masked ~42 % of the moving time and the
    skating gait faded, so later runs let only part of the envs jump."""
    auto_trigger: bool = True
    """Random triggers (training); off in play, where :meth:`JumpCommand.trigger` presses the button."""
    twist_command_name: str = "twist"
    sensor_name: str = "feet_ground_contact"
    wheel_body_names: tuple[str, ...] = ("left_wheel_link", "right_wheel_link")
    wheel_radius: float = 0.075

    def build(self, env: ManagerBasedRlEnv) -> JumpCommand:
        return JumpCommand(self, env)


def jump_feedforward_alpha(env: ManagerBasedRlEnv, env_ids: torch.Tensor, action_name: str = "joint_pos") -> torch.Tensor:
    """Curriculum-style log term: the feed-forward alpha (and nothing else), shown as Curriculum/jump_ff_alpha."""
    del env_ids
    return torch.tensor(env.action_manager.get_term(action_name).alpha)


def _jump(env: ManagerBasedRlEnv, command_name: str) -> JumpCommand:
    term = env.command_manager.get_term(command_name)
    assert isinstance(term, JumpCommand)
    return term


def _airborne(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Both wheels off the ground."""
    return (env.scene[sensor_name].data.found == 0).all(dim=1)


def _wheel_bottom(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, radius: float) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    return (robot.data.body_link_pos_w[:, asset_cfg.body_ids, 2] - radius).clamp_min(0.0)


def jump_air_height(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    wheel_radius: float = 0.075,
) -> torch.Tensor:
    """In the window, while both wheels are off the ground: min(lower wheel's clearance / target, 1)."""
    h = _wheel_bottom(env, asset_cfg, wheel_radius).min(dim=1).values
    pay = (h / target_height).clamp(0.0, 1.0)
    return pay * (_airborne(env, sensor_name) & _jump(env, command_name).active).float()


def jump_height_excess(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    limit: float,
    scale: float,
    wheel_radius: float = 0.075,
) -> torch.Tensor:
    """Bound on the clearance: sum over wheels of max(h - limit, 0) / scale (any time)."""
    del command_name
    return torch.sum((_wheel_bottom(env, asset_cfg, wheel_radius) - limit).clamp_min(0.0), dim=1) / scale


def jump_takeoff_velocity(
    env: ManagerBasedRlEnv,
    command_name: str,
    max_phase: float,
    v_cap: float,
    min_phase: float = 0.0,
    asset_name: str = "robot",
) -> torch.Tensor:
    """Take-off shaping: the trunk's upward velocity (clipped to [0, v_cap]) for phase in [min_phase, max_phase)."""
    term = _jump(env, command_name)
    vz = env.scene[asset_name].data.root_link_lin_vel_w[:, 2].clamp(0.0, v_cap)
    return vz * (term.active & (term.phase >= min_phase) & (term.phase < max_phase)).float()


def jump_crouch(
    env: ManagerBasedRlEnv, command_name: str, end_phase: float, depth: float, asset_name: str = "robot"
) -> torch.Tensor:
    """Crouch before the push-off (J03, user: "跳跃理论上要先弯曲腿再跳起"): for phase < end_phase, the trunk's drop
    below its height at the press, clipped to [0, depth] and divided by depth."""
    term = _jump(env, command_name)
    drop = (term.z0 - env.scene[asset_name].data.root_link_pos_w[:, 2]).clamp(0.0, depth) / depth
    return drop * (term.active & (term.phase < end_phase)).float()


def jump_trunk_rise(
    env: ManagerBasedRlEnv, command_name: str, cap: float, min_phase: float = 0.0, asset_name: str = "robot"
) -> torch.Tensor:
    """In the window (from ``min_phase`` on): the trunk's rise above its height at the press, clipped to [0, cap] and
    divided by cap. Dense (it pays for straightening the legs before the wheels leave the ground), unlike
    ``jump_air_height``."""
    term = _jump(env, command_name)
    rise = (env.scene[asset_name].data.root_link_pos_w[:, 2] - term.z0).clamp(0.0, cap) / cap
    return rise * (term.active & (term.phase >= min_phase)).float()


def jump_leg_template(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    keyframes: tuple[tuple[float, tuple[float, ...]], ...],
    std: float,
) -> torch.Tensor:
    """In the window: exp(-mean((q - q_ref(phase))^2) / std^2) over the joints of ``asset_cfg``, with q_ref linearly
    interpolated between ``keyframes`` = ((phase, (angle per joint, in asset_cfg.joint_names order)), ...) -- a crouch /
    push-off / recover template for the legs (J04). Guidance only: the jump itself is paid by ``jump_air_height``."""
    term = _jump(env, command_name)
    q = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    err = torch.mean(torch.square(q - _template(term.phase, keyframes)), dim=1)
    return torch.exp(-err / std**2) * term.active.float()


def _template(phase: torch.Tensor, keyframes: tuple[tuple[float, tuple[float, ...]], ...]) -> torch.Tensor:
    """Joint angles of a keyframed template at ``phase`` (N,), linear between keyframes: (N, J)."""
    ph = torch.tensor([k[0] for k in keyframes], device=phase.device)
    ang = torch.tensor([k[1] for k in keyframes], device=phase.device)  # (K, J)
    p = phase.clamp(0.0, 1.0)
    i = torch.searchsorted(ph, p, right=True).clamp(1, len(keyframes) - 1)
    w = ((p - ph[i - 1]) / (ph[i] - ph[i - 1]).clamp_min(1e-6)).clamp(0.0, 1.0).unsqueeze(1)
    return ang[i - 1] * (1.0 - w) + ang[i] * w


class JumpFeedforwardJointPositionAction(JointPositionAction):
    """Joint-position action that, inside the jump window, adds ``alpha * (template(phase) - default)`` to the targets of
    the template joints (J06). ``alpha`` falls linearly from 1 to 0 between the env steps of ``anneal_steps``, so the
    robot jumps from the first iteration (the scripted push-off of jump_feasibility.py leaves the ground) while the
    policy learns to balance, land and keep skating, then has to produce the push-off itself. 0 in play (the deployed
    policy has no feed-forward)."""

    cfg: JumpFeedforwardJointPositionActionCfg

    def __init__(self, cfg: JumpFeedforwardJointPositionActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)
        names = list(self.target_names)
        self._ff_idx = torch.tensor([names.index(j) for j in cfg.ff_joint_names], device=self.device)
        self._alpha = 1.0 if cfg.adaptive is not None else 0.0
        self._last_update = 0

    @property
    def alpha(self) -> float:
        if self.cfg.adaptive is not None:
            return self._alpha
        lo, hi = self.cfg.anneal_steps
        step = self._env.common_step_counter
        return float(min(max((hi - step) / max(hi - lo, 1), 0.0), 1.0))

    def _adapt(self) -> None:
        """Adaptive anneal (J07): every ``period`` env steps, the success rate of the jumps that finished since the last
        update moves alpha down by ``down`` (>= ``hi_rate``) or up by ``up`` (< ``lo_rate``); needs >= 20 jumps."""
        period, hi_rate, lo_rate, down, up = self.cfg.adaptive
        step = self._env.common_step_counter
        if step - self._last_update < period:
            return
        term = self._env.command_manager.get_term(self.cfg.command_name)
        if term.finished < 20:
            return
        rate = term.succeeded / term.finished
        if rate >= hi_rate:
            self._alpha = max(self._alpha - down, 0.0)
        elif rate < lo_rate:
            self._alpha = min(self._alpha + up, 1.0)
        self.success_rate = rate
        term.finished = term.succeeded = 0.0
        self._last_update = step

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        if self.cfg.adaptive is not None:
            self._adapt()
        a = self.alpha
        if a <= 0.0:
            return
        term = self._env.command_manager.get_term(self.cfg.command_name)
        offset = self._offset[:, self._ff_idx] if torch.is_tensor(self._offset) else self._offset
        ff = (_template(term.phase, self.cfg.keyframes) - offset) * (a * term.active.float()).unsqueeze(1)
        self._processed_actions[:, self._ff_idx] += ff


@dataclass(kw_only=True)
class JumpFeedforwardJointPositionActionCfg(JointPositionActionCfg):
    command_name: str = "jump"
    ff_joint_names: tuple[str, ...] = ()
    """Template joints, in the order of the keyframe angles."""
    keyframes: tuple[tuple[float, tuple[float, ...]], ...] = ()
    anneal_steps: tuple[int, int] = (0, 0)
    """alpha = 1 before the first env step, 0 after the second (env steps = iterations x 24); (0, 0) = no feed-forward."""
    adaptive: tuple[int, float, float, float, float] | None = None
    """(period env steps, hi_rate, lo_rate, down, up): alpha starts at 1 and follows the jump success rate instead of
    ``anneal_steps`` (J07+)."""

    def build(self, env: ManagerBasedRlEnv) -> JumpFeedforwardJointPositionAction:
        return JumpFeedforwardJointPositionAction(self, env)


def hop_outside_jump(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    asset_cfg: SceneEntityCfg,
    min_height: float,
    wheel_radius: float = 0.075,
) -> torch.Tensor:
    """1 while both wheels are more than ``min_height`` off the ground outside the jump window and its recovery
    (unasked hops). The height gate ignores the one-step contact flicker of fast skating (Exp 90 at 1.3 m/s: ~12 both-
    wheels-free steps per minute without any visible hop)."""
    high = _wheel_bottom(env, asset_cfg, wheel_radius).min(dim=1).values > min_height
    return (_airborne(env, sensor_name) & high & (_jump(env, command_name).style_mask > 0)).float()
