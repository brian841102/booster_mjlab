"""Motor-target timing dynamics ("tau delay"), ported from flamingo_rl_train (mdp/motor_target_dynamics.py).

The user's other framework, whose Nemov3 policies transfer to the hardware, runs every action target through what a
slower low-level motor controller does to it before the PD law sees it: per environment, (1) a motor update rate (the
target is held between motor ticks), (2) a transport delay, (3) a first-order low-pass with time constant tau (kept, but it did not help: Exp 84 with
tau 0-30 ms was strange and asymmetric on the hardware; Exp 83 runs with tau = 0). All three are drawn at reset and kept for the episode. This is separate from the actuator command delay
(get_nemov3_robot_cfg(actuator_delay_steps)); the two stack.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.actions.actions import (
    JointPositionAction,
    JointPositionActionCfg,
    JointVelocityAction,
    JointVelocityActionCfg,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass
class MotorTargetDynamicsCfg:
    """Per-episode motor update rate (Hz, one of the choices), transport delay (s) and low-pass tau (s)."""

    control_hz_choices: tuple[float, ...] = (200.0,)
    delay_range_s: tuple[float, float] = (0.0, 0.03)
    # Shown not to help in Exp 84 (0-30 ms: strange, asymmetric motion on the hardware); Exp 83 uses (0, 0) = no low-pass.
    tau_range_s: tuple[float, float] = (0.0, 0.0)
    initialize_to_first_target: bool = True


@dataclass
class ActuatorDelayCfg:
    """One per-episode command delay per actuator, in physics steps, set by the reset event sample_actuator_delay:
    a lag uniform in [0, max_lag] per environment and joint group (``groups``), plus, with ``cmd_groups``, a second lag
    drawn like MotorTargetDynamics' transport delay (``cmd_delay_range_s``) per environment and cmd group."""

    groups: tuple[str, ...]
    max_lag: int
    cmd_groups: tuple[str, ...] = ()
    cmd_delay_range_s: tuple[float, float] = (0.0, 0.0)

    def buffer_steps(self, dt: float) -> int:
        """Longest total lag (the delay buffers' size)."""
        return self.max_lag + (int(round(self.cmd_delay_range_s[1] / dt)) if self.cmd_groups else 0)


class MotorTargetDynamics:
    """Hold, delay and first-order low-pass applied to action targets, one setting per environment."""

    def __init__(self, cfg: MotorTargetDynamicsCfg, num_envs: int, action_dim: int, sim_dt: float, device):
        self.cfg = cfg
        self.num_envs = num_envs
        self.sim_dt = float(sim_dt)
        self.device = torch.device(device)
        shape = (num_envs, action_dim)
        self.target = torch.zeros(shape, device=self.device)
        self.hold_steps = torch.ones(num_envs, dtype=torch.long, device=self.device)
        self.hold_counter = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.delay_steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.tau_s = torch.zeros(num_envs, 1, device=self.device)
        self.initialized = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        # Longest delay in motor ticks (+2 slack, as in flamingo).
        max_motor_dt = max(1.0 / max(min(cfg.control_hz_choices), 1e-9), self.sim_dt)
        self.max_delay_steps = int(round(max(cfg.delay_range_s) / max_motor_dt)) + 2
        self.buffer = torch.zeros(self.max_delay_steps + 1, *shape, device=self.device)
        self.write_idx = torch.zeros(num_envs, dtype=torch.long, device=self.device)

    def reset(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        n = env_ids.numel()
        self.target[env_ids] = 0.0
        self.hold_counter[env_ids] = 0
        self.write_idx[env_ids] = 0
        self.initialized[env_ids] = False
        self.buffer[:, env_ids] = 0.0
        hz = torch.tensor(self.cfg.control_hz_choices, device=self.device, dtype=torch.float32)
        hz = hz[torch.randint(0, hz.numel(), (n,), device=self.device)]
        self.hold_steps[env_ids] = torch.clamp(torch.round((1.0 / hz) / self.sim_dt).long(), min=1)
        self.tau_s[env_ids, 0] = self._uniform(self.cfg.tau_range_s, n)
        motor_dt = self.sim_dt * self.hold_steps[env_ids].float()
        delay = self._uniform(self.cfg.delay_range_s, n)
        self.delay_steps[env_ids] = torch.clamp(torch.round(delay / motor_dt).long(), 0, self.max_delay_steps)

    def step(self, nominal: torch.Tensor) -> torch.Tensor:
        """Advance one physics step with the policy's current target ``nominal``; returns the executed target."""
        fresh = ~self.initialized
        if fresh.any():
            ids = fresh.nonzero(as_tuple=False).squeeze(-1)
            if self.cfg.initialize_to_first_target:
                self.target[ids] = nominal[ids]
                self.buffer[:, ids] = nominal[ids].unsqueeze(0)
            self.initialized[ids] = True
        self.hold_counter += 1
        tick = self.hold_counter >= self.hold_steps
        if not tick.any():
            return self.target
        ids = tick.nonzero(as_tuple=False).squeeze(-1)
        self.hold_counter[ids] = 0
        size = self.max_delay_steps + 1
        self.write_idx[ids] = (self.write_idx[ids] + 1) % size
        self.buffer[self.write_idx[ids], ids] = nominal[ids]
        delayed = self.buffer[(self.write_idx[ids] - self.delay_steps[ids]) % size, ids]
        motor_dt = (self.sim_dt * self.hold_steps[ids].float()).unsqueeze(-1)
        alpha = motor_dt / torch.clamp(self.tau_s[ids] + motor_dt, min=1e-9)
        self.target[ids] += alpha * (delayed - self.target[ids])
        return self.target

    def _uniform(self, bounds: tuple[float, float], n: int) -> torch.Tensor:
        lo, hi = bounds
        return torch.empty(n, device=self.device).uniform_(float(lo), float(hi))


def sample_actuator_delay(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    groups: tuple[str, ...],
    max_lag: int,
    cmd_groups: tuple[str, ...] = (),
    cmd_delay_range_s: tuple[float, float] = (0.0, 0.0),
    asset_name: str = "robot",
) -> None:
    """Reset event: one command lag in [0, max_lag] physics steps per environment and actuator group, kept for the
    episode (flamingo_rl_train's DelayedPD actuators: a group shares one lag). ``groups`` are joint regexes; each
    actuator joins the first group its joint matches. The actuators' delay buffers must not resample on their own
    (get_nemov3_robot_cfg sets delay_hold_prob=1); the scene reset zeroes their lags before reset events run.

    With ``cmd_groups`` a second lag is added on top, drawn like MotorTargetDynamics' transport delay (uniform in
    ``cmd_delay_range_s``, rounded to physics steps) once per environment and cmd group: the two delays of Exp 83
    (actuator delay per joint group + motor-target delay per action term) merged into one buffer (Exp 85)."""
    ids = _env_ids(env_ids, env.num_envs, env.device)
    all_delayed = [a for a in env.scene[asset_name].actuators if a.has_delay]
    lags = {id(a): torch.zeros(ids.numel(), dtype=torch.long, device=env.device) for a in all_delayed}
    for patterns, draw in (
        (groups, lambda n: torch.randint(0, max_lag + 1, (n,), device=env.device)),
        (cmd_groups, lambda n: _cmd_lag(cmd_delay_range_s, env.physics_dt, n, env.device)),
    ):
        actuators = list(all_delayed)
        for pattern in patterns:
            members = [a for a in actuators if any(re.fullmatch(pattern, name) for name in a.target_names)]
            lag = draw(ids.numel())
            for actuator in members:
                lags[id(actuator)] += lag
            actuators = [a for a in actuators if a not in members]
        assert not patterns or not actuators, f"actuators in no delay group: {[a.target_names for a in actuators]}"
    for actuator in all_delayed:
        actuator.set_lags(lags[id(actuator)], ids)


def _cmd_lag(delay_range_s: tuple[float, float], dt: float, n: int, device) -> torch.Tensor:
    """MotorTargetDynamics' delay draw at a 1-physics-step motor tick: round(uniform(range) / dt)."""
    delay = torch.empty(n, device=device).uniform_(float(delay_range_s[0]), float(delay_range_s[1]))
    return torch.round(delay / dt).long()


def _env_ids(env_ids, num_envs: int, device) -> torch.Tensor:
    all_ids = torch.arange(num_envs, device=device)
    if env_ids is None:
        return all_ids
    if isinstance(env_ids, slice):
        return all_ids[env_ids]
    return torch.as_tensor(env_ids, device=device, dtype=torch.long)


@dataclass(kw_only=True)
class MotorDynamicsJointPositionActionCfg(JointPositionActionCfg):
    motor_dynamics: MotorTargetDynamicsCfg = field(default_factory=MotorTargetDynamicsCfg)

    def build(self, env: ManagerBasedRlEnv) -> MotorDynamicsJointPositionAction:
        return MotorDynamicsJointPositionAction(self, env)


@dataclass(kw_only=True)
class MotorDynamicsJointVelocityActionCfg(JointVelocityActionCfg):
    motor_dynamics: MotorTargetDynamicsCfg = field(default_factory=MotorTargetDynamicsCfg)

    def build(self, env: ManagerBasedRlEnv) -> MotorDynamicsJointVelocityAction:
        return MotorDynamicsJointVelocityAction(self, env)


class MotorDynamicsJointPositionAction(JointPositionAction):
    """JointPositionAction whose position targets go through MotorTargetDynamics (encoder bias as in the base)."""

    cfg: MotorDynamicsJointPositionActionCfg

    def __init__(self, cfg: MotorDynamicsJointPositionActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)
        self._motor = MotorTargetDynamics(cfg.motor_dynamics, self.num_envs, self.action_dim, env.physics_dt, self.device)
        self._motor.reset(_env_ids(None, self.num_envs, self.device))

    def apply_actions(self) -> None:
        encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
        target = self._motor.step(self._processed_actions) - encoder_bias
        self._entity.set_joint_position_target(target, joint_ids=self._target_ids)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        self._motor.reset(_env_ids(env_ids, self.num_envs, self.device))


class MotorDynamicsJointVelocityAction(JointVelocityAction):
    """JointVelocityAction whose velocity targets go through MotorTargetDynamics."""

    cfg: MotorDynamicsJointVelocityActionCfg

    def __init__(self, cfg: MotorDynamicsJointVelocityActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)
        self._motor = MotorTargetDynamics(cfg.motor_dynamics, self.num_envs, self.action_dim, env.physics_dt, self.device)
        self._motor.reset(_env_ids(None, self.num_envs, self.device))

    def apply_actions(self) -> None:
        self._entity.set_joint_velocity_target(self._motor.step(self._processed_actions), joint_ids=self._target_ids)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        self._motor.reset(_env_ids(env_ids, self.num_envs, self.device))
