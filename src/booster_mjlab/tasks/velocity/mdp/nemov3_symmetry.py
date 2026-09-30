"""Left/right symmetry data augmentation for the Nemov3 velocity tasks.

Same contract as ``mdp.observations.augment_symmetries`` (K1), which is hard-coded to K1's observation
layout and 22-DOF actions: returns the batch doubled, with the mirrored copy in the second half. The PPO
side (``RslRlSymmetryCfg`` + ``AmpPPO.augment_batch``) is reused unchanged.

The per-term mirror rules are looked up by observation-term name from the environment's observation
manager, so the clock / arms / narrow-stance variants all work. Conventions (verified against the
simulator in tools/nemov3_tools/symmetry_check.py):
  * joints: left <-> right partner with the sign from ``nemov3_mirror_maps`` (derived from the XML axes), in the
    order the environment actually uses (the joint_pos / joint_vel terms' joints, the action terms' targets), so a
    robot with welded joints (fixed_joints) works as well;
  * angular velocity (IMU gyro, base-aligned) is a pseudovector: (-x, +y, -z);
  * gravity / linear velocity / contact forces are polar vectors: (+x, -y, +z);
  * command (vx, vy, wz) -> (vx, -vy, -wz);
  * gait clock (sin, cos) -> (-sin, -cos): the mirror of "right leg at phase p" is "left leg at p + pi".
Requires a mirror-symmetric default pose and action scale (joint_pos is observed relative to the default
and actions are offsets from it).
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from booster_mjlab.robots.nemov3.nemov3_constants import (
    NEMOV3_POSITION_JOINT_NAMES,
    NEMOV3_WHEEL_JOINT_NAMES,
    nemov3_mirror_maps,
)

# Action layout of the full 32-joint robot, for callers without an environment (mirror_actions(actions)).
_ACTION_JOINT_NAMES = (*NEMOV3_POSITION_JOINT_NAMES, "right_wheel_joint", "left_wheel_joint")
_JOINT_TERMS = ("joint_pos", "joint_vel")
_VECTOR_TERMS = {
    "base_ang_vel": (-1.0, 1.0, -1.0),
    "projected_gravity": (1.0, -1.0, 1.0),
    "base_lin_vel": (1.0, -1.0, 1.0),
    "command": (1.0, -1.0, -1.0),
    "gait_clock": (-1.0, -1.0),
}
_POLICY_GROUPS = ("actor", "critic")  # other groups (e.g. "amp") are not used by the PPO update: copied as is
_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

assert set(NEMOV3_WHEEL_JOINT_NAMES) == {"right_wheel_joint", "left_wheel_joint"}


def _action_joint_names(env) -> list[str]:
    """Joint names of the action vector: the action terms' targets in order."""
    manager = getattr(env, "unwrapped", env).action_manager
    return [name for term in manager._terms.values() for name in term.target_names]


def _obs_joint_names(env, group: str, name: str) -> list[str]:
    """Joints of observation term ``name`` in ``group`` (its asset_cfg's resolved joints)."""
    unwrapped = getattr(env, "unwrapped", env)
    manager = unwrapped.observation_manager
    cfg = manager._group_obs_term_cfgs[group][manager._group_obs_term_names[group].index(name)]
    asset_cfg = cfg.params.get("asset_cfg")
    robot = unwrapped.scene[asset_cfg.name if asset_cfg is not None else "robot"]
    ids = asset_cfg.joint_ids if asset_cfg is not None else slice(None)
    return list(robot.joint_names) if isinstance(ids, slice) else [robot.joint_names[i] for i in ids]


def _term_map(env, group: str, name: str, dim: int) -> tuple[list[int], list[float]]:
    if name in _JOINT_TERMS or name == "actions":
        joints = _obs_joint_names(env, group, name) if name in _JOINT_TERMS else _action_joint_names(env)
        if len(joints) == dim:
            return nemov3_mirror_maps(joints)
    if name in _VECTOR_TERMS and dim == len(_VECTOR_TERMS[name]):
        return list(range(dim)), list(_VECTOR_TERMS[name])
    if name == "foot_contact" and dim == 2:  # (right, left)
        return [1, 0], [1.0, 1.0]
    if name == "foot_contact_forces" and dim == 6:  # (right xyz, left xyz), odd transform of the force
        return [3, 4, 5, 0, 1, 2], [1.0, -1.0, 1.0, 1.0, -1.0, 1.0]
    raise ValueError(f"No mirror rule for observation term '{name}' with dim {dim}")


def _group_map(env, group: str) -> tuple[torch.Tensor, torch.Tensor]:
    manager = getattr(env, "unwrapped", env).observation_manager
    names = manager._group_obs_term_names[group]
    dims = [int(torch.tensor(d).prod()) for d in manager._group_obs_term_dim[group]]
    key = (group, tuple(names), tuple(dims))
    if key not in _cache:
        perm, sign, offset = [], [], 0
        for name, dim in zip(names, dims):
            p, s = _term_map(env, group, name, dim)
            perm += [offset + i for i in p]
            sign += s
            offset += dim
        _cache[key] = (torch.tensor(perm), torch.tensor(sign))
    return _cache[key]


def _flip(x: torch.Tensor, perm: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    return x[:, perm.to(x.device)] * sign.to(x.device, x.dtype)


def mirror_obs(env, group: str, x: torch.Tensor) -> torch.Tensor:
    perm, sign = _group_map(env, group)
    return _flip(x, perm, sign)


def mirror_actions(actions: torch.Tensor, env=None) -> torch.Tensor:
    """Mirror an action batch; with ``env`` in its action layout, otherwise the full 32-joint layout."""
    perm, sign = nemov3_mirror_maps(_action_joint_names(env) if env is not None else _ACTION_JOINT_NAMES)
    return _flip(actions, torch.tensor(perm), torch.tensor(sign))


def augment_symmetries(env, obs: TensorDict | None, actions: torch.Tensor | None):
    """Return (obs, actions) with the mirrored copy appended (batch size doubled)."""
    if obs is not None:
        n = obs.batch_size[0]
        groups = {
            g: torch.cat([obs[g], mirror_obs(env, g, obs[g]) if g in _POLICY_GROUPS else obs[g]], dim=0)
            for g in obs.keys()
        }
        obs = TensorDict(groups, batch_size=(2 * n,))
    if actions is not None:
        actions = torch.cat([actions, mirror_actions(actions, env)], dim=0)
    return obs, actions
