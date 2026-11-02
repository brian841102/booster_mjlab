"""The SuperMimic simulator state and its heading-invariant features.

State tuple ``S = (p, R, v, w, q, qd)``: root position, rotation matrix, world linear / angular velocity of the root
link frame, joint positions and velocities (robot joint order). Flattened as ``[p, quat wxyz, v, w, q, qd]``.
"""

from __future__ import annotations

import torch

from booster_mjlab.supermimic.utils.rotations import mat_to_6d, mat_to_quat, quat_to_mat, yaw_mat


def state_dim(num_joints: int) -> int:
    return 13 + 2 * num_joints


def invariant_dim(num_joints: int) -> int:
    return 1 + 6 + 3 + 3 + 2 * num_joints


def flatten_state(S) -> torch.Tensor:
    p, R, v, w, q, qd = S
    return torch.cat((p, mat_to_quat(R), v, w, q, qd), -1)


def unflatten_state(x: torch.Tensor, num_joints: int):
    J = num_joints
    return x[..., 0:3], quat_to_mat(x[..., 3:7]), x[..., 7:10], x[..., 10:13], x[..., 13 : 13 + J], x[..., 13 + J : 13 + 2 * J]


def invariant_features(S, pos_mask: torch.Tensor) -> torch.Tensor:
    """Heading- and position-invariant state features: height, tilt (6D), body-frame velocities, joints.

    ``pos_mask`` is 1 for position-controlled joints and 0 for velocity-controlled ones (wheels), whose angle carries
    no information."""
    p, R, v, w, q, qd = S
    H = yaw_mat(R)
    Rt = R.transpose(-1, -2)
    return torch.cat(
        (
            p[..., 2:3],
            mat_to_6d(H.transpose(-1, -2) @ R),
            (Rt @ v.unsqueeze(-1)).squeeze(-1),
            (Rt @ w.unsqueeze(-1)).squeeze(-1),
            q * pos_mask,
            qd,
        ),
        -1,
    )
