# Copyright (c) 2024-2025, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom event functions for the pick-and-place environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_block_and_target_positions(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    workspace_range: dict[str, tuple[float, float]],
    min_separation: float,
    block_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target_area"),
) -> None:
    """Reset block and target area positions ensuring a minimum XY separation.

    Both positions are sampled uniformly within *workspace_range* relative to
    each asset's default root position (set in InitialStateCfg).  Rejection
    sampling guarantees the block never spawns on top of the target area.

    Args:
        env: The environment instance.
        env_ids: Indices of environments being reset.
        workspace_range: Dict with "x" and "y" keys, each a (min, max) offset
            tuple applied relative to the asset's default XY position.
        min_separation: Minimum Euclidean XY distance between the block spawn
            position and the target area centre.
        block_cfg: Scene entity config for the block.
        target_cfg: Scene entity config for the target area marker.
    """
    block: RigidObject = env.scene[block_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    n = len(env_ids)
    device = env.device

    x_lo, x_hi = workspace_range["x"]
    y_lo, y_hi = workspace_range["y"]

    def _sample_xy(size: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = (x_hi - x_lo) * torch.rand(size, device=device) + x_lo
        y = (y_hi - y_lo) * torch.rand(size, device=device) + y_lo
        return x, y

    # Sample target positions
    tgt_x, tgt_y = _sample_xy(n)

    # Sample block positions; reject any that are too close to the target
    blk_x, blk_y = _sample_xy(n)
    for _ in range(50):
        dist = torch.sqrt((blk_x - tgt_x) ** 2 + (blk_y - tgt_y) ** 2)
        too_close = dist < min_separation
        if not too_close.any():
            break
        new_x, new_y = _sample_xy(n)
        blk_x = torch.where(too_close, new_x, blk_x)
        blk_y = torch.where(too_close, new_y, blk_y)

    # Apply block state
    # default_root_state is in asset-local space; add env origin to get world space.
    block_states = block.data.default_root_state[env_ids].clone()
    block_states[:, :3] += env.scene.env_origins[env_ids]
    block_states[:, 0] += blk_x
    block_states[:, 1] += blk_y
    block.write_root_state_to_sim(block_states, env_ids=env_ids)

    # Apply target state
    target_states = target.data.default_root_state[env_ids].clone()
    target_states[:, :3] += env.scene.env_origins[env_ids]
    target_states[:, 0] += tgt_x
    target_states[:, 1] += tgt_y
    target.write_root_state_to_sim(target_states, env_ids=env_ids)