# Copyright (c) 2024-2025, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination functions for the pick-and-place environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_placed_success(
    env: ManagerBasedRLEnv,
    threshold: float = 0.015,
    placed_height_max: float = 0.03,
    target_cfg: SceneEntityCfg = SceneEntityCfg("target_area"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Success termination: block is resting within *threshold* m of the target.

    Deliberately named differently from the reward ``object_placed_at_goal`` to
    avoid the symbol being overwritten when both modules are star-imported into
    the mdp namespace.

    Args:
        threshold: Maximum XY distance from target centre for success.
        placed_height_max: Maximum block world-z that counts as resting.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    xy_dist = torch.norm(obj.data.root_pos_w[:, :2] - target.data.root_pos_w[:, :2], dim=1)
    is_at_rest = obj.data.root_pos_w[:, 2] < placed_height_max
    return is_at_rest & (xy_dist < threshold)