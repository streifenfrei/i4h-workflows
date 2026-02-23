# Copyright (c) 2024-2025, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for the pick-and-place environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward for moving the end-effector toward the block (tanh kernel)."""
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    dist = torch.norm(obj.data.root_pos_w - ee_frame.data.target_pos_w[..., 0, :], dim=1)
    return 1.0 - torch.tanh(dist / std)


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Binary reward: 1 when the block world-z exceeds *minimal_height*.

    The threshold is set above the resting height (~0.025 m) so this only
    fires once the robot has actually grasped and lifted the block.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    return torch.where(obj.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_goal_distance_xy(
    env: ManagerBasedRLEnv,
    std: float,
    lifted_height: float,
    target_cfg: SceneEntityCfg = SceneEntityCfg("target_area"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for transporting the lifted block toward the target (XY only).

    Gated on *lifted_height* so it only activates while the block is in the
    air, encouraging the agent to keep it elevated during transport.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    xy_dist = torch.norm(obj.data.root_pos_w[:, :2] - target.data.root_pos_w[:, :2], dim=1)
    is_lifted = (obj.data.root_pos_w[:, 2] > lifted_height).float()
    return is_lifted * (1.0 - torch.tanh(xy_dist / std))


def object_placed_at_goal(
    env: ManagerBasedRLEnv,
    std: float,
    placed_height_max: float,
    target_cfg: SceneEntityCfg = SceneEntityCfg("target_area"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for placing the block at rest on the target area.

    Combines two conditions:
    - Block z < *placed_height_max*  (resting on the table, not being held).
    - Block XY close to target centre (tanh kernel with *std*).
    """
    obj: RigidObject = env.scene[object_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    xy_dist = torch.norm(obj.data.root_pos_w[:, :2] - target.data.root_pos_w[:, :2], dim=1)
    is_at_rest = (obj.data.root_pos_w[:, 2] < placed_height_max).float()
    return is_at_rest * (1.0 - torch.tanh(xy_dist / std))