# Copyright (c) 2024-2025, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, ik_abs_env_cfg, joint_pos_env_cfg

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-Place-Block-PSM-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.BlockPlaceEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PlaceBlockPPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Place-Block-PSM-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.BlockPlaceEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PlaceBlockPPORunnerCfg,
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics – Absolute Pose Control
##

gym.register(
    id="Isaac-Place-Block-PSM-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_abs_env_cfg.BlockPlaceEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PlaceBlockPPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Place-Block-PSM-IK-Abs-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_abs_env_cfg.BlockPlaceEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PlaceBlockPPORunnerCfg,
    },
    disable_env_checker=True,
)