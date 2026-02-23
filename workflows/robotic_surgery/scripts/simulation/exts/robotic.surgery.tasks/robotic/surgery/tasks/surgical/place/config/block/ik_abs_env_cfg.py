# Copyright (c) 2024-2025, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IK absolute-pose control variant of the block pick-and-place environment."""

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import joint_pos_env_cfg

##
# Pre-defined configs
##
from robotic.surgery.assets.psm import PSM_HIGH_PD_CFG  # isort: skip


@configclass
class BlockPlaceEnvCfg(joint_pos_env_cfg.BlockPlaceEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Stiffer PD gains for better IK tracking
        self.scene.robot = PSM_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Replace joint-position action with IK absolute-pose action
        self.actions.body_joint_pos = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[
                "psm_yaw_joint",
                "psm_pitch_end_joint",
                "psm_main_insertion_joint",
                "psm_tool_roll_joint",
                "psm_tool_pitch_joint",
                "psm_tool_yaw_joint",
            ],
            body_name="psm_tool_tip_link",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
            ),
        )


@configclass
class BlockPlaceEnvCfg_PLAY(BlockPlaceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False