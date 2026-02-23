# Copyright (c) 2024-2025, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint-position control variant of the block pick-and-place environment."""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg, TiledCameraCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from robotic.surgery.tasks.surgical.place import mdp
from robotic.surgery.tasks.surgical.place.place_env_cfg import PlaceEnvCfg
from simulation.utils.assets import robotic_surgery_assets

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from robotic.surgery.assets.psm import PSM_CFG  # isort: skip


@configclass
class BlockPlaceEnvCfg(PlaceEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ------------------------------------------------------------------ #
        # Robot
        # ------------------------------------------------------------------ #
        self.scene.robot = PSM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ------------------------------------------------------------------ #
        # Actions
        # ------------------------------------------------------------------ #
        self.actions.body_joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "psm_yaw_joint",
                "psm_pitch_end_joint",
                "psm_main_insertion_joint",
                "psm_tool_roll_joint",
                "psm_tool_pitch_joint",
                "psm_tool_yaw_joint",
            ],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.finger_joint_pos = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["psm_tool_gripper.*_joint"],
            open_command_expr={"psm_tool_gripper1_joint": -0.5, "psm_tool_gripper2_joint": 0.5},
            close_command_expr={"psm_tool_gripper1_joint": -0.1, "psm_tool_gripper2_joint": 0.1},
        )

        # ------------------------------------------------------------------ #
        # Block (rigid object to be grasped)
        # ------------------------------------------------------------------ #
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.025), rot=(1, 0, 0, 0)),
            spawn=UsdFileCfg(
                usd_path=robotic_surgery_assets.Block,
                scale=(0.011, 0.011, 0.011),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=8,
                    max_angular_velocity=200,
                    max_linear_velocity=200,
                    max_depenetration_velocity=1.0,
                    disable_gravity=False,
                ),
            ),
        )

        # ------------------------------------------------------------------ #
        # Target area – blue kinematic flat square on the table surface.
        #
        # * kinematic_enabled=True  → physics cannot push it around.
        # * No collision_props      → block slides onto the table surface
        #                             without being stopped by the marker edge.
        # * PreviewSurfaceCfg blue  → clearly visible in the viewport.
        # ------------------------------------------------------------------ #
        self.scene.target_area = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/TargetArea",
            # Default position on the table surface; the event randomises both
            # block and target together at each reset.
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.001), rot=(1, 0, 0, 0)),
            spawn=sim_utils.CuboidCfg(
                size=(0.03, 0.03, 0.002),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 0.2, 1.0),
                    opacity=0.9,
                    metallic=0.0,
                    roughness=1.0,
                ),
            ),
        )

        # ------------------------------------------------------------------ #
        # Overhead camera for visual observations.
        #
        # Positioned 15 cm above the env origin (≈ table surface), looking
        # straight down (USD cameras look along local -Z; identity rotation
        # aligns local -Z with world -Z → downward view).
        #
        # At this height, focal_length=24 gives a horizontal FOV of ~47° and
        # a ground footprint of ~0.13 m, which comfortably covers the ±0.05 m
        # workspace.  With 64×64 pixels each pixel covers ~2 mm – enough to
        # resolve the 11 mm block and 30 mm target square.
        #
        # Render interval is inherited from decimation (every 4 sim steps).
        # ------------------------------------------------------------------ #
        self.scene.camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            offset=TiledCameraCfg.OffsetCfg(
                # 15 cm back (-Y) and 15 cm up gives a 45° viewing angle.
                # Rotating +45° around X tilts the look direction from straight
                # down (0,0,-1) to (0,+0.707,-0.707), so the optical axis passes
                # exactly through the env-local origin.
                pos=(0.0, -0.15, 0.15),
                rot=(0.9239, 0.3827, 0.0, 0.0),  # +45° around X
                convention="world",
            ),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 1.0e5),
            ),
            data_types=["rgb"],
            width=280,
            height=150,
        )

        # ------------------------------------------------------------------ #
        # End-effector frame transformer
        # ------------------------------------------------------------------ #
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/psm_base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/psm_tool_tip_link",
                    name="end_effector",
                ),
            ],
        )


@configclass
class BlockPlaceEnvCfg_PLAY(BlockPlaceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False