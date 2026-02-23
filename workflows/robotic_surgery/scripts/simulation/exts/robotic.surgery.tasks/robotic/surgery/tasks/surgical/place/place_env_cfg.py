# Copyright (c) 2024-2025, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration for the pick-and-place environment.

The robot picks a block from a random position within a fixed workspace and
places it onto a randomly positioned target area – a blue flat square visible
on the table surface.  Both the block and the target are randomised each
episode with a guaranteed minimum XY separation so the block never spawns on
the goal zone.
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from simulation.utils.assets import robotic_surgery_assets

from . import mdp

##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Scene for the pick-and-place task.

    Derived classes must set ``robot``, ``ee_frame``, ``object`` (the block),
    and ``target_area`` (the kinematic blue marker).
    """

    # Populated by the agent env cfg
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    # Rigid block to be grasped and moved
    object: RigidObjectCfg = MISSING
    # Kinematic visual marker defining the goal placement zone
    target_area: RigidObjectCfg = MISSING
    # Overhead camera; populated by the agent env cfg
    camera: TiledCameraCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.457)),
        spawn=UsdFileCfg(usd_path=robotic_surgery_assets.Table),
    )

    # Ground plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -0.95)),
        spawn=GroundPlaneCfg(),
    )

    # Lighting
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """No command generators – target position is tracked via the scene entity."""

    pass


@configclass
class ActionsCfg:
    """Action specifications; filled in by the agent env cfg."""

    body_joint_pos: mdp.JointPositionActionCfg = MISSING
    finger_joint_pos: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observations for the pick-and-place policy."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Concatenated policy observation vector (per env):

        joint_pos (6) | joint_vel (6) | block_pos_robot_frame (3) |
        target_pos_robot_frame (3) | last_action (6 or 7) |
        camera_features (576)

        Total: ~601 floats (IK-abs) / ~600 floats (joint-pos).
        The 576-D visual features come from a frozen MobileNetV3-Small
        encoder applied to the 64×64 overhead camera image.
        """

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_position = ObsTerm(func=mdp.target_position_in_robot_root_frame)
        actions = ObsTerm(func=mdp.last_action)
        camera_features = ObsTerm(func=mdp.camera_features)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Events executed at every episode reset."""

    # 1. Reset the entire scene (robot joints + all objects) to defaults
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # 2. Randomise block AND target together, guaranteeing min_separation
    reset_block_and_target = EventTerm(
        func=mdp.reset_block_and_target_positions,
        mode="reset",
        params={
            # XY offsets relative to each asset's default position.
            # Range chosen to stay within the PSM reachable workspace on table.
            "workspace_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            "min_separation": 0.04,
            "block_cfg": SceneEntityCfg("object"),
            "target_cfg": SceneEntityCfg("target_area"),
        },
    )


@configclass
class RewardsCfg:
    """Reward shaping for the four implicit phases of pick-and-place.

    Phase 1 – Approach : move EE to block.
    Phase 2 – Grasp    : lift block above resting height.
    Phase 3 – Transport: move lifted block over target (XY).
    Phase 4 – Place    : lower block onto target and release.
    """

    # Phase 1 – approach
    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.1},
        weight=1.0,
    )

    # Phase 2 – grasp / lift
    # Threshold (0.04 m) > resting height (~0.025 m) so this only fires after
    # the block has actually been picked up.
    lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        params={"minimal_height": 0.04},
        weight=10.0,
    )

    # Phase 3 – transport (XY alignment while still lifted)
    object_transport = RewTerm(
        func=mdp.object_goal_distance_xy,
        params={"std": 0.1, "lifted_height": 0.04},
        weight=15.0,
    )

    # Phase 4 – place (block resting near target centre)
    object_placed = RewTerm(
        func=mdp.object_placed_at_goal,
        params={"std": 0.05, "placed_height_max": 0.03},
        weight=20.0,
    )

    # Efficiency penalties
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-3)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Episode termination conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Failure: block fell off the table
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )

    # Success: block is resting at target (early termination accelerates training)
    object_placed = DoneTerm(
        func=mdp.object_placed_success,
        params={
            "threshold": 0.015,
            "placed_height_max": 0.03,
            "target_cfg": SceneEntityCfg("target_area"),
            "object_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class CurriculumCfg:
    """Gradually tighten efficiency penalties over training."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000},
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000},
    )


##
# Environment configuration
##


@configclass
class PlaceEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the pick-and-place environment."""

    # Scene
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    # MDP components
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.sim.render_interval = self.decimation
        # Pick-and-place needs more time than a pure lift task
        self.episode_length_s = 5.0
        self.sim.dt = 1.0 / 200.0
        self.viewer.eye = (0.2, 0.2, 0.1)
        self.viewer.lookat = (0.0, 0.0, 0.04)