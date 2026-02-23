# Copyright (c) 2024-2025, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom observation functions for the pick-and-place environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# ---------------------------------------------------------------------------
# Frozen image encoder (MobileNetV3-Small, pretrained on ImageNet).
#
# Lazy-initialised on the first call so that torchvision is imported only
# after Isaac Sim is up and the CUDA context is available.
# ---------------------------------------------------------------------------
_encoder: torch.nn.Module | None = None


def _get_encoder(device: torch.device) -> torch.nn.Module:
    """Return the shared frozen MobileNetV3-Small feature extractor.

    The network is loaded once, moved to *device*, and reused for every
    subsequent call within the same process.  Output dimension: 576.
    """
    global _encoder
    if _encoder is None:
        try:
            import torchvision.models as tv_models
        except ImportError as e:
            raise ImportError(
                "torchvision is required for the camera_features observation. "
                "Install it with:  pip install torchvision"
            ) from e

        weights = tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        backbone = tv_models.mobilenet_v3_small(weights=weights)

        # Keep only the convolutional feature extractor + global average pool.
        # Output shape: (N, 576) for any input resolution.
        _encoder = torch.nn.Sequential(
            backbone.features,   # (N, 576, H', W')
            backbone.avgpool,    # AdaptiveAvgPool2d → (N, 576, 1, 1)
            torch.nn.Flatten(1), # (N, 576)
        )
        for param in _encoder.parameters():
            param.requires_grad_(False)
        _encoder.eval()

    return _encoder.to(device)


# ImageNet normalisation constants (shape broadcastable to (N, 3, H, W))
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Debug flag: set to True to save one image per env on the first observation call.
DEBUG_SAVE_IMAGES: bool = False
_debug_images_saved: bool = False


def _save_debug_images(images: torch.Tensor) -> None:
    """Save raw [0,1] camera images for every env to /tmp/place_task_debug/.

    Saves one PNG per environment plus a single overview grid (up to 64 envs).
    Called at most once per process (guarded by _debug_images_saved).

    Args:
        images: Float32 tensor of shape (N, 3, H, W) with values in [0, 1].
    """
    import os

    from torchvision.utils import make_grid, save_image

    out_dir = "/root/Documents/debug/place_task"
    os.makedirs(out_dir, exist_ok=True)

    imgs_cpu = images.cpu()
    n = imgs_cpu.shape[0]

    # Individual per-env images
    for i in range(n):
        save_image(imgs_cpu[i], os.path.join(out_dir, f"env_{i:04d}.png"))

    # Overview grid (first 64 envs at most)
    grid_imgs = imgs_cpu[: min(n, 64)]
    nrow = min(8, n)
    grid = make_grid(grid_imgs, nrow=nrow, padding=2)
    save_image(grid, os.path.join(out_dir, "grid.png"))

    print(f"[place_task] Saved {n} debug camera images to {out_dir}/")


# ---------------------------------------------------------------------------
# Proprioceptive observations
# ---------------------------------------------------------------------------


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Position of the block expressed in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    obj_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3],
        robot.data.root_state_w[:, 3:7],
        obj.data.root_pos_w[:, :3],
    )
    return obj_pos_b


def target_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target_area"),
) -> torch.Tensor:
    """Position of the target area centre expressed in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]
    target_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3],
        robot.data.root_state_w[:, 3:7],
        target.data.root_pos_w[:, :3],
    )
    return target_pos_b


# ---------------------------------------------------------------------------
# Visual observation
# ---------------------------------------------------------------------------


def camera_features(
    env: ManagerBasedRLEnv,
    camera_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
) -> torch.Tensor:
    """Encode the overhead camera image with a frozen MobileNetV3-Small.

    Pipeline (per training step, all on GPU):
      1. Read RGB uint8 tensor  (N, H, W, 4)  from the tiled camera.
      2. Drop the alpha channel, normalise to [0, 1].
      3. Reorder to  (N, 3, H, W)  and apply ImageNet mean/std.
      4. Forward-pass through the frozen encoder  →  (N, 576).

    The 576-D feature vector is concatenated with the proprioceptive
    observations by Isaac Lab's ObservationManager before being fed to the
    policy MLP.

    Note: the encoder runs at the camera's native resolution (64 × 64 by
    default).  MobileNetV3-Small was trained on 224 × 224 but generalises
    well to smaller inputs; the adaptive average pool ensures a fixed output
    size regardless of resolution.
    """
    from isaaclab.sensors import TiledCamera

    camera: TiledCamera = env.scene[camera_cfg.name]

    # rgb: (num_envs, H, W, 4)  uint8  [0, 255]
    rgb = camera.data.output["rgb"]

    # → float32  (N, 3, H, W)  [0, 1]
    x = rgb[..., :3].float().div_(255.0).permute(0, 3, 1, 2).contiguous()

    # One-shot debug: save raw [0,1] images before normalisation alters them.
    global _debug_images_saved
    if DEBUG_SAVE_IMAGES and not _debug_images_saved:
        _save_debug_images(x.clone())
        _debug_images_saved = True

    # ImageNet normalisation
    mean = _IMAGENET_MEAN.to(x.device)
    std  = _IMAGENET_STD.to(x.device)
    x = x.sub_(mean).div_(std)

    # Frozen forward pass – no gradient tracking needed
    encoder = _get_encoder(x.device)
    with torch.no_grad():
        features = encoder(x)   # (N, 576)

    return features