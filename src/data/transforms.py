from __future__ import annotations
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torchvision import transforms

ArrayLike = Union[np.ndarray, torch.Tensor]

# ImageNet defaults (safe baseline for most pretrained 2D backbones)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def make_transforms(
    train: bool,
    image_size: int = 224,
    *,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> Callable[[np.ndarray], torch.Tensor]:
    """
    Returns a transform callable that converts an RGB uint8 frame (H,W,3) to a
    float tensor (C,H,W) normalised by mean/std.

    Train:
      - Resize -> RandomResizedCrop -> RandomHorizontalFlip
    Val/Test:
      - Resize -> CenterCrop
    """
    if train:
        t = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),               # uint8 [0,255] -> float [0,1]
            transforms.Normalize(mean, std),
        ])
    else:
        # Typical evaluation: resize shorter side to 256 then center crop 224
        t = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return t


def apply_framewise(
    frames: np.ndarray,
    transform: Callable[[np.ndarray], torch.Tensor],
) -> torch.Tensor:
    """
    Apply a per-frame transform to:
      - single frame: (H,W,3) -> (C,H,W)
      - multiple frames: (T,H,W,3) -> (T,C,H,W)

    Input must be RGB uint8 frames.
    """
    if frames.ndim == 3:
        return transform(frames)  # (C,H,W)

    if frames.ndim != 4:
        raise ValueError(f"Expected (H,W,3) or (T,H,W,3), got shape {frames.shape}")

    outs = [transform(frames[i]) for i in range(frames.shape[0])]
    return torch.stack(outs, dim=0)  # (T,C,H,W)
