from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader

from .dataset import VideoDataset
from .transforms import make_transforms


@dataclass
class DataConfig:
    dataset_root: str
    train_csv: str
    val_csv: str
    test_csv: str
    image_size: int = 224


def make_loaders(
    cfg: DataConfig,
    *,
    mode: str,
    n_frames: int = 16,
    clip_len: int = 16,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = VideoDataset(
        csv_path=cfg.train_csv,
        dataset_root=cfg.dataset_root,
        train=True,
        transform=make_transforms(train=True, image_size=cfg.image_size),
        mode=mode,
        n_frames=n_frames,
        clip_len=clip_len,
    )
    val_ds = VideoDataset(
        csv_path=cfg.val_csv,
        dataset_root=cfg.dataset_root,
        train=False,
        transform=make_transforms(train=False, image_size=cfg.image_size),
        mode=mode,
        n_frames=n_frames,
        clip_len=clip_len,
    )
    test_ds = VideoDataset(
        csv_path=cfg.test_csv,
        dataset_root=cfg.dataset_root,
        train=False,
        transform=make_transforms(train=False, image_size=cfg.image_size),
        mode=mode,
        n_frames=n_frames,
        clip_len=clip_len,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader
