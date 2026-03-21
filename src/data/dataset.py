from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .sampling import uniform_indices, segment_clip_indices, center_clip_indices
from .video_io import read_frames_by_indices, stack_frames
from .transforms import apply_framewise


PathLike = Union[str, Path]


@dataclass
class SampleMeta:
    video_id: str
    source: str
    label_str: str
    rel_path: str


class VideoDataset(Dataset):
    """
    CSV dataset for IVY-Fake.

    Expects split CSVs with:
      video_id, source, label_str, label, rel_path

    dataset_root should point to directory containing video_train/ and video_test/ dirs.
    rel_path is: dataset_root / rel_path
    """

    def __init__(
        self,
        csv_path: PathLike,
        dataset_root: PathLike,
        *,
        train: bool,
        transform: Callable[[np.ndarray], torch.Tensor],
        mode: str = "frames",            
        n_frames: int = 16,              
        clip_len: int = 16,              
        num_segments: int = 3,         
        seed: int = 42,
        allow_partial_decode: bool = True,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.dataset_root = Path(dataset_root)
        self.train = train
        self.transform = transform
        self.mode = mode
        self.n_frames = int(n_frames)
        self.clip_len = int(clip_len)
        self.num_segments = int(num_segments)
        self.seed = int(seed)
        self.allow_partial_decode = bool(allow_partial_decode)

        df = pd.read_csv(self.csv_path)

        required = {"video_id", "source", "label_str", "label", "rel_path"}
        missing = required - set(df.columns)
        if missing:
            raise RuntimeError(f"{self.csv_path} missing columns: {missing}")

        # Drop any rows with parse errors if present
        if "parse_error" in df.columns:
            df = df[df["parse_error"].fillna("").astype(str).str.strip() == ""].copy()

        self.df = df.reset_index(drop=True)

        if self.mode not in {"frames", "clip"}:
            raise ValueError(f"mode must be 'frames' or 'clip', got: {self.mode}")

    def __len__(self) -> int:
        return len(self.df)

    def _get_abs_path(self, rel_path: str) -> Path:
        return self.dataset_root / rel_path

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        row = self.df.iloc[idx]
        rel_path = str(row["rel_path"])
        abs_path = self._get_abs_path(rel_path)

        y = int(row["label"])
        meta = SampleMeta(
            video_id=str(row["video_id"]),
            source=str(row["source"]),
            label_str=str(row["label_str"]),
            rel_path=rel_path,
        )

        import cv2
        cap = cv2.VideoCapture(str(abs_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {abs_path}")
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        cap.release()

        # --- Sampling ---
        if self.mode == "frames":
            indices = uniform_indices(num_frames, self.n_frames)
            frames = read_frames_by_indices(abs_path, indices, allow_partial=self.allow_partial_decode)
            arr = stack_frames(frames)  # (T,H,W,3) uint8
            x = apply_framewise(arr, self.transform)  # (T,C,H,W)
            # return frame stack as (N,C,H,W)
            out = x

        else:
            if self.train:
                clip_idx = segment_clip_indices(num_frames, num_segments=1, clip_len=self.clip_len, rng=self.seed + idx)[0]
            else:
                clip_idx = center_clip_indices(num_frames, self.clip_len)

            frames = read_frames_by_indices(abs_path, clip_idx, allow_partial=self.allow_partial_decode)
            arr = stack_frames(frames)               # (T,H,W,3)
            x = apply_framewise(arr, self.transform) # (T,C,H,W)
            out = x.permute(1, 0, 2, 3).contiguous() # (C,T,H,W)

        meta_dict: Dict[str, Any] = {
            "video_id": meta.video_id,
            "source": meta.source,
            "label_str": meta.label_str,
            "rel_path": meta.rel_path,
            "abs_path": str(abs_path),
            "split_csv": str(self.csv_path),
        }
        return out, y, meta_dict
