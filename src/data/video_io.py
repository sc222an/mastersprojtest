from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
cv2.setNumThreads(0) #
cv2.ocl.setUseOpenCL(False) #
import numpy as np


PathLike = Union[str, Path]


@dataclass
class VideoProbe:
    ok_open: bool
    ok_decode: bool
    num_frames: int
    fps: float
    width: int
    height: int
    error: str = ""


def probe_video(video_path: PathLike) -> VideoProbe:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return VideoProbe(False, False, 0, 0.0, 0, 0, "open_failed")

    try:
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0

        ok, _ = cap.read()
        if not ok:
            return VideoProbe(True, False, num_frames, fps, width, height, "decode_failed_first_frame")

        return VideoProbe(True, True, num_frames, fps, width, height, "")
    finally:
        cap.release()


def _ensure_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR (OpenCV default) -> RGB."""
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _fallback_frame(frames: List[Optional[np.ndarray]], shape: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
    """
    Choose a fallback frame when decoding fails.
    - If there is a previous valid frame, reuse it.
    - Else, create a black frame with known shape.
    """
    for f in reversed(frames):
        if f is not None:
            return f
    if shape is None:
        # last resort default
        return np.zeros((224, 224, 3), dtype=np.uint8)
    return np.zeros(shape, dtype=np.uint8)


def read_frames_by_indices(
    video_path: PathLike,
    indices: List[int],
    *,
    allow_partial: bool = True,
) -> List[np.ndarray]:
    """
    Read frames at specific indices.
    Returns a list of RGB uint8 frames, same length as indices.

    If a frame can't be decoded:
      - allow_partial=True: substitutes with previous valid frame (or black frame).
      - allow_partial=False: raises RuntimeError.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames: List[Optional[np.ndarray]] = []
    out_shape: Optional[Tuple[int, int, int]] = None

    try:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame_bgr = cap.read()

            if not ok or frame_bgr is None:
                if not allow_partial:
                    raise RuntimeError(f"Failed to decode frame {idx} from {video_path}")
                frames.append(None)
                continue

            frame_rgb = _ensure_rgb(frame_bgr)
            if out_shape is None:
                out_shape = frame_rgb.shape  # (H,W,3)
            frames.append(frame_rgb)

    finally:
        cap.release()

    # Replace missing frames
    fixed: List[np.ndarray] = []
    for f in frames:
        if f is None:
            fixed.append(_fallback_frame(fixed, out_shape))
        else:
            fixed.append(f)
    return fixed


def stack_frames(frames: List[np.ndarray]) -> np.ndarray:
    """
    Stack list of frames into a single numpy array: (T, H, W, 3), dtype uint8.
    """
    if len(frames) == 0:
        return np.zeros((0, 0, 0, 3), dtype=np.uint8)
    return np.stack(frames, axis=0)
