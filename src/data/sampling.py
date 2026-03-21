from __future__ import annotations
import numpy as np
from typing import List, Optional

def uniform_indices(num_frames: int, n: int) -> List[int]:
    """
    Select n frame indices uniformly across the video.
    If video shorter than n, pad by repeating the last frame index.
    """
    if n <= 0:
        return []
    if num_frames <= 0:
        return [0] * n
    if num_frames >= n:
        return np.linspace(0, num_frames - 1, n).round().astype(int).tolist()
    idx = list(range(num_frames))
    idx += [num_frames - 1] * (n - num_frames)
    return idx


def segment_clip_indices(
    num_frames: int,
    num_segments: int,
    clip_len: int,
    rng: Optional[int] = None,
) -> List[List[int]]:
    """
    Split video into num_segments equal segments.
    Sample one contiguous clip of length clip_len per segment.

    Train-time: pass rng (seed or Generator) for random start.
    """
    if num_segments <= 0:
        return []
    if clip_len <= 0:
        return [[] for _ in range(num_segments)]

    rng = np.random.default_rng(rng)

    if num_frames <= 0:
        return [[0] * clip_len for _ in range(num_segments)]

    seg_size = num_frames / num_segments
    clips: List[List[int]] = []

    for s in range(num_segments):
        start = int(round(s * seg_size))
        end = int(round((s + 1) * seg_size)) - 1
        end = max(end, start)

        max_start = end - clip_len + 1
        if max_start >= start:
            clip_start = int(rng.integers(start, max_start + 1))
            clip = list(range(clip_start, clip_start + clip_len))
        else:
            # segment too short: clamp and repeat last frame
            clip = list(range(start, end + 1))
            clip += [end] * (clip_len - len(clip))
        clips.append(clip)

    return clips


def center_clip_indices(num_frames: int, clip_len: int) -> List[int]:
    """
    Deterministic contiguous clip (centered) for val/test.
    Pads by repeating last frame index if video shorter than clip_len.
    """
    if clip_len <= 0:
        return []
    if num_frames <= 0:
        return [0] * clip_len
    if num_frames >= clip_len:
        start = (num_frames - clip_len) // 2
        return list(range(start, start + clip_len))
    # short video: all frames then pad with last
    idx = list(range(num_frames))
    idx += [num_frames - 1] * (clip_len - num_frames)
    return idx


def flatten_clips(clips: List[List[int]]) -> List[int]:
    """Flatten list-of-clips into a single list of indices."""
    return [i for clip in clips for i in clip]
