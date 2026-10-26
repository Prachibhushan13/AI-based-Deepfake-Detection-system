"""Temporal frame selection helpers shared by inference and tests."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def evenly_spaced_indices(frame_count: int, sequence_length: int) -> np.ndarray:
    if frame_count < 0:
        raise ValueError("frame_count must be non-negative")
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    if frame_count == 0:
        return np.empty(0, dtype=np.int64)
    return np.linspace(
        0,
        frame_count - 1,
        num=min(frame_count, sequence_length),
        dtype=np.int64,
    )


def select_evenly_spaced_frames(
    frames: Sequence[np.ndarray], sequence_length: int
) -> list[np.ndarray]:
    indices = evenly_spaced_indices(len(frames), sequence_length)
    return [frames[int(index)] for index in indices]
