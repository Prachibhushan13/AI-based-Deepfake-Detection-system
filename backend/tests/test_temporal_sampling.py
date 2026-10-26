from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.ml.temporal_sampling import (  # noqa: E402
    evenly_spaced_indices,
    select_evenly_spaced_frames,
)
from app.ml.torch_inference import faces_to_tensor  # noqa: E402


class TemporalSamplingTests(unittest.TestCase):
    def test_long_video_has_exactly_twenty_in_range_indices(self) -> None:
        indices = evenly_spaced_indices(101, 20)
        self.assertEqual(len(indices), 20)
        self.assertEqual(len(set(indices.tolist())), 20)
        self.assertTrue(np.all(np.diff(indices) > 0))
        self.assertTrue(np.all((indices >= 0) & (indices < 101)))

    def test_selects_ordered_twenty_frames_from_long_video(self) -> None:
        frames = [np.full((1, 1, 1), index) for index in range(100)]
        selected = select_evenly_spaced_frames(frames, 20)
        self.assertEqual(len(selected), 20)
        self.assertEqual(
            [int(frame[0, 0, 0]) for frame in selected],
            np.linspace(0, 99, 20, dtype=np.int64).tolist(),
        )

    def test_short_video_returns_all_decodable_frames_in_order(self) -> None:
        frames = [np.full((1, 1, 1), index) for index in range(7)]
        selected = select_evenly_spaced_frames(frames, 20)
        self.assertEqual(len(selected), 7)
        self.assertEqual(
            [int(frame[0, 0, 0]) for frame in selected],
            list(range(7)),
        )

    def test_selection_is_deterministic(self) -> None:
        first = evenly_spaced_indices(137, 20)
        second = evenly_spaced_indices(137, 20)
        self.assertTrue(np.array_equal(first, second))

    def test_rgb_dimensions_and_tensor_layout_are_preserved(self) -> None:
        frames = [
            np.zeros((224, 224, 3), dtype=np.float32)
            for _ in range(20)
        ]
        tensor = faces_to_tensor(frames)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tuple(tensor.shape), (20, 3, 224, 224))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_empty_video_returns_no_frames(self) -> None:
        self.assertEqual(evenly_spaced_indices(0, 20).tolist(), [])
        self.assertEqual(select_evenly_spaced_frames([], 20), [])

    def test_invalid_lengths_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            evenly_spaced_indices(-1, 20)
        with self.assertRaises(ValueError):
            evenly_spaced_indices(20, 0)


if __name__ == "__main__":
    unittest.main()
