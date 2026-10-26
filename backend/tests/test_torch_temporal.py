from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from torch_model import CachedFeatureTemporalClassifier  # noqa: E402
from train_torch_temporal import CachedFeatureDataset  # noqa: E402


def make_index(tmp_path: Path) -> pd.DataFrame:
    feature_path = tmp_path / "features.npy"
    np.save(feature_path, np.zeros((20, 1280), dtype=np.float32))
    return pd.DataFrame(
        [
            {
                "relative_path": "video.mp4",
                "feature_path": str(feature_path),
                "label": 1,
                "split": "train",
                "status": "ok",
            }
        ]
    )


def test_cached_feature_dataset_shape_and_label(tmp_path: Path) -> None:
    frame = make_index(tmp_path)
    features, label = CachedFeatureDataset(frame, tmp_path)[0]
    assert features.shape == (20, 1280)
    assert features.dtype == torch.float32
    assert label.item() == 1.0


def test_cached_feature_batch_forward_backward_updates_parameters(
    tmp_path: Path,
) -> None:
    frame = pd.concat([make_index(tmp_path)] * 2, ignore_index=True)
    dataset = CachedFeatureDataset(frame, tmp_path)
    features = torch.stack([dataset[0][0], dataset[1][0]])
    labels = torch.tensor([0.0, 1.0])
    model = CachedFeatureTemporalClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    loss = torch.nn.BCEWithLogitsLoss()(model(features), labels)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, model.parameters())
    )


def test_cached_feature_dataset_rejects_corrupt_file(tmp_path: Path) -> None:
    frame = make_index(tmp_path)
    np.save(tmp_path / "features.npy", np.zeros((3, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="Invalid cached features"):
        CachedFeatureDataset(frame, tmp_path)[0]
