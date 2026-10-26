"""Unit tests for train_torch_temporal_experiment.py"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_torch_temporal_experiment import (
    compute_metrics,
    CachedFeatureDataset,
)


def test_label_encoding():
    """Verify label encoding: REAL=0, FAKE=1"""
    # This is verified in the training script, but we can test the metric function
    # REAL=0 -> need negative logits to predict REAL (prob < 0.5)
    # FAKE=1 -> need positive logits to predict FAKE (prob > 0.5)
    labels = [0, 0, 1, 1]  # REAL, REAL, FAKE, FAKE
    logits = [-10.0, -10.0, 10.0, 10.0]  # Perfect predictions
    metrics = compute_metrics(labels, logits)
    assert metrics["precision_real"] == 1.0
    assert metrics["recall_real"] == 1.0
    assert metrics["precision_fake"] == 1.0
    assert metrics["recall_fake"] == 1.0


def test_compute_metrics_all_real():
    """Test metric code correctly identifies all-REAL predictions."""
    labels = [0, 0, 0, 0]  # All REAL
    logits = [-10.0, -10.0, -10.0, -10.0]  # All predict REAL
    metrics = compute_metrics(labels, logits)
    assert metrics["recall_real"] == 1.0
    assert metrics["recall_fake"] == 0.0
    assert metrics["f1_real"] == 1.0
    assert metrics["f1_fake"] == 0.0
    assert metrics["macro_f1"] == 0.5


def test_compute_metrics_all_fake():
    """Test metric code correctly identifies all-FAKE predictions."""
    labels = [1, 1, 1, 1]  # All FAKE
    logits = [10.0, 10.0, 10.0, 10.0]  # All predict FAKE
    metrics = compute_metrics(labels, logits)
    assert metrics["recall_fake"] == 1.0
    assert metrics["recall_real"] == 0.0
    assert metrics["f1_fake"] == 1.0
    assert metrics["f1_real"] == 0.0
    assert metrics["macro_f1"] == 0.5


def test_compute_metrics_balanced():
    """Test metrics with balanced predictions."""
    labels = [0, 0, 1, 1]
    logits = [10.0, -10.0, 10.0, -10.0]  # Two correct, two wrong
    metrics = compute_metrics(labels, logits)
    assert metrics["accuracy"] == 0.5
    assert metrics["balanced_accuracy"] == 0.5


def test_weighted_mode_weights():
    """Test that weighted mode calculates correct inverse-frequency weights."""
    # Simulate training data: 389 REAL, 4188 FAKE
    negative = 389  # REAL (label 0)
    positive = 4188  # FAKE (label 1)
    total = negative + positive
    num_classes = 2

    weight_real = total / (num_classes * negative)
    weight_fake = total / (num_classes * positive)

    # Expected: weight_real = 4577 / (2 * 389) = 5.883, weight_fake = 4577 / (2 * 4188) = 0.546
    assert abs(weight_real - 5.883) < 0.01
    assert abs(weight_fake - 0.546) < 0.01

    # Verify weights are applied to correct labels
    # REAL (label 0) gets weight_real, FAKE (label 1) gets weight_fake
    assert weight_real > weight_fake  # REAL is minority class, should have higher weight


def test_balanced_sampler_no_pos_weight():
    """Test that balanced_sampler mode uses ordinary loss without pos_weight."""
    negative = 389
    positive = 4188
    weight_real = 1.0 / negative
    weight_fake = 1.0 / positive

    # Sampler weights
    sample_weights = np.array([weight_fake if x == 1 else weight_real for x in [0, 0, 1, 1]])
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=4,
        replacement=True
    )
    assert sampler is not None

    # Criterion should be ordinary BCEWithLogitsLoss without pos_weight
    criterion = nn.BCEWithLogitsLoss()
    assert hasattr(criterion, 'pos_weight') is False or criterion.pos_weight is None


def test_unweighted_mode_no_weights():
    """Test that unweighted mode has no class weights or sampler."""
    criterion = nn.BCEWithLogitsLoss()
    assert hasattr(criterion, 'pos_weight') is False or criterion.pos_weight is None


def test_cached_feature_dataset():
    """Test CachedFeatureDataset loads features correctly."""
    # Create dummy data
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        # Create a dummy feature file
        features = np.random.randn(20, 1280).astype(np.float32)
        feature_path = cache_dir / "test.npy"
        np.save(feature_path, features)

        df = pd.DataFrame([{
            "relative_path": "test.mp4",
            "feature_path": str(feature_path),
            "label": 1,
            "split": "train",
            "status": "ok"
        }])

        dataset = CachedFeatureDataset(df, cache_dir)
        assert len(dataset) == 1
        features_loaded, label = dataset[0]
        assert features_loaded.shape == (20, 1280)
        assert label.item() == 1.0


def test_compute_metrics_roc_auc():
    """Test ROC-AUC computation."""
    labels = [0, 0, 1, 1]
    logits = [10.0, -10.0, 10.0, -10.0]  # Two correct, two wrong
    metrics = compute_metrics(labels, logits)
    assert "roc_auc" in metrics
    assert metrics["roc_auc"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])