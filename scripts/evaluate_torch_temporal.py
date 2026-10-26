"""Evaluate one temporal checkpoint on the untouched official test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_torch_temporal import CachedFeatureDataset
from torch_model import CachedFeatureTemporalClassifier

DEFAULT_INDEX = ROOT / "datasets" / "celebdf_features" / "features_index.csv"
DEFAULT_CACHE = ROOT / "datasets" / "celebdf_features"
DEFAULT_CHECKPOINT = ROOT / "backend" / "artifacts" / "models" / "torch" / "best_checkpoint.pt"


def evaluate(checkpoint_path: Path, index_path: Path, cache_dir: Path, output: Path) -> None:
    index = pd.read_csv(index_path)
    test_frame = index[(index["split"] == "test") & (index["status"] == "ok")].copy()
    if test_frame.empty:
        raise ValueError("No successful official test entries are available")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model = CachedFeatureTemporalClassifier().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    loader = DataLoader(
        CachedFeatureDataset(test_frame, cache_dir),
        batch_size=32,
        shuffle=False,
    )
    criterion = nn.BCEWithLogitsLoss()
    labels: list[int] = []
    probabilities: list[float] = []
    total_loss = 0.0
    with torch.inference_mode():
        for features, targets in loader:
            logits = model(features.to(device))
            total_loss += criterion(logits, targets.to(device)).item() * len(targets)
            labels.extend(targets.int().tolist())
            probabilities.extend(torch.sigmoid(logits).cpu().tolist())
    y_true = np.asarray(labels)
    probs = np.asarray(probabilities)
    predictions = (probs >= 0.5).astype(int)
    metrics = {
        "checkpoint": str(checkpoint_path),
        "test_rows": len(test_frame),
        "real_count": int((y_true == 0).sum()),
        "fake_count": int((y_true == 1).sum()),
        "test_loss": total_loss / len(test_frame),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
        "false_positive_count": int(((y_true == 0) & (predictions == 1)).sum()),
        "false_negative_count": int(((y_true == 1) & (predictions == 0)).sum()),
        "classification_report": classification_report(
            y_true, predictions, output_dict=True, zero_division=0
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(metrics, indent=2))
    np.savez(
        output.with_suffix(".npz"),
        probabilities=probs,
        labels=y_true,
        relative_paths=test_frame["relative_path"].to_numpy(),
    )
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=ROOT / "backend/artifacts/models/torch/test_metrics")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.index, args.cache_dir, args.output)


if __name__ == "__main__":
    main()
