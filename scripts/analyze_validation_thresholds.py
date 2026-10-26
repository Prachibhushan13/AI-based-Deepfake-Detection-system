
"""Analyze classification thresholds on the validation split only."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_torch_temporal import CachedFeatureDataset
from torch_model import CachedFeatureTemporalClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "backend/artifacts/models/torch/gpu_experiment/best_checkpoint.pt",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=ROOT / "datasets/celebdf_features/features_index.csv",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "datasets/celebdf_features",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index = pd.read_csv(args.index)
    val = index[(index["split"] == "val") & (index["status"] == "ok")].copy()

    if val.empty:
        raise ValueError("No usable validation videos found.")

    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    model = CachedFeatureTemporalClassifier().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    loader = DataLoader(
        CachedFeatureDataset(val, args.cache_dir),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    labels, probabilities = [], []
    with torch.inference_mode():
        for features, targets in loader:
            logits = model(features.to(device))
            probabilities.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            labels.extend(targets.numpy().astype(int).tolist())

    y_true = np.asarray(labels)
    probs = np.asarray(probabilities)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}; validation videos: {len(y_true)}")
    print("Labels: 0=real, 1=fake")
    print()
    print(
        f"{'Threshold':>9} {'Accuracy':>9} {'Bal.Acc':>9} "
        f"{'Precision':>10} {'Recall':>9} {'F1':>8} "
        f"{'FP(real)':>9} {'FN(fake)':>9}"
    )

    for threshold in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        predictions = (probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).ravel()

        print(
            f"{threshold:9.2f} "
            f"{accuracy_score(y_true, predictions):9.3f} "
            f"{balanced_accuracy_score(y_true, predictions):9.3f} "
            f"{precision_score(y_true, predictions, zero_division=0):10.3f} "
            f"{recall_score(y_true, predictions, zero_division=0):9.3f} "
            f"{f1_score(y_true, predictions, zero_division=0):8.3f} "
            f"{fp:9d} {fn:9d}"
        )

    print("\nNo files were modified; the official test split was not used.")


if __name__ == "__main__":
    main()
