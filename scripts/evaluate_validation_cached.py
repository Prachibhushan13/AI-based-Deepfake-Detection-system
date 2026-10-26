"""Evaluate the temporal checkpoint on cached validation features only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_torch_temporal import CachedFeatureDataset
from torch_model import CachedFeatureTemporalClassifier

DEFAULT_INDEX = ROOT / "datasets" / "celebdf_features" / "features_index.csv"
DEFAULT_CACHE = ROOT / "datasets" / "celebdf_features"
DEFAULT_CHECKPOINT = (
    ROOT / "backend" / "artifacts" / "models" / "torch" / "best_checkpoint_wsl.pt"
)
DEFAULT_OUTPUT_ROOT = ROOT / "evaluation_results"
THRESHOLDS = np.round(np.arange(0.10, 0.901, 0.05), 2)
MIN_FAKE_RECALL = 0.90


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_row(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict:
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    result = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    if np.unique(labels).size == 2:
        result["roc_auc"] = float(roc_auc_score(labels, probabilities))
        result["average_precision"] = float(
            average_precision_score(labels, probabilities)
        )
    else:
        result["roc_auc"] = None
        result["average_precision"] = None
    return result


def select_threshold(rows: list[dict]) -> tuple[dict | None, dict]:
    eligible = [
        row for row in rows if row["recall"] >= MIN_FAKE_RECALL
    ]
    if not eligible:
        return None, {
            "minimum_fake_recall": MIN_FAKE_RECALL,
            "selected": False,
            "reason": "No candidate threshold met the predeclared recall constraint.",
        }
    selected = sorted(
        eligible,
        key=lambda row: (
            -(row["specificity"] if row["specificity"] is not None else -1.0),
            -row["balanced_accuracy"],
            -row["f1"],
            abs(row["threshold"] - 0.5),
            row["threshold"],
        ),
    )[0]
    return selected, {
        "minimum_fake_recall": MIN_FAKE_RECALL,
        "selected": True,
        "selected_threshold": selected["threshold"],
        "tie_breaking": [
            "highest specificity",
            "highest balanced accuracy",
            "highest F1-score",
            "closest threshold to 0.50",
            "lowest threshold",
        ],
    }


def evaluate(args: argparse.Namespace) -> Path:
    index = pd.read_csv(args.index)
    required = {"relative_path", "feature_path", "label", "split", "status"}
    missing = required - set(index.columns)
    if missing:
        raise ValueError(f"Index is missing columns: {sorted(missing)}")
    validation = index[
        index["split"].eq("val") & index["status"].eq("ok")
    ].copy()
    if validation.empty:
        raise ValueError("No successful validation entries are available")
    validation = validation.sort_values("relative_path").reset_index(drop=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = args.output_root / f"{timestamp}_validation_cached"
    output_dir.mkdir(parents=True, exist_ok=False)
    validation.to_csv(output_dir / "validation_manifest.csv", index=False)

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = CachedFeatureTemporalClassifier().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    loader = DataLoader(
        CachedFeatureDataset(validation, args.cache_dir),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    labels: list[int] = []
    probabilities: list[float] = []
    statuses: dict[str, dict] = {
        path: {"status": "pending", "error": ""}
        for path in validation["relative_path"]
    }
    started = time.perf_counter()
    position = 0
    with torch.inference_mode():
        for features, targets in loader:
            batch_started = time.perf_counter()
            try:
                logits = model(features.to(device))
                batch_probabilities = torch.sigmoid(logits).cpu().numpy()
            except Exception as exc:
                raise RuntimeError(
                    "Cached validation inference failed; no partial result "
                    "will be reported."
                ) from exc
            elapsed = time.perf_counter() - batch_started
            batch_size = len(targets)
            labels.extend(targets.int().tolist())
            probabilities.extend(batch_probabilities.tolist())
            for offset in range(batch_size):
                path = str(validation.iloc[position + offset]["relative_path"])
                statuses[path] = {
                    "status": "ok",
                    "error": "",
                    "inference_time_seconds": elapsed / batch_size,
                }
            position += batch_size

    y_true = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    threshold_metrics = [
        metric_row(y_true, probs, float(threshold)) for threshold in THRESHOLDS
    ]
    selected_threshold, selection = select_threshold(threshold_metrics)
    fixed_metrics = metric_row(y_true, probs, args.threshold)

    prediction_rows = []
    for row, label, probability in zip(
        validation.to_dict("records"), y_true, probs
    ):
        path = str(row["relative_path"])
        status = statuses[path]
        prediction_rows.append(
            {
                "relative_path": path,
                "true_label": int(label),
                "fake_probability": float(probability),
                "predicted_label_at_0.50": int(probability >= args.threshold),
                "source_folder": str(row.get("folder", Path(path).parts[0])),
                "evaluation_status": status["status"],
                "error": status["error"],
                "inference_time_seconds": status.get(
                    "inference_time_seconds", None
                ),
            }
        )
    pd.DataFrame(prediction_rows).to_csv(
        output_dir / "validation_predictions.csv", index=False
    )

    source_metrics = {}
    for source, group in pd.DataFrame(prediction_rows).groupby(
        "source_folder", sort=True
    ):
        group_labels = group["true_label"].to_numpy(dtype=np.int64)
        group_probs = group["fake_probability"].to_numpy(dtype=np.float64)
        source_metrics[source] = {
            "rows": len(group),
            "real": int((group_labels == 0).sum()),
            "fake": int((group_labels == 1).sum()),
            "metrics_at_0.50": metric_row(
                group_labels, group_probs, args.threshold
            ),
            "roc_auc_available": np.unique(group_labels).size == 2,
        }

    (output_dir / "thresholds.csv").write_text(
        pd.DataFrame(threshold_metrics).to_csv(index=False), encoding="utf-8"
    )
    metrics_payload = {
        "validation_rows": len(validation),
        "successful_videos": len(prediction_rows),
        "skipped_videos": 0,
        "threshold_0.50": fixed_metrics,
        "selected_validation_threshold": selected_threshold,
        "threshold_selection": selection,
        "per_source": source_metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2), encoding="utf-8"
    )
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "index_path": str(args.index.resolve()),
        "index_sha256": sha256_file(args.index),
        "validation_manifest_path": str(
            (output_dir / "validation_manifest.csv").resolve()
        ),
        "validation_manifest_sha256": sha256_file(
            output_dir / "validation_manifest.csv"
        ),
        "split": "val",
        "threshold": args.threshold,
        "threshold_is_tuned": False,
        "minimum_fake_recall_for_selection": MIN_FAKE_RECALL,
        "requested_videos": len(validation),
        "successful_videos": len(prediction_rows),
        "skipped_videos": 0,
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
        "batch_size": args.batch_size,
        "preprocessing": {
            "input": "cached EfficientNet features used by training",
            "feature_shape": [20, 1280],
            "original_sampling": "20 evenly spaced frames over original video",
            "color_order": "BGR decoded by OpenCV, converted to RGB",
            "resize": "224x224",
            "pixel_range_before_backbone": "[0, 1] float32",
            "normalization": "ImageNet mean/std before EfficientNet",
            "label_mapping": {"0": "REAL", "1": "FAKE"},
        },
        "test_separation": (
            "Validation split only. Official test rows and saved test "
            "predictions were not read or used for threshold selection."
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics_payload, indent=2))
    print(f"Results saved to: {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        raise ValueError("--threshold must be between 0 and 1")
    evaluate(args)


if __name__ == "__main__":
    main()
