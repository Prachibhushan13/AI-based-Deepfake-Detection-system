"""Reproducible held-out video evaluation for the PyTorch detector.

This script is intentionally standalone and does not import TensorFlow. It
defaults to the official Celeb-DF test split from the project manifest, but
the result is only video-held-out (not identity-held-out) unless a stronger
manifest is supplied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None
    plt = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "app" / "ml"))
from torch_inference import TorchTemporalInference

DEFAULT_MANIFEST = ROOT / "datasets" / "celebdf_manifest.csv"
DEFAULT_DATASET = ROOT / "datasets" / "Celeb-DF-v2"
DEFAULT_CHECKPOINT = (
    ROOT / "backend" / "artifacts" / "models" / "torch" / "best_checkpoint_wsl.pt"
)
DEFAULT_OUTPUT_ROOT = ROOT / "evaluation_results"
SEQUENCE_LENGTH = 20
IMAGE_SIZE = 224
THRESHOLDS = np.round(np.arange(0.10, 0.901, 0.05), 2)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_video_frames(path: Path) -> np.ndarray:
    """Sample 20 raw video frames using the cache-generation convention."""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError("video could not be opened")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 1:
            raise RuntimeError("video has no reported frames")
        targets = np.linspace(
            0, frame_count - 1, SEQUENCE_LENGTH, dtype=np.int64
        )
        target_positions = {}
        for output_index, frame_index in enumerate(targets):
            target_positions.setdefault(int(frame_index), []).append(output_index)
        frames: list[np.ndarray | None] = [None] * SEQUENCE_LENGTH
        frame_index = 0
        while frame_index <= int(targets[-1]):
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index in target_positions:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(
                    rgb,
                    (IMAGE_SIZE, IMAGE_SIZE),
                    interpolation=cv2.INTER_LINEAR,
                )
                normalized = resized.astype(np.float32) / 255.0
                for output_index in target_positions[frame_index]:
                    frames[output_index] = normalized
            frame_index += 1
        if any(frame is None for frame in frames):
            decoded = sum(frame is not None for frame in frames)
            raise RuntimeError(
                f"could not decode all selected frames ({decoded}/{SEQUENCE_LENGTH})"
            )
        return np.stack(frames)
    finally:
        capture.release()


def resolve_video_path(row: pd.Series, dataset_root: Path) -> Path:
    relative = Path(str(row["relative_path"]))
    candidate = dataset_root / relative
    if candidate.is_file():
        return candidate
    recorded = Path(str(row.get("video_path", "")))
    if recorded.is_file():
        return recorded
    return candidate


def classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict:
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if tn + fp else None
    result = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": specificity,
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "classification_report": classification_report(
            labels,
            predictions,
            labels=[0, 1],
            target_names=["REAL", "FAKE"],
            output_dict=True,
            zero_division=0,
        ),
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


def save_curves(
    labels: np.ndarray, probabilities: np.ndarray, output_dir: Path
) -> None:
    if plt is None or np.unique(labels).size != 2:
        return
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
    precision, recall, _ = precision_recall_curve(labels, probabilities)

    plt.figure()
    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"ROC-AUC = {roc_auc_score(labels, probabilities):.4f}",
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False-positive rate")
    plt.ylabel("True-positive rate")
    plt.title("Held-out ROC curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(
        recall,
        precision,
        label=f"Average precision = {average_precision_score(labels, probabilities):.4f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Held-out precision-recall curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_curve.png", dpi=160)
    plt.close()


def save_confusion_matrix(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float, output_dir: Path
) -> None:
    if plt is None:
        return
    predictions = (probabilities >= threshold).astype(np.int64)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(4.5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["REAL", "FAKE"],
        yticklabels=["REAL", "FAKE"],
        xlabel="Predicted label",
        ylabel="Actual label",
        title=f"Confusion matrix (threshold={threshold:.2f})",
    )
    for row in range(2):
        for column in range(2):
            axis.text(column, row, int(matrix[row, column]), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(figure)


def evaluate(args: argparse.Namespace) -> Path:
    random.seed(args.seed)
    np.random.seed(args.seed)
    manifest = pd.read_csv(args.manifest)
    required = {"relative_path", "label", "split"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    selected = manifest[manifest["split"].eq(args.split)].copy()
    if selected.empty:
        raise ValueError(f"No manifest rows found for split={args.split!r}")

    if args.sample_per_class is not None:
        sampled = []
        for label in (0, 1):
            group = selected[selected["label"].eq(label)]
            if len(group) < args.sample_per_class:
                raise ValueError(
                    f"Split {args.split!r} has fewer than "
                    f"{args.sample_per_class} examples for label {label}"
                )
            sampled.append(
                group.sample(
                    n=args.sample_per_class,
                    random_state=args.seed + label,
                )
            )
        selected = pd.concat(sampled).sort_values("relative_path")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_root / f"{timestamp}_{args.split}"
    output_dir.mkdir(parents=True, exist_ok=False)
    selected.to_csv(output_dir / "evaluation_manifest.csv", index=False)

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    checkpoint_metadata = {
        key: checkpoint[key]
        for key in ("epoch", "best_val_loss", "config")
        if key in checkpoint
    }
    model = TorchTemporalInference(args.checkpoint, device=args.device)
    if plt is None:
        print(
            "WARNING: Matplotlib is unavailable; skipping plot generation. "
            "Metrics, CSV, JSON, and confusion-matrix calculations will continue.",
            flush=True,
        )
    labels: list[int] = []
    probabilities: list[float] = []
    prediction_rows: list[dict] = []
    skipped_rows: list[dict] = []
    started = time.perf_counter()

    for position, (_, row) in enumerate(selected.iterrows(), start=1):
        relative_path = str(row["relative_path"])
        video_path = resolve_video_path(row, args.dataset_root)
        print(
            f"[{position}/{len(selected)}] {relative_path}",
            flush=True,
        )
        try:
            frames = sample_video_frames(video_path)
            probability = model.predict(list(frames))
            if not np.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError(f"invalid probability {probability!r}")
        except Exception as exc:
            skipped_rows.append(
                {
                    "relative_path": relative_path,
                    "label": int(row["label"]),
                    "video_path": str(video_path),
                    "error": str(exc),
                }
            )
            print(f"  SKIP: {exc}", flush=True)
            continue
        labels.append(int(row["label"]))
        probabilities.append(float(probability))
        prediction_rows.append(
            {
                "relative_path": relative_path,
                "video_path": str(video_path),
                "label": int(row["label"]),
                "fake_probability": float(probability),
                "prediction_at_threshold": int(probability >= args.threshold),
            }
        )

    if not labels:
        raise RuntimeError("No videos were evaluated successfully")

    y_true = np.asarray(labels, dtype=np.int64)
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities_array >= args.threshold).astype(np.int64)
    for row in prediction_rows:
        row["prediction_at_threshold"] = int(
            float(row["fake_probability"]) >= args.threshold
        )
    pd.DataFrame(prediction_rows).to_csv(
        output_dir / "predictions.csv", index=False
    )
    pd.DataFrame(skipped_rows).to_csv(output_dir / "skipped.csv", index=False)
    np.savez(
        output_dir / "predictions.npz",
        labels=y_true,
        probabilities=probabilities_array,
        predictions=predictions,
    )

    metrics = classification_metrics(
        y_true, probabilities_array, args.threshold
    )
    threshold_rows = [
        classification_metrics(y_true, probabilities_array, float(threshold))
        for threshold in THRESHOLDS
    ]
    pd.DataFrame(
        [
            {
                key: value
                for key, value in row.items()
                if key not in {"confusion_matrix", "classification_report"}
            }
            for row in threshold_rows
        ]
    ).to_csv(output_dir / "thresholds.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    save_confusion_matrix(
        y_true, probabilities_array, args.threshold, output_dir
    )
    save_curves(y_true, probabilities_array, output_dir)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_metadata": checkpoint_metadata,
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "dataset_root": str(args.dataset_root.resolve()),
        "split": args.split,
        "split_claim": (
            "official video-level test split from the manifest; "
            "not identity-disjoint and not proven source-disjoint"
            if args.split == "test"
            else "manifest split; independence must be established separately"
        ),
        "seed": args.seed,
        "threshold": args.threshold,
        "threshold_is_tuned": False,
        "sample_per_class": args.sample_per_class,
        "requested_videos": len(selected),
        "evaluated_videos": len(prediction_rows),
        "skipped_videos": len(skipped_rows),
        "elapsed_seconds": time.perf_counter() - started,
        "preprocessing": {
            "sampling": "20 evenly spaced raw video frames",
            "color_order": "BGR decoded by OpenCV, converted to RGB",
            "resize": "224x224, cv2.INTER_LINEAR",
            "pixel_range": "[0, 1] float32 before ImageNet normalization",
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
            "sequence_length": SEQUENCE_LENGTH,
            "label_mapping": {"0": "REAL", "1": "FAKE"},
            "model_score": "sigmoid(logit), treated as fake score; calibration unverified",
        },
        "metrics_note": (
            "ROC-AUC and average precision are threshold-independent; "
            "classification metrics use the configured threshold."
        ),
        "skip_bias_warning": (
            "Skipped videos are excluded from metrics. Inspect skipped.csv; "
            "systematic decode failures can bias results."
            if skipped_rows
            else "No videos were skipped."
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )

    print(json.dumps(metrics, indent=2))
    print(f"Results saved to: {output_dir}")
    print(f"Skipped videos: {len(skipped_rows)}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--split", choices=("test", "val", "train"), default="test")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample-per-class",
        type=int,
        default=None,
        help="Optional exploratory balanced sample; omitted means the full split.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        raise ValueError("--threshold must be between 0 and 1")
    if args.sample_per_class is not None and args.sample_per_class < 1:
        raise ValueError("--sample-per-class must be positive")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")
    evaluate(args)


if __name__ == "__main__":
    main()
