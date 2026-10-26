#!/usr/bin/env python3
"""
SDFVD Evaluation Script for Deepfake Detection System

Evaluates the production PyTorch temporal model on the SDFVD dataset
using the exact production inference pipeline (full-frame, no face detection).

Usage:
    python scripts/evaluate_sdfvd.py \
        --checkpoint backend/artifacts/models/torch/best_checkpoint.pt \
        --manifest datasets/sdfvd/processed/sdfvd_manifest.csv \
        --output backend/artifacts/models/torch/sdfvd_evaluation \
        --device auto --threshold 0.532962620
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
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
    roc_auc_score,
)

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.ml.temporal_sampling import select_evenly_spaced_frames
from app.ml.torch_inference import TorchTemporalInference


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def load_and_preprocess_video(
    video_path: Path,
    sequence_length: int = 20,
    image_size: int = 224,
) -> list[np.ndarray]:
    """
    Load video and apply exact production preprocessing:
    - Read frames with OpenCV
    - Select evenly spaced frames
    - Convert BGR to RGB
    - Resize to image_size x image_size
    - Scale to [0, 1] float32
    - Pad with last frame if needed
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    try:
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    finally:
        cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"Video has no readable frames: {video_path}")

    # Select evenly spaced frames (production behavior)
    selected_frames = select_evenly_spaced_frames(frames, sequence_length)

    # Resize and normalize to [0, 1]
    processed_frames = []
    for frame in selected_frames:
        resized = cv2.resize(
            frame,
            (image_size, image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        processed_frames.append(resized.astype(np.float32) / 255.0)

    # Pad if fewer frames than sequence_length (production behavior)
    while len(processed_frames) < sequence_length:
        processed_frames.append(processed_frames[-1].copy())

    return processed_frames


def evaluate_model(
    checkpoint_path: Path,
    manifest_path: Path,
    output_dir: Path,
    device: str = "auto",
    threshold: float = 0.532962620,
    sequence_length: int = 20,
    image_size: int = 224,
) -> dict[str, Any]:
    """
    Run evaluation on SDFVD dataset.
    """
    # Verify checkpoint exists and record hash
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint_hash_before = compute_sha256(checkpoint_path)
    print(f"Checkpoint SHA-256 (before): {checkpoint_hash_before}")

    # Load manifest
    manifest = pd.read_csv(manifest_path)
    print(f"Loaded manifest: {len(manifest)} videos")
    print(f"Label distribution: {manifest['numeric_label'].value_counts().to_dict()}")

    # Load model using production code
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model on {device}...")
    model = TorchTemporalInference(
        checkpoint_path,
        device=device,
        sequence_length=sequence_length,
        image_size=image_size,
    )

    # Process each video
    predictions = []
    failures = []

    for idx, row in manifest.iterrows():
        video_path = Path(row["absolute_path"])
        true_label = int(row["numeric_label"])  # 0=REAL, 1=FAKE

        try:
            # Preprocess video
            frames = load_and_preprocess_video(
                video_path,
                sequence_length=sequence_length,
                image_size=image_size,
            )

            # Run inference
            fake_prob = model.predict(frames)

            # Decision
            decision = "FAKE" if fake_prob >= threshold else "REAL"
            pred_label = 1 if fake_prob >= threshold else 0

            predictions.append({
                "filename": row["filename"],
                "video_path": str(video_path),
                "true_label": true_label,
                "pred_label": pred_label,
                "fake_probability": fake_prob,
                "decision_threshold": threshold,
                "decision": decision,
                "status": "success",
            })

        except Exception as e:
            failures.append({
                "filename": row["filename"],
                "video_path": str(video_path),
                "error": str(e),
            })
            predictions.append({
                "filename": row["filename"],
                "video_path": str(video_path),
                "true_label": true_label,
                "pred_label": None,
                "fake_probability": None,
                "decision_threshold": threshold,
                "decision": None,
                "status": "failed",
            })
            print(f"FAILED: {row['filename']} - {e}")

    # Filter successful predictions for metrics
    successful = [p for p in predictions if p["status"] == "success"]
    print(f"\nEvaluated: {len(successful)}/{len(predictions)} videos")
    print(f"Failed: {len(failures)}")

    if len(successful) == 0:
        raise RuntimeError("No videos were successfully evaluated")

    # Compute metrics
    y_true = np.array([p["true_label"] for p in successful], dtype=np.int64)
    y_pred = np.array([p["pred_label"] for p in successful], dtype=np.int64)
    y_probs = np.array([p["fake_probability"] for p in successful], dtype=np.float64)

    # Confusion matrix (rows=actual, cols=predicted, labels=[0,1])
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Per-class metrics
    precision_real = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    recall_real = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1_real = f1_score(y_true, y_pred, pos_label=0, zero_division=0)

    precision_fake = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall_fake = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_fake = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    macro_f1 = (f1_real + f1_fake) / 2.0
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_probs)

    # Threshold sweep for analysis (NOT for selection)
    thresholds = np.arange(0.05, 0.96, 0.05)
    threshold_sweep = []
    for t in thresholds:
        preds_t = (y_probs >= t).astype(int)
        cm_t = confusion_matrix(y_true, preds_t, labels=[0, 1])
        tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
        threshold_sweep.append({
            "threshold": float(t),
            "real_recall": float(tn_t / (tn_t + fp_t)) if (tn_t + fp_t) > 0 else 0.0,
            "fake_recall": float(tp_t / (tp_t + fn_t)) if (tp_t + fn_t) > 0 else 0.0,
            "real_precision": float(tn_t / (tn_t + fn_t)) if (tn_t + fn_t) > 0 else 0.0,
            "fake_precision": float(tp_t / (tp_t + fp_t)) if (tp_t + fp_t) > 0 else 0.0,
            "macro_f1": float(f1_score(y_true, preds_t, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, preds_t)),
            "confusion_matrix": cm_t.tolist(),
        })

    # Prepare results
    metrics = {
        "confusion_matrix": cm.tolist(),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "real_recall": float(recall_real),
        "fake_recall": float(recall_fake),
        "real_precision": float(precision_real),
        "fake_precision": float(precision_fake),
        "f1_real": float(f1_real),
        "f1_fake": float(f1_fake),
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(bal_acc),
        "accuracy": float(acc),
        "roc_auc": float(roc_auc),
        "threshold_used": threshold,
    }

    run_metadata = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256_before": checkpoint_hash_before,
        "checkpoint_sha256_after": compute_sha256(checkpoint_path),
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
        "device": device,
        "threshold": threshold,
        "sequence_length": sequence_length,
        "image_size": image_size,
        "total_videos": len(predictions),
        "successful": len(successful),
        "failed": len(failures),
        "failures": failures,
        "preprocessing": {
            "frame_sampling": "evenly_spaced",
            "sequence_length": sequence_length,
            "image_size": image_size,
            "color_conversion": "BGR2RGB",
            "resize_interpolation": "INTER_LINEAR",
            "normalization": "[0, 1] float32",
            "padding": "repeat_last_frame",
        },
    }

    # Verify checkpoint integrity
    if run_metadata["checkpoint_sha256_before"] != run_metadata["checkpoint_sha256_after"]:
        raise RuntimeError("Checkpoint SHA-256 changed during evaluation!")

    return {
        "predictions": predictions,
        "metrics": metrics,
        "threshold_sweep": threshold_sweep,
        "run_metadata": run_metadata,
    }


def save_results(
    results: dict[str, Any],
    output_dir: Path,
) -> None:
    """Save all evaluation outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # predictions.csv
    pred_df = pd.DataFrame(results["predictions"])
    pred_df.to_csv(output_dir / "predictions.csv", index=False)

    # metrics.json
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results["metrics"], f, indent=2)

    # threshold_sweep.json
    with open(output_dir / "threshold_sweep.json", "w") as f:
        json.dump(results["threshold_sweep"], f, indent=2)

    # run_metadata.json
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(results["run_metadata"], f, indent=2)

    print(f"\nResults saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SDFVD dataset")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--threshold", type=float, default=0.532962620)
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    results = evaluate_model(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        output_dir=args.output,
        device=args.device,
        threshold=args.threshold,
        sequence_length=args.sequence_length,
        image_size=args.image_size,
    )

    save_results(results, args.output)

    # Print summary
    m = results["metrics"]
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Confusion Matrix: {m['confusion_matrix']}")
    print(f"REAL recall: {m['real_recall']:.4f}")
    print(f"FAKE recall: {m['fake_recall']:.4f}")
    print(f"REAL precision: {m['real_precision']:.4f}")
    print(f"FAKE precision: {m['fake_precision']:.4f}")
    print(f"Macro F1: {m['macro_f1']:.4f}")
    print(f"Balanced Accuracy: {m['balanced_accuracy']:.4f}")
    print(f"Accuracy: {m['accuracy']:.4f}")
    print(f"ROC-AUC: {m['roc_auc']:.4f}")
    print(f"Threshold: {m['threshold_used']}")
    print("=" * 60)


if __name__ == "__main__":
    main()