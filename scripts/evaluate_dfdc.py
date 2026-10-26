#!/usr/bin/env python3
"""
Evaluate the trained deepfake detection model on DFDC dataset.

This script loads the trained PyTorch temporal model and evaluates it
on all valid, labelled videos from the DFDC manifest.

Usage:
    python scripts/evaluate_dfdc.py --manifest PATH

Required:
    --manifest: Path to the DFDC manifest CSV (from prepare_dfdc_manifest.py)

Optional:
    --output-dir: Output directory for predictions/report (default: same as manifest parent)
    --checkpoint-hash: Expected SHA-256 of checkpoint (default: from .env)
    --dry-run: Validate manifest and model loading without running inference
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import cv2
import torch
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, average_precision_score
)

# Add backend to path and change working directory to backend so .env is found
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
os.chdir(BACKEND_DIR)
sys.path.insert(0, str(BACKEND_DIR))

from app.ml.inference import load_model, predict_video
from app.services.video_service import extract_frames
from app.core.config import settings


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate the trained deepfake detection model on DFDC dataset"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to the DFDC manifest CSV (from prepare_dfdc_manifest.py)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for predictions/report (default: manifest parent directory)",
    )
    parser.add_argument(
        "--checkpoint-hash",
        type=str,
        default=None,
        help="Expected SHA-256 of checkpoint (default: from .env TORCH_CHECKPOINT_SHA256)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate manifest and model loading without running inference",
    )
    return parser.parse_args()


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    """Load the DFDC manifest CSV."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}. Run prepare_dfdc_manifest.py first.")

    df = pd.read_csv(manifest_path)
    return df


def filter_evaluable_videos(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to only readable videos that can be evaluated."""
    return df[df["readable"] == True].copy()


def run_inference_on_video(video_path: str) -> Dict[str, Any]:
    """Run inference on a single video file."""
    result = {
        "success": False,
        "fake_probability": None,
        "predicted_label": None,
        "predicted_numeric": None,
        "model_mode": None,
        "error": None,
    }

    try:
        # Extract frames
        frames = extract_frames(video_path)
        if not frames:
            result["error"] = "No frames extracted from video"
            return result

        # Run prediction
        pred = predict_video(frames, os.path.basename(video_path))

        result["success"] = True
        result["fake_probability"] = pred["fakeProbability"]
        result["predicted_label"] = pred["result"]  # "REAL" or "FAKE"
        result["predicted_numeric"] = 1 if pred["result"] == "FAKE" else 0
        result["model_mode"] = pred["modelMode"]

    except Exception as e:
        result["error"] = str(e)

    return result


def compute_metrics(y_true: List[int], y_pred: List[int], y_scores: List[float]) -> Dict[str, Any]:
    """Compute evaluation metrics."""
    metrics = {}

    # Basic counts
    metrics["total_evaluated"] = len(y_true)
    metrics["real_count"] = y_true.count(0)
    metrics["fake_count"] = y_true.count(1)
    metrics["predicted_real"] = y_pred.count(0)
    metrics["predicted_fake"] = y_pred.count(1)

    # Confusion matrix
    try:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        metrics["confusion_matrix"] = cm.tolist()
        metrics["tn"] = int(cm[0, 0])
        metrics["fp"] = int(cm[0, 1])
        metrics["fn"] = int(cm[1, 0])
        metrics["tp"] = int(cm[1, 1])
    except Exception as e:
        metrics["confusion_matrix"] = [[0, 0], [0, 0]]
        metrics["tn"] = metrics["fp"] = metrics["fn"] = metrics["tp"] = 0
        metrics["confusion_matrix_error"] = str(e)

    # Standard metrics
    try:
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    except Exception:
        metrics["accuracy"] = None

    try:
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    except Exception:
        metrics["balanced_accuracy"] = None

    try:
        metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    except Exception:
        metrics["precision"] = None

    try:
        metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))  # Fake-class recall
    except Exception:
        metrics["recall"] = None

    try:
        metrics["specificity"] = float(metrics["tn"] / (metrics["tn"] + metrics["fp"])) if (metrics["tn"] + metrics["fp"]) > 0 else None
    except Exception:
        metrics["specificity"] = None

    try:
        metrics["fpr"] = float(metrics["fp"] / (metrics["fp"] + metrics["tn"])) if (metrics["fp"] + metrics["tn"]) > 0 else None  # False positive rate
    except Exception:
        metrics["fpr"] = None

    try:
        metrics["fnr"] = float(metrics["fn"] / (metrics["fn"] + metrics["tp"])) if (metrics["fn"] + metrics["tp"]) > 0 else None  # False negative rate
    except Exception:
        metrics["fnr"] = None

    try:
        metrics["f1_score"] = float(f1_score(y_true, y_pred, zero_division=0))
    except Exception:
        metrics["f1_score"] = None

    # ROC-AUC and Average Precision (require both classes and valid scores)
    try:
        if len(set(y_true)) == 2 and len(y_scores) == len(y_true):
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_scores))
            metrics["average_precision"] = float(average_precision_score(y_true, y_scores))
        else:
            metrics["roc_auc"] = None
            metrics["average_precision"] = None
            metrics["roc_auc_note"] = "Requires both classes present and valid scores"
    except Exception as e:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None
        metrics["roc_auc_error"] = str(e)

    # Decision threshold used
    metrics["decision_threshold"] = float(settings.decision_threshold)

    return metrics


def main():
    args = parse_args()

    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        print(f"Error: Manifest not found: {manifest_path}")
        sys.exit(1)

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = manifest_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / "dfdc_predictions.csv"
    evaluation_report_path = output_dir / "dfdc_evaluation_report.json"

    # --- Verify checkpoint hash ---
    expected_hash = args.checkpoint_hash or settings.torch_checkpoint_sha256.strip().lower()
    if not expected_hash:
        print("Error: No expected checkpoint hash provided (--checkpoint-hash or TORCH_CHECKPOINT_SHA256 in .env)")
        sys.exit(1)

    checkpoint_path = settings.torch_model_artifact_path
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    import hashlib
    with open(checkpoint_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest().lower()

    if file_hash != expected_hash:
        print(f"Error: Checkpoint SHA-256 mismatch!")
        print(f"  Expected: {expected_hash}")
        print(f"  Got:      {file_hash}")
        sys.exit(1)

    print(f"Checkpoint SHA-256 verified: {file_hash}")

    # --- Load manifest ---
    print(f"Loading manifest: {manifest_path}")
    try:
        manifest = load_manifest(manifest_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Total videos in manifest: {len(manifest)}")

    # Filter to evaluable videos
    evaluable = filter_evaluable_videos(manifest)
    print(f"Readable/evaluable videos: {len(evaluable)}")

    if len(evaluable) == 0:
        print("No evaluable videos found. Cannot run evaluation.")
        # Create empty outputs
        pd.DataFrame(columns=[
            "filename", "ground_truth_label", "ground_truth_numeric",
            "predicted_label", "predicted_numeric", "fake_probability",
            "correct", "inference_status", "error", "model_mode"
        ]).to_csv(predictions_path, index=False)

        with open(evaluation_report_path, "w") as f:
            json.dump({
                "dataset": "DFDC",
                "total_videos_in_manifest": len(manifest),
                "evaluable_videos": 0,
                "successfully_evaluated": 0,
                "failed_inference": 0,
                "metrics": {},
                "note": "No evaluable videos in manifest"
            }, f, indent=2)
        return 1

    if args.dry_run:
        print("\nDry-run mode: Validating manifest and model loading...")
        # Load model to verify it works
        model = load_model()
        model_type = getattr(model, 'model_type', 'unknown')
        print(f"Model loaded: {model_type}")
        if model_type != 'torch_temporal':
            print(f"Error: Expected torch_temporal model, got {model_type}")
            sys.exit(1)
        print("Dry-run successful. Model and manifest are ready for evaluation.")
        return 0

    # Load model once
    print("\nLoading model...")
    model = load_model()
    model_type = getattr(model, 'model_type', 'unknown')
    print(f"Model loaded: {model_type}")
    if model_type != 'torch_temporal':
        print(f"Error: Expected torch_temporal model, got {model_type}")
        print("Refusing to evaluate with mock/fallback model.")
        sys.exit(1)
    print(f"Decision threshold: {settings.decision_threshold}")

    # Run inference on each evaluable video
    print(f"\nRunning inference on {len(evaluable)} videos...")
    predictions = []
    y_true = []
    y_pred = []
    y_scores = []

    for idx, row in evaluable.iterrows():
        filename = row["filename"]
        video_path = row["absolute_path"]
        gt_label = row["ground_truth_label"]
        gt_numeric = row["numeric_label"]

        print(f"  [{idx+1}/{len(evaluable)}] {filename} (GT: {gt_label})")

        inf_result = run_inference_on_video(video_path)

        if inf_result["success"]:
            pred_label = inf_result["predicted_label"]
            pred_numeric = inf_result["predicted_numeric"]
            fake_prob = inf_result["fake_probability"]
            correct = (pred_numeric == gt_numeric)

            y_true.append(gt_numeric)
            y_pred.append(pred_numeric)
            y_scores.append(fake_prob)

            pred_record = {
                "filename": filename,
                "ground_truth_label": gt_label,
                "ground_truth_numeric": gt_numeric,
                "predicted_label": pred_label,
                "predicted_numeric": pred_numeric,
                "fake_probability": fake_prob,
                "correct": correct,
                "inference_status": "success",
                "error": "",
                "model_mode": inf_result["model_mode"],
            }
            predictions.append(pred_record)
            print(f"      -> Pred: {pred_label} (fake_prob={fake_prob:.4f}) {'OK' if correct else 'FAIL'}")
        else:
            pred_record = {
                "filename": filename,
                "ground_truth_label": gt_label,
                "ground_truth_numeric": gt_numeric,
                "predicted_label": None,
                "predicted_numeric": None,
                "fake_probability": None,
                "correct": None,
                "inference_status": "failed",
                "error": inf_result["error"],
                "model_mode": None,
            }
            predictions.append(pred_record)
            print(f"      -> FAILED: {inf_result['error']}")

    # Save predictions
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(predictions_path, index=False)
    print(f"\nPredictions saved to: {predictions_path}")

    # Compute metrics on successful predictions only
    successful = pred_df[pred_df["inference_status"] == "success"]
    print(f"\nSuccessfully evaluated: {len(successful)} / {len(evaluable)}")
    print(f"Failed inference: {len(evaluable) - len(successful)}")

    if len(successful) > 0:
        y_true = successful["ground_truth_numeric"].tolist()
        y_pred = successful["predicted_numeric"].tolist()
        y_scores = successful["fake_probability"].tolist()

        metrics = compute_metrics(y_true, y_pred, y_scores)

        # Create evaluation report
        report = {
            "dataset": "DFDC",
            "model": {
                "type": model_type,
                "checkpoint": str(settings.torch_model_artifact_path),
                "decision_threshold": float(settings.decision_threshold),
            },
            "total_videos_in_manifest": len(manifest),
            "evaluable_videos": len(evaluable),
            "successfully_evaluated": len(successful),
            "failed_inference": len(evaluable) - len(successful),
            "metrics": metrics,
            "limitations": [
                "DFDC dataset may have different synthesis methods than Celeb-DF training data",
                "Threshold (0.66) fixed from Celeb-DF validation, not tuned on DFDC",
                "Results reflect cross-dataset generalization, not independent test performance",
                "DFDC metadata.json labels used as ground truth; label noise possible",
            ]
        }
    else:
        report = {
            "dataset": "DFDC",
            "model": {
                "type": model_type,
                "checkpoint": str(settings.torch_model_artifact_path),
                "decision_threshold": float(settings.decision_threshold),
            },
            "total_videos_in_manifest": len(manifest),
            "evaluable_videos": len(evaluable),
            "successfully_evaluated": 0,
            "failed_inference": len(evaluable),
            "metrics": {},
            "note": "All inference attempts failed"
        }

    # Save evaluation report
    with open(evaluation_report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Evaluation report saved to: {evaluation_report_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    if "metrics" in report and report["metrics"]:
        m = report["metrics"]
        print(f"Total evaluated: {m.get('total_evaluated', 0)}")
        print(f"  REAL: {m.get('real_count', 0)}, FAKE: {m.get('fake_count', 0)}")
        print(f"  Predicted REAL: {m.get('predicted_real', 0)}, Predicted FAKE: {m.get('predicted_fake', 0)}")
        print(f"Confusion Matrix: TN={m.get('tn',0)} FP={m.get('fp',0)} FN={m.get('fn',0)} TP={m.get('tp',0)}")
        print(f"Accuracy: {m.get('accuracy', 'N/A')}")
        print(f"Balanced Accuracy: {m.get('balanced_accuracy', 'N/A')}")
        print(f"Precision: {m.get('precision', 'N/A')}")
        print(f"Recall (Fake-class): {m.get('recall', 'N/A')}")
        print(f"Specificity: {m.get('specificity', 'N/A')}")
        print(f"FPR: {m.get('fpr', 'N/A')}")
        print(f"FNR: {m.get('fnr', 'N/A')}")
        print(f"F1-Score: {m.get('f1_score', 'N/A')}")
        print(f"ROC-AUC: {m.get('roc_auc', 'N/A')}")
        print(f"Average Precision: {m.get('average_precision', 'N/A')}")

    print(f"\nOutput files:")
    print(f"  Predictions: {predictions_path}")
    print(f"  Report: {evaluation_report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())