"""Compare legacy and full-video temporal sampling on one fixed manifest."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "app" / "ml"))

from app.ml.temporal_sampling import select_evenly_spaced_frames
from evaluate_heldout import (
    classification_metrics,
    resolve_video_path,
    sha256_file,
)
from torch_inference import TorchTemporalInference

DEFAULT_MANIFEST = ROOT / "datasets" / "celebdf_manifest.csv"
DEFAULT_DATASET = ROOT / "datasets" / "Celeb-DF-v2"
DEFAULT_CHECKPOINT = (
    ROOT / "backend" / "artifacts" / "models" / "torch" / "best_checkpoint_wsl.pt"
)
DEFAULT_OUTPUT_ROOT = ROOT / "evaluation_results"
SEQUENCE_LENGTH = 20
IMAGE_SIZE = 224


def decode_sampled_frames(path: Path, stride: int) -> list[np.ndarray]:
    if stride < 1:
        raise ValueError("stride must be positive")
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        if not capture.isOpened():
            raise RuntimeError("video could not be opened")
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(
                    cv2.resize(
                        rgb,
                        (IMAGE_SIZE, IMAGE_SIZE),
                        interpolation=cv2.INTER_LINEAR,
                    ).astype(np.float32)
                    / 255.0
                )
            frame_index += 1
    finally:
        capture.release()
    if not frames:
        raise RuntimeError("video has no decodable frames")
    return frames


def predict_one(
    model: TorchTemporalInference, path: Path, stride: int
) -> float:
    decoded = decode_sampled_frames(path, stride)
    selected = select_evenly_spaced_frames(decoded, SEQUENCE_LENGTH)
    return model.predict(selected)


def evaluate_mode(
    model: TorchTemporalInference,
    selected: pd.DataFrame,
    dataset_root: Path,
    stride: int,
    mode: str,
) -> tuple[list[dict], list[dict], float]:
    predictions: list[dict] = []
    skipped: list[dict] = []
    started = time.perf_counter()
    for position, (_, row) in enumerate(selected.iterrows(), start=1):
        relative_path = str(row["relative_path"])
        video_path = resolve_video_path(row, dataset_root)
        print(f"{mode} [{position}/{len(selected)}] {relative_path}", flush=True)
        video_started = time.perf_counter()
        try:
            probability = predict_one(model, video_path, stride)
            if not np.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError(f"invalid probability {probability!r}")
            predictions.append(
                {
                    "relative_path": relative_path,
                    "video_path": str(video_path),
                    "label": int(row["label"]),
                    "fake_probability": float(probability),
                    "elapsed_seconds": time.perf_counter() - video_started,
                }
            )
        except Exception as exc:
            skipped.append(
                {
                    "relative_path": relative_path,
                    "video_path": str(video_path),
                    "label": int(row["label"]),
                    "error": str(exc),
                    "elapsed_seconds": time.perf_counter() - video_started,
                }
            )
            print(f"  SKIP: {exc}", flush=True)
    return predictions, skipped, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--split", choices=("test", "val", "train"), default="test")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-per-class", type=int, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        raise ValueError("--threshold must be between 0 and 1")

    manifest = pd.read_csv(args.manifest)
    selected = manifest[manifest["split"].eq(args.split)].copy()
    if selected.empty:
        raise ValueError(f"No manifest rows found for split={args.split!r}")
    if args.sample_per_class is not None:
        groups = []
        for label in (0, 1):
            group = selected[selected["label"].eq(label)]
            if len(group) < args.sample_per_class:
                raise ValueError(f"Not enough class-{label} rows for requested sample")
            groups.append(group.sample(args.sample_per_class, random_state=args.seed + label))
        selected = pd.concat(groups).sort_values("relative_path")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_root / f"{timestamp}_sampling_comparison"
    output_dir.mkdir(parents=True, exist_ok=False)
    selected.to_csv(output_dir / "comparison_manifest.csv", index=False)

    model = TorchTemporalInference(args.checkpoint, device=args.device)
    mode_results = {}
    for mode, stride in (("legacy_stride", 5), ("full_video", 1)):
        predictions, skipped, elapsed = evaluate_mode(
            model, selected, args.dataset_root, stride, mode
        )
        pd.DataFrame(predictions).to_csv(output_dir / f"{mode}_predictions.csv", index=False)
        pd.DataFrame(skipped).to_csv(output_dir / f"{mode}_skipped.csv", index=False)
        if not predictions:
            raise RuntimeError(f"No videos evaluated successfully in {mode} mode")
        labels = np.asarray([row["label"] for row in predictions], dtype=np.int64)
        probabilities = np.asarray(
            [row["fake_probability"] for row in predictions], dtype=np.float64
        )
        mode_results[mode] = {
            "stride": stride,
            "metrics": classification_metrics(labels, probabilities, args.threshold),
            "elapsed_seconds": elapsed,
            "evaluated_videos": len(predictions),
            "skipped_videos": len(skipped),
        }

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "split": args.split,
        "threshold": args.threshold,
        "threshold_is_tuned": False,
        "requested_videos": len(selected),
        "same_input_manifest_for_both_modes": True,
        "sampling_modes": {
            "legacy_stride": "Decode every fifth frame, then select up to 20 evenly spaced decoded frames.",
            "full_video": "Decode every frame, then select up to 20 evenly spaced decoded frames.",
        },
        "results": mode_results,
        "interpretation": (
            "Differences alone do not establish improvement. Compare error trade-offs, "
            "metrics, elapsed time, and skipped-video counts. ROC-AUC is threshold-independent."
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(mode_results, indent=2))
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
