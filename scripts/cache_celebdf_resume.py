from pathlib import Path
import hashlib
import argparse

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]
T, SIZE, FEATURE_SIZE = 20, 224, 1280


def cache_name(relative_path):
    key = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:24]
    return f"{key}.npy"


def extract_frames_sequentially(path):
    """Decode sequentially and select 20 evenly spaced frames."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError("Cannot open video")

        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count < 1:
            raise RuntimeError("Video has no reported frames")

        targets = np.linspace(0, count - 1, T, dtype=np.int64)
        targets_by_frame = {}
        for output_index, frame_index in enumerate(targets):
            targets_by_frame.setdefault(int(frame_index), []).append(output_index)

        frames = [None] * T
        frame_index = 0

        while frame_index <= int(targets[-1]):
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index in targets_by_frame:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (SIZE, SIZE))
                frame = frame.astype(np.float32) / 255.0

                for output_index in targets_by_frame[frame_index]:
                    frames[output_index] = frame

            frame_index += 1

        if any(frame is None for frame in frames):
            raise RuntimeError(
                f"Could not decode all selected frames "
                f"({sum(frame is not None for frame in frames)}/{T})"
            )

        return np.stack(frames)
    finally:
        cap.release()


def valid_cache(path):
    try:
        arr = np.load(path, mmap_mode="r")
        return arr.shape == (T, FEATURE_SIZE) and np.isfinite(arr).all()
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "datasets" / "celebdf_manifest.csv",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "datasets" / "celebdf_features",
    )
    parser.add_argument("--index", type=Path, default=None)
    parser.add_argument("--frame-batch-size", type=int, default=20)
    args = parser.parse_args()
    if args.frame_batch_size < 1 or args.frame_batch_size > T:
        raise ValueError(f"--frame-batch-size must be between 1 and {T}")

    manifest = args.manifest.resolve()
    cache_dir = args.cache_dir.resolve()
    index = (
        args.index.resolve()
        if args.index is not None
        else cache_dir / "features_index.csv"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(manifest)
    required = {"video_path", "relative_path", "label", "split"}
    if not required.issubset(df.columns):
        raise ValueError(f"Manifest must contain: {sorted(required)}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("WARNING: CUDA unavailable; feature extraction will use CPU.")

    backbone = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT
    )
    backbone.classifier = nn.Identity()
    backbone = backbone.to(device).eval()

    mean = torch.tensor(
        [0.485, 0.456, 0.406], device=device
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        [0.229, 0.224, 0.225], device=device
    ).view(1, 3, 1, 1)

    records, failures = [], []
    reused = newly_cached = 0

    def save_index():
        pd.DataFrame(records).to_csv(index, index=False)

    try:
        for i, row in enumerate(df.itertuples(index=False), start=1):
            rel = str(row.relative_path)
            output = cache_dir / cache_name(rel)
            status = "ok"
            error = ""

            try:
                if valid_cache(output):
                    reused += 1
                else:
                    frames = extract_frames_sequentially(row.video_path)
                    tensor = torch.from_numpy(frames).permute(
                        0, 3, 1, 2
                    ).to(device)
                    tensor = (tensor - mean) / std

                    chunks = []
                    with torch.inference_mode():
                        for start in range(0, T, args.frame_batch_size):
                            chunks.append(
                                backbone(
                                    tensor[start:start + args.frame_batch_size]
                                ).cpu()
                            )

                    features = torch.cat(chunks).numpy().astype(np.float32)
                    if features.shape != (T, FEATURE_SIZE):
                        raise RuntimeError(f"Unexpected shape {features.shape}")
                    if not np.isfinite(features).all():
                        raise RuntimeError("Non-finite feature values")

                    temp = output.with_suffix(".tmp.npy")
                    np.save(temp, features)
                    temp.replace(output)
                    newly_cached += 1

            except Exception as exc:
                status = "failed"
                error = str(exc)
                failures.append((rel, error))
                print(f"\nFAILED [{i}/{len(df)}] {rel}: {error}")

            records.append({
                "relative_path": rel,
                "feature_path": str(output) if status == "ok" else "",
                "label": int(row.label),
                "split": row.split,
                "status": status,
                "error": error,
            })

            if i % 25 == 0 or i == len(df):
                save_index()
                print(
                    f"Processed {i}/{len(df)} | "
                    f"new: {newly_cached} | reused: {reused} | "
                    f"failed: {len(failures)}",
                    flush=True,
                )

    except KeyboardInterrupt:
        save_index()
        print("\nInterrupted safely. Progress index saved.")
        print("Run this same script again to resume.")
        return

    save_index()
    print("\n--- Summary ---")
    print("Manifest rows:", len(df))
    print("Newly cached:", newly_cached)
    print("Reused valid caches:", reused)
    print("Failed:", len(failures))
    print("Index:", index)

    if failures:
        print("Failed video paths are recorded in the index CSV.")


if __name__ == "__main__":
    main()
