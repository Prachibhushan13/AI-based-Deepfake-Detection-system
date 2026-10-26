from pathlib import Path
import argparse

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision import models
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "datasets" / "celebdf_manifest.csv"
CACHE_DIR = PROJECT_ROOT / "datasets" / "celebdf_features"

SEQUENCE_LENGTH = 20
IMAGE_SIZE = 224
FEATURE_SIZE = 1280
BATCH_SIZE = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_video_frames(video_path: str) -> np.ndarray:
    """Read 20 evenly spaced frames and return RGB float32 frames in [0, 1]."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_count <= 0:
            raise RuntimeError(f"Video has no readable frames: {video_path}")

        indices = np.linspace(
            0, frame_count - 1, SEQUENCE_LENGTH, dtype=np.int64
        )

        frames = []

        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()

            if not ok:
                raise RuntimeError(
                    f"Could not read frame {index} from {video_path}"
                )

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(
                frame, (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=cv2.INTER_AREA
            )
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)

        return np.stack(frames)

    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=4,
        help="Number of videos to process; defaults to four for testing."
    )
    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    if args.limit < 1:
        raise ValueError("--limit must be at least 1")

    df = pd.read_csv(MANIFEST_PATH)

    required = {"video_path", "relative_path", "label", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("WARNING: CUDA unavailable; using CPU.")

    weights = models.EfficientNet_B0_Weights.DEFAULT
    backbone = models.efficientnet_b0(weights=weights)
    backbone.classifier = nn.Identity()
    backbone = backbone.to(DEVICE).eval()

    mean = torch.tensor(
        [0.485, 0.456, 0.406], device=DEVICE
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225], device=DEVICE
    ).view(1, 3, 1, 1)

    # Select two real and two fake videos when testing the default limit.
    if args.limit == 4:
        selected = pd.concat([
            df[df["label"] == 0].head(2),
            df[df["label"] == 1].head(2),
        ])
    else:
        selected = df.head(args.limit)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    failures = []
    successful = 0

    for row in selected.itertuples(index=False):
        video_path = Path(row.video_path)
        cache_name = video_path.stem + "_" + str(row.label) + ".npy"
        feature_path = CACHE_DIR / cache_name

        try:
            frames = sample_video_frames(str(video_path))

            # Convert (T, H, W, C) to (T, C, H, W).
            frames_tensor = torch.from_numpy(frames).permute(
                0, 3, 1, 2
            ).to(DEVICE)

            normalized = (frames_tensor - mean) / std
            frame_features = []

            with torch.inference_mode():
                for start in range(0, SEQUENCE_LENGTH, BATCH_SIZE):
                    batch = normalized[start:start + BATCH_SIZE]
                    features = backbone(batch)
                    frame_features.append(features.cpu())

            features = torch.cat(frame_features, dim=0)
            features = features.numpy().astype(np.float32)

            if features.shape != (SEQUENCE_LENGTH, FEATURE_SIZE):
                raise RuntimeError(
                    f"Unexpected feature shape: {features.shape}"
                )

            # Save atomically so an interrupted run won't leave a partial file.
            temporary_path = feature_path.with_suffix(".tmp.npy")
            np.save(temporary_path, features)
            temporary_path.replace(feature_path)

            successful += 1
            print(
                f"[{successful}/{len(selected)}] "
                f"{row.relative_path} | label={row.label} | "
                f"shape={features.shape} | "
                f"{features.nbytes / 1024:.1f} KiB"
            )

        except Exception as exc:
            failures.append((str(video_path), str(exc)))
            print(f"FAILED: {video_path}\n  {exc}")

    print("\n--- Cache test summary ---")
    print("Requested:", len(selected))
    print("Successful:", successful)
    print("Failed:", len(failures))
    print("Cache directory:", CACHE_DIR)

    if failures:
        for path, error in failures:
            print(f"\nFailure: {path}\nReason: {error}")

    if successful == 0:
        raise SystemExit("No videos were cached successfully.")

    if failures:
        raise SystemExit("Some videos failed; inspect the errors above.")


if __name__ == "__main__":
    main()
