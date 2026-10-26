from pathlib import Path
import hashlib

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import models


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "datasets" / "celebdf_manifest.csv"
CACHE = ROOT / "datasets" / "celebdf_features"
INDEX = CACHE / "features_index.csv"

T = 20
SIZE = 224
FEATURES = 1280
BATCH = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cache_name(relative_path):
    key = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:24]
    return f"{key}.npy"


def extract_frames(path):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError("Cannot open video")

        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count < 1:
            raise RuntimeError("Video has no frames")

        indices = np.linspace(0, count - 1, T, dtype=np.int64)
        frames = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Cannot read frame {idx}")

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (SIZE, SIZE))
            frames.append(frame.astype(np.float32) / 255.0)

        return np.stack(frames)

    finally:
        cap.release()


def valid_cache(path):
    if not path.is_file():
        return False
    try:
        arr = np.load(path, mmap_mode="r")
        return arr.shape == (T, FEATURES) and np.isfinite(arr).all()
    except Exception:
        return False


def main():
    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)

    df = pd.read_csv(MANIFEST)
    required = {"video_path", "relative_path", "label", "split"}
    if not required.issubset(df.columns):
        raise ValueError(f"Required columns: {sorted(required)}")

    CACHE.mkdir(parents=True, exist_ok=True)

    print("Device:", DEVICE)
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

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

    records = []
    failures = []
    skipped = 0
    completed = 0

    for i, row in enumerate(df.itertuples(index=False), start=1):
        path = Path(row.video_path)
        rel = str(row.relative_path)
        output = CACHE / cache_name(rel)

        status = "ok"

        try:
            if valid_cache(output):
                skipped += 1
            else:
                frames = extract_frames(path)
                tensor = torch.from_numpy(frames).permute(
                    0, 3, 1, 2
                ).to(DEVICE)

                tensor = (tensor - mean) / std
                chunks = []

                with torch.inference_mode():
                    for start in range(0, T, BATCH):
                        chunks.append(
                            backbone(tensor[start:start + BATCH]).cpu()
                        )

                features = torch.cat(chunks).numpy().astype(np.float32)

                if features.shape != (T, FEATURES):
                    raise RuntimeError(
                        f"Unexpected feature shape {features.shape}"
                    )
                if not np.isfinite(features).all():
                    raise RuntimeError("Non-finite feature values")

                temp = output.with_suffix(".tmp.npy")
                np.save(temp, features)
                temp.replace(output)
                completed += 1

        except Exception as exc:
            status = "failed"
            failures.append((rel, str(exc)))
            print(f"\nFAILED [{i}/{len(df)}]: {rel}: {exc}")

        records.append({
            "relative_path": rel,
            "feature_path": str(output) if status == "ok" else "",
            "label": int(row.label),
            "split": row.split,
            "status": status,
        })

        if i % 25 == 0 or i == len(df):
            print(
                f"\rProcessed {i}/{len(df)} | newly cached: {completed} "
                f"| reused: {skipped} | failed: {len(failures)}",
                end="\n" if i % 25 == 0 or i == len(df) else "",
                flush=True,
            )

    pd.DataFrame(records).to_csv(INDEX, index=False)

    print("\n--- Full cache summary ---")
    print("Manifest rows:", len(df))
    print("Newly cached:", completed)
    print("Reused valid caches:", skipped)
    print("Failed:", len(failures))
    print("Index:", INDEX)

    if failures:
        print("\nFailed videos:")
        for rel, error in failures:
            print(f"- {rel}: {error}")
        raise SystemExit(
            "Some videos failed. Fix the reported issue and rerun; "
            "valid cached features will be reused."
        )


if __name__ == "__main__":
    main()
