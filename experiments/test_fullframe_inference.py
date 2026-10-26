from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from torch_model import CachedFeatureTemporalClassifier

INDEX = ROOT / "datasets/celebdf_features/features_index.csv"
CHECKPOINT = ROOT / "backend/artifacts/models/torch/best_checkpoint_wsl.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIZE = 224
T = 20


def extract_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count < 1:
            raise RuntimeError(f"Cannot read video: {video_path}")

        frames = []
        for idx in np.linspace(0, count - 1, T, dtype=np.int64):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Cannot read frame {idx}: {video_path}")

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (SIZE, SIZE))
            frames.append(frame.astype(np.float32) / 255.0)

        return np.stack(frames)
    finally:
        cap.release()


def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)

    df = pd.read_csv(INDEX)
    df = df[(df["status"] == "ok") & (df["split"] == "val")].copy()

    # Select a small balanced sample for a first pipeline check.
    sample = pd.concat([
        df[df["label"] == 0].head(10),
        df[df["label"] == 1].head(10),
    ])

    model = CachedFeatureTemporalClassifier().to(DEVICE)
    checkpoint = torch.load(
        CHECKPOINT, map_location=DEVICE, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

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

    results = []

    for row in sample.itertuples(index=False):
        video_path = ROOT / "datasets" / "Celeb-DF-v2" / str(row.relative_path)

        if not video_path.is_file():
            print(f"SKIP: video not found: {video_path}")
            continue

        frames = extract_frames(video_path)
        tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).to(DEVICE)
        tensor = (tensor - mean) / std

        with torch.inference_mode():
            chunks = [
                backbone(tensor[start:start + 4])
                for start in range(0, T, 4)
            ]
            features = torch.cat(chunks).unsqueeze(0)
            probability = torch.sigmoid(model(features)).item()

        results.append({
            "label": int(row.label),
            "prob_fake": probability,
            "predicted_label": int(probability >= 0.5),
            "correct_at_0.5": int((probability >= 0.5) == int(row.label)),
        })

        print(
            f"true={int(row.label)} "
            f"prob_fake={probability:.4f} "
            f"predicted={int(probability >= 0.5)}"
        )

    if not results:
        raise RuntimeError(
            "No videos were evaluated. Check video_path entries in the index."
        )

    out = pd.DataFrame(results)
    print("\nSummary")
    print("Evaluated:", len(out))
    print("Correct at threshold 0.5:", int(out["correct_at_0.5"].sum()))
    print("Accuracy on this small sample:", round(out["correct_at_0.5"].mean(), 4))
    print("Note: this is a small validation diagnostic, not a benchmark.")


if __name__ == "__main__":
    main()
