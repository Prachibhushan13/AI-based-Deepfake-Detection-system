import sys
import csv
from pathlib import Path
import random
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

sys.path.insert(0, "backend")

from app.ml.inference import load_model, predict_video
from app.services.video_service import extract_frames

ROOT = Path("datasets/Celeb-DF-v2")
SAMPLE_PER_CLASS = 20
SEED = 42

random.seed(SEED)

# Label: REAL = 0, FAKE = 1
real_dirs = [ROOT / "Celeb-real", ROOT / "YouTube-real"]
fake_dir = ROOT / "Celeb-synthesis"

real_videos = []
for folder in real_dirs:
    real_videos.extend(folder.glob("*.mp4"))

fake_videos = list(fake_dir.glob("*.mp4"))

if len(real_videos) < SAMPLE_PER_CLASS or len(fake_videos) < SAMPLE_PER_CLASS:
    raise SystemExit("Not enough videos for the requested sample.")

sample = (
    [(p, 0) for p in random.sample(real_videos, SAMPLE_PER_CLASS)]
    + [(p, 1) for p in random.sample(fake_videos, SAMPLE_PER_CLASS)]
)
random.shuffle(sample)

print("Loading model...")
model = load_model()
if model is None:
    raise SystemExit("Model loading failed.")
print("Model type:", getattr(model, "model_type", "unknown"))
print("Device:", getattr(model, "device", "unknown"))
print(f"\nEvaluating {len(sample)} videos...\n")

y_true, y_pred, y_score = [], [], []
start = time.time()

for i, (path, label) in enumerate(sample, 1):
    try:
        frames = extract_frames(str(path))
        if not frames:
            print(f"[{i}/{len(sample)}] SKIP: no frames — {path.name}")
            continue

        result = predict_video(frames, filename=str(path))
        score = float(result["fakeProbability"])
        prediction = 1 if score >= 0.5 else 0

        y_true.append(label)
        y_pred.append(prediction)
        y_score.append(score)

        actual = "FAKE" if label else "REAL"
        predicted = "FAKE" if prediction else "REAL"
        print(
            f"[{i}/{len(sample)}] {path.name}: "
            f"actual={actual}, predicted={predicted}, "
            f"fake_probability={score:.4f}"
        )
    except Exception as exc:
        print(f"[{i}/{len(sample)}] ERROR: {path.name}: {exc}")

if not y_true:
    raise SystemExit("No videos were evaluated successfully.")

# Save actual labels and fake probabilities for threshold analysis.
with open("evaluation_predictions.csv", "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["actual_label", "predicted_fake_probability"])
    writer.writerows(zip(y_true, y_score))

print("Saved predictions to evaluation_predictions.csv")


print("\n========== EVALUATION RESULTS ==========")
print("Videos evaluated:", len(y_true))
print(f"Elapsed time: {time.time() - start:.2f} seconds")
print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
print(f"F1-score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")
print("\nConfusion matrix (rows=actual, columns=predicted; REAL, FAKE):")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
print("\nClassification report:")
print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["REAL", "FAKE"],
    zero_division=0,
))

if len(set(y_true)) == 2:
    print(f"ROC-AUC:   {roc_auc_score(y_true, y_score):.4f}")
else:
    print("ROC-AUC unavailable: only one actual class was evaluated.")
