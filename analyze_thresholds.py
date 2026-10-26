import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

csv_path = Path("evaluation_predictions.csv")
if not csv_path.exists():
    raise SystemExit(
        "Missing evaluation_predictions.csv. "
        "Run: python evaluate_model.py"
    )

y_true, y_score = [], []
with csv_path.open(newline="") as f:
    reader = csv.DictReader(f)
    required = {"actual_label", "predicted_fake_probability"}
    if not reader.fieldnames or not required.issubset(reader.fieldnames):
        raise SystemExit("CSV headers are missing or incorrect.")
    for row in reader:
        y_true.append(int(row["actual_label"]))
        y_score.append(float(row["predicted_fake_probability"]))

if not y_true:
    raise SystemExit("The CSV contains no predictions.")

y_true = np.asarray(y_true, dtype=int)
y_score = np.asarray(y_score, dtype=float)

print(f"Loaded {len(y_true)} saved predictions.")
print("Labels: REAL=0, FAKE=1")
print()
print(
    f"{'Threshold':>9} {'Accuracy':>9} {'Precision':>10} "
    f"{'Recall':>8} {'Specificity':>11} {'F1':>8} {'FP':>5} {'FN':>5}"
)
print("-" * 76)

for threshold in np.arange(0.3, 0.71, 0.1):
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if tn + fp else 0.0
    print(
        f"{threshold:9.2f} "
        f"{accuracy_score(y_true, y_pred):9.3f} "
        f"{precision_score(y_true, y_pred, zero_division=0):10.3f} "
        f"{recall_score(y_true, y_pred, zero_division=0):8.3f} "
        f"{specificity:11.3f} "
        f"{f1_score(y_true, y_pred, zero_division=0):8.3f} "
        f"{fp:5d} {fn:5d}"
    )

if len(np.unique(y_true)) == 2:
    print(f"\nROC-AUC (threshold independent): {roc_auc_score(y_true, y_score):.3f}")

print(
    "\nReminder: these results describe the saved sample only. "
    "Choose a threshold using validation data, not the final test set."
)
