from pathlib import Path
from datetime import datetime
import shutil
import sys

root = Path.cwd()
evaluation = root / "evaluate_model.py"
thresholds = root / "analyze_thresholds.py"

if not evaluation.exists():
    sys.exit("ERROR: evaluate_model.py not found. Run this from your project root.")

# Make timestamped backups before changing existing files.
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = evaluation.with_name(f"evaluate_model.py.backup_{stamp}")
shutil.copy2(evaluation, backup)
print(f"Backup created: {backup.name}")

source = evaluation.read_text()

# Ensure the CSV module is imported.
if "import csv" not in source:
    lines = source.splitlines(keepends=True)
    insert_at = 0
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            insert_at = i + 1
        elif insert_at and line.strip() == "":
            continue
        elif insert_at:
            break
    lines.insert(insert_at, "import csv\n")
    source = "".join(lines)
    print("Added csv import.")
else:
    print("CSV import already present.")

# Add CSV export after the evaluation has confirmed there are results.
if 'evaluation_predictions.csv' not in source:
    anchor = '    raise SystemExit("No videos were evaluated successfully.")'
    if anchor not in source:
        sys.exit(
            "ERROR: Could not find the expected evaluation anchor. "
            "No evaluation-script changes were saved. Backup is available."
        )

    export = '''

# Save actual labels and fake probabilities for threshold analysis.
with open("evaluation_predictions.csv", "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["actual_label", "predicted_fake_probability"])
    writer.writerows(zip(y_true, y_score))

print("Saved predictions to evaluation_predictions.csv")
'''
    source = source.replace(anchor, anchor + export, 1)
    print("Added CSV export.")
else:
    print("CSV export already present; not adding a duplicate.")

evaluation.write_text(source)

# Create the threshold analysis script.
threshold_source = r'''import csv
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
'''

if thresholds.exists():
    threshold_backup = thresholds.with_name(
        f"analyze_thresholds.py.backup_{stamp}"
    )
    shutil.copy2(thresholds, threshold_backup)
    print(f"Threshold-script backup: {threshold_backup.name}")

thresholds.write_text(threshold_source)
print("Created/updated analyze_thresholds.py")

# Verify syntax before finishing.
import py_compile

try:
    py_compile.compile(str(evaluation), doraise=True)
    py_compile.compile(str(thresholds), doraise=True)
except py_compile.PyCompileError as exc:
    sys.exit(f"Syntax verification failed: {exc}")

print("\nSUCCESS: both Python scripts pass syntax checks.")
print("No model checkpoints or dataset files were modified.")
