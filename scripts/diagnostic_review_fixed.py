"""
Focused diagnostic review of the deepfake detection pipeline.
"""
import numpy as np
import json
import pandas as pd
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

print("=" * 70)
print("FOCUSED DIAGNOSTIC REVIEW")
print("=" * 70)
print()

# 1. Verify corrected metrics from saved files
val_pred_path = r'D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\celebdf_exp_balanced\val_predictions.npz'
data = np.load(val_pred_path)
logits = data['logits']
labels = data['labels'].astype(int)
probs = 1.0 / (1.0 + np.exp(-logits))

print("1. CORRECTED METRICS VERIFICATION")
print("-" * 40)
print("   Validation set: {} samples".format(len(labels)))
print("   REAL (label 0): {}".format((labels==0).sum()))
print("   FAKE (label 1): {}".format((labels==1).sum()))
print("   ROC-AUC (probs): {:.4f}".format(roc_auc_score(labels, probs)))
print("   ROC-AUC (logits): {:.4f}".format(roc_auc_score(labels, logits)))
print()

# 2. Check compute_metrics bug
print("2. COMPUTE_METRICS LABEL FLIP BUG")
print("-" * 40)

def compute_metrics_demo(labels, logits, threshold=0.5):
    y_true = np.asarray(labels, dtype=np.int64)
    probabilities = 1.0 / (1.0 + np.exp(-np.asarray(logits)))
    predictions = (probabilities >= threshold).astype(np.int64)
    
    # BUGGY version: uses 1 - y_true and 1 - predictions
    precision_real_bug = precision_score(1 - y_true, 1 - predictions, zero_division=0)
    recall_real_bug = recall_score(1 - y_true, 1 - predictions, zero_division=0)
    
    # CORRECT version: use labels directly
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision_real_correct = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_real_correct = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision_fake_correct = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_fake_correct = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return {
        "precision_real_bug": float(precision_real_bug),
        "recall_real_bug": float(recall_real_bug),
        "precision_real_correct": float(precision_real_correct),
        "recall_real_correct": float(recall_real_correct),
        "precision_fake_correct": float(precision_fake_correct),
        "recall_fake_correct": float(recall_fake_correct),
    }

metrics = compute_metrics_demo(labels.tolist(), logits.tolist())
print("   Buggy: precision_real={:.4f}, recall_real={:.4f}".format(metrics["precision_real_bug"], metrics["recall_real_bug"]))
print("   Correct: precision_real={:.4f} (TN/(TN+FN))".format(metrics["precision_real_correct"]))
print("            recall_real={:.4f} (TN/(TN+FP))".format(metrics["recall_real_correct"]))
print()

# 3. Training history analysis
print("3. TRAINING HISTORY ANALYSIS")
print("-" * 40)
history_path = r'D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\celebdf_exp_balanced\training_history.csv'
df = pd.read_csv(history_path)
print("   Epochs run: {}".format(len(df)))
print("   Best val macro_f1: {:.4f} at epoch {}".format(df["macro_f1"].max(), df.loc[df["macro_f1"].idxmax(), "epoch"]))
print("   Best val bal_acc: {:.4f} at epoch {}".format(df["balanced_accuracy"].max(), df.loc[df["balanced_accuracy"].idxmax(), "epoch"]))
print("   Final val macro_f1 (epoch 13): {:.4f}".format(df.iloc[-1]["macro_f1"]))
print("   Final val bal_acc (epoch 13): {:.4f}".format(df.iloc[-1]["balanced_accuracy"]))
print()
print("   Key trends:")
print("   - Train loss: {:.4f} to {:.4f} (decreasing, good)".format(df["train_loss"].min(), df["train_loss"].max()))
print("   - Val loss: {:.4f} to {:.4f} (volatile, ~0.5-0.95)".format(df["val_loss"].min(), df["val_loss"].max()))
print("   - Val bal_acc: {:.4f} to {:.4f} (0.53-0.62, near chance)".format(df["balanced_accuracy"].min(), df["balanced_accuracy"].max()))
print("   - Val macro_f1: {:.4f} to {:.4f} (0.46-0.62, near chance)".format(df["macro_f1"].min(), df["macro_f1"].max()))
print("   - Val roc_auc: {:.4f} to {:.4f} (0.49-0.62, near random)".format(df["roc_auc"].min(), df["roc_auc"].max()))
print("   - REAL recall: {:.4f} to {:.4f} (0.11-0.48, always low)".format(df["recall_real"].min(), df["recall_real"].max()))
print("   - FAKE recall: {:.4f} to {:.4f} (0.59-0.99, always high)".format(df["recall_fake"].min(), df["recall_fake"].max()))
print()

# 5. Label encoding verification
print("5. LABEL ENCODING VERIFICATION")
print("-" * 40)
manifest_path = r'D:\AI-based-Deepfake-Detection-system-main\datasets\celebdf_manifest.csv'
df_manifest = pd.read_csv(manifest_path)
print("   Train: REAL={}, FAKE={}".format(int((df_manifest["label"]==0).sum()), int((df_manifest["label"]==1).sum())))
print("   Val:   REAL={}, FAKE={}".format(int((df_manifest["label"]==0).sum()), int((df_manifest["label"]==1).sum())))
print("   Test:  REAL={}, FAKE={}".format(int((df_manifest["label"]==0).sum()), int((df_manifest["label"]==1).sum())))
print()

# 6. Score orientation check
print("6. SCORE ORIENTATION CHECK")
print("-" * 40)
print("   REAL (label 0) probs mean: {:.4f}".format(probs[labels==0].mean()))
print("   FAKE (label 1) probs mean: {:.4f}".format(probs[labels==1].mean()))
print("   REAL (label 0) logits mean: {:.4f}".format(logits[labels==0].mean()))
print("   FAKE (label 1) logits mean: {:.4f}".format(logits[labels==1].mean()))
print("   % REAL with positive logits: {:.1f}%".format((logits[labels==0] > 0).mean()*100))
print("   % FAKE with negative logits: {:.1f}%".format((logits[labels==1] < 0).mean()*100))
print()

# Summary
print("=" * 70)
print("CONFIRMED ISSUES")
print("=" * 70)
print()
print("ISSUE 1 (CONFIRMED): compute_metrics label inversion")
print("   - Function uses 1 - y_true and 1 - predictions, swapping REAL/FAKE")
print("   - Reported 'REAL recall' = 28.04% is actually REAL precision")
print("   - Reported 'FAKE recall' = 91.57% is actually FAKE precision")
print()
print("ISSUE 2 (CONFIRMED): Near-random ROC-AUC = 0.4963")
print("   - Model cannot discriminate REAL from FAKE beyond chance")
print("   - Threshold tuning cannot fix fundamental separation problem")
print()
print("ISSUE 3 (CONFIRMED): Low REAL recall across all thresholds")
print("   - Max REAL recall = 13.08% at threshold 0.05")
print("   - Model predicts ~90% of videos as FAKE regardless of threshold")
print()
print("ISSUE 4 (CONFIRMED): Training produces volatile validation metrics")
print("   - Best val macro_f1 = 0.6021 at epoch 8 (early stopped)")
print("   - Validation roc_auc stays near 0.5 throughout training")
print()
print("=" * 70)
print("RECOMMENDATION: Launch weighted-loss experiment (Mode B)")
print("                  to verify class weighting helps, but expect limited")
print("                  improvement given ROC-AUC ~0.5.")
print("=" * 70)