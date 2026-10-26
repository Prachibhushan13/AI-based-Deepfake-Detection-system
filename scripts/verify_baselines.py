import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

# Load neural model predictions (BiLSTM)
val_neural = np.load(r'D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\identity_audit_v2\neural_val_predictions.npz')
test_neural = np.load(r'D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\identity_audit_v2\neural_test_predictions.npz')

val_probs = val_neural['probs']
val_labels = val_neural['labels']
test_probs = test_neural['probs']
test_labels = test_neural['labels']

print('=== NEURAL MODEL (BiLSTM) RECOMPUTED METRICS ===')
print()

# Threshold 0.5
val_preds_05 = (val_probs >= 0.5).astype(int)
test_preds_05 = (test_probs >= 0.5).astype(int)

print('=== THRESHOLD 0.5 ===')
for name, probs, labels, preds in [('VAL', val_probs, val_labels, val_preds_05), ('TEST', test_probs, test_labels, test_preds_05)]:
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print('{}: CM={}, REAL_recall={:.4f}, FAKE_recall={:.4f}, REAL_prec={:.4f}, FAKE_prec={:.4f}, bal_acc={:.4f}, macro_f1={:.4f}, roc_auc={:.4f}'.format(
        name, cm.tolist(), tn/(tn+fp), tp/(tp+fn), tn/(tn+fn), tp/(tp+fp), balanced_accuracy_score(labels, preds), f1_score(labels, preds, average='macro', zero_division=0), roc_auc_score(labels, probs)))

print()

# Threshold 0.85 (validation-selected)
val_preds_85 = (val_probs >= 0.85).astype(int)
test_preds_85 = (test_probs >= 0.85).astype(int)

print('=== THRESHOLD 0.85 (validation-selected) ===')
for name, probs, labels, preds in [('VAL', val_probs, val_labels, val_preds_85), ('TEST', test_probs, test_labels, test_preds_85)]:
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print('{}: CM={}, REAL_recall={:.4f}, FAKE_recall={:.4f}, REAL_prec={:.4f}, FAKE_prec={:.4f}, bal_acc={:.4f}, macro_f1={:.4f}, roc_auc={:.4f}'.format(
        name, cm.tolist(), tn/(tn+fp), tp/(tp+fn), tn/(tn+fn), tp/(tp+fp), balanced_accuracy_score(labels, preds), f1_score(labels, preds, average='macro', zero_division=0), roc_auc_score(labels, probs)))

print()

# Verify threshold 0.85 was selected on validation only
thresholds = np.arange(0.05, 0.96, 0.05)
best_bal_acc = -1
best_t = 0.5
for t in thresholds:
    preds_t = (val_probs >= t).astype(int)
    bal_acc = balanced_accuracy_score(val_labels, preds_t)
    if bal_acc > best_bal_acc:
        best_bal_acc = bal_acc
        best_t = t
print('Best threshold on validation (balanced accuracy): {} (bal_acc={:.4f})'.format(best_t, best_bal_acc))

# Logistic regression
print()
print('=== LOGISTIC REGRESSION (from saved file) ===')
import json
with open(r'D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\identity_audit_v2\logreg_identity_disjoint.json') as f:
    lr_results = json.load(f)

    print('Validation (threshold=0.5):')
    for k, v in lr_results['val_metrics'].items():
        print('  {}: {}'.format(k, v))
    print()
    print('Test (threshold=0.5):')
    for k, v in lr_results['test_metrics'].items():
        print('  {}: {}'.format(k, v))