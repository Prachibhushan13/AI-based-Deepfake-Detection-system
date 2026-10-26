"""
Corrected threshold analysis with explicit confusion matrix conventions and assertions.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve
)

val_pred_path = r'D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\celebdf_exp_balanced\val_predictions.npz'
data = np.load(val_pred_path)
logits = data['logits']
labels = data['labels'].astype(int)

# Verify data integrity
assert len(labels) == 902, f"Expected 902 samples, got {len(labels)}"
assert np.array_equal(np.unique(labels), [0, 1]), f"Labels must be [0,1], got {np.unique(labels)}"
assert (labels == 0).sum() == 107, f"Expected 107 REAL (label 0), got {(labels==0).sum()}"
assert (labels == 1).sum() == 795, f"Expected 795 FAKE (label 1), got {(labels==1).sum()}"
assert not np.isnan(logits).any(), "Logits contain NaN"
assert not np.isinf(logits).any(), "Logits contain Inf"

probs = 1.0 / (1.0 + np.exp(-logits))

print("=== DATA VERIFICATION ===")
print(f"Total samples: {len(labels)}")
print(f"REAL (label 0): {(labels==0).sum()}")
print(f"FAKE (label 1): {(labels==1).sum()}")
print(f"Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
print(f"Probs range: [{probs.min():.4f}, {probs.max():.4f}]")
print()

# ROC-AUC verification
roc_auc_probs = roc_auc_score(labels, probs)
roc_auc_logits = roc_auc_score(labels, logits)
print(f"ROC-AUC (probs): {roc_auc_probs:.4f}")
print(f"ROC-AUC (logits): {roc_auc_logits:.4f}")
print()

# Define threshold evaluation function with explicit CM convention
def evaluate_threshold(threshold, probs, labels):
    """Evaluate at given probability threshold.
    
    Confusion matrix convention (sklearn default with labels=[0,1]):
    - Rows = actual labels (0=REAL, 1=FAKE)
    - Columns = predicted labels (0=REAL, 1=FAKE)
    - CM[0,0] = TN (REAL correctly predicted as REAL)
    - CM[0,1] = FP (REAL incorrectly predicted as FAKE)
    - CM[1,0] = FN (FAKE incorrectly predicted as REAL)
    - CM[1,1] = TP (FAKE correctly predicted as FAKE)
    """
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    
    # Row totals must equal actual class counts
    assert cm[0, :].sum() == 107, f"REAL row total {cm[0,:].sum()} != 107"
    assert cm[1, :].sum() == 795, f"FAKE row total {cm[1,:].sum()} != 795"
    
    # Extract counts
    tn = cm[0, 0]  # REAL -> REAL
    fp = cm[0, 1]  # REAL -> FAKE
    fn = cm[1, 0]  # FAKE -> REAL
    tp = cm[1, 1]  # FAKE -> FAKE
    
    # Recall = TP / (TP + FN) for each class
    # REAL recall = TN / (TN + FP) = CM[0,0] / sum(CM[0,:])
    real_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    # FAKE recall = TP / (TP + FN) = CM[1,1] / sum(CM[1,:])
    fake_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Precision = TP / (TP + FP) for each class
    # REAL precision = TN / (TN + FN)
    real_precision = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    # FAKE precision = TP / (TP + FP)
    fake_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Verify recalls are in [0, 1]
    assert 0.0 <= real_recall <= 1.0, f"REAL recall out of bounds: {real_recall}"
    assert 0.0 <= fake_recall <= 1.0, f"FAKE recall out of bounds: {fake_recall}"
    assert 0.0 <= real_precision <= 1.0, f"REAL precision out of bounds: {real_precision}"
    assert 0.0 <= fake_precision <= 1.0, f"FAKE precision out of bounds: {fake_precision}"
    
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    bal_acc = balanced_accuracy_score(labels, preds)
    acc = accuracy_score(labels, preds)
    
    return {
        'threshold': float(threshold),
        'confusion_matrix': cm.tolist(),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'real_recall': float(real_recall),
        'fake_recall': float(fake_recall),
        'real_precision': float(real_precision),
        'fake_precision': float(fake_precision),
        'macro_f1': float(macro_f1),
        'balanced_accuracy': float(bal_acc),
        'accuracy': float(acc)
    }

# Test with the reported confusion matrix [[28,79],[17,778]]
print("=== UNIT TEST: Reported CM [[28,79],[17,778]] ===")
cm_test = np.array([[28, 79], [17, 778]])
assert cm_test[0, :].sum() == 107, "REAL row total mismatch"
assert cm_test[1, :].sum() == 795, "FAKE row total mismatch"

tn_test, fp_test, fn_test, tp_test = cm_test[0,0], cm_test[0,1], cm_test[1,0], cm_test[1,1]
real_recall_test = tn_test / (tn_test + fp_test)
fake_recall_test = tp_test / (tp_test + fn_test)
real_prec_test = tn_test / (tn_test + fn_test)
fake_prec_test = tp_test / (tp_test + fp_test)

print(f"CM: {cm_test.tolist()}")
print(f"  REAL recall (row 0): {real_recall_test:.4f} = {tn_test}/{tn_test+fp_test}")
print(f"  FAKE recall (row 1): {fake_recall_test:.4f} = {tp_test}/{tp_test+fn_test}")
print(f"  REAL precision (col 0): {real_prec_test:.4f} = {tn_test}/{tn_test+fn_test}")
print(f"  FAKE precision (col 1): {fake_prec_test:.4f} = {tp_test}/{tp_test+fp_test}")
print()

# The earlier report incorrectly claimed:
# "REAL recall = 62.22%" — this is actually REAL PRECISION (28/45)
# "FAKE recall = 90.78%" — this is actually FAKE PRECISION (778/857)
print("=== CORRECTION ===")
print(f"Earlier report claimed: REAL recall=62.22%, FAKE recall=90.78%")
print(f"CORRECT: REAL recall={real_recall_test:.4f} (26.17%), FAKE recall={fake_recall_test:.4f} (97.86%)")
print(f"         REAL precision={real_prec_test:.4f} (62.22%), FAKE precision={fake_prec_test:.4f} (90.78%)")
print()

# Full threshold sweep
print("=== FULL THRESHOLD SWEEP (corrected) ===")
thresholds = np.arange(0.05, 0.96, 0.05)
results = []

for t in thresholds:
    result = evaluate_threshold(t, probs, labels)
    results.append(result)
    print(f"t={t:.2f}: CM={result['confusion_matrix']}, REAL_recall={result['real_recall']:.4f}, FAKE_recall={result['fake_recall']:.4f}, REAL_prec={result['real_precision']:.4f}, FAKE_prec={result['fake_precision']:.4f}, macro_f1={result['macro_f1']:.4f}, bal_acc={result['balanced_accuracy']:.4f}")

# Find best thresholds
best_bal = max(results, key=lambda x: x['balanced_accuracy'])
best_macro = max(results, key=lambda x: x['macro_f1'])

print()
print("=== BEST THRESHOLDS ===")
print(f"Best balanced accuracy: {best_bal['balanced_accuracy']:.4f} at threshold={best_bal['threshold']:.2f}")
print(f"  REAL recall={best_bal['real_recall']:.4f}, FAKE recall={best_bal['fake_recall']:.4f}")
print(f"  REAL precision={best_bal['real_precision']:.4f}, FAKE precision={best_bal['fake_precision']:.4f}")
print(f"  macro_f1={best_bal['macro_f1']:.4f}")
print()
print(f"Best macro F1: {best_macro['macro_f1']:.4f} at threshold={best_macro['threshold']:.2f}")
print(f"  REAL recall={best_macro['real_recall']:.4f}, FAKE recall={best_macro['fake_recall']:.4f}")
print(f"  REAL precision={best_macro['real_precision']:.4f}, FAKE precision={best_macro['fake_precision']:.4f}")
print(f"  balanced_accuracy={best_macro['balanced_accuracy']:.4f}")

# Thresholds with REAL recall >= 60%
print()
print("=== THRESHOLDS WITH REAL RECALL >= 60% ===")
for r in results:
    if r['real_recall'] >= 0.60:
        print(f"t={r['threshold']:.2f}: REAL_recall={r['real_recall']:.4f}, FAKE_recall={r['fake_recall']:.4f}, REAL_prec={r['real_precision']:.4f}, FAKE_prec={r['fake_precision']:.4f}, macro_f1={r['macro_f1']:.4f}, bal_acc={r['balanced_accuracy']:.4f}")

# Save corrected results
import json
output = {
    'validation_set_info': {
        'n_total': int(len(labels)),
        'n_real': int((labels == 0).sum()),
        'n_fake': int((labels == 1).sum()),
        'label_encoding': {'0': 'REAL', '1': 'FAKE'}
    },
    'confusion_matrix_convention': {
        'rows': 'actual labels (0=REAL, 1=FAKE)',
        'columns': 'predicted labels (0=REAL, 1=FAKE)',
        'cm[0,0]': 'TN (REAL->REAL)',
        'cm[0,1]': 'FP (REAL->FAKE)',
        'cm[1,0]': 'FN (FAKE->REAL)',
        'cm[1,1]': 'TP (FAKE->FAKE)'
    },
    'roc_auc': {
        'using_probabilities': float(roc_auc_probs),
        'using_logits': float(roc_auc_logits)
    },
    'threshold_sweep': results,
    'best_balanced_accuracy': best_bal,
    'best_macro_f1': best_macro,
    'correction_note': 'Earlier report swapped recall and precision. REAL recall = CM[0,0]/sum(CM[0,:]), FAKE recall = CM[1,1]/sum(CM[1,:]). At threshold 0.40: REAL recall=26.17%, FAKE recall=97.86%.'
}

with open(r'D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\threshold_analysis_20261026\corrected_threshold_analysis.json', 'w') as f:
    json.dump(output, f, indent=2)

print()
print("Saved corrected analysis to corrected_threshold_analysis.json")