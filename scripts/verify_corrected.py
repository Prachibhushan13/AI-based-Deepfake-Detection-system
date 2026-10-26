import numpy as np
import json

# Verify corrected metrics from val_predictions.npz
val_pred_path = r'D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\celebdf_exp_balanced\val_predictions.npz'
data = np.load(val_pred_path)
logits = data['logits']
labels = data['labels'].astype(int)
probs = 1.0 / (1.0 + np.exp(-logits))

print('=== VALIDATION PREDICTIONS VERIFICATION ===')
print('Total: {}, REAL: {}, FAKE: {}'.format(len(labels), (labels==0).sum(), (labels==1).sum()))
print('Logits: min={:.4f}, max={:.4f}, mean={:.4f}'.format(logits.min(), logits.max(), logits.mean()))
print('Probs: min={:.4f}, max={:.4f}, mean={:.4f}'.format(probs.min(), probs.max(), probs.mean()))
print()

# Verify corrected_threshold_analysis.json
with open(r'D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\threshold_analysis_20261026\corrected_threshold_analysis.json') as f:
    corr = json.load(f)

print('=== CORRECTED ANALYSIS VERIFICATION ===')
print('ROC-AUC (probs): {:.4f}'.format(corr["roc_auc"]["using_probabilities"]))
print('ROC-AUC (logits): {:.4f}'.format(corr["roc_auc"]["using_logits"]))
print()

# Check threshold 0.40 metrics
for r in corr['threshold_sweep']:
    if abs(r['threshold'] - 0.40) < 0.001:
        print('Threshold 0.40:')
        print('  CM: {}'.format(r["confusion_matrix"]))
        print('  REAL recall: {:.4f}'.format(r["real_recall"]))
        print('  FAKE recall: {:.4f}'.format(r["fake_recall"]))
        print('  REAL precision: {:.4f}'.format(r["real_precision"]))
        print('  FAKE precision: {:.4f}'.format(r["fake_precision"]))
        print('  macro_f1: {:.4f}'.format(r["macro_f1"]))
        print('  bal_acc: {:.4f}'.format(r["balanced_accuracy"]))
        break