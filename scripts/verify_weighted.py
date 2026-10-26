import numpy as np
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

# Recompute validation metrics from weighted experiment
val_pred_path = r'D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\celebdf_exp_weighted\val_predictions.npz'
data = np.load(val_pred_path)
logits = data['logits']
labels = data['labels'].astype(int)
probs = 1.0 / (1.0 + np.exp(-logits))

print('=== RECOMPUTED VALIDATION METRICS (threshold=0.5) ===')
preds = (probs >= 0.5).astype(int)
cm = confusion_matrix(labels, preds, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
print('Confusion matrix: {}'.format(cm.tolist()))
print('  TN={}, FP={}, FN={}, TP={}'.format(tn, fp, fn, tp))
print('  REAL recall (label 0): {:.4f} = {}/{}'.format(tn/(tn+fp), tn, tn+fp))
print('  FAKE recall (label 1): {:.4f} = {}/{}'.format(tp/(tp+fn), tp, tp+fn))
print('  REAL precision: {:.4f} = {}/{}'.format(tn/(tn+fn), tn, tn+fn))
print('  FAKE precision: {:.4f} = {}/{}'.format(tp/(tp+fp), tp, tp+fp))
print('  Macro F1: {:.4f}'.format(f1_score(labels, preds, average='macro', zero_division=0)))
print('  Balanced accuracy: {:.4f}'.format(balanced_accuracy_score(labels, preds)))
print('  Accuracy: {:.4f}'.format(accuracy_score(labels, preds)))
print('  ROC-AUC: {:.4f}'.format(roc_auc_score(labels, probs)))
print()

# Check score orientation
print('=== SCORE ORIENTATION ===')
print('REAL (label 0) probs: mean={:.4f}, median={:.4f}'.format(probs[labels==0].mean(), np.median(probs[labels==0])))
print('FAKE (label 1) probs: mean={:.4f}, median={:.4f}'.format(probs[labels==1].mean(), np.median(probs[labels==1])))
print('% REAL with prob > 0.5: {:.1f}%'.format((probs[labels==0] > 0.5).mean()*100))
print('% FAKE with prob > 0.5: {:.1f}%'.format((probs[labels==1] > 0.5).mean()*100))
print()

# Check test confusion matrix from test_metrics.json
print('=== TEST CONFUSION MATRIX VERIFICATION ===')
print('Test CM: [[47, 131], [7, 333]]')
print('  REAL recall: 47/(47+131) = 47/178 = 0.2640')
print('  FAKE recall: 333/(333+7) = 333/340 = 0.9794')
print('  REAL precision: 47/(47+7) = 47/54 = 0.8704')
print('  FAKE precision: 333/(333+131) = 333/464 = 0.7177')
print('  Matches test_metrics.json: YES')