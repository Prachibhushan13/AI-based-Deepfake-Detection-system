import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score
)

val_pred_path = r'D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\celebdf_exp_balanced\val_predictions.npz'
data = np.load(val_pred_path)
logits = data['logits']
labels = data['labels'].astype(int)

probs = 1.0 / (1.0 + np.exp(-logits))

print('=== Label encoding verification ===')
print('Label 0 count:', (labels == 0).sum(), '(should be REAL)')
print('Label 1 count:', (labels == 1).sum(), '(should be FAKE)')
print()

# Check training data label encoding from index
import pandas as pd
index_path = r'D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\celebdf_retrain_20261023\features_index.csv'
df = pd.read_csv(index_path)
print('Training split label distribution:')
print(df[df['split']=='train']['label'].value_counts().to_dict())
print('Validation split label distribution:')
print(df[df['split']=='val']['label'].value_counts().to_dict())
print()

# Check score orientation
print('=== Score orientation ===')
print('REAL (label 0) probs: mean={:.4f}, median={:.4f}, min={:.4f}, max={:.4f}'.format(
    probs[labels==0].mean(), np.median(probs[labels==0]), probs[labels==0].min(), probs[labels==0].max()))
print('FAKE (label 1) probs: mean={:.4f}, median={:.4f}, min={:.4f}, max={:.4f}'.format(
    probs[labels==1].mean(), np.median(probs[labels==1]), probs[labels==1].min(), probs[labels==1].max()))
print()

# Compute at threshold 0.50
print('=== THRESHOLD 0.50 ===')
preds_50 = (probs >= 0.50).astype(int)
cm_50 = confusion_matrix(labels, preds_50, labels=[0, 1])
tn, fp, fn, tp = cm_50.ravel()
print('Confusion matrix (rows=true, cols=pred):')
print(cm_50)
print('  TN={}, FP={}, FN={}, TP={}'.format(tn, fp, fn, tp))
print('  REAL recall (label 0): {:.4f} = {}/{}'.format(tn/(tn+fn), tn, tn+fn))
print('  FAKE recall (label 1): {:.4f} = {}/{}'.format(tp/(tp+fp), tp, tp+fp))
print('  REAL precision: {:.4f} = {}/{}'.format(tn/(tn+fp), tn, tn+fp))
print('  FAKE precision: {:.4f} = {}/{}'.format(tp/(tp+fn), tp, tp+fn))
print('  Macro F1: {:.4f}'.format(f1_score(labels, preds_50, average='macro', zero_division=0)))
print('  Balanced accuracy: {:.4f}'.format(balanced_accuracy_score(labels, preds_50)))
print('  Accuracy: {:.4f}'.format(accuracy_score(labels, preds_50)))
print('  ROC-AUC: {:.4f}'.format(roc_auc_score(labels, probs)))
print()

# Compute at threshold 0.40
print('=== THRESHOLD 0.40 ===')
preds_40 = (probs >= 0.40).astype(int)
cm_40 = confusion_matrix(labels, preds_40, labels=[0, 1])
tn, fp, fn, tp = cm_40.ravel()
print('Confusion matrix (rows=true, cols=pred):')
print(cm_40)
print('  TN={}, FP={}, FN={}, TP={}'.format(tn, fp, fn, tp))
print('  REAL recall (label 0): {:.4f} = {}/{}'.format(tn/(tn+fn), tn, tn+fn))
print('  FAKE recall (label 1): {:.4f} = {}/{}'.format(tp/(tp+fp), tp, tp+fp))
print('  REAL precision: {:.4f} = {}/{}'.format(tn/(tn+fp), tn, tn+fp))
print('  FAKE precision: {:.4f} = {}/{}'.format(tp/(tp+fn), tp, tp+fn))
print('  Macro F1: {:.4f}'.format(f1_score(labels, preds_40, average='macro', zero_division=0)))
print('  Balanced accuracy: {:.4f}'.format(balanced_accuracy_score(labels, preds_40)))
print('  Accuracy: {:.4f}'.format(accuracy_score(labels, preds_40)))
print()

# Earlier validation report from training output:
# Validation: loss=0.5830 acc=0.8404 bal_acc=0.5980 macro_f1=0.6021 prec_real=0.3093 rec_real=0.2804 prec_fake=0.9043 rec_fake=0.9157 macro_f1=0.6021 auc=0.49633809439840126 cm=[[30, 77], [67, 728]]
print('=== Earlier validation report (from training output) ===')
print('Reported: rec_real=0.2804, rec_fake=0.9157, prec_real=0.3093, prec_fake=0.9043')
print('Reported CM: [[30, 77], [67, 728]]')
print('Our computed at 0.50: rec_real={:.4f}, rec_fake={:.4f}, prec_real={:.4f}, prec_fake={:.4f}'.format(
    tn/(tn+fn), tp/(tp+fp), tn/(tn+fp), tp/(tp+fn)))
print('Our CM at 0.50:', cm_50.tolist())

# Also check logits at threshold 0 (equivalent to prob 0.5)
print()
print('=== Logit threshold 0.0 (equiv to prob 0.5) ===')
preds_logit0 = (logits >= 0.0).astype(int)
cm_logit0 = confusion_matrix(labels, preds_logit0, labels=[0, 1])
tn, fp, fn, tp = cm_logit0.ravel()
print('Confusion matrix:', cm_logit0.tolist())
print('  REAL recall: {:.4f}, FAKE recall: {:.4f}'.format(tn/(tn+fn), tp/(tp+fp)))

# ROC-AUC using logits vs probs
print()
print('=== ROC-AUC comparison ===')
print('ROC-AUC (probs): {:.4f}'.format(roc_auc_score(labels, probs)))
print('ROC-AUC (logits): {:.4f}'.format(roc_auc_score(labels, logits)))

# Find thresholds for REAL recall >= 60%
print()
print('=== Thresholds achieving REAL recall >= 60% ===')
thresholds = np.arange(0.05, 0.96, 0.05)
for t in thresholds:
    preds = (probs >= t).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    real_recall = tn/(tn+fn) if (tn+fn) > 0 else 0
    fake_recall = tp/(tp+fp) if (tp+fp) > 0 else 0
    real_prec = tn/(tn+fp) if (tn+fp) > 0 else 0
    fake_prec = tp/(tp+fn) if (tp+fn) > 0 else 0
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    bal_acc = balanced_accuracy_score(labels, preds)
    if real_recall >= 0.60:
        print('  t={:.2f}: REAL_recall={:.4f}, FAKE_recall={:.4f}, REAL_prec={:.4f}, FAKE_prec={:.4f}, macro_f1={:.4f}, bal_acc={:.4f}, CM={}'.format(
            t, real_recall, fake_recall, real_prec, fake_prec, macro_f1, bal_acc, cm.tolist()))

# Find best balanced accuracy and macro F1
print()
print('=== Best thresholds ===')
best_bal = -1
best_bal_t = 0
best_macro = -1
best_macro_t = 0
for t in thresholds:
    preds = (probs >= t).astype(int)
    bal_acc = balanced_accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    if bal_acc > best_bal:
        best_bal = bal_acc
        best_bal_t = t
    if macro_f1 > best_macro:
        best_macro = macro_f1
        best_macro_t = t
print('Best balanced accuracy: {:.4f} at threshold={:.2f}'.format(best_bal, best_bal_t))
print('Best macro F1: {:.4f} at threshold={:.2f}'.format(best_macro, best_macro_t))