import numpy as np
import json
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_curve, roc_auc_score

val_pred_path = r'D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\celebdf_exp_balanced\val_predictions.npz'
data = np.load(val_pred_path)
logits = data['logits']
labels = data['labels'].astype(int)
probs = 1.0 / (1.0 + np.exp(-logits))

n_total = len(labels)
n_real = int((labels == 0).sum())
n_fake = int((labels == 1).sum())

# Score distributions
real_logits = logits[labels == 0]
fake_logits = logits[labels == 1]

# Threshold sweep on probabilities
thresholds = np.arange(0.05, 0.96, 0.05)
results = []

for t in thresholds:
    preds = (probs >= t).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    real_recall = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    fake_recall = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    real_prec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fake_prec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    
    macro_f1 = float(f1_score(labels, preds, average='macro', zero_division=0))
    bal_acc = float(balanced_accuracy_score(labels, preds))
    acc = float(accuracy_score(labels, preds))
    
    results.append({
        'threshold': float(t),
        'real_recall': real_recall,
        'fake_recall': fake_recall,
        'real_precision': real_prec,
        'fake_precision': fake_prec,
        'macro_f1': macro_f1,
        'balanced_accuracy': bal_acc,
        'accuracy': acc,
        'confusion_matrix': cm.tolist(),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    })

# Best thresholds
best_bal_acc = max(results, key=lambda x: x['balanced_accuracy'])
best_macro_f1 = max(results, key=lambda x: x['macro_f1'])

# Logit thresholds
logit_thresholds = np.arange(-4, 5, 0.5)
logit_results = []
for t in logit_thresholds:
    preds = (logits >= t).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    real_recall = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    fake_recall = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    real_prec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fake_prec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    macro_f1 = float(f1_score(labels, preds, average='macro', zero_division=0))
    bal_acc = float(balanced_accuracy_score(labels, preds))
    acc = float(accuracy_score(labels, preds))
    logit_results.append({
        'logit_threshold': float(t),
        'real_recall': real_recall,
        'fake_recall': fake_recall,
        'real_precision': real_prec,
        'fake_precision': fake_prec,
        'macro_f1': macro_f1,
        'balanced_accuracy': bal_acc,
        'accuracy': acc,
        'confusion_matrix': cm.tolist()
    })

# Save
output = {
    'validation_set_info': {
        'n_total': n_total,
        'n_real': n_real,
        'n_fake': n_fake,
        'real_proportion': n_real / n_total,
        'fake_proportion': n_fake / n_total
    },
    'score_distributions': {
        'real_logits': {
            'mean': float(real_logits.mean()),
            'std': float(real_logits.std()),
            'median': float(np.median(real_logits)),
            'min': float(real_logits.min()),
            'max': float(real_logits.max()),
            'percent_positive': float((real_logits > 0).mean())
        },
        'fake_logits': {
            'mean': float(fake_logits.mean()),
            'std': float(fake_logits.std()),
            'median': float(np.median(fake_logits)),
            'min': float(fake_logits.min()),
            'max': float(fake_logits.max()),
            'percent_negative': float((fake_logits < 0).mean())
        },
        'roc_auc': float(roc_auc_score(labels, probs)),
        'pr_auc': float(np.trapz(*precision_recall_curve(labels, probs)[:2][::-1]))
    },
    'probability_threshold_sweep': results,
    'logit_threshold_sweep': logit_results,
    'best_balanced_accuracy': best_bal_acc,
    'best_macro_f1': best_macro_f1,
    'current_threshold_05': next(r for r in results if abs(r['threshold'] - 0.5) < 0.001)
}

with open(r'D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\threshold_analysis_20261026\threshold_analysis.json', 'w') as f:
    json.dump(output, f, indent=2)

print('Results saved successfully')
print()
print('KEY FINDINGS:')
print(f'  ROC-AUC: {output["score_distributions"]["roc_auc"]:.4f} (near random ~0.5)')
print(f'  Best balanced accuracy: {best_bal_acc["balanced_accuracy"]:.4f} at threshold={best_bal_acc["threshold"]:.2f}')
print(f'    REAL recall: {best_bal_acc["real_recall"]:.4f}, FAKE recall: {best_bal_acc["fake_recall"]:.4f}')
print(f'  Best macro F1: {best_macro_f1["macro_f1"]:.4f} at threshold={best_macro_f1["threshold"]:.2f}')
print(f'    REAL recall: {best_macro_f1["real_recall"]:.4f}, FAKE recall: {best_macro_f1["fake_recall"]:.4f}')
print(f'  Current threshold 0.5: REAL recall={output["current_threshold_05"]["real_recall"]:.4f}, FAKE recall={output["current_threshold_05"]["fake_recall"]:.4f}')
print()
print('SCORE DISTRIBUTION ISSUE:')
print(f'  REAL logits mean={real_logits.mean():.4f}, median={np.median(real_logits):.4f}, %>0: {(real_logits>0).mean()*100:.1f}%')
print(f'  FAKE logits mean={fake_logits.mean():.4f}, median={np.median(fake_logits):.4f}, %<0: {(fake_logits<0).mean()*100:.1f}%')
print(f'  Overlap: REAL max={real_logits.max():.4f}, FAKE min={fake_logits.min():.4f}')