import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.preprocessing import StandardScaler

# Load feature index
index_path = r'D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\celebdf_retrain_20261023\features_index.csv'
index = pd.read_csv(index_path)

cache_dir = Path(r'D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\celebdf_retrain_20261023\celebdf_features')

# Split data
train_idx = index[index['split'] == 'train']
val_idx = index[index['split'] == 'val']
test_idx = index[index['split'] == 'test']

print('Train:', len(train_idx), 'Val:', len(val_idx), 'Test:', len(test_idx))
print('Train labels:', train_idx['label'].value_counts().to_dict())
print('Val labels:', val_idx['label'].value_counts().to_dict())
print('Test labels:', test_idx['label'].value_counts().to_dict())

# Load features
def load_features(df):
    features = []
    labels = []
    for _, row in df.iterrows():
        fpath = Path(row['feature_path'])
        if not fpath.exists():
            fpath = cache_dir / Path(row['feature_path']).name
        arr = np.load(fpath)  # shape (20, 1280)
        features.append(arr)
        labels.append(row['label'])
    return np.array(features), np.array(labels)

print('Loading training features...')
X_train, y_train = load_features(train_idx)
print('Loading validation features...')
X_val, y_val = load_features(val_idx)
print('Loading test features...')
X_test, y_test = load_features(test_idx)

print('X_train shape:', X_train.shape)  # (N, 20, 1280)
print('X_val shape:', X_val.shape)
print('X_test shape:', X_test.shape)

# Aggregate temporal features: mean pooling over time
X_train_agg = X_train.mean(axis=1)  # (N, 1280)
X_val_agg = X_val.mean(axis=1)
X_test_agg = X_test.mean(axis=1)

print('Aggregated shapes:', X_train_agg.shape, X_val_agg.shape, X_test_agg.shape)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_agg)
X_val_scaled = scaler.transform(X_val_agg)
X_test_scaled = scaler.transform(X_test_agg)

# Train logistic regression with class weighting
class_weight = {0: 1.0, 1: 1.0}  # Start with balanced
# Compute inverse frequency weights
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
total = len(y_train)
class_weight = {0: total / (2 * neg), 1: total / (2 * pos)}
print('Class weights:', class_weight)

lr = LogisticRegression(
    class_weight=class_weight,
    max_iter=1000,
    random_state=42,
    C=1.0,
    solver='lbfgs'
)
lr.fit(X_train_scaled, y_train)

# Evaluate on validation
val_probs = lr.predict_proba(X_val_scaled)[:, 1]
val_preds = (val_probs >= 0.5).astype(int)

cm = confusion_matrix(y_val, val_preds, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

print()
print('=== VALIDATION RESULTS (Logistic Regression, mean pooling) ===')
print('Confusion matrix:', cm.tolist())
print('REAL recall:', tn/(tn+fp))
print('FAKE recall:', tp/(tp+fn))
print('REAL precision:', tn/(tn+fn))
print('FAKE precision:', tp/(tp+fp))
print('Macro F1:', f1_score(y_val, val_preds, average='macro', zero_division=0))
print('Balanced accuracy:', balanced_accuracy_score(y_val, val_preds))
print('Accuracy:', accuracy_score(y_val, val_preds))
print('ROC-AUC:', roc_auc_score(y_val, val_probs))

# Threshold sweep
print()
print('=== THRESHOLD SWEEP ===')
thresholds = np.arange(0.05, 0.96, 0.05)
for t in thresholds:
    preds_t = (val_probs >= t).astype(int)
    cm_t = confusion_matrix(y_val, preds_t, labels=[0, 1])
    tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
    real_recall = tn_t/(tn_t+fp_t) if (tn_t+fp_t) > 0 else 0
    fake_recall = tp_t/(tp_t+fn_t) if (tp_t+fn_t) > 0 else 0
    real_prec = tn_t/(tn_t+fn_t) if (tn_t+fn_t) > 0 else 0
    fake_prec = tp_t/(tp_t+fp_t) if (tp_t+fp_t) > 0 else 0
    macro_f1 = f1_score(y_val, preds_t, average='macro', zero_division=0)
    bal_acc = balanced_accuracy_score(y_val, preds_t)
    if real_recall >= 0.5:
        print('t={:.2f}: REAL_recall={:.4f}, FAKE_recall={:.4f}, REAL_prec={:.4f}, FAKE_prec={:.4f}, macro_f1={:.4f}, bal_acc={:.4f}'.format(
            t, real_recall, fake_recall, real_prec, fake_prec, macro_f1, bal_acc))

# Test evaluation
test_probs = lr.predict_proba(X_test_scaled)[:, 1]
test_preds = (test_probs >= 0.5).astype(int)
cm_test = confusion_matrix(y_test, test_preds, labels=[0, 1])
tn_t, fp_t, fn_t, tp_t = cm_test.ravel()

print()
print('=== TEST RESULTS (Logistic Regression, mean pooling) ===')
print('Confusion matrix:', cm_test.tolist())
print('REAL recall:', tn_t/(tn_t+fp_t))
print('FAKE recall:', tp_t/(tp_t+fn_t))
print('REAL precision:', tn_t/(tn_t+fn_t))
print('FAKE precision:', tp_t/(tp_t+fp_t))
print('Macro F1:', f1_score(y_test, test_preds, average='macro', zero_division=0))
print('Balanced accuracy:', balanced_accuracy_score(y_test, test_preds))
print('Accuracy:', accuracy_score(y_test, test_preds))
print('ROC-AUC:', roc_auc_score(y_test, test_probs))

# Save results
output_dir = Path(r'D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\feature_diagnostics')
output_dir.mkdir(parents=True, exist_ok=True)

results = {
    'method': 'LogisticRegression_mean_pooling',
    'class_weights': class_weight,
    'val_metrics': {
        'confusion_matrix': cm.tolist(),
        'real_recall': float(tn/(tn+fp)),
        'fake_recall': float(tp/(tp+fn)),
        'real_precision': float(tn/(tn+fn)),
        'fake_precision': float(tp/(tp+fp)),
        'macro_f1': float(f1_score(y_val, val_preds, average='macro', zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(y_val, val_preds)),
        'accuracy': float(accuracy_score(y_val, val_preds)),
        'roc_auc': float(roc_auc_score(y_val, val_probs)),
    },
    'test_metrics': {
        'confusion_matrix': cm_test.tolist(),
        'real_recall': float(tn_t/(tn_t+fp_t)),
        'fake_recall': float(tp_t/(tp_t+fn_t)),
        'real_precision': float(tn_t/(tn_t+fn_t)),
        'fake_precision': float(tp_t/(tp_t+fp_t)),
        'macro_f1': float(f1_score(y_test, test_preds, average='macro', zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(y_test, test_preds)),
        'accuracy': float(accuracy_score(y_test, test_preds)),
        'roc_auc': float(roc_auc_score(y_test, test_probs)),
    },
    'val_threshold_sweep': []
}

with open(output_dir / 'logreg_baseline.json', 'w') as f:
    json.dump(results, f, indent=2)

print('Results saved to', output_dir / 'logreg_baseline.json')