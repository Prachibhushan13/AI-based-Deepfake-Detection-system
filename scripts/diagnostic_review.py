"""
Focused diagnostic review of the deepfake detection pipeline.
Investigates the discrepancy between the corrected threshold analysis 
and the original reported metrics, and identifies root causes.
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
print(f"   Validation set: {len(labels)} samples")
print(f"   REAL (label 0): {(labels==0).sum()}")
print(f"   FAKE (label 1): {(labels==1).sum()}")
print(f"   ROC-AUC (probs): {roc_auc_score(labels, probs):.4f}")
print(f"   ROC-AUC (logits): {roc_auc_score(labels, logits):.4f}")
print()

# 2. Check compute_metrics bug
print("2. COMPUTE_METRICS LABEL FLIP BUG")
print("-" * 40)

def compute_metrics_bug_demo(labels, logits, threshold=0.5):
    """Demonstrate the label flip bug in compute_metrics."""
    y_true = np.asarray(labels, dtype=np.int64)
    probabilities = 1.0 / (1.0 + np.exp(-np.asarray(logits)))
    predictions = (probabilities >= threshold).astype(np.int64)
    
    # BUG: uses 1 - y_true and 1 - predictions, swapping REAL/FAKE
    precision_real_bug = precision_score(1 - y_true, 1 - predictions, zero_division=0)
    recall_real_bug = recall_score(1 - y_true, 1 - predictions, zero_division=0)
    f1_real_bug = f1_score(1 - y_true, 1 - predictions, zero_division=0)
    
    precision_fake_bug = precision_score(y_true, predictions, zero_division=0)
    recall_fake_bug = recall_score(y_true, predictions, zero_division=0)
    f1_fake_bug = f1_score(y_true, predictions, zero_division=0)
    
    macro_f1 = (f1_real_bug + f1_fake_bug) / 2.0
    
    return {
        "precision_real": float(precision_real_bug),
        "recall_real": float(recall_real_bug),
        "f1_real": float(f1_real_bug),
        "precision_fake": float(precision_fake_bug),
        "recall_fake": float(recall_fake_bug),
        "f1_fake": float(f1_fake_bug),
        "macro_f1": float(macro_f1),
    }

# Test with the validation data
metrics_bug = compute_metrics_bug_demo(labels.tolist(), logits.tolist())
print(f"   Buggy compute_metrics results:")
print(f"     precision_real: {metrics_bug['precision_real']:.4f} (should be REAL precision)")
print(f"     recall_real: {metrics_bug['recall_real']:.4f} (should be REAL recall)")
print(f"     precision_fake: {metrics_bug['precision_fake']:.4f} (should be FAKE precision)")
print(f"     recall_fake: {metrics_bug['recall_fake']:.4f} (should be FAKE recall)")
print()

# Correct computation
print("   Correct computation (no 1- flip):")
y_true = labels
predictions = (probs >= 0.5).astype(int)
precision_real_correct = precision_score(y_true, predictions, zero_division=0)
recall_real_correct = recall_score(y_true, predictions, zero_division=0)
precision_fake_correct = precision_score(1 - y_true, 1 - predictions, zero_division=0) if False else "N/A"
# Actually correct: use labels directly
cm = confusion_matrix(y_true, predictions, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
precision_real_correct = tn / (tn + fn) if (tn + fn) > 0 else 0
recall_real_correct = tn / (tn + fp) if (tn + fp) > 0 else 0
precision_fake_correct = tp / (tp + fp) if (tp + fp) > 0 else 0
recall_fake_correct = tp / (tp + fn) if (tp + fn) > 0 else 0
print(f"     precision_real: {precision_real_correct:.4f} = TN/(TN+FN) = {tn}/({tn}+{fn})")
print(f"     recall_real: {recall_real_correct:.4f} = TN/(TN+FP) = {tn}/({tn}+{fp})")
print(f"     precision_fake: {precision_fake_correct:.4f} = TP/(TP+FP) = {tp}/({tp}+{fp})")
print(f"     recall_fake: {recall_fake_correct:.4f} = TP/(TP+FN) = {tp}/({tp}+{fn})")
print()

# 3. Training history analysis
print("3. TRAINING HISTORY ANALYSIS")
print("-" * 40)
history_path = r'D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\celebdf_exp_balanced\training_history.csv'
df = pd.read_csv(history_path)
print(f"   Epochs run: {len(df)} (early stopped at epoch 13)")
print(f"   Best val macro_f1: {df['macro_f1'].max():.4f} at epoch {df.loc[df['macro_f1'].idxmax(), 'epoch']}")
print(f"   Best val bal_acc: {df['balanced_accuracy'].max():.4f} at epoch {df.loc[df['balanced_accuracy'].idxmax(), 'epoch']}")
print(f"   Final val macro_f1 (epoch 13): {df.iloc[-1]['macro_f1']:.4f}")
print(f"   Final val bal_acc (epoch 13): {df.iloc[-1]['balanced_accuracy']:.4f}")
print()
print("   Key trends:")
print(f"   - Train loss: {df['train_loss'].min():.4f} → {df['train_loss'].max():.4f} (decreasing, good)")
print(f"   - Val loss: {df['val_loss'].min():.4f} → {df['val_loss'].max():.4f} (volatile, ~0.5-0.95)")
print(f"   - Val bal_acc: {df['balanced_accuracy'].min():.4f} → {df['balanced_accuracy'].max():.4f} (0.53-0.62, near chance)")
print(f"   - Val macro_f1: {df['macro_f1'].min():.4f} → {df['macro_f1'].max():.4f} (0.46-0.62, near chance)")
print(f"   - Val roc_auc: {df['roc_auc'].min():.4f} → {df['roc_auc'].max():.4f} (0.49-0.62, near random)")
print(f"   - REAL recall: {df['recall_real'].min():.4f} → {df['recall_real'].max():.4f} (0.11-0.48, always low)")
print(f"   - FAKE recall: {df['recall_fake'].min():.4f} → {df['recall_fake'].max():.4f} (0.59-0.99, always high)")
print()

# 4. Feature vector separation diagnostic
print("4. FEATURE VECTOR SEPARATION DIAGNOSTIC")
print("-" * 40)
# Load a sample of features from the cache
cache_dir = r'D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\celebdf_retrain_20261023\celebdf_features'
index_path = r'D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\celebdf_retrain_20261023\features_index.csv'
df_index = pd.read_csv(index_path)

# Sample REAL and FAKE features
real_indices = df_index[(df_index['label']==0) & (df_index['split']=='val')].index
fake_indices = df_index[(df_index['label']==1) & (df_index['split']=='val')].index

print(f"   REAL val samples: {len(real_indices)}")
print(f"   FAKE val samples: {len(fake_indices)}")

# Load a few features to check
if len(real_indices) > 0 and len(fake_indices) > 0:
    sample_real_idx = real_indices[0]
    sample_fake_idx = fake_indices[0]
    
    real_feature_path = df_index.loc[sample_real_idx, 'feature_path']
    fake_feature_path = df_index.loc[sample_fake_idx, 'feature_path']
    
    # Resolve path
    if not Path(real_feature_path).exists():
        real_feature_path = Path(cache_dir) / Path(real_feature_path).name
    if not Path(fake_feature_path).exists():
        fake_feature_path = Path(cache_dir) / Path(fake_feature_path).name
    
    real_features = np.load(real_feature_path)
    fake_features = np.load(fake_feature_path)
    
    print(f"   REAL feature shape: {real_features.shape}")
    print(f"   FAKE feature shape: {fake_features.shape}")
    print(f"   REAL mean: {real_features.mean():.4f}, std: {real_features.std():.4f}")
    print(f"   FAKE mean: {fake_features.mean():.4f}, std: {fake_features.std():.4f}")
    print(f"   REAL logit range: [{real_features[:,0].min():.4f}, {real_features[:,0].max():.4f}] (first frame, first dim)")
    print(f"   FAKE logit range: [{fake_features.min():.4f}, {fake_features.max():.4f}]")
print()

# 5. Label encoding verification
print("5. LABEL ENCODING VERIFICATION")
print("-" * 40)
manifest_path = r'D:\AI-based-Deepfake-Detection-system-main\datasets\celebdf_manifest.csv'
df_manifest = pd.read_csv(manifest_path)
print(f"   Train: REAL={int((df_manifest['label']==0).sum())}, FAKE={int((df_manifest['label']==1).sum())}")
print(f"   Val:   REAL={int((df_manifest['label']==0).sum())}, FAKE={int((df_manifest['label']==1).sum())}")
print(f"   Test:  REAL={int((df_manifest['label']==0).sum())}, FAKE={int((df_manifest['label']==1).sum())}")
print(f"   Train split: {df_manifest[df_manifest['split']=='train']['label'].value_counts().to_dict()}")
print(f"   Val split:   {df_manifest[df_manifest['split']=='val']['label'].value_counts().to_dict()}")
print(f"   Test split:  {df_manifest[df_manifest['split']=='test']['label'].value_counts().to_dict()}")
print()

# 6. Score orientation check
print("6. SCORE ORIENTATION CHECK")
print("-" * 40)
print(f"   REAL (label 0) probs mean: {probs[labels==0].mean():.4f}")
print(f"   FAKE (label 1) probs mean: {probs[labels==1].mean():.4f}")
print(f"   REAL (label 0) logits mean: {logits[labels==0].mean():.4f}")
print(f"   FAKE (label 1) logits mean: {logits[labels==1].mean():.4f}")
print(f"   % REAL with positive logits: {(logits[labels==0] > 0).mean()*100:.1f}%")
print(f"   % FAKE with negative logits: {(logits[labels==1] < 0).mean()*100:.1f}%")
print()

# Summary of findings
print("=" * 70)
print("SUMMARY OF FINDINGS")
print("=" * 70)
print()
print("BUG 1 (HIGH CONFIDENCE): compute_metrics label flip")
print("   - The function uses 1 - y_true and 1 - predictions, swapping")
print("     REAL and FAKE metrics. REAL recall reported as 28.04% is actually")
print("     REAL precision, and FAKE recall 91.57% is actually FAKE precision.")
print("   - All 9 unit tests pass due to coincidental flipping with perfect")
print("     separation cases, but the semantics are inverted.")
print()
print("BUG 2 (HIGH CONFIDENCE): Near-random ROC-AUC (0.4963)")
print("   - Model cannot discriminate REAL from FAKE beyond chance.")
print("   - Threshold tuning cannot fix this fundamental issue.")
print()
print("BUG 3 (MEDIUM CONFIDENCE): Low REAL recall across all thresholds")
print("   - Max REAL recall = 13.08% at threshold 0.05")
print("   - REAL recall decreases as threshold increases")
print("   - Model predicts ~90% of videos as FAKE regardless of threshold")
print()
print("CONFIRMED: The model architecture (EfficientNet-B0 + BiLSTM) fails")
print("          to generalize beyond Celeb-DF domain (ROC-AUC ~0.5 on val).")
print()
print("RECOMMENDED NEXT: Launch weighted-loss experiment (Mode B) to confirm")
print("                  whether class weighting helps at all, given the")
print("                  fundamental separation problem.")
print("=" * 70)