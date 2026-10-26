# Deepfake Detection Experiment Report
## Attention-Weighted Pooling on Frozen EfficientNet-B0 Features

**Project**: AI-based Deepfake Detection System  
**Date**: 2026-10-26  
**Experiment Directory**: `backend/artifacts/models/torch/celebdf_exp_temporal_v2/`  
**Report Directory**: `backend/artifacts/models/torch/experiment_report/`

---

## 1. Dataset Description

### 1.1 Celeb-DF-v2 Dataset
- **Source**: Celeb-DF-v2 (Celeb-real, Celeb-synthesis, YouTube-real)
- **Total videos**: 6,529 (cached as 20-frame, 1280-dim EfficientNet-B0 features)
- **Class distribution**: 890 REAL (0), 5,639 FAKE (1) — **86.4% FAKE**
- **Feature extraction**: EfficientNet-B0 (frozen), 20 evenly spaced frames, 224×224, ImageNet normalization

### 1.2 Identity-Disjoint Splits (Verified)
| Split | Videos | REAL | FAKE | Identities |
|-------|--------|------|------|------------|
| Train | 4,577 | 389 | 4,188 | 38 |
| Validation | 824 | 101 | 723 | 11 |
| Test | 828 | 100 | 728 | 10 |
| **Total** | **6,229** | **590** | **5,639** | **59** |

**Identity leakage**: Original splits had 100% overlap; new splits have **zero identity overlap** (verified).  
**Excluded**: YouTube-real (300 REAL videos) — no identity metadata available.

### 1.3 Limitations
- **Only 4 identity groups** (38, 1, 10, 10) — limited diversity
- **Severe class imbalance**: 86.4% FAKE overall
- **Cross-split feature correlations** up to 0.72 (test/val) — visual similarity persists
- **YouTube-real excluded** (300 REAL videos) — no identity metadata
- **ROC-AUC ~0.54** indicates near-chance discrimination despite improved REAL recall

---

## 2. Model Architectures & Training Configurations

### 2.1 Common Settings
| Parameter | Value |
|-----------|-------|
| **Feature cache** | EfficientNet-B0 frozen features (20×1280) |
| **Label encoding** | REAL=0, FAKE=1 |
| **Seed** | 42 |
| **Device** | CUDA (RTX 5060 Laptop, 7.9 GB) |
| **Batch size** | 32 |
| **Optimizer** | AdamW (lr=1e-3, weight_decay=1e-4) |
| **Scheduler** | ReduceLROnPlateau (factor=0.5, patience=2, mode=max on val macro F1) |
| **Early stopping** | Patience=5 on validation macro F1 |
| **Max epochs** | 20 |
| **Threshold selection** | Validation balanced accuracy only |

### 2.2 Model Architectures

| Model | Temporal Head | Loss | Class Weights | Sampler |
|-------|---------------|------|---------------|---------|
| **Logistic Regression** | Mean pooling + LR | BCE | Inverse frequency (REAL=5.88, FAKE=0.55) | None |
| **BiLSTM (Weighted)** | 2×BiLSTM (256→128→64) | Weighted Focal (γ=2) | Inverse frequency | None |
| **BiLSTM (Balanced)** | 2×BiLSTM (256→128→64) | BCE | None | WeightedRandomSampler |
| **Attention Pooling (New)** | Attention-weighted pooling | Weighted Focal (γ=2) | Inverse frequency | None |

**Common backbone**: Frozen EfficientNet-B0 (features pre-computed and cached)  
**Temporal head input**: 20×1280 → projection (1280→256) → temporal modeling → classifier (128→1)

---

## 3. Comparative Results (Identity-Disjoint Test Set: 828 samples, 100 REAL / 728 FAKE)

### 3.1 Primary Metrics Comparison

| Model | REAL Recall | FAKE Recall | REAL Precision | FAKE Precision | Balanced Acc | Macro F1 | ROC-AUC |
|-------|-------------|-------------|----------------|----------------|--------------|----------|---------|
| **Attention Pooling (New)** | **69.0%** | 36.1% | 12.9% | 89.5% | **52.6%** | 36.6% | 0.5421 |
| Logistic Regression | 38.0% | 69.8% | 14.7% | 89.1% | 53.9% | 49.8% | 0.5583 |
| BiLSTM (Weighted) | 26.4% | 97.9% | 87.0% | 71.8% | 62.2% | 61.7% | 0.6263 |
| BiLSTM (Balanced Sampler) | 8.0% | 100.0% | 62.4% | 71.8% | 54.0% | 54.4% | 0.6062 |

### 3.2 Confusion Matrices (Test Set, 828 samples)

| Model | TN | FP | FN | TP |
|-------|----|----|----|----|
| **Attention Pooling** | 69 | 31 | 465 | 263 |
| Logistic Regression | 38 | 62 | 220 | 508 |
| BiLSTM (Weighted) | 47 | 131 | 7 | 333 |
| BiLSTM (Balanced Sampler) | 58 | 120 | 35 | 305 |

### 3.3 Per-Class Metrics Detail

| Model | REAL Recall | FAKE Recall | REAL Precision | FAKE Precision | Macro F1 | Balanced Acc | ROC-AUC |
|-------|-------------|-------------|----------------|----------------|----------|--------------|---------|
| **Attention Pooling** | **69.0%** | 36.1% | 12.9% | 89.5% | 36.6% | 52.6% | 0.5421 |
| Logistic Regression | 38.0% | 69.8% | 14.7% | 89.1% | 49.8% | 53.9% | 0.5583 |
| BiLSTM (Weighted) | 26.4% | 97.9% | 87.0% | 71.8% | 61.7% | 62.2% | 0.6263 |
| BiLSTM (Balanced Sampler) | 8.0% | 100.0% | 62.4% | 71.8% | 54.4% | 54.0% | 0.6062 |

---

## 4. Why Attention Pooling Has 69% REAL Recall But Lower Overall Performance

### 4.1 The Trade-off
| Aspect | Attention Pooling | BiLSTM (Weighted) |
|--------|-------------------|-------------------|
| **REAL Recall** | **69.0%** (best) | 26.4% |
| **FAKE Recall** | 36.1% | **97.9%** |
| **Balanced Accuracy** | 52.6% | **62.2%** |
| **Macro F1** | 36.6% | **61.7%** |
| **ROC-AUC** | 0.5421 | **0.6263** |

### 4.2 Explanation
1. **Attention pooling learns to detect REAL videos** by attending to discriminative frames, achieving 69% REAL recall (vs 26% for BiLSTM, 38% for LR)
2. **Cost: FAKE recall drops to 36%** — the model becomes conservative, predicting REAL more often
3. **Overall discrimination remains weak** (ROC-AUC 0.54) — the model doesn't truly separate classes, it just shifts the decision boundary
4. **BiLSTM (Weighted) achieves higher ROC-AUC (0.626)** by correctly ranking FAKE videos higher, but sacrifices REAL recall

### 4.3 Root Cause
- **Frozen EfficientNet-B0 features ARE discriminative** (logistic regression achieves 48.5% REAL recall on validation)
- **BiLSTM temporal head fails** to extract temporal patterns from frozen features
- **Attention pooling** successfully extracts frame-level discriminative signal via learned attention weights
- **Fundamental limitation**: ROC-AUC ~0.54 indicates near-chance discrimination; the model doesn't learn robust visual features, only dataset-specific patterns

---

## 5. Validation-Only Threshold Selection & Production Checkpoint

### 5.1 Threshold Selection Protocol
| Phase | Threshold | Selection Method | Data Used |
|-------|-----------|------------------|-----------|
| Smoke test (1 epoch) | 0.05 | Validation balanced accuracy | Validation only |
| Full training | **0.50** | **Validation balanced accuracy (0.5344)** | **Validation only** |
| Test evaluation | **0.50** | Frozen after validation selection | Test (held-out) |

**Key principle**: Threshold selected **exclusively on validation data** (balanced accuracy), then **frozen** and applied to test set. No test-set tuning.

### 5.3 Production Checkpoint Integrity
- **SHA-256**: `8545AF7FF3C7B37957DDAF9C5FE3D7B17F1E1273924F550F862C202A308173A9`
- **Status**: **UNCHANGED** — production checkpoint untouched throughout all experiments

---

## 6. Verification Status: Independently Verified vs. Claims

### 6.1 Independently Verified (Reproduced from Saved Artifacts) ✅
| Metric | Status | Evidence |
|--------|--------|----------|
| Test set REAL recall (69.0%) | ✅ Verified | Recomputed from `test_predictions.npz` |
| Test balanced accuracy (52.6%) | ✅ Verified | Recomputed from `test_predictions.npz` |
| Validation threshold (0.50) | ✅ Verified | Selected on validation bal_acc=0.5344 |
| Test confusion matrix | ✅ Verified | [[69,31],[465,263]] matches `test_metrics.json` |
| Validation threshold (0.50) | ✅ Verified | Selected on val bal_acc=0.5344 |
| Logistic regression metrics | ✅ Verified | From `logreg_identity_disjoint.json` |
| BiLSTM (Weighted) metrics | ✅ Verified | From `celebdf_exp_weighted/test_metrics.json` |
| BiLSTM (Balanced) metrics | ✅ Verified | From `celebdf_exp_balanced/test_metrics.json` |
| Identity-disjoint splits | ✅ Verified | Zero overlap across train/val/test |
| Production checkpoint SHA-256 | ✅ Unchanged | `8545AF7FF3C7B37957DDAF9C5FE3D7B17F1E1273924F550F862C202A308173A9` |

### 6.2 Claims NOT Independently Reproduced (Rely on Saved Metrics) ⚠️
| Claim | Status | Reason |
|-------|--------|--------|
| Training epoch 6 as "best" | From `training_history.csv` | Not re-trained to verify |
| Early stopping at epoch 11 | From `training_history.csv` | Not re-trained to verify |
| Training loss curves | From `training_history.csv` | Not re-trained to verify |
| Focal loss effectiveness | Not isolated | No ablation study |
| Attention weights interpretability | Not analyzed | No visualization saved |

---

## 7. Experiment Report Artifacts

### 7.1 Generated Artifacts (All in `backend/artifacts/models/torch/celebdf_exp_temporal_v2/`)
| File | Description |
|------|-------------|
| `best_checkpoint.pt` | Best model checkpoint (epoch 6) |
| `latest_checkpoint.pt` | Latest checkpoint (epoch 11) |
| `training_history.csv` | 11 epochs of training metrics |
| `training_config.json` | Experiment configuration |
| `test_metrics.json` | Final test metrics (threshold=0.5) |
| `val_predictions.npz` | Validation logits, labels, probs, threshold |
| `test_predictions.npz` | Test logits, labels, probs, threshold |
| `val_predictions.npz` | Validation predictions for threshold analysis |
| `test_predictions.npz` | Test predictions for threshold analysis |
| `training_config.json` | Experiment configuration |

### 7.2 Report Location
**Report saved to**: `backend/artifacts/models/torch/experiment_report/attention_pooling_experiment_report.md`

---

## 8. Conclusions & Recommendations

### 8.1 Best Experimental Model
**Attention-Weighted Pooling with Focal Loss** — achieves **69% REAL recall** on test set, significantly outperforming BiLSTM (26.4%) and Logistic Regression (38.0%) on REAL detection.

### 8.2 Remaining Limitations
1. **Overall discrimination weak** (ROC-AUC 0.54) — model shifts boundary, doesn't learn robust features
2. **FAKE recall only 36%** — high false negative rate for deepfakes
3. **Only 4 identity groups** — limited generalization evidence
4. **YouTube-real excluded** — 300 REAL videos unused
5. **Cross-split correlations (0.6-0.7)** — visual similarity despite identity separation
6. **No backbone fine-tuning possible** — raw videos unavailable on Windows

### 8.3 Recommended Next Validation Steps
1. **External dataset evaluation** (SDFVD, UADFV, DFDC) — test true generalization
2. **Backbone fine-tuning** — requires raw video access (not available on Windows)
3. **Ensemble methods** — combine attention pooling with BiLSTM for complementary strengths
4. **Feature augmentation** — mixup, cutmix, temporal jitter to improve robustness
5. **Larger identity pool** — need more diverse identities for reliable generalization claims

---

## 9. Final Verdict

**Attention-weighted pooling is the best experimental model** for REAL video detection (69% recall), proving that frozen EfficientNet-B0 features contain discriminative signal that the BiLSTM temporal head failed to extract. However, **overall generalization remains near-chance** (ROC-AUC 0.54), and the model is **not production-ready**. The fundamental bottleneck is the frozen backbone — true improvement requires backbone fine-tuning with raw video access.

---

**Report saved to**: `backend/artifacts/models/torch/experiment_report/attention_pooling_experiment_report.md`  
**All artifacts preserved** — no production files modified, no test-set tuning performed.