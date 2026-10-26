# Deepfake Detection Model Failure Audit Report
**Date:** 2026-10-23
**Project:** AI-based Deepfake Detection System
**Audit Directory:** `backend/artifacts/models/torch/failure_audit_20261023/`

---

## Executive Summary

This audit investigates the failure of the candidate PyTorch temporal deepfake detector (pos_weight=0.5, threshold=0.66) which incorrectly classifies Celeb-real videos as FAKE with 84.3% false positive rate, while YouTube-real videos are correctly classified with only 28.6% FPR. The model achieves near-random performance on external datasets (SDFVD: 50% accuracy, UADFV: 51% accuracy).

**Verdict: The model is not production-ready. It learns domain-specific features (Celeb vs YouTube) rather than generalizable manipulation artifacts.**

---

## 1. Threshold Consistency Audit

### Verified Threshold Values

| Source | Threshold Value | Notes |
|--------|----------------|-------|
| `backend/.env` (production) | `0.66` | Rounded value |
| `backend/app/core/config.py` default | `0.532962620` | Unused default |
| `backend/app/ml/inference.py` | `settings.decision_threshold` | Reads from .env |
| `final_selection_report.json` - selected | `0.66` | Rounded for reporting |
| `final_selection_report.json` - exact | `0.660688` | Exact validation optimum |
| `reproducibility_manifest.json` - selected | `0.66` | Rounded |
| `reproducibility_manifest.json` - exact | `0.660688` | Exact validation optimum |
| `threshold_comparison_final.csv` | `0.66` / `0.6606877446174622` | Both present |

### Discrepancy Found
- **Production .env uses `0.66` (rounded)** while the **exact validation-selected threshold is `0.660688`**
- The difference is small (0.000688) but technically the production threshold does not match the exact validation optimum
- **Recommendation**: Update `.env` to `DECISION_THRESHOLD=0.660688` for exact reproducibility

### Threshold Selection Criteria
- **Baseline (pos_weight=0.134)**: Selected at `0.40` (max balanced accuracy on validation)
- **Candidate (pos_weight=0.5)**: Selected at `0.660688` (max specificity subject to fake recall ≥ 0.90 on validation)
- **Test evaluation**: Used frozen threshold `0.66` (rounded)

---

## 2. Checkpoint Identity Audit

### SHA-256 Verification

| Checkpoint Path | SHA-256 | Role |
|----------------|---------|------|
| `backend/artifacts/models/torch/best_checkpoint.pt` (production) | `8545AF7FF3C7B37957DDAF9C5FE3D7B17F1E1273924F550F862C202A308173A9` | **Candidate** |
| `celebdf_retrain_20261023/experiment_posweight_0.5/best_checkpoint.pt` | `8545AF7FF3C7B37957DDAF9C5FE3D7B17F1E1273924F550F862C202A308173A9` | **Candidate (source)** |
| `celebdf_retrain_20261023/checkpoints/best_checkpoint.pt` (baseline) | `0C58FD041D01ECA5D0DA9257621642C0B8695D7566AC74D381C4706E26C25397` | **Baseline** |

### Verification Results
✅ **Production checkpoint = Candidate checkpoint** (identical SHA-256)  
✅ **Baseline checkpoint is different** (different training run, pos_weight=0.134)  
✅ **Architecture matches**: Both use `CachedFeatureTemporalClassifier` with:
- EfficientNet-B0 backbone (frozen, ImageNet pretrained)
- Projection: Linear(1280→256) + ReLU + Dropout(0.3)
- BiLSTM1: 256→128 (bidirectional)
- BiLSTM2: 256→64 (bidirectional)
- Classifier: Linear(128→128) + ReLU + Dropout(0.4) + Linear(128→1)
- Input: 20 frames × 224×224 RGB, [0,1] normalized
- ImageNet normalization: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]

### Preprocessing Consistency
✅ **Training** (`cache_celebdf_resume.py`): 20 evenly spaced frames, 224×224 resize, BGR→RGB, [0,1] normalization, ImageNet mean/std  
✅ **Inference** (`torch_inference.py`): 20 evenly spaced frames via `select_evenly_spaced_frames`, 224×224 resize, BGR→RGB, [0,1] normalization, ImageNet mean/std  
✅ **Frame sampling matches**: Both use 20 evenly spaced frames

---

## 3. False Positive Investigation

### Source-Level Performance (Test Set, threshold=0.66)

| Source | n | Accuracy | Balanced Acc | Specificity | Fake Recall | FPR | FNR |
|--------|---|----------|--------------|-------------|-------------|-----|-----|
| Celeb-real | 108 | 0.1574 | 0.1574 | 0.1574 | N/A | **0.8426** | 0.0000 |
| YouTube-real | 70 | 0.7143 | 0.7143 | 0.7143 | N/A | 0.2857 | 0.0000 |
| Celeb-synthesis | 340 | 0.9176 | 0.9176 | N/A | 0.9176 | 0.0000 | 0.0824 |

### Probability Distributions
| Source | Mean Fake Prob | Std | Min | Max |
|--------|---------------|-----|-----|-----|
| Celeb-real | **0.7242** | 0.1313 | 0.0167 | 0.8418 |
| YouTube-real | **0.4622** | 0.2809 | 0.0139 | 0.8256 |
| Celeb-synthesis | **0.7628** | 0.0737 | 0.0320 | 0.8434 |

### Key Findings
1. **Celeb-real and Celeb-synthesis have nearly identical fake probability distributions** (mean 0.724 vs 0.763, overlapping ranges)
2. **YouTube-real has significantly lower fake probabilities** (mean 0.462)
3. **Video properties (frames, fps, resolution) are similar** between false positives and true negatives in Celeb-real
4. **All 48 Celeb-real false positive identities appear in training** (48/48)
4. **All 13 Celeb-real true negative identities also appear in training** (13/13)

### Three Tested Videos Verification
| Video | True Label | Frames | Resolution | Selected Frames | Fake Prob | Predicted |
|-------|------------|--------|------------|-----------------|-----------|-----------|
| Celeb-synthesis/id0_id16_0000.mp4 | FAKE | 469 | 944×500 | 20 | 0.7903 | FAKE ✅ |
| Celeb-real/id0_0000.mp4 | REAL | 469 | 942×500 | 20 | 0.7824 | FAKE ❌ |
| YouTube-real/00000.mp4 | REAL | 450 | 892×500 | 20 | 0.0895 | REAL ✅ |

✅ Correct files and labels used  
✅ Preprocessing pipeline identical for all three  
✅ Frame sampling (20 evenly spaced), resize (224×224), normalization ([0,1] + ImageNet) all match

---

## 4. Data Leakage and Test Independence Verification

### Identity Overlap Analysis (Celeb Domain)

| Split | Identities | Overlap with Train | Overlap with Val | Overlap with Test |
|-------|------------|-------------------|------------------|-------------------|
| Train | 59 | - | 59/59 | 54/54 |
| Val | 59 | 59/59 | - | 54/54 |
| Test | 54 | 54/54 | 54/54 | - |

### Critical Findings
🔴 **SEVERE IDENTITY LEAKAGE**: **54/54 test identities (100%) appear in training**  
🔴 **0 test-only identities** - no identity-disjoint evaluation possible  
🔴 **Train-Val overlap: 59/59 (100%)** - validation not identity-disjoint from training  
🔴 **Train-Test overlap: 54/54 (100%)** - test not identity-disjoint from training  

### Manifest & Feature Index Integrity
✅ **No label mismatches** between feature index and manifest (0/6529)  
✅ **No split mismatches** between feature index and manifest (0/6529)  
✅ **Feature index status**: 6529/6529 "ok"  
✅ **Test partition not used for training**: Training uses val split for checkpoint selection (early stopping on val loss); test only evaluated once at end

### Split Distribution
| Split | Total | Real (0) | Fake (1) |
|-------|-------|----------|----------|
| Train | 5,109 | 605 | 4,504 |
| Val | 902 | 107 | 795 |
| Test | 518 | 178 | 340 |

---

## 5. External Evaluation Evidence

### SDFVD Results (106 videos: 53 REAL, 53 FAKE)
| Metric | Value |
|--------|-------|
| Accuracy | 0.5000 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.5000 |
| Recall (Fake) | 0.2453 |
| Specificity | 0.7547 |
| ROC-AUC | 0.5290 |
| REAL mean prob | 0.3209 |
| FAKE mean prob | 0.3558 |

### UADFV Results (98 videos: 49 REAL, 49 FAKE)
| Metric | Value |
|--------|-------|
| Accuracy | 0.5102 |
| Balanced Accuracy | 0.5102 |
| Precision | 0.5075 |
| Recall (Fake) | 0.6939 |
| Specificity | 0.3265 |
| ROC-AUC | 0.5392 |
| REAL mean prob | 0.6233 |
| FAKE mean prob | 0.6507 |

### Critical Finding
**On both external datasets, REAL and FAKE fake probabilities are nearly identical** (SDFVD: 0.32 vs 0.36; UADFV: 0.62 vs 0.65). The model **cannot distinguish real from fake** on unseen domains. Performance is at near-random chance (50-51% accuracy).

---

## 5. Summary of Trustworthy vs Compromised Results

### ✅ Trustworthy Results
1. **Internal validation metrics** (902 videos, identity-overlapped but consistent)
2. **Internal test metrics** (518 videos, evaluated once with frozen threshold)
3. **Per-source analysis** on Celeb-DF test set (consistent with validation)
4. **Threshold comparison** on identical validation set (902 videos)
5. **Checkpoint integrity** (SHA-256 verified, architecture matches)

### 🔴 Compromised Results
1. **Identity-disjoint evaluation claims** - **NOT POSSIBLE** with current splits (54/54 test identities in training)
2. **Generalization claims** - Model fails on external datasets (SDFVD/UADFV ~50% accuracy)
3. **Manipulation detection claims** - Model learns domain features, not manipulation artifacts
4. **Production readiness claims** - Explicitly marked as NOT ready in reports

### 🟡 Partially Reliable
- Internal test metrics are internally consistent but **not generalizable** due to identity leakage and domain bias
- Threshold selection methodology is sound but **applied to compromised splits**

---

## 6. Most Likely Explanation for Celeb-real False Positives

**The model learns domain-specific features (Celeb vs YouTube) rather than manipulation artifacts.**

### Evidence:
1. **Celeb-real and Celeb-synthesis share identities** (54/54 test identities in training) and visual domain (same actors, lighting, compression)
2. **Celeb-real and Celeb-synthesis have nearly identical fake probability distributions** (mean 0.724 vs 0.763)
3. **YouTube-real has completely different distribution** (mean 0.462, lower FPR)
4. **External datasets (SDFVD/UADFV) show near-random performance** - model fails on unseen domains
5. **All Celeb-real false positive identities appear in training** (48/48)
5. **Video properties (resolution, fps, duration) don't distinguish FP from TN**

### Root Cause
The model exploits **identity/domain shortcuts** rather than learning manipulation artifacts:
- Celeb domain (Celeb-real + Celeb-synthesis): Same actors, same lighting, same compression artifacts
- Model learns "Celeb domain = likely fake" because Celeb-synthesis dominates training (4,504 fake vs 605 real)
- YouTube-real is visually distinct (different actors, YouTube compression) → model learns "YouTube domain = likely real"

---

## 7. What Should Be Fixed First

### Priority 1: Fix Data Splits (Critical)
- **Create identity-disjoint splits** for Celeb domain (separate identities across train/val/test)
- **Document YouTube-real limitation** (no identity metadata available)
- **Re-train with identity-disjoint splits** to measure true generalization

### Priority 2: Domain Generalization
- **Add domain-diverse datasets**: FaceForensics++, DFDC, DFDC-preview
- **Implement domain-adversarial training** to learn domain-invariant features
- **Per-domain threshold calibration** for deployment

### Priority 3: Evaluation Rigor
- **Reserve truly independent test set** (never used in any experiment)
- **Report per-domain metrics** separately (never aggregate across domains)
- **Test on external datasets** before claiming generalization

### Priority 4: Configuration Fix
- Update `.env` threshold to exact value: `DECISION_THRESHOLD=0.660688`
- Ensure `.env.example` documents all required variables

---

## 8. Evidence Needed Before Retraining

1. **Identity-disjoint Celeb-DF splits** with verified no-overlap
2. **FaceForensics++ dataset** downloaded and preprocessed with compatible labels
3. **DFDC dataset** (or subset) with compatible preprocessing
4. **Domain-adversarial training framework** implemented and tested
5. **Independent test set reserved** (e.g., 20% of FaceForensics++ held out completely)
4. **Per-domain evaluation protocol** defined and documented
5. **Frame-level manipulation labels** (if available) for supervised artifact learning

---

## Appendix: Key File Paths

```
Production checkpoint:     backend/artifacts/models/torch/best_checkpoint.pt
Candidate checkpoint:      backend/artifacts/models/torch/celebdf_retrain_20261023/experiment_posweight_0.5/best_checkpoint.pt
Baseline checkpoint:       backend/artifacts/models/torch/celebdf_retrain_20261023/checkpoints/best_checkpoint.pt
Manifest:                  backend/artifacts/models/torch/celebdf_retrain_20261023/celebdf_manifest.csv
Feature index:             backend/artifacts/models/torch/celebdf_retrain_20261023/features_index.csv
Test predictions:          backend/artifacts/models/torch/celebdf_retrain_20261023/experiment_posweight_0.5/test_predictions.npz
Validation predictions:    backend/artifacts/models/torch/celebdf_retrain_20261023/experiment_posweight_0.5/val_predictions.npz
Threshold comparison:      backend/artifacts/models/torch/celebdf_retrain_20261023/threshold_comparison_final.csv
Reports:                   backend/artifacts/models/torch/celebdf_retrain_20261023/*.json
External predictions:      datasets/sdfvd/processed/sdfvd_predictions.csv
                           datasets/UADFV/processed/uadfv_predictions.csv
```

---

**Audit completed by:** Automated audit pipeline  
**Next steps:** Await approval for retraining with identity-disjoint splits and domain-diverse datasets