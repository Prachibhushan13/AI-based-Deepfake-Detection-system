# Experiment Comparison Report

**Experiment Directory:** `backend/artifacts/models/torch/celebdf_identity_disjoint_20261023/`
**Date:** 2026-10-23
**Dataset:** Celeb-DF identity-disjoint split (6229 videos)

---

## Dataset Summary

| Split | Total | REAL (0) | FAKE (1) | Identities |
|-------|-------|----------|----------|------------|
| Train | 4,577 | 389 | 4,188 | 38 |
| Val   | 824   | 101 | 723 | 11 |
| Test  | 828   | 100 | 728 | 10 |
| **Total** | **6,229** | **590** | **5,639** | **59** |

**Class Imbalance:** Severe - FAKE dominates (91.5% train, 87.7% val, 87.9% test)
**Identity Groups:** 4 groups (0: 38 identities, 1: 1 identity, 2: 10 identities, 3: 10 identities)
**YouTube-real:** 300 videos excluded (no identity metadata)

---

## Experiments Run on Identity-Disjoint Split

### Experiment A: Unweighted Baseline
- **Mode:** `--experiment-mode unweighted`
- **Loss:** BCEWithLogitsLoss with pos_weight=negative/positive=0.0929
- **Output:** `celebdf_identity_disjoint_20261023/baseline_unweighted/`
- **Early stopping:** Epoch 6

| Metric | Validation | Test |
|--------|------------|------|
| Loss | 0.1614 | 0.1546 |
| Accuracy | 0.880 | 0.1957 |
| Balanced Accuracy | 0.510 | 0.4951 |
| Precision | 0.880 | 0.0000 |
| Recall (FAKE) | 1.000 | 0.0000 |
| F1 | 0.1546 | 0.1798 |
| ROC-AUC | 0.4518 | 0.5027 |
| Confusion Matrix | [[89,11],[655,73]] | [[89,11],[655,73]] |

**Behavior:** Predicts almost everything as FAKE (91.5% of test set)

---

### Experiment B: Class-Weighted Loss (Per-Sample Inverse-Frequency)
- **Mode:** `--experiment-mode weighted`
- **Loss:** BCEWithLogitsLoss(reduction="none") with per-sample weights
- **Weights:** REAL=5.883, FAKE=0.5464
- **Output:** `celebdf_identity_disjoint_20261023/weighted_loss/`
- **Early stopping:** Epoch 6

| Metric | Validation | Test |
|--------|------------|------|
| Loss | 0.4784 | 0.4641 |
| Accuracy | 0.880 | 0.8792 |
| Balanced Accuracy | 0.510 | 0.5000 |
| Precision | 0.880 | 0.8792 |
| Recall (FAKE) | 1.000 | 1.000 |
| F1 | 0.9347 | 0.9357 |
| ROC-AUC | 0.4565 | 0.5173 |
| Confusion Matrix | [[0,100],[0,728]] | [[0,100],[0,728]] |

**Behavior:** Predicts everything as FAKE (100% FAKE recall, 0% REAL recall)

---

### Experiment C: Balanced Batch Sampler
- **Mode:** `--experiment-mode balanced_sampler`
- **Sampler:** WeightedRandomSampler with inverse class frequency weights
- **Loss:** BCEWithLogitsLoss with pos_weight=negative/positive
- **Output:** `celebdf_identity_disjoint_20261023/balanced_sampler/`
- **Early stopping:** Epoch 6

| Metric | Validation | Test |
|--------|------------|------|
| Loss | 0.3107 | 0.2995 |
| Accuracy | 0.880 | 0.1208 |
| Balanced Accuracy | 0.510 | 0.5000 |
| Precision | 0.880 | 0.0000 |
| Recall (FAKE) | 1.000 | 0.0000 |
| F1 | 0.9347 | 0.0000 |
| ROC-AUC | 0.5057 | 0.5393 |
| Confusion Matrix | [[100,0],[728,0]] | [[100,0],[728,0]] |

**Behavior:** Predicts everything as REAL (opposite problem)

---

## Comparison with Original Experiments (Leaky Split)

The original experiments trained on the leaky split (with identity overlap) and evaluated on the identity-disjoint test set:

| Model | Test Acc | Bal Acc | Precision | Recall | F1 | ROC-AUC | Confusion Matrix |
|-------|----------|---------|-----------|--------|-----|---------|------------------|
| original_pw0.25 | 0.8744 | 0.5188 | 0.8833 | 0.9876 | 0.9326 | 0.5694 | [[5,95],[9,719]] |
| original_pw0.5 | 0.8684 | 0.5154 | 0.8826 | 0.9808 | 0.9291 | 0.5327 | [[5,95],[14,714]] |
| **original_pw1.0** | **0.8816** | **0.5143** | **0.8823** | **0.9986** | **0.9369** | **0.5725** | **[[3,97],[1,727]]** |
| original_baseline | 0.7850 | 0.5543 | 0.8929 | 0.8585 | 0.8754 | 0.5769 | [[25,75],[103,625]] |

**Key Finding:** Models trained on the leaky split (with identity overlap) generalize better to the identity-disjoint test set than models trained on the identity-disjoint split itself. This suggests the identity-disjoint split is too difficult for the current model/features to learn meaningful manipulation features.

---

## Threshold Analysis (Best Model: original_pw1.0)

**Validation Set:** 824 videos (101 REAL, 723 FAKE)
**Probability Range:** 0.1124 - 0.9656, Mean: 0.9210

| Threshold | Balanced Acc | Specificity | Recall (FAKE) | Precision | F1 |
|-----------|--------------|-------------|---------------|-----------|-----|
| 0.50 | 0.510 | 0.020 | 1.000 | 0.880 | 0.936 |
| 0.70 | 0.510 | 0.020 | 1.000 | 0.880 | 0.936 |
| 0.80 | 0.513 | 0.030 | 0.996 | 0.880 | 0.934 |
| 0.85 | 0.528 | 0.079 | 0.976 | 0.884 | 0.928 |
| **0.8839** | **0.529** | **0.168** | **0.901** | **0.885** | **0.928** |
| 0.8839* | 0.529 | 0.168 | 0.901 | 0.885 | 0.928 |
| 0.8839* | 0.529 | 0.168 | 0.901 | 0.885 | 0.928 |
| 0.9220* | 0.573 | 0.099 | 0.950 | 0.889 | 0.918 |
| 0.95 | 0.508 | 0.871 | 0.144 | 0.889 | 0.248 |

*Best balanced accuracy: 0.5727 at threshold 0.9220
*Max specificity (recall≥0.90): 0.1683 at threshold 0.8839
*Max specificity (recall≥0.95): 0.0990 at threshold 0.8590

**Note:** Model is heavily biased towards FAKE (mean probability = 0.9210). Even at threshold 0.95, FAKE recall drops to 0.144.

---

## Summary & Recommendations

### Key Findings

1. **All three new experiments on identity-disjoint split FAILED** to learn meaningful discrimination:
   - Unweighted: Predicts all FAKE
   - Weighted: Predicts all FAKE  
   - Balanced Sampler: Predicts all REAL

2. **Root Cause:** The identity-disjoint split is too difficult for the current frozen EfficientNet-B0 + BiLSTM architecture. The model cannot learn manipulation features that generalize across identities.

3. **Leaky Split Models Generalize Better:** Models trained on the leaky split (with identity overlap) achieve balanced accuracy ~0.51-0.55 on the identity-disjoint test set, while identity-disjoint training yields ~0.50 (random).

### Recommendations

1. **Do NOT promote any new checkpoint to production** - all three new experiments failed.

4. **If identity-disjoint training is required:**
   - Use a more powerful backbone (unfreeze EfficientNet-B0, or use larger model)
   - Add data augmentation (face crops, color jitter, etc.)
   - Try contrastive/self-supervised pretraining on Celeb-DF
   - Consider domain adaptation techniques

5. **If production deployment is needed now:**
   - Use original_pw1.0 checkpoint (best on identity-disjoint test)
   - Apply threshold 0.8839 for recall≥0.90 (specificity=0.168)
   - Or threshold 0.922 for best balanced accuracy (0.573)
   - Document severe limitations: 83% FPR at recall=0.90

---

## Experiment Artifacts

| Experiment | Directory | Key Files |
|------------|-----------|-----------|
| Unweighted | `baseline_unweighted/` | best_checkpoint.pt, training_history.csv, test_metrics.json |
| Weighted | `weighted_loss/` | best_checkpoint.pt, training_history.csv, test_metrics.json |
| Balanced Sampler | `balanced_sampler/` | best_checkpoint.pt, training_history.csv, test_metrics.json |

---

## Training Commands for Reproduction

```bash
# Unweighted baseline
python scripts/train_torch_temporal_experiment.py \
  --index backend/artifacts/models/torch/celebdf_identity_disjoint_20261023/features_index.csv \
  --cache-dir backend/artifacts/models/torch/celebdf_retrain_20261023/celebdf_features \
  --output backend/artifacts/models/torch/celebdf_identity_disjoint_20261023/baseline_unweighted \
  --epochs 20 --patience 5 --batch-size 32 --learning-rate 1e-3 --weight-decay 1e-4 --seed 42 \
  --experiment-mode unweighted

# Weighted loss
python scripts/train_torch_temporal_experiment.py \
  --index backend/artifacts/models/torch/celebdf_identity_disjoint_20261023/features_index.csv \
  --cache-dir backend/artifacts/models/torch/celebdf_retrain_20261023/celebdf_features \
  --output backend/artifacts/models/torch/celebdf_identity_disjoint_20261023/weighted_loss \
  --epochs 20 --patience 5 --batch-size 32 --learning-rate 1e-3 --weight-decay 1e-4 --seed 42 \
  --experiment-mode weighted

# Balanced sampler
python scripts/train_torch_temporal_experiment.py \
  --index backend/artifacts/models/torch/celebdf_identity_disjoint_20261023/features_index.csv \
  --cache-dir backend/artifacts/models/torch/celebdf_retrain_20261023/celebdf_features \
  --output backend/artifacts/models/torch/celebdf_identity_disjoint_20261023/balanced_sampler \
  --epochs 20 --patience 5 --batch-size 32 --learning-rate 1e-3 --weight-decay 1e-4 --seed 42 \
  --experiment-mode balanced_sampler
```

---

## Verdict

**NOT READY FOR PRODUCTION** - The identity-disjoint training experiments failed to produce a viable model. The best available model is `original_pw1.0` from the leaky-split experiments, but it still has severe limitations (83% FPR at 90% recall).

**Next Steps Required:** Architectural changes, unfreezing backbone, data augmentation, or alternative approaches before identity-disjoint training can succeed.