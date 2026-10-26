# Pre-Training Readiness Report

**Experiment:** `celebdf_identity_disjoint_20261023`
**Date:** 2026-10-23
**Overall Status:** ✅ **READY FOR TRAINING** (with warnings)

---

## Mandatory Checks Summary

| Check | Status | Notes |
|-------|--------|-------|
| Zero identity overlap (train/val/test) | ✅ PASS | 0 overlap across all partitions |
| No source/target conflicts | ✅ PASS | 0 cross-partition source/target pairs |
| Deterministic split generation | ✅ PASS | Connected components of identity graph |
| No external test data used | ✅ PASS | Official test list not used; excluded due to overlap |
| Training loader correct | ✅ PASS | Uses --index, --cache-dir, --output; reads split column |
| Output directory isolated | ✅ PASS | New experiment dir; production checkpoint safe |
| Feature files exist | ✅ PASS | 6229/6229 files present |
| Feature shapes correct | ✅ PASS | All (20, 1280), float32, finite |
| Labels/splits match | ✅ PASS | 0 mismatches |

---

## Detailed Audit Results

### 1. Cache Filenames ✅ PASS
- **Algorithm:** SHA256(relative_path)[:24] + ".npy"
- **Verified:** 6229/6229 filenames match exactly
- **Missing files:** 0
- **Label/split mismatches:** 0

### 2. Feature Quality ✅ PASS
- **Total samples:** 6229 (train: 4577, val: 824, test: 828)
- **Shape:** All (20, 1280)
- **Dtype:** float32
- **Finite values:** All samples finite
- **Sequence length:** 20 frames
- **Feature dimension:** 1280 (EfficientNet-B0)
- **YouTube-real:** Excluded (0 samples in manifest)

### 3. Provenance ⚠️ PARTIAL
- **Pipeline match:** ✅ Cache generation and inference preprocessing match exactly
  - EfficientNet-B0 backbone (ImageNet pretrained, frozen)
  - 20 evenly spaced frames via `np.linspace`
  - 224×224 resize, BGR→RGB, [0,1] normalization
  - ImageNet mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
- **Training script compatibility:** ✅ Uses `--index`, `--cache-dir`, `--output`; reads `split` column
- **Provenance limitation:** ⚠️ `.npy` files contain only features (20,1280) - no embedded source-video hash, frame indices, or preprocessing metadata. Provenance traceable via index/manifest chain only.

### 4. Training Loader ✅ PASS
- **Arguments:** `--index`, `--cache-dir`, `--output`, `--epochs`, `--batch-size`, `--learning-rate`, `--weight-decay`, `--patience`, `--seed`
- **Split handling:** Reads `split` column from index for train/val/test
- **Checkpoint selection:** Uses validation loss only (`val_result["loss"]`)
- **Test split:** NOT used for checkpoint selection or early stopping
- **Output directory:** New experiment directory (isolated from production)

### 5. Label Balance ⚠️ PASS WITH WARNING
| Split | Total | Real | Fake | Ratio (R/F) | Fake % |
|-------|-------|------|------|-------------|--------|
| Train | 4,577 | 389 | 4,188 | 0.093 | 91.5% |
| Val | 824 | 101 | 723 | 0.140 | 87.7% |
| Test | 828 | 100 | 728 | 0.137 | 87.9% |

- **All partitions have both classes:** ✅
- **Identity counts:** Train 38, Val 11, Test 10
- **Min videos per identity:** Train 32, Val 10, Test 131
- **Warning:** Severe class imbalance (fake-dominated) in all partitions

### 6. Output Directory ✅ PASS
- **Experiment dir:** `backend/artifacts/models/torch/celebdf_identity_disjoint_20261023/`
- **No checkpoints subdirectory** (will be created during training)
- **Production checkpoint safe:** `backend/artifacts/models/torch/best_checkpoint.pt` exists and will not be overwritten

### 7. Determinism ⚠️ WARNING
- **Seed:** 42 (set via `set_seed()` → random, numpy, torch, torch.cuda)
- **CUDA determinism NOT fully configured:**
  - `torch.backends.cudnn.deterministic = False` (default)
  - `torch.backends.cudnn.benchmark = False` (default)
  - `CUBLAS_WORKSPACE_CONFIG` not set
  - `torch.use_deterministic_algorithms(True)` not set
- **DataLoader workers:** 0 (deterministic)
- **Impact:** GPU execution may produce slightly different results across runs
- **Recommendation:** Add deterministic settings or use CPU-only training for strict reproducibility

---

## Warnings & Limitations

1. **CUDA determinism not fully configured** - results may vary slightly across GPU runs
2. **Severe class imbalance** (fake-dominated) in all partitions
2. **Imbalanced partition sizes:** Train 4,577 >> Val 824 ≈ Test 828
3. **Group 0 dominates** (38/59 identities in training)
4. **YouTube-real excluded** (300 videos, no identity metadata)
5. **Official test list incompatible** with identity-disjoint splits
6. **Feature-to-video provenance** not embedded in .npy files (traceable via index/manifest only)

---

## Output Paths

| Artifact | Path |
|----------|------|
| Experiment directory | `backend/artifacts/models/torch/celebdf_identity_disjoint_20261023/` |
| Manifest | `celebdf_manifest.csv` |
| Feature index | `features_index.csv` |
| Excluded videos | `excluded_videos.csv` |
| Identity group assignment | `identity_group_assignment.csv` |
| Validation report (JSON) | `validation_report.json` |
| Validation report (MD) | `validation_report.md` |
| Pre-training report (JSON) | `pretraining_readiness_report.json` |
| Pre-training report (MD) | `pretraining_readiness_report.md` |

---

## Training Command

```bash
python scripts/train_torch_temporal.py \
  --index backend/artifacts/models/torch/celebdf_identity_disjoint_20261023/features_index.csv \
  --cache-dir backend/artifacts/models/torch/celebdf_retrain_20261023/celebdf_features \
  --output backend/artifacts/models/torch/celebdf_identity_disjoint_20261023/checkpoints \
  --epochs 20 --batch-size 32 --learning-rate 1e-3 --weight-decay 1e-4 --patience 5 --seed 42
```

---

## Verdict

**READY FOR TRAINING** - All mandatory checks pass. Proceed with training pending approval. The warnings (CUDA determinism, class imbalance) are documented and do not prevent training but should be considered when interpreting results.