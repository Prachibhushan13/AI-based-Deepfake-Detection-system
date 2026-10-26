# Identity-Disjoint Celeb-DF Experiment Validation Report

**Experiment:** `celebdf_identity_disjoint_20261023`
**Date:** 2026-10-23
**Status:** ✅ READY FOR TRAINING

---

## Summary

This experiment creates an identity-disjoint train/validation/test split for the Celeb-DF dataset, ensuring zero known identity overlap across partitions. The YouTube-real subset (300 videos) is excluded due to lack of identity metadata.

---

## Manifest Summary

| Split | Total | Real | Fake | Celeb-real | Celeb-synthesis |
|-------|-------|------|------|------------|-----------------|
| Train | 4,577 | 389 | 4,188 | 389 | 4,188 |
| Val   | 824   | 101 | 723   | 101 | 723   |
| Test  | 828   | 100 | 728   | 100 | 728   |
| **Total** | **6,229** | **590** | **5,639** | **590** | **5,639** |

**Excluded:** YouTube-real (300 videos) - no identity metadata

---

## Identity Groups & Partition Assignment

| Group | Identities | # Identities | Videos | Partition | Notes |
|-------|------------|--------------|--------|-----------|-------|
| 0 | 0-13, 16-13, 16-17, 19-38, 59-61 | 38 | 4,577 | Train | Largest connected component |
| 1 | 36 | 1 | 10 | Val | Only in Celeb-real |
| 2 | 39-48 | 10 | 814 | Val | |
| 3 | 49-58 | 10 | 828 | Test | |
| -1 | unknown | 0 | 300 | Excluded | YouTube-real, no identity metadata |

---

## Validation Results

### ✅ Identity Overlap Checks
- Train-Val overlap: **0 identities**
- Train-Test overlap: **0 identities**
- Val-Test overlap: **0 identities**

### ✅ Source/Target Relationships
- Cross-partition source/target pairs: **0**
- All Celeb-synthesis videos have both source and target identities in the same partition
- Train synthesis: 4,188 videos
- Val synthesis: 723 videos
- Test synthesis: 728 videos

### ✅ Manifest Validation
- Shape: 6,229 × 5
- Missing columns: 0
- Duplicate relative paths: 0
- Missing video files: 0
- Label mismatches (manifest vs index): 0
- Split mismatches: 0

### ✅ Feature Index Validation
- Total entries: 6,229
- All status: "ok"
- Missing feature files: 0
- Label mismatches: 0
- Split mismatches: 0

### ✅ Training Script Compatibility
- Required columns present: ✅ (relative_path, feature_path, label, split, status)
- All entries status "ok": ✅ (6,229/6,229)
- Split values match expected: ✅ (train, val, test)
- All feature files exist: ✅ (6,229/6,229)
- Train/Val/Test splits match expected counts

---

## Official Test List Compatibility

| Metric | Value |
|--------|-------|
| Official test list entries | 518 |
| In new manifest | 448 (70 YouTube-real excluded) |
| In our test partition | 43 |
| In train partition | 368 |
| In val partition | 37 |
| Label mismatches | 0 |

**Note:** The official Celeb-DF test list is **incompatible** with identity-disjoint splits due to severe identity overlap with training data. It is not used for training or model selection in this experiment.

---

## Excluded Data

| Dataset | Videos | Reason |
|---------|--------|--------|
| YouTube-real | 300 | No identity metadata available; cannot verify identity independence |

---

## Output Paths

| Artifact | Path |
|----------|------|
| Experiment directory | `backend/artifacts/models/torch/celebdf_identity_disjoint_20261023/` |
| Manifest | `celebdf_manifest.csv` |
| Feature index | `features_index.csv` |
| Excluded videos | `excluded_videos.csv` |
| Identity group assignment | `identity_group_assignment.csv` |
| Identity groups (source) | `../celebdf_retrain_20261023/identity_groups.csv` |

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

## Notes & Limitations

1. **Zero known identity overlap** across train/val/test partitions
2. **No known source/target grouping conflicts** across partitions
3. **Deterministic split generation** via connected components of identity graph
4. **No external test data** used for training or model selection
5. **YouTube-real excluded** due to lack of identity metadata
6. **Official test list not used** due to identity overlap with training data
7. **Training script compatible** with new feature index and split column

---

## Verdict

✅ **READY FOR TRAINING** - All mandatory requirements met:
- Zero known identity overlap across train/val/test
- No known source/target grouping conflicts across partitions
- Deterministic split generation
- No external test data used for training or model selection
- All validation checks pass