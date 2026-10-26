import pandas as pd
import numpy as np
from pathlib import Path

# Load manifest and feature index
manifest_path = r"D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\celebdf_identity_disjoint_20261023\celebdf_manifest.csv"
index_path = r"D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\celebdf_identity_disjoint_20261023\features_index.csv"

manifest = pd.read_csv(manifest_path)
index = pd.read_csv(index_path)

# Merge for folder info
merged = manifest.merge(index[["relative_path", "feature_path", "status"]], on="relative_path", how="left")

print("=== FEATURE QUALITY AUDIT ===")

# Load samples from each split/folder
for split_name in ["train", "val", "test"]:
    for folder_name in ["Celeb-real", "Celeb-synthesis"]:
        subset = merged[(merged["split"] == split_name) & (merged["folder"] == folder_name) & (merged["status"] == "ok")]
        if len(subset) == 0:
            print(f"\n{split_name}/{folder_name}: NO SAMPLES")
            continue
        
        # Load first 3 samples
        samples_loaded = 0
        for _, row in subset.head(3).iterrows():
            features = np.load(row["feature_path"])
            print(f"\n{split_name}/{folder_name} - {row['relative_path']}:")
            print(f"  Shape: {features.shape}")
            print(f"  Dtype: {features.dtype}")
            print(f"  Finite: {np.isfinite(features).all()}")
            print(f"  Sequence length: {features.shape[0]}")
            print(f"  Feature dim: {features.shape[1]}")
            print(f"  Mean: {features.mean():.4f}, Std: {features.std():.4f}")
            print(f"  Min: {features.min():.4f}, Max: {features.max():.4f}")
            samples_loaded += 1
        
        # Check all samples in this split/folder
        all_finite = True
        shapes = []
        for _, row in subset.iterrows():
            features = np.load(row["feature_path"])
            if not np.isfinite(features).all():
                all_finite = False
                print(f"  NON-FINITE: {row['relative_path']}")
            if features.shape != (20, 1280):
                print(f"  WRONG SHAPE: {row['relative_path']} -> {features.shape}")
            shapes.append(features.shape)
        
        print(f"  All finite: {all_finite}")
        print(f"  All correct shape (20,1280): {all(s == (20, 1280) for s in shapes)}")
        print(f"  Total samples in {split_name}/{folder_name}: {len(subset)}")

# Also check YouTube-real (excluded)
print("\n=== YOUTUBE-REAL (EXCLUDED) ===")
yt_subset = merged[(merged["folder"] == "YouTube-real") & (merged["status"] == "ok")]
print(f"YouTube-real samples: {len(yt_subset)}")
if len(yt_subset) > 0:
    for _, row in yt_subset.head(3).iterrows():
        features = np.load(row["feature_path"])
        print(f"  {row['relative_path']}: shape={features.shape}, finite={np.isfinite(features).all()}")