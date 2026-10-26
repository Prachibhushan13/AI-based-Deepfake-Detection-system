import pandas as pd
import hashlib
from pathlib import Path

# Load new manifest
manifest_path = r"D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\celebdf_identity_disjoint_20261023\celebdf_manifest.csv"
df = pd.read_csv(manifest_path)

# Load original feature index to get feature paths
orig_index_path = r"D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\celebdf_retrain_20261023\features_index.csv"
orig_index = pd.read_csv(orig_index_path)

# Create feature path mapping
def cache_name(relative_path):
    key = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:24]
    return key + ".npy"

# Build feature index for new manifest
feature_index_rows = []
cache_dir = Path(r"D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\celebdf_retrain_20261023\celebdf_features")

for _, row in df.iterrows():
    rel_path = row["relative_path"]
    cache_file = cache_dir / cache_name(rel_path)
    
    # Check if feature file exists
    if cache_file.exists():
        status = "ok"
        error = ""
    else:
        status = "missing"
        error = "Feature file not found"
    
    feature_index_rows.append({
        "relative_path": rel_path,
        "feature_path": str(cache_file),
        "label": row["label"],
        "split": row["split"],
        "status": status,
        "error": error
    })

feature_index = pd.DataFrame(feature_index_rows)

# Save feature index
output_dir = Path(r"D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\celebdf_identity_disjoint_20261023")
index_path = output_dir / "features_index.csv"
feature_index.to_csv(index_path, index=False)
print("Saved feature index to " + str(index_path))

print("\nFeature index status:")
print(feature_index["status"].value_counts())

print("\nSplit distribution:")
print(feature_index.groupby(["split", "status"]).size().unstack(fill_value=0))