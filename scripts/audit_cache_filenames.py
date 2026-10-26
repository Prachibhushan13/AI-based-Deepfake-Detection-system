import pandas as pd
import hashlib
from pathlib import Path

# Load manifest and feature index
manifest_path = r"D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\celebdf_identity_disjoint_20261023\celebdf_manifest.csv"
index_path = r"D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\celebdf_identity_disjoint_20261023\features_index.csv"

manifest = pd.read_csv(manifest_path)
index = pd.read_csv(index_path)

print("Manifest shape:", manifest.shape)
print("Index shape:", index.shape)
print("Index columns:", index.columns.tolist())
print("Index status:", index["status"].value_counts().to_dict())

# Check cache filename algorithm
def cache_name(relative_path):
    key = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:24]
    return key + ".npy"

# Verify each row
mismatches = 0
for _, row in index.iterrows():
    expected_name = cache_name(row["relative_path"])
    actual_name = Path(row["feature_path"]).name
    if expected_name != actual_name:
        print("MISMATCH: " + row["relative_path"] + " -> expected " + expected_name + ", got " + actual_name)
        mismatches += 1

print("\nCache filename mismatches: " + str(mismatches))

# Check feature path resolution
missing = 0
for _, row in index.iterrows():
    if not Path(row["feature_path"]).exists():
        print("MISSING: " + row["feature_path"])
        missing += 1
print("Missing feature files: " + str(missing))

# Check manifest-index alignment
merged = manifest.merge(index[["relative_path", "feature_path", "label", "split", "status"]], on="relative_path", how="left", suffixes=("_m", "_i"))
label_mismatch = (merged["label_m"] != merged["label_i"]).sum()
split_mismatch = (merged["split_m"] != merged["split_i"]).sum()
print("Label mismatches: " + str(label_mismatch))
print("Split mismatches: " + str(split_mismatch))