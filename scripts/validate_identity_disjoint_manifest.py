import pandas as pd
import ast
from pathlib import Path

# Load new manifest
manifest_path = r"D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\celebdf_identity_disjoint_20261023\celebdf_manifest.csv"
df = pd.read_csv(manifest_path)

print("=== MANIFEST VALIDATION ===")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# Check required columns
required_cols = ["video_path", "relative_path", "folder", "label", "split"]
missing = set(required_cols) - set(df.columns)
print("Missing columns: " + str(missing))

# Check for duplicates
dup_paths = df["relative_path"].duplicated().sum()
print("Duplicate relative_paths: " + str(dup_paths))

# Check file existence
from pathlib import Path
missing_files = 0
for _, row in df.iterrows():
    if not Path(row["video_path"]).exists():
        missing_files += 1
print("Missing video files: " + str(missing_files))

# Label distribution
print("\nLabel distribution by split:")
print(df.groupby(["split", "label"]).size().unstack(fill_value=0))

# Folder distribution
print("\nFolder distribution by split:")
print(df.groupby(["split", "folder"]).size().unstack(fill_value=0))

# Identity overlap check
def extract_ids(path):
    filename = path.split("/")[-1]
    if filename.startswith("id"):
        ids = [int(p[2:]) for p in filename.split("_")[:2] if p.startswith("id")]
        return ids
    return []

df["identities"] = df["relative_path"].apply(extract_ids)

train_ids = set()
val_ids = set()
test_ids = set()

for _, row in df[df["split"] == "train"].iterrows():
    for id_ in row["identities"]:
        if id_ is not None:
            train_ids.add(id_)
for _, row in df[df["split"] == "val"].iterrows():
    for id_ in row["identities"]:
        if id_ is not None:
            val_ids.add(id_)
for _, row in df[df["split"] == "test"].iterrows():
    for id_ in row["identities"]:
        if id_ is not None:
            test_ids.add(id_)

print("\nTrain identities: " + str(len(train_ids)))
print("Val identities: " + str(len(val_ids)))
print("Test identities: " + str(len(test_ids)))
print("Train-Val overlap: " + str(len(train_ids & val_ids)))
print("Train-Test overlap: " + str(len(train_ids & test_ids)))
print("Val-Test overlap: " + str(len(val_ids & test_ids)))

# Source/target relationship check
synthesis = df[df["folder"] == "Celeb-synthesis"].copy()

def get_source_target(path):
    filename = path.split("/")[-1]
    parts = filename.split("_")
    if len(parts) >= 2 and parts[0].startswith("id") and parts[1].startswith("id"):
        return int(parts[0][2:]), int(parts[1][2:])
    return None, None

synthesis[["source_id", "target_id"]] = synthesis["relative_path"].apply(
    lambda x: pd.Series([int(x.split("/")[-1].split("_")[0][2:]), int(x.split("/")[-1].split("_")[1][2:])])
)

# Partition mapping
train_ids_set = set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38, 59, 60, 61])
val_ids_set = set([36, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48])
test_ids_set = set([49, 50, 51, 52, 53, 54, 55, 56, 57, 58])

def get_partition(id_):
    if id_ in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38, 59, 60, 61]:
        return "train"
    elif id_ in [36, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]:
        return "val"
    elif id_ in [49, 50, 51, 52, 53, 54, 55, 56, 57, 58]:
        return "test"
    return "unknown"

synthesis["source_partition"] = synthesis["relative_path"].apply(
    lambda x: "train" if int(x.split("/")[-1].split("_")[0][2:]) in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38, 59, 60, 61] else
    ("val" if int(x.split("/")[-1].split("_")[0][2:]) in [36, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48] else
    ("test" if int(x.split("/")[-1].split("_")[0][2:]) in [49, 50, 51, 52, 53, 54, 55, 56, 57, 58] else "unknown"))
)

synthesis["target_partition"] = synthesis["relative_path"].apply(
    lambda x: "train" if int(x.split("/")[-1].split("_")[1][2:]) in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38, 59, 60, 61] else
    ("val" if int(x.split("/")[-1].split("_")[1][2:]) in [36, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48] else
    ("test" if int(x.split("/")[-1].split("_")[1][2:]) in [49, 50, 51, 52, 53, 54, 55, 56, 57, 58] else "unknown"))
)

cross = synthesis[synthesis["source_partition"] != synthesis["target_partition"]]
print("\nCross-partition source/target pairs: " + str(len(cross)))

# Feature index validation
index_path = r"D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\celebdf_identity_disjoint_20261023\features_index.csv"
index_df = pd.read_csv(index_path)

print("\n=== FEATURE INDEX VALIDATION ===")
print("Shape: " + str(index_df.shape))
print("Status: " + str(index_df["status"].value_counts().to_dict()))

# Check feature file existence
from pathlib import Path
missing_features = 0
for _, row in index_df.iterrows():
    if not Path(row["feature_path"]).exists():
        missing_features += 1
print("Missing feature files: " + str(missing_features))

# Check mapping consistency
merged = df.merge(index_df[["relative_path", "feature_path", "label", "split", "status"]], on="relative_path", how="left", suffixes=("_manifest", "_index"))
label_mismatch = (merged["label_manifest"] != merged["label_index"]).sum()
split_mismatch = (merged["split_manifest"] != merged["split_index"]).sum()
print("Label mismatches: " + str(label_mismatch))
print("Split mismatches: " + str(split_mismatch))

print("\n=== VALIDATION COMPLETE ===")