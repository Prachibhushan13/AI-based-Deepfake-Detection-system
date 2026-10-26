import pandas as pd
import numpy as np
import ast
from pathlib import Path

# Load the identity groups
df = pd.read_csv(r"D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\celebdf_retrain_20261023\identity_groups.csv")

# Parse identities from string representation
df["identities"] = df["identities"].apply(ast.literal_eval)

# Assign identity groups to partitions
# Group 0 -> train (largest group)
# Group 2 -> val
# Group 3 -> test
# Group 1 -> val (small, all real)
# YouTube-real (group -1) -> excluded from identity-disjoint experiment

partition_map = {
    0: "train",
    1: "val",   # Group 1 (identity 36) - all real, small
    2: "val",   # Group 2
    3: "test",  # Group 3
    -1: "excluded"  # YouTube-real - excluded from identity-disjoint experiment
}

df["new_split"] = df["identity_group"].map(partition_map)

# Verify no identity overlap across partitions
for split_name in ["train", "val", "test"]:
    split_ids = set()
    for _, row in df[df["new_split"] == split_name].iterrows():
        for id_ in row["identities"]:
            split_ids.add(id_)
    print(f"{split_name} identities: {sorted(split_ids)}")

# Check overlap
train_ids = set()
for ids in df[df["new_split"] == "train"]["identities"]:
    train_ids.update(ids)
val_ids = set()
for ids in df[df["new_split"] == "val"]["identities"]:
    val_ids.update(ids)
test_ids = set()
for ids in df[df["new_split"] == "test"]["identities"]:
    test_ids.update(ids)

print(f"\nTrain-Val overlap: {len(train_ids & val_ids)}")
print(f"Train-Test overlap: {len(train_ids & test_ids)}")
print(f"Val-Test overlap: {len(val_ids & test_ids)}")

# Count videos per partition
print("\nPartition sizes:")
print(df.groupby(["new_split", "folder"]).size().unstack(fill_value=0))

# Merge with original manifest to get video_path
orig_manifest = pd.read_csv(r"D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\celebdf_retrain_20261023\celebdf_manifest.csv")
new_manifest = df[df["new_split"] != "excluded"].copy()
new_manifest = new_manifest.merge(
    orig_manifest[["relative_path", "video_path"]], 
    on="relative_path", 
    how="left"
)
new_manifest = new_manifest[["video_path", "relative_path", "folder", "label", "new_split"]].copy()
new_manifest = new_manifest.rename(columns={"new_split": "split"})

# Sort by split then relative_path
new_manifest = new_manifest.sort_values(["split", "relative_path"]).reset_index(drop=True)

print("\nNew manifest split distribution:")
print(new_manifest.groupby(["split", "label"]).size().unstack(fill_value=0))

# Save new manifest
output_dir = Path(r"D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\celebdf_identity_disjoint_20261023")
output_dir.mkdir(parents=True, exist_ok=True)

manifest_path = output_dir / "celebdf_manifest.csv"
new_manifest.to_csv(manifest_path, index=False)
print(f"\nSaved new manifest to {manifest_path}")

# Save excluded videos info
excluded = df[df["new_split"] == "excluded"]
excluded_path = output_dir / "excluded_videos.csv"
excluded[["relative_path", "folder", "label", "identities", "identity_group"]].to_csv(excluded_path, index=False)
print(f"Saved excluded videos to {excluded_path}")

# Save identity group assignment
group_assignment = pd.DataFrame({
    "identity_group": [0, 1, 2, 3, -1],
    "partition": ["train", "val", "val", "test", "excluded"],
    "identities": [
        "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38, 59, 60, 61]",
        "[36]",
        "[39, 40, 41, 42, 43, 44, 45, 46, 47, 48]",
        "[49, 50, 51, 52, 53, 54, 55, 56, 57, 58]",
        "unknown (YouTube-real)"
    ],
    "num_identities": [38, 1, 10, 10, 0],
    "num_videos": [4577, 10, 814, 828, 300],
    "notes": [
        "Largest connected component, assigned to train",
        "Identity 36 only in Celeb-real, assigned to val",
        "Assigned to val",
        "Assigned to test",
        "No identity metadata, excluded from identity-disjoint experiment"
    ]
})
group_assignment_path = output_dir / "identity_group_assignment.csv"
group_assignment.to_csv(group_assignment_path, index=False)
print(f"Saved group assignment to {group_assignment_path}")

# Print summary
print("\n=== PARTITION SUMMARY ===")
for split_name in ["train", "val", "test"]:
    subset = new_manifest[new_manifest["split"] == split_name]
    print(f"\n{split_name.upper()}: {len(subset)} videos")
    print(f"  Real: {(subset['label']==0).sum()}, Fake: {(subset['label']==1).sum()}")
    print(f"  Celeb-real: {(subset['folder']=='Celeb-real').sum()}")
    print(f"  Celeb-synthesis: {(subset['folder']=='Celeb-synthesis').sum()}")