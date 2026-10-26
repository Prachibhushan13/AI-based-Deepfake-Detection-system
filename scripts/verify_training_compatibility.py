import pandas as pd
from pathlib import Path

# Load new feature index
index_path = r"D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\celebdf_identity_disjoint_20261023\features_index.csv"
index = pd.read_csv(index_path)

print("Feature index columns:", index.columns.tolist())
print("Required columns check:")
required = {"relative_path", "feature_path", "label", "split", "status"}
missing = required - set(index.columns)
print("Missing: " + str(missing))

# Check split values
print("\nSplit values:", index["split"].unique())
print("Split counts:")
print(index["split"].value_counts())

# Check status
print("\nStatus:", index["status"].unique())

# Simulate training script loading
usable = index[index["status"] == "ok"].copy()
print("\nUsable entries: " + str(len(usable)))

train_frame = usable[usable["split"] == "train"]
val_frame = usable[usable["split"] == "val"]
test_frame = usable[usable["split"] == "test"]

print("Train: " + str(len(train_frame)) + " (real=" + str((train_frame["label"]==0).sum()) + ", fake=" + str((train_frame["label"]==1).sum()) + ")")
print("Val: " + str(len(val_frame)) + " (real=" + str((val_frame["label"]==0).sum()) + ", fake=" + str((val_frame["label"]==1).sum()) + ")")
print("Test: " + str(len(test_frame)) + " (real=" + str((test_frame["label"]==0).sum()) + ", fake=" + str((test_frame["label"]==1).sum()) + ")")

# Check feature paths exist
from pathlib import Path
missing = 0
for _, row in index.iterrows():
    if not Path(row["feature_path"]).exists():
        print("Missing: " + row["feature_path"])
        missing += 1
print("\nMissing feature files: " + str(missing))