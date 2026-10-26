import pandas as pd

# Load official test list
test_list_path = r"C:\Users\laksh\Downloads\AI-based-Deepfake-Detection-system-main\AI-based-Deepfake-Detection-system-main\datasets\Celeb-DF-v2\List_of_testing_videos.txt"
official_test = set()
official_labels = {}
with open(test_list_path, "r") as f:
    for line in f:
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            label, path = parts
            official_test.add(path)
            official_labels[path] = int(label)

print("Official test list entries: " + str(len(official_test)))

# Load new manifest
df = pd.read_csv(r"D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\celebdf_identity_disjoint_20261023\celebdf_manifest.csv")

# Check overlap with official test list
df["in_official_test"] = df["relative_path"].isin(official_test)

print("New manifest videos in official test list: " + str(df["in_official_test"].sum()))
print("By split:")
print(df.groupby(["split", "in_official_test"]).size().unstack(fill_value=0))

# Check if official test list videos are in our test partition
official_in_our_test = df[(df["in_official_test"]) & (df["split"] == "test")]
print("\nOfficial test videos in our test partition: " + str(len(official_in_our_test)))

official_in_our_train = df[(df["in_official_test"]) & (df["split"] == "train")]
print("Official test videos in our train partition: " + str(len(official_in_our_train)))

official_in_our_val = df[(df["in_official_test"]) & (df["split"] == "val")]
print("Official test videos in our val partition: " + str(len(official_in_our_val)))

# Check label consistency (official test list uses 1=real, 0=fake)
# Our manifest uses 0=real, 1=fake
mismatch = 0
for _, row in df[df["in_official_test"]].iterrows():
    official_label = official_labels.get(row["relative_path"])
    if official_label is not None:
        # Official: 1=real, 0=fake; Our: 0=real, 1=fake
        our_label = row["label"]
        expected_our = 0 if official_label == 1 else 1
        if our_label != expected_our:
            mismatch += 1
            print("Label mismatch: " + row["relative_path"] + " official=" + str(official_label) + " our=" + str(our_label))

print("\nLabel mismatches: " + str(mismatch))