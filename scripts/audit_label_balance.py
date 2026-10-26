import pandas as pd

# Load manifest
manifest_path = r"D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\celebdf_identity_disjoint_20261023\celebdf_manifest.csv"
manifest = pd.read_csv(manifest_path)

print("=== LABEL BALANCE AUDIT ===")

for split_name in ["train", "val", "test"]:
    subset = manifest[manifest["split"] == split_name]
    n = len(subset)
    real = (subset["label"] == 0).sum()
    fake = (subset["label"] == 1).sum()
    balance = real / fake if fake > 0 else float("inf")
    
    print(f"\n{split_name.upper()}:")
    print(f"  Total: {n}")
    print(f"  Real (0): {real}")
    print(f"  Fake (1): {fake}")
    print(f"  Real/Fake ratio: {balance:.4f}")
    print(f"  Fake proportion: {fake/n:.4f}")

# Check identity counts per partition
def extract_ids(path):
    filename = path.split("/")[-1]
    if filename.startswith("id"):
        ids = [int(p[2:]) for p in filename.split("_")[:2] if p.startswith("id")]
        return ids
    return []

manifest["identities"] = manifest["relative_path"].apply(lambda x: [int(p[2:]) for p in x.split("/")[-1].split("_")[:2] if p.startswith("id")])

for split_name in ["train", "val", "test"]:
    subset = manifest[manifest["split"] == split_name]
    ids = set()
    for ids_list in subset["identities"]:
        for id_ in ids_list:
            ids.add(id_)
    print(f"\n{split_name} identities: {len(ids)} - {sorted(ids)}")

# Check class balance per identity group
print("\n=== CLASS BALANCE PER IDENTITY GROUP ===")
for split_name in ["train", "val", "test"]:
    subset = manifest[manifest["split"] == split_name]
    print(f"\n{split_name}:")
    for folder in ["Celeb-real", "Celeb-synthesis"]:
        sub = subset[subset["folder"] == folder]
        print(f"  {folder}: {len(sub)} videos")

# Check if any partition lacks a class
print("\n=== CLASS PRESENCE CHECK ===")
for split_name in ["train", "val", "test"]:
    subset = manifest[manifest["split"] == split_name]
    has_real = (subset["label"] == 0).any()
    has_fake = (subset["label"] == 1).any()
    print(f"{split_name}: has_real={has_real}, has_fake={has_fake}")

# Check identity counts
print("\n=== IDENTITY COUNT PER PARTITION ===")
for split_name in ["train", "val", "test"]:
    subset = manifest[manifest["split"] == split_name]
    ids = set()
    for ids_list in subset["identities"]:
        for id_ in ids_list:
            ids.add(id_)
    print(f"{split_name}: {len(ids)} unique identities")
    
    # Check minimum videos per identity
    id_counts = {}
    for _, row in subset.iterrows():
        for id_ in row["identities"]:
            id_counts[id_] = id_counts.get(id_, 0) + 1
    min_videos = min(id_counts.values()) if id_counts else 0
    max_videos = max(id_counts.values()) if id_counts else 0
    print(f"  Min videos per identity: {min_videos}, Max: {max_videos}")