import pandas as pd
import re

df = pd.read_csv(r"D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\celebdf_retrain_20261023\celebdf_manifest.csv")

def extract_identities(path):
    filename = path.split("/")[-1]
    if filename.startswith("id"):
        ids = re.findall(r"id(\d+)", filename)
        return [int(x) for x in ids]
    else:
        return []

df["identities"] = df["relative_path"].apply(extract_identities)

# Build identity graph from Celeb-synthesis
synthesis = df[df["folder"] == "Celeb-synthesis"]
edges = []
for _, row in synthesis.iterrows():
    ids = row["identities"]
    if len(ids) == 2:
        edges.append((ids[0], ids[1]))

# Find connected components (identity groups)
from collections import defaultdict

graph = defaultdict(set)
for u, v in edges:
    graph[u].add(v)
    graph[v].add(u)

# Add all identities as nodes (including those only in Celeb-real)
all_identities = set()
for ids in df["identities"]:
    all_identities.update(ids)

# Find connected components
visited = set()
components = []
for node in all_identities:
    if node not in visited:
        stack = [node]
        component = set()
        while stack:
            n = stack.pop()
            if n not in visited:
                visited.add(n)
                component.add(n)
                stack.extend(graph[n] - visited)
        components.append(component)

print("Number of identity groups: " + str(len(components)))
for i, comp in enumerate(components):
    print("  Group " + str(i) + ": " + str(sorted(comp)) + " (size=" + str(len(comp)) + ")")

# Now count videos per group
df["identity_group"] = None
for i, comp in enumerate(components):
    for idx, row in df.iterrows():
        if any(id_ in comp for id_ in row["identities"]):
            df.at[idx, "identity_group"] = i

# YouTube-real gets group -1
df.loc[df["folder"] == "YouTube-real", "identity_group"] = -1

print("\nVideos per identity group:")
group_counts = df.groupby(["identity_group", "folder"]).size().unstack(fill_value=0)
print(group_counts)

print("\nGroup sizes:")
for i, comp in enumerate(components):
    subset = df[df["identity_group"] == i]
    print("  Group " + str(i) + ": " + str(len(subset)) + " videos, identities=" + str(sorted(comp)))
    print("    Celeb-real: " + str((subset["folder"]=="Celeb-real").sum()))
    print("    Celeb-synthesis: " + str((subset["folder"]=="Celeb-synthesis").sum()))

# YouTube-real
yt = df[df["identity_group"] == -1]
print("\nYouTube-real (group -1): " + str(len(yt)) + " videos")

# Save the identity group mapping
df[["relative_path", "folder", "label", "split", "identities", "identity_group"]].to_csv(
    r"D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\celebdf_retrain_20261023\identity_groups.csv",
    index=False
)
print("\nSaved identity_groups.csv")