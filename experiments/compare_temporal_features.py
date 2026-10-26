from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "datasets/celebdf_features/features_index.csv"

df = pd.read_csv(INDEX)
df = df[
    (df["status"] == "ok")
    & df["split"].isin(["train", "val"])
].copy().reset_index(drop=True)

features = {
    "mean": [],
    "std": [],
    "mean_std": [],
    "mean_temporal_diff": [],
}

labels = []
splits = []

for row in df.itertuples(index=False):
    path = Path(str(row.feature_path))
    if not path.is_absolute() or not path.exists():
        path = ROOT / "datasets/celebdf_features" / path.name

    a = np.load(path).astype(np.float32)

    if a.shape != (20, 1280) or not np.isfinite(a).all():
        raise ValueError(f"Invalid features: {path}")

    mean = a.mean(axis=0)
    std = a.std(axis=0)
    temporal_diff = np.abs(np.diff(a, axis=0)).mean(axis=0)

    features["mean"].append(mean)
    features["std"].append(std)
    features["mean_std"].append(np.concatenate([mean, std]))
    features["mean_temporal_diff"].append(
        np.concatenate([mean, temporal_diff])
    )

    labels.append(int(row.label))
    splits.append(row.split)

y = np.asarray(labels)
splits = np.asarray(splits)
train = splits == "train"
val = splits == "val"

print(f"Train: {train.sum()} | Validation: {val.sum()}")
print("Official test split excluded.")
print("\nResults:")

for name, values in features.items():
    X = np.asarray(values, dtype=np.float32)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        ),
    )

    model.fit(X[train], y[train])
    scores = model.predict_proba(X[val])[:, 1]

    thresholds = np.arange(0.05, 0.951, 0.005)
    results = [
        (balanced_accuracy_score(y[val], scores >= t), t)
        for t in thresholds
    ]
    best_bal, best_threshold = max(results)

    print(
        f"{name:24s} "
        f"AUC={roc_auc_score(y[val], scores):.4f}  "
        f"Best validation balanced accuracy={best_bal:.4f}  "
        f"Threshold={best_threshold:.3f}"
    )
