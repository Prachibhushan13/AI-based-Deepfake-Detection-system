from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "datasets/celebdf_features/features_index.csv"
OUT = ROOT / "experiments/results"

def main():
    df = pd.read_csv(INDEX)
    df = df[
        (df["status"] == "ok")
        & df["split"].isin(["train", "val"])
    ].copy()

    if df["relative_path"].duplicated().any():
        raise ValueError("Duplicate video paths found")

    train_mask = df["split"].to_numpy() == "train"
    val_mask = df["split"].to_numpy() == "val"

    X, y = [], []
    for row in df.itertuples(index=False):
        p = Path(str(row.feature_path))
        if not p.is_absolute() or not p.exists():
            p = ROOT / "datasets/celebdf_features" / p.name

        a = np.load(p).astype(np.float32)
        if a.shape != (20, 1280) or not np.isfinite(a).all():
            raise ValueError(f"Invalid feature array: {p}")

        # Average frame embeddings into one vector per video.
        X.append(a.mean(axis=0))
        y.append(int(row.label))

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    X_train, X_val = X[train_mask], X[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]

    print(f"Train: {len(y_train)} | Validation: {len(y_val)}")
    print("Official test split excluded.")
    print("Train labels:", dict(zip(*np.unique(y_train, return_counts=True))))

    models = {
        "logistic_balanced": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=42,
            ),
        ),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42,
        ),
        "extra_trees_balanced": ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced",
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42,
        ),
    }

    results = []
    for name, model in models.items():
        print(f"\nFitting {name}...", flush=True)
        model.fit(X_train, y_train)
        scores = model.predict_proba(X_val)[:, 1]

        # Select threshold on validation data only.
        candidates = []
        for threshold in np.arange(0.05, 0.951, 0.005):
            pred = (scores >= threshold).astype(int)
            candidates.append(
                (balanced_accuracy_score(y_val, pred), threshold)
            )
        best_bal, best_t = max(candidates)
        pred = (scores >= best_t).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_val, pred, labels=[0, 1]
        ).ravel()

        row = {
            "model": name,
            "roc_auc": float(roc_auc_score(y_val, scores)),
            "balanced_accuracy": float(best_bal),
            "threshold": float(best_t),
            "accuracy": float(accuracy_score(y_val, pred)),
            "real_recall": float(tn / (tn + fp)),
            "fake_recall": float(tp / (tp + fn)),
            "TN": int(tn), "FP": int(fp),
            "FN": int(fn), "TP": int(tp),
        }
        results.append(row)
        print(json.dumps(row, indent=2))

    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "classifier_comparison.json"
    result_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to: {result_path}")
    print("Existing checkpoints and feature cache were not modified.")

if __name__ == "__main__":
    main()
