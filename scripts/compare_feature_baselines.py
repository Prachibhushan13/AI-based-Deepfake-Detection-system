
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def extract_features(index: pd.DataFrame):
    mean_features = []
    temporal_changes = []
    labels = []

    for row in index.itertuples(index=False):
        features = np.load(row.feature_path).astype(np.float32)

        if features.shape != (20, 1280):
            raise ValueError(
                f"Unexpected shape {features.shape}: {row.feature_path}"
            )

        if not np.isfinite(features).all():
            raise ValueError(f"Non-finite values: {row.feature_path}")

        mean_features.append(features.mean(axis=0))

        changes = np.linalg.norm(
            np.diff(features, axis=0), axis=1
        )
        temporal_changes.append([changes.mean()])

        labels.append(int(row.label))

    return (
        np.asarray(mean_features),
        np.asarray(temporal_changes),
        np.asarray(labels),
    )


def evaluate(name, x_train, y_train, x_val, y_val):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
    )
    model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_val)[:, 1]
    auc = roc_auc_score(y_val, probabilities)

    # Select the threshold using validation data only.
    thresholds = np.linspace(0.05, 0.95, 181)
    candidates = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        score = balanced_accuracy_score(y_val, predictions)
        candidates.append((score, threshold))

    best_score, best_threshold = max(candidates)
    predictions = (probabilities >= best_threshold).astype(int)

    print(f"\n{'=' * 60}")
    print(name)
    print(f"Validation ROC-AUC: {auc:.4f}")
    print(f"Selected threshold: {best_threshold:.3f}")
    print(f"Validation accuracy: {accuracy_score(y_val, predictions):.4f}")
    print(f"Validation balanced accuracy: {best_score:.4f}")
    print(f"Validation F1 (fake class): {f1_score(y_val, predictions):.4f}")
    print("Confusion matrix (rows=true, columns=predicted; 0=real, 1=fake):")
    print(confusion_matrix(y_val, predictions))
    print(classification_report(
        y_val,
        predictions,
        labels=[0, 1],
        target_names=["REAL", "FAKE"],
        zero_division=0,
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index",
        default="datasets/celebdf_features/features_index.csv",
    )
    args = parser.parse_args()

    index = pd.read_csv(args.index)
    index = index[
        (index["status"] == "ok")
        & (index["split"].isin(["train", "val"]))
    ].copy()

    if index.empty:
        raise ValueError("No train/validation feature rows found.")

    if index.duplicated("relative_path").any():
        raise ValueError("Duplicate video paths found in the index.")

    print("Loading train/validation features only...")
    mean_features, temporal_changes, labels = extract_features(index)

    train_mask = index["split"].to_numpy() == "train"
    val_mask = index["split"].to_numpy() == "val"

    y_train = labels[train_mask]
    y_val = labels[val_mask]

    x_train_mean = mean_features[train_mask]
    x_val_mean = mean_features[val_mask]

    x_train_temporal = temporal_changes[train_mask]
    x_val_temporal = temporal_changes[val_mask]

    x_train_combined = np.concatenate(
        [x_train_mean, x_train_temporal], axis=1
    )
    x_val_combined = np.concatenate(
        [x_val_mean, x_val_temporal], axis=1
    )

    print(f"Training videos: {len(y_train)}")
    print(f"Validation videos: {len(y_val)}")
    print("The test split is excluded.")

    evaluate(
        "Baseline A: mean frame features",
        x_train_mean, y_train, x_val_mean, y_val,
    )
    evaluate(
        "Baseline B: temporal variation only",
        x_train_temporal, y_train, x_val_temporal, y_val,
    )
    evaluate(
        "Baseline C: mean features + temporal variation",
        x_train_combined, y_train, x_val_combined, y_val,
    )


if __name__ == "__main__":
    main()
