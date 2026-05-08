from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DatasetConfig:
    root: Path
    split_csv: Path


def build_manifest(dataset_root: str, output_csv: str) -> pd.DataFrame:
    root = Path(dataset_root)
    rows = []
    for label_dir, label in [("real", 0), ("fake", 1)]:
        for suffix in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
            for video_path in (root / label_dir).rglob(suffix):
                rows.append(
                    {
                        "video_path": str(video_path),
                        "label": label,
                        "dataset": root.name,
                    }
                )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_csv, index=False)
    return manifest


def create_split_manifest(manifest_path: str, output_csv: str) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    train_df, temp_df = train_test_split(
        manifest,
        test_size=0.3,
        stratify=manifest["label"],
        random_state=42,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["label"],
        random_state=42,
    )
    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")
    split_manifest = pd.concat([train_df, val_df, test_df], ignore_index=True)
    split_manifest.to_csv(output_csv, index=False)
    return split_manifest
