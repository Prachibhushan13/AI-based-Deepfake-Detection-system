#!/usr/bin/env python3
"""
Create identity-disjoint manifest for Celeb-DF dataset.

This script creates a new manifest with identity-disjoint splits (train/val/test)
that ensures no identity crosses splits. It handles:
- Celeb-real: id{number}_{number}.mp4 -> identity = id{number}
- Celeb-synthesis: id{source}_id{target}_{number}.mp4 -> tracks BOTH source and target identities
- YouTube-real: {number}.mp4 -> no clear identity, treated as unique per video

The script creates identity-disjoint splits (41 train / 9 val / 9 test identities)
while preserving the official test list where possible.
"""

import argparse
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Official test list path (may not exist locally)
OFFICIAL_TEST_LIST = Path("datasets/Celeb-DF-v2/List_of_testing_videos.txt")


def parse_identity_from_filename(folder: str, filename: str) -> Set[str]:
    """
    Parse identity/identities from filename based on folder type.
    Returns a set of identities (can be multiple for Celeb-synthesis).
    """
    stem = Path(filename).stem
    
    if folder == "Celeb-real":
        # Format: id{number}_{number}.mp4 -> identity = id{number}
        # e.g., id0_0001.mp4 -> id0
        if stem.startswith("id") and "_" in stem:
            identity = stem.split("_")[0]
            return {identity}
        return {stem}
    
    elif folder == "Celeb-synthesis":
        # Format: id{source}_id{target}_{number}.mp4
        # e.g., id0_id16_0003.mp4 -> identities: id0 (source), id16 (target)
        # Both source and target identities must be tracked for leakage prevention
        parts = stem.split("_")
        identities = set()
        for part in parts:
            if part.startswith("id") and part[2:].isdigit():
                identities.add(part)
        return identities
    
    elif folder == "YouTube-real":
        # No clear identity, treat each video as unique
        return {f"youtube_{Path(filename).stem}"}
    
    return {Path(filename).stem}


def load_official_test_list(test_list_path: Path) -> Set[str]:
    """Load official test list from Celeb-DF."""
    official_test = set()
    if test_list_path.exists():
        for line in test_list_path.read_text(encoding="utf-8").splitlines():
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                relative_path = parts[1].strip()
                official_test.add(relative_path)
    return official_test


def load_existing_manifest(manifest_path: Path) -> pd.DataFrame:
    """Load existing manifest and parse identities from relative_path."""
    df = pd.read_csv(manifest_path)
    
    # Parse identities from relative_path
    def parse_identities(row):
        folder = row["folder"]
        filename = Path(row["relative_path"]).name
        stem = Path(filename).stem
        
        if folder == "Celeb-real":
            if "_" in stem:
                identity = stem.split("_")[0]
                return [identity]
            return [stem]
        elif folder == "Celeb-synthesis":
            parts = stem.split("_")
            identities = set()
            for part in parts:
                if part.startswith("id") and part[2:].isdigit():
                    identities.add(part)
            return list(identities)
        elif folder == "YouTube-real":
            return [f"youtube_{Path(filename).stem}"]
        return [Path(filename).stem]
    
    df["identities"] = df.apply(parse_identities, axis=1)
    df["primary_identity"] = df["identities"].apply(lambda x: min(x) if x else "unknown")
    
    # Ensure required columns exist
    if "video_path" not in df.columns:
        df["video_path"] = df["relative_path"]
    if "folder" not in df.columns:
        df["folder"] = df["relative_path"].apply(lambda x: x.split("/")[0])
    if "label" not in df.columns:
        df["label"] = df["folder"].map({"Celeb-real": 0, "YouTube-real": 0, "Celeb-synthesis": 1})
    if "split" not in df.columns:
        df["split"] = "pending"
    
    return df


def create_identity_disjoint_splits(
    df: pd.DataFrame,
    official_test: Set[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> pd.DataFrame:
    """
    Create identity-disjoint splits.
    
    Strategy:
    1. First, assign official test videos to test split (preserving official test list)
    2. Collect all unique identities from remaining videos
    3. Split identities into train/val/test groups
    4. Assign videos to splits based on their identities
    7. Ensure no identity crosses splits
    """
    import random
    random.seed(seed)
    
    df = df.copy()
    df["in_official_test"] = df["relative_path"].isin(official_test)
    
    # Separate official test videos
    official_test_df = df[df["in_official_test"]].copy()
    remaining_df = df[~df["in_official_test"]].copy()
    
    # Collect all identities from remaining videos
    all_identities = set()
    for identities in remaining_df["identities"]:
        all_identities.update(identities)
    
    all_identities = list(all_identities)
    random.shuffle(all_identities)
    
    # Calculate split sizes (aiming for ~41/9/9 identities)
    n_identities = len(all_identities)
    n_train = max(1, int(n_identities * train_ratio))
    n_val = max(1, int(n_identities * val_ratio))
    n_test = n_identities - n_train - n_val
    
    train_identities = set(all_identities[:n_train])
    val_identities = set(all_identities[n_train:n_train + n_val])
    test_identities = set(all_identities[n_train + n_val:])
    
    print(f"Identity split: train={len(train_identities)}, val={len(val_identities)}, test={len(test_identities)}")
    
    # Assign splits based on identities
    # Priority: test > val > train (if a video has identities in multiple splits, assign to highest priority)
    def assign_split(identities: List[str]) -> str:
        identity_set = set(identities)
        if identity_set & test_identities:
            return "test"
        elif identity_set & val_identities:
            return "val"
        elif identity_set & train_identities:
            return "train"
        return "train"  # default
    
    # Assign splits for remaining videos
    remaining_df["split"] = remaining_df["identities"].apply(assign_split)
    official_test_df["split"] = "test"
    
    # Combine
    result = pd.concat([remaining_df, official_test_df], ignore_index=True)
    
    # Verify no identity overlap
    for split in ["train", "val", "test"]:
        split_ids = set()
        for identities in result[result["split"] == split]["identities"]:
            split_ids.update(identities)
        print(f"{split} identities: {len(split_ids)}")
    
    # Check overlaps
    train_ids = set().union(*[set(ids) for ids in result[result["split"]=="train"]["identities"]])
    val_ids = set().union(*[set(ids) for ids in result[result["split"]=="val"]["identities"]])
    test_ids = set().union(*[set(ids) for ids in result[result["split"]=="test"]["identities"]])
    
    print(f"Train-Val overlap: {len(train_ids & val_ids)}")
    print(f"Train-Test overlap: {len(train_ids & test_ids)}")
    print(f"Val-Test overlap: {len(val_ids & test_ids)}")
    
    return result


def validate_manifest(df: pd.DataFrame, dataset_root: Path) -> Dict:
    """Validate the manifest and return validation report."""
    report = {
        "total_videos": int(len(df)),
        "required_columns": ["video_path", "relative_path", "folder", "label", "split", "identities"],
        "checks": {}
    }
    
    # Check required columns
    required_cols = ["video_path", "relative_path", "folder", "label", "split", "identities"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    report["checks"]["required_columns"] = {
        "passed": len(missing_cols) == 0,
        "missing": missing_cols
    }
    
    # Check valid labels
    valid_labels = {0, 1}
    invalid_labels = set(df["label"].unique()) - valid_labels
    report["checks"]["valid_labels"] = {
        "passed": len(invalid_labels) == 0,
        "invalid": list(invalid_labels)
    }
    
    # Check valid splits
    valid_splits = {"train", "val", "test"}
    invalid_splits = set(df["split"].unique()) - valid_splits
    report["checks"]["valid_splits"] = {
        "passed": len(invalid_splits) == 0,
        "invalid": list(invalid_splits)
    }
    
    # Check duplicate videos
    dup_count = int(df["relative_path"].duplicated().sum())
    report["checks"]["duplicate_videos"] = {
        "passed": dup_count == 0,
        "count": int(dup_count)
    }
    
    # Check identity overlap across splits
    train_ids = set().union(*[set(ids) for ids in df[df["split"]=="train"]["identities"]])
    val_ids = set().union(*[set(ids) for ids in df[df["split"]=="val"]["identities"]])
    test_ids = set().union(*[set(ids) for ids in df[df["split"]=="test"]["identities"]])
    
    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids
    
    report["checks"]["identity_overlap"] = {
        "passed": len(train_ids & val_ids) == 0 and len(train_ids & test_ids) == 0 and len(val_ids & test_ids) == 0,
        "train_val_overlap": int(len(train_ids & val_ids)),
        "train_test_overlap": int(len(train_ids & test_ids)),
        "val_test_overlap": int(len(val_ids & test_ids)),
        "train_identities": int(len(train_ids)),
        "val_identities": int(len(val_ids)),
        "test_identities": int(len(test_ids))
    }
    
    # Check class distribution
    split_label_counts = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    report["checks"]["class_distribution"] = {k: {str(k2): int(v2) for k2, v2 in v.items()} for k, v in split_label_counts.to_dict().items()}
    
    # Check feature paths if they exist
    if "feature_path" in df.columns:
        missing_features = df[~df["feature_path"].apply(lambda x: Path(x).exists())]
        report["checks"]["feature_paths_exist"] = {
            "passed": len(missing_features) == 0,
            "missing_count": int(len(missing_features))
        }
    
    # Per-split stats
    stats = {}
    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split]
        identities = set()
        for ids in split_df["identities"]:
            identities.update(ids)
        stats[split] = {
            "videos": int(len(split_df)),
            "identities": int(len(identities)),
            "real": int((split_df["label"] == 0).sum()),
            "fake": int((split_df["label"] == 1).sum())
        }
    report["split_stats"] = stats
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Create identity-disjoint Celeb-DF manifest")
    parser.add_argument("--input-manifest", type=Path, default=Path("datasets/celebdf_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("backend/artifacts/models/torch/celebdf_id_disjoint_20261023"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args()
    
    input_manifest = args.input_manifest.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_path = output_dir / "celebdf_id_disjoint_manifest.csv"
    validation_report_path = output_dir / "validation_report.json"
    
    print("=" * 60)
    print("Creating Identity-Disjoint Celeb-DF Manifest")
    print("=" * 60)
    print(f"Input manifest: {input_manifest}")
    print(f"Output directory: {output_dir}")
    print(f"Seed: {args.seed}")
    
    # Load official test list (may not exist locally)
    official_test = load_official_test_list(ROOT / "datasets/Celeb-DF-v2/List_of_testing_videos.txt")
    print(f"Official test videos: {len(official_test)}")
    
    # Load existing manifest
    print("\nLoading existing manifest...")
    df = load_existing_manifest(input_manifest)
    print(f"Total videos: {len(df)}")
    print(f"Folders: {df['folder'].value_counts().to_dict()}")
    print(f"Labels: {df['label'].value_counts().to_dict()}")
    print(f"Existing splits: {df['split'].value_counts().to_dict()}")
    
    # Create identity-disjoint splits
    print("\nCreating identity-disjoint splits...")
    df = create_identity_disjoint_splits(
        df, official_test, 
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    # Validate
    print("\nValidating manifest...")
    report = validate_manifest(df, Path("."))
    
    # Save manifest
    df.to_csv(manifest_path, index=False)
    print(f"\nManifest saved: {manifest_path}")
    
    # Save validation report
    with open(validation_report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Validation report saved: {validation_report_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    for check, result in report["checks"].items():
        status = "PASS" if result.get("passed", False) else "FAIL"
        print(f"  {check}: {status}")
        if not result.get("passed", True):
            print(f"    Details: {result}")
    
    print(f"\nSplit Statistics:")
    for split, stats in report.get("split_stats", {}).items():
        print(f"  {split}: {stats['videos']} videos, {stats['identities']} identities, "
              f"{stats['real']} REAL, {stats['fake']} FAKE")
    
    # Check if any critical checks failed
    critical_failed = any(not check.get("passed", True) for check in report["checks"].values())
    if critical_failed:
        print("\n⚠️  CRITICAL CHECKS FAILED - DO NOT PROCEED TO TRAINING")
        return 1
    else:
        print("\n✅ All critical checks passed. Ready for training.")
        return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())