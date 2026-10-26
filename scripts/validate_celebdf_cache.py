"""Validate the Celeb-DF cached EfficientNet feature dataset."""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "datasets" / "celebdf_manifest.csv"
DEFAULT_CACHE = ROOT / "datasets" / "celebdf_features"
DEFAULT_INDEX = DEFAULT_CACHE / "features_index.csv"
EXPECTED_SHAPE = (20, 1280)


def cache_name(relative_path: str) -> str:
    digest = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:24]
    return f"{digest}.npy"


def resolve_feature_path(
    row: pd.Series, cache_dir: Path
) -> Path:
    """Resolve portable cache paths while preserving the index convention."""
    recorded = Path(str(row["feature_path"]))
    if recorded.exists():
        return recorded
    return cache_dir / cache_name(str(row["relative_path"]))


def validate(manifest_path: Path, index_path: Path, cache_dir: Path) -> int:
    manifest = pd.read_csv(manifest_path)
    index = pd.read_csv(index_path)
    required = {"relative_path", "label", "split"}
    for name, frame in (("manifest", manifest), ("index", index)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    if "feature_path" not in index.columns or "status" not in index.columns:
        raise ValueError("index must contain feature_path and status columns")

    issues: list[str] = []
    manifest_counts = manifest["relative_path"].value_counts()
    index_counts = index["relative_path"].value_counts()
    duplicate_manifest = manifest_counts[manifest_counts != 1]
    duplicate_index = index_counts[index_counts != 1]
    if not duplicate_manifest.empty:
        issues.append(f"manifest duplicate paths: {len(duplicate_manifest)}")
    if not duplicate_index.empty:
        issues.append(f"index duplicate paths: {len(duplicate_index)}")

    manifest_paths = set(manifest["relative_path"].astype(str))
    index_paths = set(index["relative_path"].astype(str))
    missing_index = sorted(manifest_paths - index_paths)
    orphan_index = sorted(index_paths - manifest_paths)
    if missing_index:
        issues.append(f"missing index entries: {len(missing_index)}")
    if orphan_index:
        issues.append(f"orphan index entries: {len(orphan_index)}")

    expected_files = {cache_name(path) for path in manifest_paths}
    actual_files = {path.name for path in cache_dir.glob("*.npy")}
    orphan_files = sorted(actual_files - expected_files)
    missing_files = sorted(expected_files - actual_files)
    if orphan_files:
        issues.append(f"orphan feature files: {len(orphan_files)}")
    if missing_files:
        issues.append(f"missing feature files: {len(missing_files)}")

    valid = Counter()
    invalid = Counter()
    index_by_path = {
        str(row.relative_path): row
        for row in index.itertuples(index=False)
    }
    for row in manifest.itertuples(index=False):
        key = str(row.relative_path)
        category = (str(row.split), int(row.label))
        indexed = index_by_path.get(key)
        if indexed is None:
            invalid[category] += 1
            continue
        if int(indexed.label) != int(row.label) or str(indexed.split) != str(row.split):
            issues.append(f"label/split mismatch: {key}")
            invalid[category] += 1
            continue
        if str(indexed.status) != "ok":
            invalid[category] += 1
            issues.append(f"index status is not ok: {key}")
            continue
        feature_path = resolve_feature_path(
            pd.Series({
                "feature_path": indexed.feature_path,
                "relative_path": key,
            }),
            cache_dir,
        )
        try:
            if not feature_path.exists():
                raise FileNotFoundError(str(feature_path))
            array = np.load(feature_path, mmap_mode="r")
            if array.shape != EXPECTED_SHAPE:
                raise ValueError(f"shape {array.shape}")
            if not np.isfinite(array).all():
                raise ValueError("non-finite values")
        except Exception as exc:
            invalid[category] += 1
            issues.append(f"invalid feature {key}: {exc}")
        else:
            valid[category] += 1

    print(f"Manifest rows: {len(manifest)}")
    print(f"Index rows: {len(index)}")
    print(f"Feature files: {len(actual_files)}")
    print("Validated by split and label:")
    for split in ("train", "val", "test"):
        for label in (0, 1):
            category = (split, label)
            print(
                f"  {split:5} label={label}: "
                f"valid={valid[category]} invalid={invalid[category]}"
            )
    if orphan_files:
        print("Orphan files (not removed):")
        for name in orphan_files:
            print(f"  {name}")
    if issues:
        print("CACHE VALIDATION FAILED")
        for issue in issues:
            print(f"- {issue}")
        return 1
    print("CACHE VALIDATION PASSED")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()
    return validate(args.manifest, args.index, args.cache_dir)


if __name__ == "__main__":
    sys.exit(main())
