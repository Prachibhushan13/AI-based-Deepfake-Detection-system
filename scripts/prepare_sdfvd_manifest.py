#!/usr/bin/env python3
"""
Prepare SDFVD dataset manifest for evaluation.

This script scans the SDFVD dataset directories, validates video files,
and creates a manifest CSV with metadata for evaluation.

Usage:
    python scripts/prepare_sdfvd_manifest.py
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import pandas as pd

# Add backend to path for config
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

try:
    from app.core.config import settings
except ImportError:
    # Fallback if config not available
    class MockSettings:
        sequence_length = 20
        image_size = 224
    settings = MockSettings()


# Dataset paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SDFVD_RAW_REAL = PROJECT_ROOT / "datasets" / "SDFVD" / "videos_real"
SDFVD_RAW_FAKE = PROJECT_ROOT / "datasets" / "SDFVD" / "videos_fake"
SDFVD_PROCESSED = PROJECT_ROOT / "datasets" / "SDFVD" / "processed"
MANIFEST_PATH = SDFVD_PROCESSED / "sdfvd_manifest.csv"
VALIDATION_REPORT_PATH = SDFVD_PROCESSED / "sdfvd_validation_report.json"

# Supported video extensions
SUPPORTED_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV"}


def get_video_info(video_path: Path) -> Dict[str, Any]:
    """Extract metadata from a video file."""
    info = {
        "readable": False,
        "frame_count": 0,
        "fps": 0.0,
        "duration": 0.0,
        "width": 0,
        "height": 0,
        "error": None,
    }
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            info["error"] = "Cannot open video file"
            return info

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Try reading first frame to verify readability
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            info["error"] = "Cannot read frames from video"
            return info

        info["readable"] = True
        info["frame_count"] = frame_count
        info["fps"] = float(fps) if fps > 0 else 0.0
        info["duration"] = float(frame_count / fps) if fps > 0 else 0.0
        info["width"] = width
        info["height"] = height

    except Exception as e:
        info["error"] = str(e)

    return info


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file for duplicate detection."""
    hasher = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return ""


def scan_dataset_directory(real_dir: Path, fake_dir: Path) -> List[Dict[str, Any]]:
    """Scan real and fake directories and collect video metadata."""
    records = []
    seen_hashes = {}

    for label_dir, label_name, numeric_label in [
        (real_dir, "REAL", 0),
        (fake_dir, "FAKE", 1),
    ]:
        if not label_dir.exists():
            print(f"Warning: Directory not found: {label_dir}")
            continue

        video_files = [f for f in label_dir.iterdir()
                       if f.is_file() and f.suffix in SUPPORTED_EXTS]

        print(f"Found {len(video_files)} video files in {label_dir}")

        for video_path in video_files:
            print(f"  Processing: {video_path.name}")

            # Get video metadata
            video_info = get_video_info(video_path)

            # Compute hash for duplicate detection
            file_hash = compute_file_hash(video_path)

            # Check for duplicate content
            duplicate_of = None
            if file_hash and file_hash in seen_hashes:
                duplicate_of = seen_hashes[file_hash]
            elif file_hash:
                seen_hashes[file_hash] = video_path.name

            record = {
                "filename": video_path.name,
                "absolute_path": str(video_path.resolve()),
                "ground_truth_label": label_name,
                "numeric_label": numeric_label,
                "dataset_source": "SDFVD",
                "readable": video_info["readable"],
                "frame_count": video_info["frame_count"],
                "fps": video_info["fps"],
                "duration_seconds": video_info["duration"],
                "width": video_info["width"],
                "height": video_info["height"],
                "file_size_bytes": video_path.stat().st_size if video_path.exists() else 0,
                "file_hash_sha256": file_hash,
                "duplicate_of": duplicate_of,
                "validation_error": video_info["error"],
            }
            records.append(record)

    return records


def generate_validation_report(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate a validation summary report."""
    total = len(records)
    readable = sum(1 for r in records if r["readable"])
    unreadable = total - readable

    real_count = sum(1 for r in records if r["numeric_label"] == 0)
    fake_count = sum(1 for r in records if r["numeric_label"] == 1)
    real_readable = sum(1 for r in records if r["numeric_label"] == 0 and r["readable"])
    fake_readable = sum(1 for r in records if r["numeric_label"] == 1 and r["readable"])

    duplicates = [r for r in records if r["duplicate_of"] is not None]
    empty_files = [r for r in records if r["file_size_bytes"] == 0]
    unsupported = [r for r in records if not r["readable"] and r["validation_error"] is not None]

    error_types = {}
    for r in records:
        if r["validation_error"]:
            error_types[r["validation_error"]] = error_types.get(r["validation_error"], 0) + 1

    report = {
        "dataset": "SDFVD",
        "total_videos_found": total,
        "readable_videos": readable,
        "unreadable_videos": unreadable,
        "class_distribution": {
            "REAL": {"total": real_count, "readable": real_readable},
            "FAKE": {"total": fake_count, "readable": fake_readable},
        },
        "duplicates_found": len(duplicates),
        "duplicate_details": [{"filename": d["filename"], "duplicate_of": d["duplicate_of"]} for d in duplicates],
        "empty_files": len(empty_files),
        "empty_file_details": [e["filename"] for e in empty_files],
        "unsupported_or_corrupt": len(unsupported),
        "error_types": error_types,
        "evaluable_videos": readable,  # Only readable videos can be evaluated
    }

    return report


def main():
    print("=" * 60)
    print("SDFVD Dataset Manifest Preparation")
    print("=" * 60)

    # Ensure output directory exists
    SDFVD_PROCESSED.mkdir(parents=True, exist_ok=True)

    # Scan directories
    print(f"\nScanning real videos: {SDFVD_RAW_REAL}")
    print(f"Scanning fake videos: {SDFVD_RAW_FAKE}")

    records = scan_dataset_directory(SDFVD_RAW_REAL, SDFVD_RAW_FAKE)

    if not records:
        print("\nNo video files found in SDFVD directories!")
        print(f"Expected directories:")
        print(f"  {SDFVD_RAW_REAL}")
        print(f"  {SDFVD_RAW_FAKE}")
        print("\nPlease place your SDFVD dataset videos in these directories.")
        print("Structure should be:")
        print("  datasets/sdfvd/raw/videos_real/*.mp4 (or .avi, .mov, .mkv)")
        print("  datasets/sdfvd/raw/videos_fake/*.mp4 (or .avi, .mov, .mkv)")

        # Create empty manifest with headers
        df = pd.DataFrame(columns=[
            "filename", "absolute_path", "ground_truth_label", "numeric_label",
            "dataset_source", "readable", "frame_count", "fps", "duration_seconds",
            "width", "height", "file_size_bytes", "file_hash_sha256",
            "duplicate_of", "validation_error"
        ])
        df.to_csv(MANIFEST_PATH, index=False)

        # Create empty validation report
        report = {
            "dataset": "SDFVD",
            "total_videos_found": 0,
            "readable_videos": 0,
            "unreadable_videos": 0,
            "class_distribution": {"REAL": {"total": 0, "readable": 0}, "FAKE": {"total": 0, "readable": 0}},
            "duplicates_found": 0,
            "empty_files": 0,
            "unsupported_or_corrupt": 0,
            "error_types": {},
            "evaluable_videos": 0,
        }
        with open(VALIDATION_REPORT_PATH, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\nEmpty manifest created at: {MANIFEST_PATH}")
        print(f"Empty validation report created at: {VALIDATION_REPORT_PATH}")
        return 1

    # Create DataFrame and save manifest
    df = pd.DataFrame(records)
    df.to_csv(MANIFEST_PATH, index=False)
    print(f"\nManifest saved to: {MANIFEST_PATH}")
    print(f"Total records: {len(df)}")

    # Generate and save validation report
    report = generate_validation_report(records)
    with open(VALIDATION_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Validation report saved to: {VALIDATION_REPORT_PATH}")

    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total videos found: {report['total_videos_found']}")
    print(f"Readable videos: {report['readable_videos']}")
    print(f"Unreadable videos: {report['unreadable_videos']}")
    print(f"REAL videos: {report['class_distribution']['REAL']['total']} (readable: {report['class_distribution']['REAL']['readable']})")
    print(f"FAKE videos: {report['class_distribution']['FAKE']['total']} (readable: {report['class_distribution']['FAKE']['readable']})")
    print(f"Duplicates: {report['duplicates_found']}")
    print(f"Empty files: {report['empty_files']}")
    print(f"Evaluable videos: {report['evaluable_videos']}")

    if report["duplicates_found"] > 0:
        print("\nDuplicate files (content):")
        for dup in report["duplicate_details"]:
            print(f"  {dup['filename']} -> duplicate of {dup['duplicate_of']}")

    if report["error_types"]:
        print("\nError types:")
        for err, count in report["error_types"].items():
            print(f"  {err}: {count}")

    return 0


if __name__ == "__main__":
    sys.exit(main())