#!/usr/bin/env python3
"""
Prepare UADFV dataset manifest for evaluation.

This script scans the UADFV dataset directories (real/ and fake/ folders),
validates video files, and creates a manifest CSV with metadata for evaluation.

Usage:
    python scripts/prepare_uadfv_manifest.py --real-dir PATH --fake-dir PATH

Required:
    --real-dir: Directory containing REAL videos
    --fake-dir: Directory containing FAKE videos

Optional:
    --output-dir: Output directory for manifest (default: datasets/UADFV/processed)
    --dry-run: Validate without requiring all videos to be readable
"""

import argparse
import json
import hashlib
import sys
from pathlib import Path

import cv2
import pandas as pd

# Add backend to path for config
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

try:
    from app.core.config import settings
except ImportError:
    class MockSettings:
        sequence_length = 20
        image_size = 224
    settings = MockSettings()


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare UADFV dataset manifest for evaluation"
    )
    parser.add_argument(
        "--real-dir",
        type=str,
        required=True,
        help="Directory containing REAL videos",
    )
    parser.add_argument(
        "--fake-dir",
        type=str,
        required=True,
        help="Directory containing FAKE videos",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for manifest (default: datasets/UADFV/processed)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate structure without requiring all videos to be readable",
    )
    return parser.parse_args()


def get_video_info(video_path: Path) -> dict:
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


def scan_directory(dir_path: Path, label_name: str, numeric_label: int, source_dir: Path) -> list:
    """Scan a directory for video files and collect metadata."""
    records = []
    supported_exts = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV"}

    if not dir_path.exists():
        print(f"Error: Directory not found: {dir_path}")
        return records

    video_files = [f for f in dir_path.iterdir()
                   if f.is_file() and f.suffix in supported_exts]

    print(f"Found {len(video_files)} video files in {dir_path}")

    for video_path in sorted(video_files):
        print(f"  Processing: {video_path.name}")

        # Get video metadata
        video_info = get_video_info(video_path)

        # Compute hash for duplicate detection
        file_hash = compute_file_hash(video_path)

        record = {
            "filename": video_path.name,
            "absolute_path": str(video_path.resolve()),
            "ground_truth_label": label_name,
            "numeric_label": numeric_label,
            "dataset_source": "UADFV",
            "readable": video_info["readable"],
            "frame_count": video_info["frame_count"],
            "fps": video_info["fps"],
            "duration_seconds": video_info["duration"],
            "width": video_info["width"],
            "height": video_info["height"],
            "file_size_bytes": video_path.stat().st_size if video_path.exists() else 0,
            "file_hash_sha256": file_hash,
            "duplicate_of": None,
            "validation_error": video_info["error"],
        }
        records.append(record)

    return records


def generate_validation_report(records: list, real_dir: Path, fake_dir: Path) -> dict:
    """Generate a validation summary report."""
    total = len(records)
    readable = sum(1 for r in records if r["readable"])
    unreadable = total - readable

    real_count = sum(1 for r in records if r["numeric_label"] == 0)
    fake_count = sum(1 for r in records if r["numeric_label"] == 1)
    real_readable = sum(1 for r in records if r["numeric_label"] == 0 and r["readable"])
    fake_readable = sum(1 for r in records if r["numeric_label"] == 1 and r["readable"])

    # Check for duplicate content
    seen_hashes = {}
    duplicates = []
    for r in records:
        if r["file_hash_sha256"] and r["file_hash_sha256"] in seen_hashes:
            r["duplicate_of"] = seen_hashes[r["file_hash_sha256"]]
            duplicates.append(r)
        elif r["file_hash_sha256"]:
            seen_hashes[r["file_hash_sha256"]] = r["filename"]

    empty_files = [r for r in records if r["file_size_bytes"] == 0]
    unsupported_or_corrupt = sum(
        1 for r in records if not r["readable"] and r["validation_error"] is not None
    )

    error_types = {}
    for r in records:
        if r["validation_error"]:
            et = r["validation_error"]
            error_types[et] = error_types.get(et, 0) + 1

    report = {
        "dataset": "UADFV",
        "real_dir": str(real_dir),
        "fake_dir": str(fake_dir),
        "total_videos_found": total,
        "readable_videos": readable,
        "unreadable_videos": unreadable,
        "class_distribution": {
            "REAL": {"total": real_count, "readable": real_readable},
            "FAKE": {"total": fake_count, "readable": fake_readable},
        },
        "duplicates_found": len(duplicates),
        "duplicate_details": [
            {"filename": d["filename"], "duplicate_of": d["duplicate_of"]}
            for d in duplicates
        ],
        "empty_files": len(empty_files),
        "empty_file_details": [e["filename"] for e in empty_files],
        "unsupported_or_corrupt": unsupported_or_corrupt,
        "error_types": error_types,
        "evaluable_videos": readable,
    }

    return report


def main():
    args = parse_args()

    real_dir = Path(args.real_dir).resolve()
    fake_dir = Path(args.fake_dir).resolve()

    if not real_dir.exists():
        print(f"Error: REAL directory not found: {real_dir}")
        sys.exit(1)
    if not fake_dir.exists():
        print(f"Error: FAKE directory not found: {fake_dir}")
        sys.exit(1)

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = Path("datasets") / "UADFV" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "uadfv_manifest.csv"
    validation_report_path = output_dir / "uadfv_validation_report.json"

    print("=" * 60)
    print("UADFV Dataset Manifest Preparation")
    print("=" * 60)
    print(f"\nREAL directory: {real_dir}")
    print(f"FAKE directory: {fake_dir}")
    print(f"Output directory: {output_dir}")

    # Scan directories
    print(f"\nScanning REAL videos: {real_dir}")
    real_records = scan_directory(real_dir, "REAL", 0, real_dir)

    print(f"\nScanning FAKE videos: {fake_dir}")
    fake_records = scan_directory(fake_dir, "FAKE", 1, fake_dir)

    records = real_records + fake_records

    if not records:
        print("\nNo video files found in either directory!")
        sys.exit(1)

    # Check for duplicate content across all records
    seen_hashes = {}
    for r in records:
        if r["file_hash_sha256"] and r["file_hash_sha256"] in seen_hashes:
            r["duplicate_of"] = seen_hashes[r["file_hash_sha256"]]
        elif r["file_hash_sha256"]:
            seen_hashes[r["file_hash_sha256"]] = r["filename"]

    # Generate validation report
    report = generate_validation_report(records, real_dir, fake_dir)

    # Save manifest
    df = pd.DataFrame(records)
    df.to_csv(manifest_path, index=False)
    print(f"\nManifest saved to: {manifest_path}")
    print(f"Total records: {len(df)}")

    # Save validation report
    with open(validation_report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Validation report saved to: {validation_report_path}")

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

    if report["error_types"]:
        print("\nError types:")
        for err, count in report["error_types"].items():
            print(f"  {err}: {count}")

    if report["duplicates_found"] > 0:
        print("\nDuplicate files (content):")
        for dup in report["duplicate_details"]:
            print(f"  {dup['filename']} -> duplicate of {dup['duplicate_of']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())