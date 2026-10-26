#!/usr/bin/env python3
"""
Prepare DFDC dataset manifest for evaluation.

This script scans the DFDC dataset directory, reads metadata.json,
validates videos, and creates a manifest CSV with metadata for evaluation.

Usage:
    python scripts/prepare_dfdc_manifest.py --dfdc-root PATH

Required:
    --dfdc-root: Root directory containing DFDC data (metadata.json + videos/)
    
Optional:
    --output-dir: Output directory for manifest (default: datasets/dfdc/processed)
    --dry-run: Validate without requiring all videos to be readable
"""

import argparse
import json
import hashlib
import os
import sys
from pathlib import Path

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


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare DFDC dataset manifest for evaluation"
    )
    parser.add_argument(
        "--dfdc-root",
        type=str,
        required=True,
        help="Root directory containing DFDC data (metadata.json + videos/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for manifest (default: datasets/dfdc/processed)",
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


def main():
    args = parse_args()

    dfdc_root = Path(args.dfdc_root).resolve()
    if not dfdc_root.exists():
        print(f"Error: DFDC root directory not found: {dfdc_root}")
        sys.exit(1)

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = dfdc_root.parent / "dfdc" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "dfdc_manifest.csv"
    validation_report_path = output_dir / "dfdc_validation_report.json"

    # --- Load metadata.json ---
    metadata_path = dfdc_root / "metadata.json"
    if not metadata_path.exists():
        print(f"Error: metadata.json not found at {metadata_path}")
        print(
            "DFDC metadata.json is typically in the extracted competition data folder."
        )
        sys.exit(1)

    print(f"Loading metadata from: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"Total entries in metadata.json: {len(metadata)}")

    # --- Discover video files ---
    videos_dir = dfdc_root / "videos"
    supported_exts = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV"}

    # Find all video files in the videos directory (recursive or flat)
    video_files = []
    if videos_dir.exists():
        for root, dirs, files in os.walk(videos_dir):
            for fname in files:
                if Path(fname).suffix.lower() in supported_exts:
                    video_files.append(Path(root) / fname)
    else:
        print(f"Warning: videos/ directory not found at {videos_dir}")
        print("Looking for video files directly under DFDC root...")
        # Try flat structure
        for fname in dfdc_root.iterdir():
            if fname.suffix.lower() in supported_exts:
                video_files.append(fname)

    print(f"Found {len(video_files)} video files in dataset directory")

    # --- Build manifest records ---
    records = []
    seen_hashes = {}
    errors = {"missing": 0, "unreadable": 0, "duplicates": 0, "invalid_label": 0}

    for video_path in video_files:
        filename = video_path.name

        # Skip if not in metadata
        if filename not in metadata:
            errors["invalid_label"] += 1
            records.append(
                {
                    "filename": filename,
                    "absolute_path": str(video_path.resolve()),
                    "ground_truth_label": "UNKNOWN",
                    "numeric_label": -1,
                    "dataset_source": "DFDC",
                    "readable": False,
                    "frame_count": 0,
                    "fps": 0.0,
                    "duration_seconds": 0.0,
                    "width": 0,
                    "height": 0,
                    "file_size_bytes": video_path.stat().st_size if video_path.exists() else 0,
                    "file_hash_sha256": compute_file_hash(video_path),
                    "duplicate_of": None,
                    "validation_error": "not_in_metadata",
                }
            )
            continue

        meta = metadata[filename]

        # Parse label
        label = meta.get("label", "").strip().upper()
        if label == "REAL":
            numeric_label = 0
            ground_truth = "REAL"
        elif label == "FAKE":
            numeric_label = 1
            ground_truth = "FAKE"
        else:
            errors["invalid_label"] += 1
            records.append(
                {
                    "filename": filename,
                    "absolute_path": str(video_path.resolve()),
                    "ground_truth_label": label,
                    "numeric_label": -1,
                    "dataset_source": "DFDC",
                    "readable": False,
                    "frame_count": 0,
                    "fps": 0.0,
                    "duration_seconds": 0.0,
                    "width": 0,
                    "height": 0,
                    "file_size_bytes": video_path.stat().st_size if video_path.exists() else 0,
                    "file_hash_sha256": compute_file_hash(video_path),
                    "duplicate_of": None,
                    "validation_error": "invalid_label",
                }
            )
            continue

        # Get original video (for original/fake pairing)
        original = meta.get("original", None)

        # Get video metadata
        video_info = get_video_info(video_path)

        # Compute hash for duplicate detection
        file_hash = compute_file_hash(video_path)

        # Check for duplicate content
        duplicate_of = None
        if file_hash and file_hash in seen_hashes:
            duplicate_of = seen_hashes[file_hash]
        elif file_hash:
            seen_hashes[file_hash] = filename

        record = {
            "filename": filename,
            "absolute_path": str(video_path.resolve()),
            "ground_truth_label": ground_truth,
            "numeric_label": numeric_label,
            "dataset_source": "DFDC",
            "readable": video_info["readable"],
            "frame_count": video_info["frame_count"],
            "fps": video_info["fps"],
            "duration_seconds": video_info["duration"],
            "width": video_info["width"],
            "height": video_info["height"],
            "file_size_bytes": video_path.stat().st_size if video_path.exists() else 0,
            "file_hash_sha256": file_hash,
            "duplicate_of": duplicate_of,
            "original": original,  # Preserve original/fake relationship
            "validation_error": video_info["error"],
        }
        records.append(record)

    # --- Generate validation report ---
    total = len(records)
    readable = sum(1 for r in records if r["readable"])
    unreadable = total - readable

    real_count = sum(1 for r in records if r["numeric_label"] == 0)
    fake_count = sum(1 for r in records if r["numeric_label"] == 1)
    real_readable = sum(
        1 for r in records if r["numeric_label"] == 0 and r["readable"]
    )
    fake_readable = sum(
        1 for r in records if r["numeric_label"] == 1 and r["readable"]
    )

    duplicates = [r for r in records if r["duplicate_of"] is not None]
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
        "dataset": "DFDC",
        "dfdc_root": str(dfdc_root),
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
        "evaluable_videos": readable,  # Only readable videos can be evaluated
        "original_fake_pairs": len(
            [r for r in records if r.get("original") is not None and r["numeric_label"] == 1]
        ),
    }

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
    print(f"Invalid labels: {errors['invalid_label']}")
    print(f"Original/fake pairs (FAKE with original): {report['original_fake_pairs']}")
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