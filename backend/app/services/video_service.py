import logging
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile, status

from app.core.config import settings

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
logger = logging.getLogger(__name__)


def ensure_directories() -> None:
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    (settings.upload_path / "frames").mkdir(parents=True, exist_ok=True)
    (settings.upload_path / "faces").mkdir(parents=True, exist_ok=True)
    (settings.upload_path / "reports").mkdir(parents=True, exist_ok=True)
    (settings.artifact_path / "models").mkdir(parents=True, exist_ok=True)


def validate_upload(file: UploadFile) -> None:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported video format '{suffix}'. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )


def save_upload(file: UploadFile) -> Path:
    ensure_directories()
    validate_upload(file)
    safe_name = Path(file.filename or "upload.mp4").name
    target = settings.upload_path / f"{uuid.uuid4()}_{safe_name}"
    with target.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    if not target.exists() or target.stat().st_size == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded video file is empty")
    logger.info("Saved uploaded video '%s' to %s (%d bytes)", safe_name, target, target.stat().st_size)
    return target


def compress_video(input_path: Path) -> Path:
    output_path = input_path.with_name(f"compressed_{input_path.name}")
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vcodec",
        "libx264",
        "-crf",
        "28",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=False, capture_output=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.warning("ffmpeg skipped (%s). Using original upload: %s", exc, input_path)
        return input_path
    return output_path if (output_path.exists() and output_path.stat().st_size > 0) else input_path


def format_timestamp(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:05.2f}"


def extract_frames_with_metadata(
    video_path: Path,
    stride: int | None = None,
    max_frames: int | None = None,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    limit = max_frames or settings.sequence_length
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not open or decode video file: {video_path.name}",
        )

    raw_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    fps = raw_fps if raw_fps > 0.5 and raw_fps < 240.0 else 25.0
    total_frames_reported = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    # Calculate an adaptive stride if the video is longer than limit * default_stride
    if stride is not None and stride > 0:
        effective_stride = stride
    elif total_frames_reported > limit:
        effective_stride = max(1, total_frames_reported // limit)
    else:
        effective_stride = max(1, settings.frame_stride)

    frames: list[np.ndarray] = []
    frame_indices: list[int] = []
    timestamps: list[float] = []

    frame_idx = 0
    while capture.isOpened() and len(frames) < limit:
        success, frame = capture.read()
        if not success or frame is None:
            break
        if frame_idx % effective_stride == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
            frame_indices.append(frame_idx)
            ts = round(frame_idx / fps, 3)
            timestamps.append(ts)
        frame_idx += 1

    capture.release()

    if not frames:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No readable video frames could be extracted from '{video_path.name}'.",
        )

    actual_total_frames = max(total_frames_reported, frame_idx)
    duration_sec = round(actual_total_frames / fps, 2) if fps > 0 else 0.0
    if (width == 0 or height == 0) and frames:
        height, width = frames[0].shape[:2]

    metadata: dict[str, Any] = {
        "fps": round(fps, 2),
        "totalFrames": actual_total_frames,
        "sampledFrames": len(frames),
        "stride": effective_stride,
        "durationSeconds": duration_sec,
        "resolution": f"{width}x{height}",
        "frameIndices": frame_indices,
        "timestamps": timestamps,
    }

    logger.info(
        "Extracted %d frames from '%s' (fps=%.2f, total_frames=%d, duration=%.2fs, stride=%d, resolution=%dx%d)",
        len(frames),
        video_path.name,
        fps,
        actual_total_frames,
        duration_sec,
        effective_stride,
        width,
        height,
    )
    return frames, metadata


def extract_frames(
    video_path: Path,
    stride: int | None = None,
    max_frames: int | None = None,
) -> list[np.ndarray]:
    frames, _ = extract_frames_with_metadata(video_path, stride=stride, max_frames=max_frames)
    return frames
