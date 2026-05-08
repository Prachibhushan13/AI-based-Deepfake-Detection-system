import shutil
import subprocess
import uuid
import logging
from pathlib import Path

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
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported video format")


def save_upload(file: UploadFile) -> Path:
    ensure_directories()
    validate_upload(file)
    target = settings.upload_path / f"{uuid.uuid4()}_{file.filename}"
    with target.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
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
        subprocess.run(command, check=False, capture_output=True)
    except FileNotFoundError:
        logger.warning("ffmpeg not found. Skipping compression and using original upload.")
        return input_path
    return output_path if output_path.exists() else input_path


def extract_frames(video_path: Path, stride: int | None = None) -> list[np.ndarray]:
    stride = stride or settings.frame_stride
    capture = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    frame_idx = 0
    while capture.isOpened():
        success, frame = capture.read()
        if not success:
            break
        if frame_idx % stride == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_idx += 1
    capture.release()
    return frames
