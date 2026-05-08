from collections.abc import Iterable

import cv2
import numpy as np
from mtcnn import MTCNN

from app.core.config import settings

detector = MTCNN()


def detect_and_crop_faces(frames: Iterable[np.ndarray]) -> tuple[list[np.ndarray], list[int]]:
    faces: list[np.ndarray] = []
    suspicious_indices: list[int] = []
    for index, frame in enumerate(frames):
        detections = detector.detect_faces(frame)
        if not detections:
            continue
        x, y, width, height = detections[0]["box"]
        x, y = max(x, 0), max(y, 0)
        crop = frame[y : y + height, x : x + width]
        if crop.size == 0:
            continue
        resized = cv2.resize(crop, (settings.image_size, settings.image_size))
        faces.append(resized.astype("float32") / 255.0)
        if detections[0].get("confidence", 0) < 0.95:
            suspicious_indices.append(index)
    return faces, suspicious_indices


def make_sequence_tensor(faces: list[np.ndarray]) -> np.ndarray:
    if not faces:
        return np.zeros((1, settings.sequence_length, settings.image_size, settings.image_size, 3), dtype="float32")
    sequence = faces[: settings.sequence_length]
    while len(sequence) < settings.sequence_length:
        sequence.append(sequence[-1])
    return np.expand_dims(np.stack(sequence, axis=0), axis=0)

