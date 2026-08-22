from collections.abc import Iterable

import cv2
import numpy as np
from mtcnn import MTCNN

from app.core.config import settings

detector = MTCNN()


def align_face(image: np.ndarray, keypoints: dict) -> np.ndarray:
    left_eye = keypoints.get("left_eye")
    right_eye = keypoints.get("right_eye")
    if not left_eye or not right_eye:
        return image
    
    # Calculate angle between eyes
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))
    
    # Rotate around the center of the eyes
    eye_center = (int((left_eye[0] + right_eye[0]) // 2), int((left_eye[1] + right_eye[1]) // 2))
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
    return rotated


def detect_and_crop_faces(frames: Iterable[np.ndarray]) -> tuple[list[np.ndarray], list[int], list[dict]]:
    faces: list[np.ndarray] = []
    suspicious_indices: list[int] = []
    all_keypoints: list[dict] = []
    for index, frame in enumerate(frames):
        detections = detector.detect_faces(frame)
        if not detections:
            continue
        
        # Align face before cropping to stabilize forensic signals
        aligned_frame = align_face(frame, detections[0].get("keypoints", {}))
        
        # Re-detect on aligned frame for precise cropping
        detections = detector.detect_faces(aligned_frame)
        if not detections:
            continue
            
        x, y, width, height = detections[0]["box"]
        x, y = max(x, 0), max(y, 0)
        crop = aligned_frame[y : y + height, x : x + width]
        if crop.size == 0:
            continue
        resized = cv2.resize(crop, (settings.image_size, settings.image_size))
        faces.append(resized.astype("float32") / 255.0)
        all_keypoints.append(detections[0].get("keypoints", {}))
        if detections[0].get("confidence", 0) < 0.95:
            suspicious_indices.append(index)
    return faces, suspicious_indices, all_keypoints


def make_sequence_tensor(faces: list[np.ndarray]) -> np.ndarray:
    if not faces:
        return np.zeros((1, settings.sequence_length, settings.image_size, settings.image_size, 3), dtype="float32")
    sequence = faces[: settings.sequence_length]
    while len(sequence) < settings.sequence_length:
        sequence.append(sequence[-1])
    return np.expand_dims(np.stack(sequence, axis=0), axis=0)

