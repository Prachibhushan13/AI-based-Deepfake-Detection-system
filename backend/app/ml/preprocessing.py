from collections.abc import Iterable
import logging

import cv2
import numpy as np
from mtcnn import MTCNN

from app.core.config import settings

logger = logging.getLogger(__name__)
detector = MTCNN()


def align_face(image: np.ndarray, keypoints: dict) -> np.ndarray:
    left_eye = keypoints.get("left_eye")
    right_eye = keypoints.get("right_eye")
    if not left_eye or not right_eye:
        return image

    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = float(np.degrees(np.arctan2(dy, dx)))

    eye_center = (
        int((left_eye[0] + right_eye[0]) // 2),
        int((left_eye[1] + right_eye[1]) // 2),
    )
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def _center_crop_fallback(frame: np.ndarray) -> tuple[np.ndarray, list[int]]:
    h, w = frame.shape[:2]
    side = int(min(h, w) * 0.7)
    y0 = max(0, (h - side) // 2)
    x0 = max(0, (w - side) // 2)
    return frame[y0 : y0 + side, x0 : x0 + side], [x0, y0, side, side]


def detect_and_crop_faces(
    frames: Iterable[np.ndarray],
    target_size: int = settings.image_size,
    is_xception: bool = False,
    return_pil: bool = False,
) -> tuple[list[np.ndarray], list[int], list[dict]]:
    faces: list = []
    low_det_indices: list[int] = []
    all_keypoints: list[dict] = []

    last_valid_box: list[int] | None = None

    for index, frame in enumerate(frames):
        detections = detector.detect_faces(frame)
        face_detected = False
        det_conf = 0.0
        kp: dict = {}
        box: list[int] = [0, 0, frame.shape[1], frame.shape[0]]

        if detections:
            # Select largest high-confidence face in frame
            best = max(detections, key=lambda d: max(0, d["box"][2]) * max(0, d["box"][3]))
            det_conf = float(best.get("confidence", 0.0))
            kp = dict(best.get("keypoints", {}))
            aligned_frame = align_face(frame, kp)
            x, y, width, height = best["box"]
            margin = int(max(width, height) * 0.12)
            x0 = max(x - margin, 0)
            y0 = max(y - margin, 0)
            w0 = min(width + margin * 2, aligned_frame.shape[1] - x0)
            h0 = min(height + margin * 2, aligned_frame.shape[0] - y0)
            crop = aligned_frame[y0 : y0 + h0, x0 : x0 + w0]
            if crop.size > 0:
                face_detected = True
                box = [int(x0), int(y0), int(w0), int(h0)]
                last_valid_box = box
            else:
                crop, box = _center_crop_fallback(frame)
        elif last_valid_box is not None:
            x0, y0, w0, h0 = last_valid_box
            crop = frame[y0 : y0 + h0, x0 : x0 + w0]
            if crop.size == 0:
                crop, box = _center_crop_fallback(frame)
            else:
                box = list(last_valid_box)
        else:
            crop, box = _center_crop_fallback(frame)

        if return_pil:
            from PIL import Image

            faces.append(Image.fromarray(crop))
        else:
            resized = cv2.resize(crop, (target_size, target_size))
            if is_xception:
                faces.append((resized.astype("float32") / 127.5) - 1.0)
            else:
                faces.append(resized.astype("float32") / 255.0)

        meta = {
            **kp,
            "box": box,
            "confidence": det_conf,
            "face_detected": face_detected,
            "sample_index": index,
        }
        all_keypoints.append(meta)
        if face_detected and det_conf < 0.95:
            low_det_indices.append(index)

    return faces, low_det_indices, all_keypoints


def make_sequence_tensor(faces: list[np.ndarray]) -> np.ndarray:
    if not faces:
        return np.zeros(
            (1, settings.sequence_length, settings.image_size, settings.image_size, 3),
            dtype="float32",
        )
    sequence = list(faces[: settings.sequence_length])
    while len(sequence) < settings.sequence_length:
        sequence.append(sequence[-1])
    return np.expand_dims(np.stack(sequence, axis=0), axis=0)
