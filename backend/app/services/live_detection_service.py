import base64
from collections import deque
from io import BytesIO

import numpy as np
from PIL import Image

from app.core.config import settings
from app.ml.inference import predict_video


def decode_base64_frame(frame_data: str) -> np.ndarray:
    encoded = frame_data.split(",", 1)[1] if "," in frame_data else frame_data
    image_bytes = base64.b64decode(encoded)
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return np.array(image)


class LiveDetectionSession:
    def __init__(self) -> None:
        self.frames: deque[np.ndarray] = deque(maxlen=settings.sequence_length)
        self.min_frames = min(5, settings.sequence_length)

    def push_frame(self, frame_data: str) -> dict:
        frame = decode_base64_frame(frame_data)
        self.frames.append(frame)

        if len(self.frames) < self.min_frames:
            return {
                "type": "status",
                "ready": False,
                "frameCount": len(self.frames),
                "requiredFrames": self.min_frames,
                "message": "Collecting frames for live analysis window.",
            }

        inference = predict_video(list(self.frames))
        return {
            "type": "prediction",
            "ready": True,
            "frameCount": len(self.frames),
            "requiredFrames": self.min_frames,
            "result": inference["result"],
            "confidence": inference["confidence"],
            "fakeProbability": inference["fakeProbability"],
            "realProbability": inference["realProbability"],
            "timeline": inference["timeline"],
            "suspiciousFrameCount": len(inference["suspiciousIndices"]),
            "message": "Live sliding-window analysis updated.",
        }
