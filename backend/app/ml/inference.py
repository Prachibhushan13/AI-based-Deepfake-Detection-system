import logging
import numpy as np
import tensorflow as tf

from app.core.config import settings
from app.ml.model import build_cnn_lstm_model
from app.ml.preprocessing import detect_and_crop_faces, make_sequence_tensor

logger = logging.getLogger(__name__)
_model: tf.keras.Model | None = None


def load_model() -> tf.keras.Model:
    global _model
    if _model is not None:
        return _model
    model_path = settings.model_artifact_path
    if model_path.exists():
        _model = tf.keras.models.load_model(model_path)
    else:
        logger.warning("Model artifact not found. Falling back to untrained architecture.")
        _model = build_cnn_lstm_model()
    return _model


def predict_video(frames: list[np.ndarray]) -> dict:
    faces, suspicious_indices = detect_and_crop_faces(frames)
    sequence_tensor = make_sequence_tensor(faces)
    using_mock_model = settings.enable_mock_model and not settings.model_artifact_path.exists()

    if using_mock_model:
        variance = float(np.var(sequence_tensor))
        frame_count = max(1, min(len(frames), settings.sequence_length))
        suspicious_ratio = len(suspicious_indices) / frame_count
        temporal_change = float(np.std(np.mean(sequence_tensor[0], axis=(1, 2, 3))))

        fake_probability = (
            0.18
            + min(0.10, variance * 1.2)
            + min(0.14, temporal_change * 6.0)
            + min(0.18, suspicious_ratio * 0.45)
        )

        if len(suspicious_indices) <= 1:
            fake_probability = min(fake_probability, 0.42)
        if suspicious_ratio < 0.2 and temporal_change < 0.03:
            fake_probability = min(fake_probability, 0.38)
        fake_probability = max(0.08, min(0.82, fake_probability))
    else:
        model = load_model()
        fake_probability = float(model.predict(sequence_tensor, verbose=0)[0][0])

    real_probability = 1.0 - fake_probability
    decision_threshold = 0.55 if using_mock_model else 0.5
    result = "FAKE" if fake_probability >= decision_threshold else "REAL"
    confidence = max(fake_probability, real_probability) * 100.0
    timeline = []
    for index in range(min(len(frames), settings.sequence_length)):
        drift = fake_probability if index in suspicious_indices else real_probability
        timeline.append({"frameIndex": index, "score": round(float(drift), 4)})

    return {
        "modelMode": "mock" if using_mock_model else "trained",
        "result": result,
        "confidence": round(confidence, 2),
        "fakeProbability": round(fake_probability, 4),
        "realProbability": round(real_probability, 4),
        "suspiciousIndices": suspicious_indices[:6],
        "timeline": timeline,
        "croppedFaces": faces,
    }
