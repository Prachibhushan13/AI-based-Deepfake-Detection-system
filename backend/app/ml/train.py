from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from app.core.config import settings
from app.ml.model import build_cnn_lstm_model
from app.ml.preprocessing import detect_and_crop_faces, make_sequence_tensor
from app.services.video_service import extract_frames


class VideoSequenceGenerator(tf.keras.utils.Sequence):
    def __init__(self, manifest: pd.DataFrame, batch_size: int = 2):
        self.manifest = manifest.reset_index(drop=True)
        self.batch_size = batch_size

    def __len__(self) -> int:
        return max(1, len(self.manifest) // self.batch_size)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        batch = self.manifest.iloc[index * self.batch_size : (index + 1) * self.batch_size]
        sequences: list[np.ndarray] = []
        labels: list[float] = []
        for row in batch.itertuples(index=False):
            frames = extract_frames(Path(row.video_path))
            faces, _ = detect_and_crop_faces(frames)
            sequence = make_sequence_tensor(faces)[0]
            sequences.append(sequence)
            labels.append(float(row.label))
        return np.asarray(sequences, dtype="float32"), np.asarray(labels, dtype="float32")


def train_model(manifest_path: str) -> dict:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    settings.model_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path)
    if "split" not in manifest.columns:
        raise ValueError("Manifest must include a split column with train, val, and test values.")
    train_df = manifest[manifest["split"] == "train"]
    val_df = manifest[manifest["split"] == "val"]
    test_df = manifest[manifest["split"] == "test"]

    model = build_cnn_lstm_model()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=6, monitor="val_loss", restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(settings.model_artifact_path), save_best_only=True, monitor="val_loss"),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.2),
        tf.keras.callbacks.TensorBoard(log_dir=str(settings.artifact_path / "logs")),
    ]
    train_gen = VideoSequenceGenerator(train_df)
    val_gen = VideoSequenceGenerator(val_df)
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20,
        callbacks=callbacks,
    )

    test_gen = VideoSequenceGenerator(test_df)
    probs = model.predict(test_gen, verbose=0).ravel()
    y_test = test_df["label"].to_numpy(dtype="float32")
    preds = (probs >= 0.5).astype("int32")
    metrics = {
        "f1_score": float(f1_score(y_test, preds)),
        "classification_report": classification_report(y_test, preds, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "history": history.history,
    }
    return metrics
