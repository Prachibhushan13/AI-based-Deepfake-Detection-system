"""Inference adapter for the cached-feature PyTorch temporal classifier."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import models


def faces_to_tensor(
    faces: list[np.ndarray],
    sequence_length: int = 20,
    image_size: int = 224,
) -> torch.Tensor:
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    if image_size < 1:
        raise ValueError("image_size must be positive")
    sequence = list(faces[:sequence_length])
    if not sequence:
        sequence = [np.zeros((image_size, image_size, 3), dtype=np.float32)]
    while len(sequence) < sequence_length:
        sequence.append(sequence[-1])
    frames = torch.from_numpy(np.stack(sequence).astype(np.float32))
    return frames.permute(0, 3, 1, 2)


class TorchTemporalInference:
    def __init__(
        self,
        checkpoint_path: Path,
        device: str | None = None,
        sequence_length: int = 20,
        image_size: int = 224,
    ):
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        if image_size < 1:
            raise ValueError("image_size must be positive")
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        self.temporal = self._build_temporal()
        self.temporal.load_state_dict(checkpoint["model_state"])
        self.temporal.to(self.device).eval()
        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
        backbone.classifier = nn.Identity()
        self.backbone = backbone.to(self.device).eval()

    @staticmethod
    def _build_temporal() -> nn.Module:
        class Temporal(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.projection = nn.Sequential(
                    nn.Linear(1280, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                )
                self.lstm1 = nn.LSTM(
                    256, 128, batch_first=True, bidirectional=True
                )
                self.lstm2 = nn.LSTM(
                    256, 64, batch_first=True, bidirectional=True
                )
                self.classifier = nn.Sequential(
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(128, 1),
                )

        return Temporal()

    def _temporal_logits(self, features: torch.Tensor) -> torch.Tensor:
        # These modules mirror CachedFeatureTemporalClassifier without a
        # second copy of its public model class in the application package.
        projection = self.temporal
        temporal, _ = projection.lstm1(projection.projection(features))
        _, (hidden2, _) = projection.lstm2(temporal)
        representation = torch.cat([hidden2[-2], hidden2[-1]], dim=1)
        return projection.classifier(representation).squeeze(1)

    def predict(self, faces: list[np.ndarray]) -> float:
        frames = faces_to_tensor(
            faces,
            sequence_length=self.sequence_length,
            image_size=self.image_size,
        ).to(self.device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        with torch.inference_mode():
            features = []
            normalized = (frames - mean) / std
            for start in range(0, self.sequence_length, 4):
                features.append(self.backbone(normalized[start : start + 4]))
            logits = self._temporal_logits(torch.cat(features).unsqueeze(0))
            return float(torch.sigmoid(logits)[0].cpu())
