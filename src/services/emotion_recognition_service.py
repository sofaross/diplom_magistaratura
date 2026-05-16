from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from configs.config import ProjectConfig
from src.models.emotion_recognizer import EmotionRecognizer


class EmotionRecognitionService:
    """Тонкий сервисный слой над текущей моделью распознавания эмоций."""

    def __init__(
        self,
        *,
        emotion_model_path: str | Path | None = None,
        emotion_map_path: str | Path | None = None,
        device: str | torch.device | None = None,
        recognizer: EmotionRecognizer | None = None,
        config: ProjectConfig | None = None,
    ) -> None:
        self.config = config or ProjectConfig()
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.emotion_model_path = Path(emotion_model_path or self.config.emotion_checkpoint_path)
        self.emotion_map_path = Path(emotion_map_path or self.config.emotion_map_path)
        self._recognizer = recognizer

    def recognize(self, audio: np.ndarray) -> tuple[str, dict[str, float]]:
        recognizer = self._get_recognizer()
        return recognizer.recognize(audio)

    def _get_recognizer(self) -> EmotionRecognizer:
        if self._recognizer is None:
            self._recognizer = EmotionRecognizer(
                emotion_model_path=self.emotion_model_path,
                emotion_map_path=self.emotion_map_path,
                device=self.device,
            )
        return self._recognizer


__all__ = ["EmotionRecognitionService"]
