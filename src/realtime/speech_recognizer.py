from __future__ import annotations

"""Распознавание речи (ASR) в realtime.

Важно:
- Раньше этот модуль использовал Whisper через `transformers.pipeline(...)`.
- Теперь логика переведена на Wav2Vec2 CTC, чтобы один и тот же стек Wav2Vec2
  можно было использовать и для ASR, и для speech-эмбеддингов (мультимодальность).

Рекомендация:
- Для новых частей кода используйте напрямую:
  `src.models.wav2vec2_multimodal.Wav2Vec2Multimodal`

Этот класс оставлен для совместимости со старым интерфейсом `SpeechRecognizer.recognize(audio)`.
"""

from typing import Any

import numpy as np
import torch

from src.models.speech_model import Wav2Vec2Multimodal
from src.realtime.audio_processor import AudioProcessor


class SpeechRecognizer:
    """Совместимый wrapper над `Wav2Vec2Multimodal` для распознавания текста.

    Основной метод:
    - `recognize(audio)` -> str

    Пример:
        asr = SpeechRecognizer(model_name="facebook/wav2vec2-base-960h", sample_rate=16000)
        text = asr.recognize(audio_np)
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        *,
        sample_rate: int = 16000,
        device: torch.device | str | None = None,
        audio_processor: AudioProcessor | None = None,
        strict: bool = False,
    ) -> None:
        self.model_name = str(model_name)
        self.sample_rate = int(sample_rate)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        # Предобработка аудио: как при обучении проекта.
        self.audio_processor = audio_processor or AudioProcessor(sample_rate=self.sample_rate)

        # Внутри используем новый класс; preprocess=False, потому что preprocess делаем выше.
        self.backend = Wav2Vec2Multimodal(
            model_name=self.model_name,
            device=self.device,
            sample_rate=self.sample_rate,
            preprocess=False,
            strict=strict,
        )

        # Для совместимости со старым поведением (soft fail).
        self.load_error: str | None = self.backend.load_error

    def load_model(self, model_name: str) -> Any:
        """Оставлено для совместимости: загружает backend заново.

        Обычно не нужно: модель загружается в `__init__`.
        """

        self.backend = Wav2Vec2Multimodal(
            model_name=str(model_name),
            device=self.device,
            sample_rate=self.sample_rate,
            preprocess=False,
            strict=False,
        )
        self.load_error = self.backend.load_error
        return self.backend

    def recognize(self, audio: np.ndarray) -> str:
        """Распознаёт текст из аудио.

        Args:
            audio: np.ndarray float32, 16kHz.

        Returns:
            Распознанный текст (строка).

        Raises:
            RuntimeError: если модель не загружена или произошла ошибка инференса.
        """

        x = np.asarray(audio, dtype=np.float32)
        x = self.audio_processor.normalize(x)
        x = self.audio_processor.trim_silence(x)

        if x.size == 0:
            return ""

        return str(self.backend.transcribe(x))


__all__ = ["SpeechRecognizer"]

