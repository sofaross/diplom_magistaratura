from __future__ import annotations

"""Предобработка аудио для моделей речи и эмоций.

Здесь собраны шаги, которые должны совпадать с обучением:
- normalize_audio
- trim_silence
- mel spectrogram (128 mel bins, librosa)
- стандартизация mel по одному примеру (mean=0, std=1)
"""

import numpy as np
import torch

from src.features.feature_extraction import extract_mel_spectrogram
from src.preprocessing.audio_preprocessing import normalize_audio, trim_silence


class AudioProcessor:
    """Класс для предобработки аудио.

    Методы:
    - normalize(audio) -> np.ndarray
    - trim_silence(audio) -> np.ndarray
    - extract_mel(audio) -> torch.Tensor [1, 128, T]
    - standardize_mel(mel) -> torch.Tensor (как в датасете)
    """

    def __init__(self, sample_rate: int = 16000):
        """Создаёт процессор аудио.

        Args:
            sample_rate: частота дискретизации входного аудио (по умолчанию 16000).
        """

        self.sample_rate = int(sample_rate)

    def normalize(self, audio: np.ndarray) -> np.ndarray:
        """Нормализует громкость (как в обучении).

        Args:
            audio: np.ndarray float/int.

        Returns:
            np.ndarray float32.
        """

        x = np.asarray(audio, dtype=np.float32)
        x = normalize_audio(x)
        return np.asarray(x, dtype=np.float32)

    def trim_silence(self, audio: np.ndarray) -> np.ndarray:
        """Обрезает тишину в начале/конце (как в обучении).

        Args:
            audio: np.ndarray float32.

        Returns:
            np.ndarray float32 (может стать пустым).
        """

        x = np.asarray(audio, dtype=np.float32)
        x = trim_silence(x)
        return np.asarray(x, dtype=np.float32)

    def extract_mel(self, audio: np.ndarray) -> torch.Tensor:
        """Считает mel-спектрограмму (в dB) как в обучении.

        Args:
            audio: np.ndarray float32 (желательно 16kHz).

        Returns:
            torch.Tensor формы [1, 128, T] float32.

        Raises:
            ValueError: если audio пустое.
        """

        x = np.asarray(audio, dtype=np.float32)
        if x.size == 0:
            raise ValueError("audio пустое: невозможно извлечь mel")

        mel_np = extract_mel_spectrogram(x, sample_rate=int(self.sample_rate))  # [128, T]
        mel = torch.from_numpy(np.asarray(mel_np, dtype=np.float32)).unsqueeze(0)  # [1, 128, T]
        return mel

    @staticmethod
    def standardize_mel(mel: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """Стандартизирует mel по одному примеру (mean=0, std=1).

        Это та же логика, что в датасете, чтобы паддинг нулём меньше вредил.

        Args:
            mel: torch.Tensor, обычно [1, 128, T].
            eps: защита от деления на 0.

        Returns:
            torch.Tensor той же формы.
        """

        mean = mel.mean()
        std = mel.std(unbiased=False)
        if not torch.isfinite(std) or float(std) < float(eps):
            return mel - mean
        return (mel - mean) / (std + float(eps))


__all__ = ["AudioProcessor"]
