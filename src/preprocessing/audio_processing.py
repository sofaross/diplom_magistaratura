from __future__ import annotations

from pathlib import Path
from typing import Union

import librosa
import numpy as np
import torch

from src.features.feature_extraction import extract_mel_spectrogram

PathLike = Union[str, Path]

# ===============================
# Единая предобработка аудио для проекта (функции + класс).
# ===============================

#Загружает аудио и приводит к нужной частоте дискретизации.
def load_audio(path: PathLike, sample_rate: int = 16000) -> np.ndarray:
    audio, _sr = librosa.load(str(path), sr=int(sample_rate))
    return np.asarray(audio, dtype=np.float32)

#Нормализует громкость
def normalize_audio(audio: np.ndarray) -> np.ndarray:

    x = np.asarray(audio, dtype=np.float32)
    if x.size == 0:
        return x

    max_abs = float(np.max(np.abs(x)))
    if not np.isfinite(max_abs) or max_abs == 0.0:
        return x

    return (x / max_abs).astype(np.float32, copy=False)

#Обрезает тишину в начале и в конце аудио
def trim_silence(audio: np.ndarray) -> np.ndarray:

    x = np.asarray(audio, dtype=np.float32)
    if x.size == 0:
        return x

    trimmed, _ = librosa.effects.trim(x)
    return np.asarray(trimmed, dtype=np.float32)


class AudioProcessor:
    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = int(sample_rate)

    #Выравнивание громкости.
    def normalize(self, audio: np.ndarray) -> np.ndarray:

        return normalize_audio(np.asarray(audio, dtype=np.float32))

    #Обрезка тишины.
    def trim_silence(self, audio: np.ndarray) -> np.ndarray:

        return trim_silence(np.asarray(audio, dtype=np.float32))

    #Считает mel-спектрограмму
    def extract_mel(self, audio: np.ndarray) -> torch.Tensor:
        x = np.asarray(audio, dtype=np.float32)
        if x.size == 0:
            raise ValueError("audio пустое: невозможно извлечь mel")

        mel_np = extract_mel_spectrogram(x, sample_rate=int(self.sample_rate))  # [128, T]
        mel = torch.from_numpy(np.asarray(mel_np, dtype=np.float32)).unsqueeze(0)  # [1, 128, T]
        return mel

    #Нормализация mel
    @staticmethod
    def standardize_mel(mel: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        mean = mel.mean()
        std = mel.std(unbiased=False)
        if not torch.isfinite(std) or float(std) < float(eps):
            return mel - mean
        return (mel - mean) / (std + float(eps))


__all__ = ["AudioProcessor", "load_audio", "normalize_audio", "trim_silence"]

