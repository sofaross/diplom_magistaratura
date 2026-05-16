from __future__ import annotations

"""Запись аудио с микрофона."""

from pathlib import Path

import numpy as np
import sounddevice as sd

from configs.config import ProjectConfig
from src.audio_io.audio_file_manager import AudioFileManager

DEFAULT_CAPTURE_CONFIG = ProjectConfig()
DEFAULT_CAPTURE_DIR = Path(DEFAULT_CAPTURE_CONFIG.clean_recordings_dir)

class MicrophoneCapture:
    """Настройки записи."""

    def __init__(self,sample_rate: int = 16000,min_rms: float = 0.005,max_clip_ratio: float = 0.01,*,file_manager: AudioFileManager | None = None,
        auto_save: bool = True,
    ) -> None:

        self.sample_rate = int(sample_rate)
        self.min_rms = float(min_rms)
        self.max_clip_ratio = float(max_clip_ratio)

        self.file_manager = file_manager or AudioFileManager(
            save_dir=DEFAULT_CAPTURE_DIR,
            sample_rate=self.sample_rate,
        )
        self.auto_save = bool(auto_save)
        self.last_saved_path: Path | None = None

    def check_microphone(self) -> None:
        """Проверяет, что микрофон доступен и поддерживает нужные параметры."""

        try:
            sd.check_input_settings(samplerate=int(self.sample_rate), channels=1)
        except Exception as e:
            raise RuntimeError(
                "Микрофон не найден или не поддерживает запись 16 kHz/mono. "
                "Проверьте устройство ввода в системе."
            ) from e

    def listen(self, duration: float = 5.0, *, save: bool | None = None, filename: str | None = None) -> np.ndarray:
        """Записывает звук с микрофона."""

        duration = float(duration)
        if duration <= 0:
            raise ValueError("длительность должна быть > 0")

        self.check_microphone()

        try:
            frames = int(round(duration * int(self.sample_rate)))
            audio = sd.rec(frames, samplerate=int(self.sample_rate), channels=1, dtype="float32")
            sd.wait()
        except Exception as e:
            raise RuntimeError("Ошибка записи с микрофона (sounddevice/PortAudio).") from e

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if not np.isfinite(audio).all():
            raise ValueError("Запись содержит нечисловые значения (NaN/Inf).")

        rms = self._calculate_rms(audio)
        if rms < float(self.min_rms):
            raise ValueError(
                f"Слишком тихо: rms={rms:.6f}. "
                "Попробуйте говорить громче или увеличьте чувствительность микрофона."
            )

        clip_ratio = self._check_clipping(audio)
        if clip_ratio > float(self.max_clip_ratio):
            raise ValueError(
                f"Слишком громко/клиппинг: {clip_ratio*100:.2f}% сэмплов близко к насыщению. "
                "Уменьшите усиление микрофона."
            )

        # Автосохранение (если включено).
        do_save = self.auto_save if save is None else bool(save)
        if do_save:
            if self.file_manager is None:
                raise RuntimeError(
                    "Запрошено сохранение записи, но AudioFileManager не задан. "
                    "Передайте file_manager=AudioFileManager(...) в MicrophoneCapture."
                )
            if int(self.file_manager.sample_rate) != int(self.sample_rate):
                raise ValueError(
                    "Нельзя сохранить запись: sample_rate MicrophoneCapture и AudioFileManager не совпадают. "
                    f"MicrophoneCapture={self.sample_rate}, AudioFileManager={self.file_manager.sample_rate}. "
                    "Создайте AudioFileManager с таким же sample_rate."
                )
            self.last_saved_path = self.file_manager.save(audio, filename=filename)

        return audio

    # Считаем громкость
    @staticmethod
    def _calculate_rms(audio: np.ndarray) -> float:
        x = np.asarray(audio, dtype=np.float32)
        if x.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))

    # Ищем перегрузки
    @staticmethod
    def _check_clipping(audio: np.ndarray, threshold: float = 0.99) -> float:
        x = np.asarray(audio, dtype=np.float32)
        if x.size == 0:
            return 0.0
        return float(np.mean(np.abs(x) > float(threshold)))

    def listen_and_save(self, duration: float = 5.0, filename: str | None = None) -> tuple[np.ndarray, Path]:
        """Записывает и сразу сохраняет в WAV файл."""

        audio = self.listen(duration=duration, save=True, filename=filename)
        if self.last_saved_path is None:
            raise RuntimeError("Не удалось сохранить запись (неожиданное состояние).")
        return audio, self.last_saved_path


__all__ = ["MicrophoneCapture"]
