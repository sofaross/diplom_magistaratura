from __future__ import annotations

"""Запись аудио с микрофона.

Модуль изолирует работу с библиотекой `sounddevice` и PortAudio. Это позволяет:
- держать основной код распознавания "чистым" и тестируемым;
- импортировать пакет даже если `sounddevice` не установлен (импорт делается внутри методов).
"""

import numpy as np


class MicrophoneCapture:
    """Класс для записи аудио с микрофона.

    Основные методы:
    - `check_microphone()` проверяет, что устройство ввода доступно и поддерживает 16kHz mono.
    - `listen(duration)` записывает аудио и возвращает numpy-массив float32.

    Пример:
        capture = MicrophoneCapture(sample_rate=16000, min_rms=0.005)
        audio = capture.listen(duration=5.0)
    """

    def __init__(self, sample_rate: int = 16000, min_rms: float = 0.005, max_clip_ratio: float = 0.01):
        """Создаёт объект для записи с микрофона.

        Args:
            sample_rate: частота дискретизации записи (должна быть 16000 для совместимости).
            min_rms: порог "слишком тихо" (RMS). Если ниже, выбрасываем ValueError.
            max_clip_ratio: максимальная доля "клиппинга" (0..1). Если выше, выбрасываем ValueError.
        """

        self.sample_rate = int(sample_rate)
        self.min_rms = float(min_rms)
        self.max_clip_ratio = float(max_clip_ratio)

    def check_microphone(self) -> None:
        """Проверяет доступность микрофона и корректность настроек.

        Raises:
            RuntimeError: если `sounddevice` не установлен или микрофон не доступен.
        """

        try:
            import sounddevice as sd
        except Exception as e:
            raise RuntimeError(
                "Не найден модуль sounddevice. Установите его: pip install sounddevice\n"
                "Если установка не удаётся, можно поставить PyAudio и заменить реализацию записи."
            ) from e

        try:
            sd.check_input_settings(samplerate=int(self.sample_rate), channels=1)
        except Exception as e:
            raise RuntimeError(
                "Микрофон не найден или не поддерживает запись 16 kHz/mono. "
                "Проверьте устройство ввода в системе."
            ) from e

    def listen(self, duration: float = 5.0) -> np.ndarray:
        """Записывает звук с микрофона и возвращает аудио.

        Args:
            duration: длительность записи в секундах.

        Returns:
            np.ndarray формы [N], dtype=float32, диапазон обычно [-1, 1].

        Raises:
            ValueError: если duration <= 0, если слишком тихо, если есть клиппинг, если NaN/Inf.
            RuntimeError: если микрофон не доступен или произошла ошибка записи.
        """

        duration = float(duration)
        if duration <= 0:
            raise ValueError("duration должен быть > 0")

        self.check_microphone()

        try:
            import sounddevice as sd
        except Exception as e:
            # Теоретически сюда не попадём (check_microphone уже проверил), но пусть будет.
            raise RuntimeError("Не удалось импортировать sounddevice.") from e

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

        return audio

    @staticmethod
    def _calculate_rms(audio: np.ndarray) -> float:
        """Считает RMS (среднеквадратичную амплитуду) сигнала."""

        x = np.asarray(audio, dtype=np.float32)
        if x.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))

    @staticmethod
    def _check_clipping(audio: np.ndarray, threshold: float = 0.99) -> float:
        """Оценивает долю сэмплов, близких к насыщению (клиппинг).

        Returns:
            Доля (0..1) сэмплов, у которых |x| > threshold.
        """

        x = np.asarray(audio, dtype=np.float32)
        if x.size == 0:
            return 0.0
        return float(np.mean(np.abs(x) > float(threshold)))


__all__ = ["MicrophoneCapture"]
