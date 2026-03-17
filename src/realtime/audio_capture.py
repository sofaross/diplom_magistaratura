from __future__ import annotations
import numpy as np
import sounddevice as sd

# ===============================
# Класс для записи аудио с микрофона
# ===============================
class MicrophoneCapture:
    # Настройки микрофона
    def __init__(self, sample_rate: int = 16000, min_rms: float = 0.005, max_clip_ratio: float = 0.01):
        self.sample_rate = int(sample_rate)
        self.min_rms = float(min_rms)
        self.max_clip_ratio = float(max_clip_ratio)

    #Проверяет доступность микрофона
    def check_microphone(self) -> None:
        try:
            sd.check_input_settings(samplerate=int(self.sample_rate), channels=1)
        except Exception as e:
            raise RuntimeError(
                "Микрофон не найден или не поддерживает запись 16 kHz/mono. "
                "Проверьте устройство ввода в системе."
            ) from e

    #Запись звука
    def listen(self, duration: float = 5.0) -> np.ndarray:
        duration = float(duration)
        if duration <= 0:
            raise ValueError("длительность должена быть > 0")

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


__all__ = ["MicrophoneCapture"]
