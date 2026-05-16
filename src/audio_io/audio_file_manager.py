from __future__ import annotations
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import re
import soundfile as sf
import sounddevice as sd
import librosa
import numpy as np
from configs.config import ProjectConfig
"""Работа с аудиофайлами: сохранение, загрузка, воспроизведение."""

DEFAULT_AUDIO_CONFIG = ProjectConfig()

"""Класс для работы с аудиофайлами: сохранение, загрузка, воспроизведение."""
class AudioFileManager:

    def __init__(
        self,
        save_dir: str | Path = DEFAULT_AUDIO_CONFIG.clean_recordings_dir,
        sample_rate: int = DEFAULT_AUDIO_CONFIG.sample_rate,
    ):
        """Инициализация менеджера аудиофайлов."""

        self.save_dir = Path(save_dir)
        self.sample_rate = int(sample_rate)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save(self, audio: np.ndarray, filename: str | None = None) -> Path:
        """Сохраняет аудио в WAV файл."""

        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        if x.size == 0:
            raise ValueError("Нельзя сохранить пустое аудио (0 сэмплов).")

        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{ts}.wav"

        filename = self.normalize_filename(str(filename))
        if not filename.lower().endswith(".wav"):
            filename = f"{filename}.wav"

        out_path = self.save_dir / filename

        if out_path.exists():
            stem = out_path.stem
            suffix = out_path.suffix
            i = 1
            while True:
                candidate = out_path.with_name(f"{stem}_{i}{suffix}")
                if not candidate.exists():
                    out_path = candidate
                    break
                i += 1

        x = np.clip(x, -1.0, 1.0)

        try:
            sf.write(str(out_path), x, int(self.sample_rate), subtype="PCM_16")
        except Exception as e:
            raise RuntimeError(f"Не удалось сохранить WAV: {out_path}") from e

        return out_path

    def load(self, filepath: str | Path, target_sample_rate: int | None = None) -> np.ndarray:
        """Загружает аудио из WAV файла."""

        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {path}")

        try:
            audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        except Exception as e:
            raise RuntimeError(f"Не удалось прочитать аудио: {path}") from e

        x = np.asarray(audio, dtype=np.float32)

        # Если многоканальный — приводим к mono (чтобы совпадало с ожиданиями моделей).
        if x.ndim == 2:
            if x.shape[1] >= 1:
                x = np.mean(x, axis=1, dtype=np.float32)
            else:
                x = x.reshape(-1)

        x = x.reshape(-1)

        target_sr = int(target_sample_rate) if target_sample_rate is not None else None
        if target_sr is not None and int(sr) != target_sr:
            try:
                x = librosa.resample(x, orig_sr=int(sr), target_sr=int(target_sr)).astype(np.float32, copy=False)
            except Exception as e:
                raise RuntimeError(
                    f"Ошибка пересемплирования {path}: {sr} -> {target_sr}"
                ) from e

        return np.asarray(x, dtype=np.float32)

    @staticmethod
    def play(audio: np.ndarray, sample_rate: int = 16000, wait: bool = True) -> None:
        """Воспроизводит аудио."""

        x = np.asarray(audio, dtype=np.float32)
        if x.size == 0:
            raise ValueError("Нельзя воспроизвести пустое аудио (0 сэмплов).")

        try:
            sd.play(x, samplerate=int(sample_rate))
            if wait:
                sd.wait()
        except Exception as e:
            raise RuntimeError("Ошибка воспроизведения (sounddevice/PortAudio).") from e

    @staticmethod
    def normalize_filename(filename: str) -> str:
        """Очищает имя файла от недопустимых символов.

        Делает имя безопасным для Windows:
        - убирает папки (оставляет только имя файла)
        - заменяет пробелы на `_`
        - удаляет символы: <>:"/\\|?* и прочие "не-словарные"
        """

        name = Path(str(filename)).name.strip()
        if not name:
            return "recording.wav"

        name = re.sub(r"\s+", "_", name)

        name = re.sub(r'[<>:"/\\\\|?*]', "", name)
        name = re.sub(r"[^A-Za-z0-9._-]", "", name)

        if not name or name in {".", ".."}:
            return "recording.wav"

        return name

    def get_info(self, filepath: str | Path) -> dict:
        """Возвращает информацию об аудиофайле."""

        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {path}")

        try:
            info = sf.info(str(path))
        except Exception as e:
            raise RuntimeError(f"Не удалось получить информацию о файле: {path}") from e

        payload = AudioFileInfo(
            filepath=str(path),
            samplerate=int(getattr(info, "samplerate", 0) or 0),
            channels=int(getattr(info, "channels", 0) or 0),
            frames=int(getattr(info, "frames", 0) or 0),
            duration=float(getattr(info, "duration", 0.0) or 0.0),
            format=str(getattr(info, "format", None)) if getattr(info, "format", None) is not None else None,
            subtype=str(getattr(info, "subtype", None)) if getattr(info, "subtype", None) is not None else None,
        )
        return asdict(payload)

@dataclass(frozen=True)
class AudioFileInfo:
    """Структурированная информация об аудиофайле."""

    filepath: str
    samplerate: int
    channels: int
    frames: int
    duration: float
    format: str | None = None
    subtype: str | None = None

__all__ = ["AudioFileManager", "AudioFileInfo"]
