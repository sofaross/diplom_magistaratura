from __future__ import annotations

from pathlib import Path

import numpy as np

from configs.config import ProjectConfig
from src.audio_io.audio_capture import MicrophoneCapture
from src.audio_io.audio_file_manager import AudioFileManager
from src.dto.multimodal_result_dto import PreparedAudioDto
from src.preprocessing.audio_processing import normalize_audio, trim_silence


class AudioInputService:
    """Принимает аудио из файла или с микрофона и готовит его к дальнейшей обработке."""

    SUPPORTED_EXTENSIONS: tuple[str, ...] = (".wav", ".flac", ".mp3", ".ogg", ".m4a")

    def __init__(
        self,
        *,
        sample_rate: int | None = None,
        source_file_manager: AudioFileManager | None = None,
        prepared_file_manager: AudioFileManager | None = None,
        microphone_capture: MicrophoneCapture | None = None,
        config: ProjectConfig | None = None,
    ) -> None:
        self.config = config or ProjectConfig()
        self.sample_rate = int(sample_rate or self.config.sample_rate)
        self.source_file_manager = source_file_manager or AudioFileManager(sample_rate=self.sample_rate)
        self.prepared_file_manager = prepared_file_manager or AudioFileManager(
            save_dir=self.config.processed_audio_dir,
            sample_rate=self.sample_rate,
        )
        self.microphone_capture = microphone_capture or MicrophoneCapture(
            sample_rate=self.sample_rate,
            auto_save=True,
        )

    def load_audio_file(
        self,
        file_path: str | Path,
        *,
        save_prepared_copy: bool = False,
        prepared_filename: str | None = None,
    ) -> PreparedAudioDto:
        path = Path(file_path).expanduser().resolve()
        self._validate_audio_file(path)

        audio = self.source_file_manager.load(path, target_sample_rate=self.sample_rate)
        prepared_audio = self._prepare_audio(audio)

        prepared_path = None
        if save_prepared_copy:
            prepared_path = self._save_prepared_audio(
                prepared_audio,
                filename=prepared_filename or f"{path.stem}_prepared.wav",
            )

        return PreparedAudioDto(
            source_type="file",
            original_file_path=path,
            audio=prepared_audio,
            sample_rate=self.sample_rate,
            duration_seconds=self._duration_seconds(prepared_audio),
            prepared_file_path=prepared_path,
        )

    def capture_voice_message(
        self,
        *,
        duration: float = 5.0,
        filename: str | None = None,
        save_prepared_copy: bool = False,
        prepared_filename: str | None = None,
    ) -> PreparedAudioDto:
        audio, original_path = self.microphone_capture.listen_and_save(duration=duration, filename=filename)
        prepared_audio = self._prepare_audio(audio)

        prepared_path = None
        if save_prepared_copy:
            default_name = prepared_filename or f"{original_path.stem}_prepared.wav"
            prepared_path = self._save_prepared_audio(prepared_audio, filename=default_name)

        return PreparedAudioDto(
            source_type="microphone",
            original_file_path=original_path.resolve(),
            audio=prepared_audio,
            sample_rate=self.sample_rate,
            duration_seconds=self._duration_seconds(prepared_audio),
            prepared_file_path=prepared_path,
        )

    def persist_prepared_audio(
        self,
        prepared_audio: PreparedAudioDto,
        *,
        filename: str | None = None,
    ) -> PreparedAudioDto:
        prepared_path = self._save_prepared_audio(
            prepared_audio.audio,
            filename=filename or f"{prepared_audio.original_file_path.stem}_prepared.wav",
        )
        prepared_audio.prepared_file_path = prepared_path
        return prepared_audio

    def _validate_audio_file(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        if not path.is_file():
            raise ValueError(f"Expected a file, got: {path}")
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported audio format {path.suffix!r}. "
                f"Supported formats: {', '.join(self.SUPPORTED_EXTENSIONS)}."
            )

    def _prepare_audio(self, audio: np.ndarray) -> np.ndarray:
        normalized = normalize_audio(np.asarray(audio, dtype=np.float32))
        trimmed = trim_silence(normalized)

        if trimmed.size > 0:
            return np.asarray(trimmed, dtype=np.float32)
        if normalized.size > 0:
            return np.asarray(normalized, dtype=np.float32)
        return np.zeros(max(1, self.sample_rate // 100), dtype=np.float32)

    def _save_prepared_audio(self, audio: np.ndarray, *, filename: str) -> Path:
        return self.prepared_file_manager.save(audio, filename=filename)

    def _duration_seconds(self, audio: np.ndarray) -> float:
        return float(np.asarray(audio, dtype=np.float32).size) / float(self.sample_rate)


__all__ = ["AudioInputService"]
