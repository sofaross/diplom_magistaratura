from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping
import random

import numpy as np
import torch

from configs.config import ProjectConfig
from src.audio_io.audio_file_manager import AudioFileManager
from src.models.emotion_recognizer import EmotionRecognizer
from src.models.speech_model import Wav2Vec2Multimodal
from src.noise.noise_manager import NoiseManager
from src.preprocessing.audio_processing import normalize_audio, trim_silence
from src.services.text_correction_service import TextCorrectionService


@dataclass(slots=True)
class ProcessingResult:
    """Результат полного прохода pipeline по одному аудиофайлу."""

    original_audio_path: str
    processed_audio_path: str | None = None
    noise_type: str | None = None
    snr_db: float | None = None
    recognized_text: str = ""
    suggested_text: str = ""
    predicted_emotion: str = ""
    emotion_probabilities: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class MultimodalPipeline:
    """Единый pipeline: аудио -> шум -> ASR -> эмоция -> итоговый результат."""

    LANGUAGE_ALIASES: dict[str, str] = {
        "en": "en",
        "eng": "en",
        "english": "en",
        "ru": "ru",
        "rus": "ru",
        "russian": "ru",
    }

    def __init__(
        self,
        audio_manager: AudioFileManager | None = None,
        noise_manager: NoiseManager | None = None,
        speech_recognizer: Wav2Vec2Multimodal | None = None,
        emotion_recognizer: EmotionRecognizer | None = None,
        *,
        config: ProjectConfig | None = None,
        audio_file_manager: AudioFileManager | None = None,
        processed_audio_manager: AudioFileManager | None = None,
        speech_recognizers: Mapping[str, Wav2Vec2Multimodal] | None = None,
        speech_model_name: str | None = None,
        speech_model_names: Mapping[str, str] | None = None,
        default_speech_language: str | None = None,
        emotion_model_path: str | Path | None = None,
        emotion_map_path: str | Path | None = None,
        text_correction_service: TextCorrectionService | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.config = config or ProjectConfig()
        self.sample_rate = int(self.config.sample_rate)
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.audio_manager = audio_manager or audio_file_manager or AudioFileManager(
            sample_rate=self.sample_rate,
        )
        self.audio_file_manager = self.audio_manager
        self.processed_audio_manager = processed_audio_manager or AudioFileManager(
            save_dir=self.config.processed_audio_dir,
            sample_rate=self.sample_rate,
        )
        self.noise_manager = noise_manager or NoiseManager(
            noise_dir=self.config.noise_dir,
            sample_rate=self.sample_rate,
        )

        self.emotion_model_path = Path(emotion_model_path or self.config.emotion_checkpoint_path)
        self.emotion_map_path = Path(emotion_map_path or self.config.emotion_map_path)
        self._emotion_recognizer = emotion_recognizer

        self._speech_recognizer = speech_recognizer
        self.speech_model_name = str(speech_model_name) if speech_model_name is not None else None
        self.speech_model_names = self._build_speech_model_map(speech_model_names)
        self.default_speech_language = self._normalize_language(
            default_speech_language or self.config.default_speech_language
        )
        self._speech_recognizers = dict(speech_recognizers or {})
        self._text_correction_service = text_correction_service

    def list_available_noises(self) -> list[str]:
        """Возвращает список доступных типов шумов."""

        return self.noise_manager.list_available_noises()

    def list_available_noise_variants(self) -> list[str]:
        """Возвращает точные варианты файловых шумов."""

        return self.noise_manager.list_available_noise_variants()

    def process_audio(
        self,
        audio_path: str | Path,
        noise_type: str | None = None,
        snr_db: float | None = None,
        noise_file: str | Path | None = None,
        use_random_noise: bool = False,
        *,
        speech_language: str | None = None,
    ) -> ProcessingResult:
        """Выполняет полный цикл обработки аудиофайла."""

        source_path = Path(audio_path).expanduser().resolve()
        result = ProcessingResult(
            original_audio_path=str(source_path),
            processed_audio_path=str(source_path),
        )

        selected_noise_modes = int(noise_type is not None) + int(noise_file is not None) + int(use_random_noise)
        if selected_noise_modes > 1:
            result.errors.append(
                "Нужно выбрать только один способ добавления шума: noise_type, noise_file или use_random_noise."
            )
            return result

        try:
            audio = self.audio_manager.load(source_path, target_sample_rate=self.sample_rate)
            prepared_audio = self._prepare_audio(audio)
        except Exception as exc:
            result.errors.append(f"Ошибка загрузки аудио: {exc}")
            return result

        processed_audio = prepared_audio
        applied_noise_type: str | None = None
        effective_snr_db: float | None = None

        if selected_noise_modes == 1:
            effective_snr_db = float(10.0 if snr_db is None else snr_db)
            try:
                processed_audio, applied_noise_type = self._apply_requested_noise(
                    prepared_audio,
                    noise_type=noise_type,
                    noise_file=noise_file,
                    use_random_noise=use_random_noise,
                    snr_db=effective_snr_db,
                )
            except Exception as exc:
                result.errors.append(f"Ошибка наложения шума: {exc}")
                processed_audio = prepared_audio
                applied_noise_type = None
                effective_snr_db = None
            else:
                try:
                    saved_path = self._save_noisy_audio(
                        audio=processed_audio,
                        original_audio_path=source_path,
                        noise_type=applied_noise_type,
                        snr_db=effective_snr_db,
                    )
                    result.processed_audio_path = str(saved_path)
                except Exception as exc:
                    result.errors.append(f"Ошибка сохранения обработанного аудио: {exc}")
                    result.processed_audio_path = None

        result.noise_type = applied_noise_type
        result.snr_db = effective_snr_db

        try:
            resolved_language = self._normalize_language(speech_language or self.default_speech_language)
            recognizer = self._get_speech_recognizer(speech_language=resolved_language)
            recognized_text = recognizer.transcribe(processed_audio)
            raw_text = recognized_text if isinstance(recognized_text, str) else recognized_text[0]
            result.recognized_text = str(raw_text or "").strip()
            result.suggested_text = self._suggest_text(result.recognized_text, language=resolved_language)
        except Exception as exc:
            result.errors.append(f"Ошибка распознавания речи: {exc}")

        try:
            emotion_recognizer = self._get_emotion_recognizer()
            emotion, probabilities = emotion_recognizer.recognize(processed_audio)
            result.predicted_emotion = emotion
            result.emotion_probabilities = probabilities
        except Exception as exc:
            result.errors.append(f"Ошибка распознавания эмоции: {exc}")

        return result

    def process_audio_file(
        self,
        audio_path: str | Path,
        *,
        noise: str | None = None,
        snr_db: float | None = None,
        noise_file: str | Path | None = None,
        speech_language: str | None = None,
    ) -> ProcessingResult:
        """Совместимость со старым именем метода."""

        return self.process_audio(
            audio_path,
            noise_type=None if noise == "random" else noise,
            snr_db=snr_db,
            noise_file=noise_file,
            use_random_noise=noise == "random",
            speech_language=speech_language,
        )

    def _prepare_audio(self, audio: np.ndarray) -> np.ndarray:
        normalized = normalize_audio(np.asarray(audio, dtype=np.float32))
        trimmed = trim_silence(normalized)

        if trimmed.size > 0:
            return np.asarray(trimmed, dtype=np.float32)
        if normalized.size > 0:
            return np.asarray(normalized, dtype=np.float32)
        return np.zeros(max(1, self.sample_rate // 100), dtype=np.float32)

    def _apply_requested_noise(
        self,
        audio: np.ndarray,
        *,
        noise_type: str | None,
        noise_file: str | Path | None,
        use_random_noise: bool,
        snr_db: float,
    ) -> tuple[np.ndarray, str]:
        if noise_file is not None:
            noise_path = Path(noise_file).expanduser().resolve()
            noise_audio = self.audio_manager.load(noise_path, target_sample_rate=self.sample_rate)
            mixed_audio = self.noise_manager.add_noise(audio, noise_audio, snr_db=snr_db)
            return mixed_audio, noise_path.stem.lower()

        selected_noise_type = noise_type
        if use_random_noise:
            available_noises = self.list_available_noises()
            if not available_noises:
                raise ValueError("В проекте не найдено доступных шумов.")
            selected_noise_type = random.choice(available_noises)

        if selected_noise_type is None:
            raise ValueError("Не задан тип шума для обработки.")

        duration = float(np.asarray(audio, dtype=np.float32).size) / float(self.sample_rate)
        normalized_noise = str(selected_noise_type).strip().lower()

        if normalized_noise in self.noise_manager.SYNTHETIC_NOISES or normalized_noise == "brownian":
            synthetic_name = "brown" if normalized_noise == "brownian" else normalized_noise
            noise_audio = self.noise_manager.generate_synthetic_noise(synthetic_name, duration)
            mixed_audio = self.noise_manager.add_noise(audio, noise_audio, snr_db=snr_db)
            return mixed_audio, synthetic_name

        noise_audio = self.noise_manager.get_real_noise(normalized_noise, duration)
        mixed_audio = self.noise_manager.add_noise(audio, noise_audio, snr_db=snr_db)
        resolved_noise_type = self.noise_manager.get_real_noise_group(normalized_noise)
        return mixed_audio, resolved_noise_type

    def _save_noisy_audio(
        self,
        *,
        audio: np.ndarray,
        original_audio_path: Path,
        noise_type: str,
        snr_db: float,
    ) -> Path:
        base_name = original_audio_path.stem
        safe_noise = str(noise_type).strip().lower().replace(" ", "_")
        snr_label = str(float(snr_db)).replace(".", "_").replace("-", "minus")
        filename = f"{base_name}_noise_{safe_noise}_snr{snr_label}.wav"
        return self.processed_audio_manager.save(audio, filename=filename)

    def _get_emotion_recognizer(self) -> EmotionRecognizer:
        if self._emotion_recognizer is None:
            self._emotion_recognizer = EmotionRecognizer(
                emotion_model_path=self.emotion_model_path,
                emotion_map_path=self.emotion_map_path,
                device=self.device,
            )
        return self._emotion_recognizer

    def _get_text_correction_service(self) -> TextCorrectionService:
        if self._text_correction_service is None:
            self._text_correction_service = TextCorrectionService(
                config=self.config,
                default_language=self.default_speech_language,
                device=self.device,
            )
        return self._text_correction_service

    def _suggest_text(self, text: str, *, language: str) -> str:
        if not str(text or "").strip():
            return ""
        return self._get_text_correction_service().suggest(text, language=language)

    def _get_speech_recognizer(self, *, speech_language: str | None = None) -> Wav2Vec2Multimodal:
        if self._speech_recognizer is not None:
            return self._speech_recognizer

        resolved_language = self._normalize_language(speech_language or self.default_speech_language)
        cache_key = self.speech_model_name or resolved_language

        if cache_key not in self._speech_recognizers:
            model_name = self._resolve_speech_model_name(resolved_language)
            self._speech_recognizers[cache_key] = Wav2Vec2Multimodal(
                model_name=model_name,
                device=self.device,
                sample_rate=self.sample_rate,
                preprocess=False,
                strict=True,
            )

        return self._speech_recognizers[cache_key]

    def _resolve_speech_model_name(self, language: str) -> str:
        if self.speech_model_name is not None:
            return self.speech_model_name

        try:
            return self.speech_model_names[language]
        except KeyError as exc:
            supported = ", ".join(sorted(self.speech_model_names.keys())) or "none"
            raise ValueError(f"Неподдерживаемый язык речи {language!r}. Поддерживаются: {supported}.") from exc

    def _build_speech_model_map(self, speech_model_names: Mapping[str, str] | None) -> dict[str, str]:
        source = dict(
            speech_model_names
            or {
                "en": self.config.speech_model_name_en,
                "ru": self.config.speech_model_name_ru,
            }
        )
        normalized: dict[str, str] = {}
        for raw_language, model_name in source.items():
            normalized[self._normalize_language(raw_language)] = str(model_name)
        return normalized

    @classmethod
    def _normalize_language(cls, language: str) -> str:
        normalized = str(language).strip().lower()
        try:
            return cls.LANGUAGE_ALIASES[normalized]
        except KeyError as exc:
            supported = ", ".join(sorted(cls.LANGUAGE_ALIASES.keys()))
            raise ValueError(f"Неподдерживаемый язык речи {language!r}. Допустимые значения: {supported}.") from exc


__all__ = ["MultimodalPipeline", "ProcessingResult"]
