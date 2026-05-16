from __future__ import annotations

from typing import Mapping

import numpy as np
import torch

from configs.config import ProjectConfig
from src.inference.wav2vec2_inference import transcribe
from src.models.wav2vec2_wrapper import Wav2Vec2Wrapper


class SpeechRecognitionService:
    """Service layer over Wav2Vec2 ASR models."""

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
        *,
        model_name: str | None = None,
        language_model_names: Mapping[str, str] | None = None,
        default_language: str | None = None,
        device: str | torch.device | None = None,
        preprocess: bool = False,
        wrapper: Wav2Vec2Wrapper | None = None,
        config: ProjectConfig | None = None,
    ) -> None:
        self.config = config or ProjectConfig()
        self.model_name = str(model_name) if model_name is not None else None
        self.language_model_names = self._build_language_model_map(language_model_names)
        self.default_language = self._normalize_language(default_language or self.config.default_speech_language)
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.preprocess = bool(preprocess)
        self._wrapper = wrapper
        self._wrappers: dict[str, Wav2Vec2Wrapper] = {}

    def recognize(
        self,
        audio: np.ndarray | torch.Tensor | list[float],
        *,
        language: str | None = None,
    ) -> str:
        wrapper = self._get_wrapper(language=language)
        result = transcribe(wrapper, audio, preprocess=self.preprocess)
        return result if isinstance(result, str) else result[0]

    def list_supported_languages(self) -> list[str]:
        return sorted(self.language_model_names.keys())

    def _get_wrapper(self, *, language: str | None = None) -> Wav2Vec2Wrapper:
        if self._wrapper is not None and self.model_name is not None:
            return self._wrapper

        resolved_language = self._normalize_language(language or self.default_language)
        if resolved_language not in self._wrappers:
            self._wrappers[resolved_language] = Wav2Vec2Wrapper.from_pretrained(
                model_name=self._resolve_model_name(resolved_language),
                device=self.device,
                sample_rate=self.config.sample_rate,
            )
        return self._wrappers[resolved_language]

    def _resolve_model_name(self, language: str) -> str:
        if self.model_name is not None:
            return self.model_name

        try:
            return self.language_model_names[language]
        except KeyError as exc:
            supported = ", ".join(self.list_supported_languages()) or "none"
            raise ValueError(f"Unsupported speech language {language!r}. Supported values: {supported}.") from exc

    def _build_language_model_map(self, language_model_names: Mapping[str, str] | None) -> dict[str, str]:
        source = dict(language_model_names or {
            "en": self.config.speech_model_name_en,
            "ru": self.config.speech_model_name_ru,
        })

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
            raise ValueError(f"Unsupported speech language {language!r}. Known aliases: {supported}.") from exc


__all__ = ["SpeechRecognitionService"]
