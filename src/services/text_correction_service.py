from __future__ import annotations

from typing import Any, Mapping

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from configs.config import ProjectConfig


class TextCorrectionService:
    """Сервис постобработки ASR-текста для поддерживаемых языков проекта."""

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
        model_names: Mapping[str, str] | None = None,
        default_language: str | None = None,
        device: str | torch.device | None = None,
        config: ProjectConfig | None = None,
    ) -> None:
        self.config = config or ProjectConfig()
        self.default_language = self._normalize_language(default_language or self.config.default_speech_language)
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_names = self._build_model_map(model_names)
        self._tokenizers: dict[str, Any] = {}
        self._models: dict[str, Any] = {}

    def suggest(self, text: str, *, language: str | None = None) -> str:
        raw_text = str(text or "").strip()
        if not raw_text:
            return ""

        resolved_language = self._normalize_language(language or self.default_language)
        fallback_text = raw_text

        try:
            tokenizer, model = self._get_model_pair(resolved_language)
        except Exception:
            return fallback_text

        try:
            corrected = self._generate_correction(raw_text, tokenizer=tokenizer, model=model)
            corrected = str(corrected or "").strip()
            return corrected or fallback_text
        except Exception:
            return fallback_text

    def _get_model_pair(self, language: str) -> tuple[Any, Any]:
        if language not in self._models:
            model_name = self.model_names[language]
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            model.to(self.device)
            model.eval()
            self._tokenizers[language] = tokenizer
            self._models[language] = model
        return self._tokenizers[language], self._models[language]

    def _generate_correction(self, text: str, *, tokenizer: Any, model: Any) -> str:
        inputs = tokenizer(
            str(text),
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        input_length = int(inputs["input_ids"].shape[1])
        max_new_tokens = max(16, min(256, int(round(input_length * 1.5))))

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                early_stopping=True,
            )

        return tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def _build_model_map(self, model_names: Mapping[str, str] | None) -> dict[str, str]:
        source = dict(
            model_names
            or {
                "en": self.config.text_correction_model_name_en,
                "ru": self.config.text_correction_model_name_ru,
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
            raise ValueError(f"Unsupported text correction language {language!r}. Known aliases: {supported}.") from exc


__all__ = ["TextCorrectionService"]
