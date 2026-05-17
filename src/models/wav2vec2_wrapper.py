from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCTC, AutoProcessor, Wav2Vec2Processor


@dataclass(slots=True)
class Wav2Vec2Wrapper:
    """Лёгкая обёртка над загруженной Wav2Vec2 CTC-моделью и её processor.

    Класс ничего не знает про предобработку аудио, декодирование или pooling.
    Его задача только одна: загрузить модель, хранить её и отдавать базовые
    свойства, которые нужны инференс-функциям.
    """

    model_name: str
    device: torch.device
    processor: Any
    model: Any
    sample_rate: int = 16000

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "facebook/wav2vec2-base-960h",
        device: str | torch.device | None = None,
        sample_rate: int = 16000,
    ) -> "Wav2Vec2Wrapper":
        """Загружает processor и CTC-модель из HuggingFace и переносит их на нужное устройство."""

        resolved_device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        try:
            processor = cls._load_processor(model_name)
            model = AutoModelForCTC.from_pretrained(model_name)
        except Exception as exc:
            raise RuntimeError(
                f"Не удалось загрузить Wav2Vec2 CTC модель {model_name!r}. "
                "Проверьте имя модели и наличие интернета/кэша HuggingFace."
            ) from exc

        model.to(resolved_device)
        model.eval()

        return cls(
            model_name=str(model_name),
            device=resolved_device,
            processor=processor,
            model=model,
            sample_rate=int(sample_rate),
        )

    @staticmethod
    def _load_processor(model_name: str) -> Any:
        try:
            return AutoProcessor.from_pretrained(model_name)
        except ImportError as exc:
            error_text = str(exc)
            if "pyctcdecode" not in error_text:
                raise
            return Wav2Vec2Processor.from_pretrained(model_name)

    @property
    def hidden_size(self) -> int:
        """Размерность скрытого представления, которое используется как speech embedding."""

        config = getattr(self.model, "config", None)
        if config is not None and hasattr(config, "hidden_size"):
            return int(config.hidden_size)
        return int(self.base_model.config.hidden_size)

    @property
    def embedding_dim(self) -> int:
        """Совместимый алиас для кода, где раньше использовалось имя embedding_dim."""

        return self.hidden_size

    @property
    def base_model(self) -> torch.nn.Module:
        """Возвращает базовую акустическую часть модели без CTC-головы."""

        if hasattr(self.model, "wav2vec2"):
            return self.model.wav2vec2
        for attr_name in ("hubert", "data2vec_audio"):
            if hasattr(self.model, attr_name):
                return getattr(self.model, attr_name)
        if isinstance(self.model, torch.nn.Module):
            return self.model
        raise RuntimeError("Не удалось определить базовую часть speech-модели.")


__all__ = ["Wav2Vec2Wrapper"]
