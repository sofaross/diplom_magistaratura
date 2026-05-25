from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from configs.model_runtime import DEFAULT_SPEECH_MODEL
from src.inference.wav2vec2_inference import extract_embedding, transcribe, transcribe_and_embed
from src.models.wav2vec2_wrapper import Wav2Vec2Wrapper


@dataclass(frozen=True)
class Wav2Vec2InferenceResult:
    """Совместимый контейнер с результатами инференса для старого API."""

    texts: list[str]
    embeddings: torch.Tensor


class Wav2Vec2Multimodal:
    """Совместимый фасад над новым API `Wav2Vec2Wrapper` + функциями инференса.

    Исторически проект использовал один большой класс, который делал всё сразу:
    загрузку модели, предобработку, прогон, pooling и декодирование. Теперь эта
    ответственность разнесена:
    - `Wav2Vec2Wrapper` отвечает только за загрузку и хранение модели
    - функции из `src.inference` выполняют конкретные задачи инференса

    Этот класс оставлен только для обратной совместимости и делегирует вызовы
    в новую архитектуру.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_SPEECH_MODEL,
        *,
        device: str | torch.device | None = None,
        sample_rate: int = 16000,
        preprocess: bool = True,
        embedding_layer: int = -1,
        strict: bool = False,
        audio_processor: Any | None = None,
    ) -> None:
        """Создаёт совместимый объект поверх нового wrapper.

        Параметр `audio_processor` сохранён только затем, чтобы старый код с
        лишним аргументом не падал; в новом API он не используется.
        """

        self.model_name = str(model_name)
        self.sample_rate = int(sample_rate)
        self.preprocess = bool(preprocess)
        self.embedding_layer = int(embedding_layer)
        self.audio_processor = audio_processor
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.wrapper: Wav2Vec2Wrapper | None = None
        self.processor: Any | None = None
        self.model: Any | None = None
        self.load_error: str | None = None

        try:
            self.wrapper = Wav2Vec2Wrapper.from_pretrained(
                model_name=self.model_name,
                device=self.device,
                sample_rate=self.sample_rate,
            )
            self.processor = self.wrapper.processor
            self.model = self.wrapper.model
        except Exception as exc:
            if strict:
                raise
            self.wrapper = None
            self.processor = None
            self.model = None
            self.load_error = str(exc)

    def load_model(self, model_name: str) -> tuple[Any, Any]:
        """Совместимый метод загрузки, повторяющий старый контракт `(processor, model)`."""

        wrapper = Wav2Vec2Wrapper.from_pretrained(
            model_name=model_name,
            device=self.device,
            sample_rate=self.sample_rate,
        )
        return wrapper.processor, wrapper.model

    def _ensure_loaded(self) -> Wav2Vec2Wrapper:
        if self.wrapper is None:
            raise RuntimeError(self.load_error or "Wav2Vec2 модель не загружена.")
        return self.wrapper

    def _base_model(self) -> torch.nn.Module:
        """Совместимый доступ к базовой части модели без CTC-головы."""

        return self._ensure_loaded().base_model

    @property
    def embedding_dim(self) -> int:
        """Размерность speech embedding в старом API."""

        return self._ensure_loaded().hidden_size

    def transcribe(self, audio: Any) -> str | list[str]:
        """Распознаёт текст из аудио через новый функциональный API."""

        return transcribe(self._ensure_loaded(), audio, preprocess=self.preprocess)

    def recognize(self, audio: Any) -> str | list[str]:
        """Алиас старого API: `recognize(...)` эквивалентен `transcribe(...)`."""

        return self.transcribe(audio)

    def extract_embedding(
        self,
        audio: Any,
        *,
        embedding_layer: int | None = None,
        pool: str = "mean",
    ) -> torch.Tensor:
        """Извлекает embedding из указанного слоя базовой модели."""

        layer = self.embedding_layer if embedding_layer is None else int(embedding_layer)
        return extract_embedding(
            self._ensure_loaded(),
            audio,
            layer=layer,
            pool=pool,
            preprocess=self.preprocess,
        )

    def transcribe_and_embed(
        self,
        audio: Any,
        *,
        embedding_layer: int | None = None,
        pool: str = "mean",
    ) -> tuple[str | list[str], torch.Tensor]:
        """Возвращает и текст, и embedding за один проход."""

        layer = self.embedding_layer if embedding_layer is None else int(embedding_layer)
        return transcribe_and_embed(
            self._ensure_loaded(),
            audio,
            layer=layer,
            pool=pool,
            preprocess=self.preprocess,
        )

    def process(
        self,
        audio: Any,
        *,
        return_embeddings: bool = True,
        embedding_layer: int | None = None,
        pool: str = "mean",
    ) -> dict[str, Any]:
        """Совместимый словарный интерфейс для старых скриптов."""

        if return_embeddings:
            text, embedding = self.transcribe_and_embed(
                audio,
                embedding_layer=embedding_layer,
                pool=pool,
            )
            return {"text": text, "embedding": embedding}

        return {"text": self.transcribe(audio)}


__all__ = ["Wav2Vec2Multimodal", "Wav2Vec2InferenceResult"]
