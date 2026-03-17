"""Единый Wav2Vec2 класс для мультимодальности: ASR (текст) + эмбеддинги.

Идея:
- Раньше в проекте ASR (текст) и speech-эмбеддинги жили в разных классах.

Этот модуль делает логичнее: один Wav2Vec2 с CTC-головой делает и то, и другое.

Важно:
- Для распознавания текста нужна модель Wav2Vec2, ДОобученная под ASR (CTC head).
  Например:
    - "facebook/wav2vec2-base-960h" (английский)
    - "jonatasgrosman/wav2vec2-large-xlsr-53-russian" (русский)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.preprocessing.audio_preprocessing import normalize_audio, trim_silence


@dataclass(frozen=True)
class Wav2Vec2InferenceResult:
    """Результат одного прогона модели: текст + эмбеддинги."""

    texts: list[str]
    embeddings: torch.Tensor  # shape: (batch, hidden_size)


class Wav2Vec2Multimodal:
    """Wav2Vec2 "2-в-1": распознавание текста (CTC) и извлечение эмбеддингов.

    Основные методы:
    - `transcribe(audio)` -> str | list[str]
    - `extract_embedding(audio)` -> torch.Tensor (batch, hidden_size)
    - `transcribe_and_embed(audio)` -> (text(s), embedding(s)) за ОДИН прогон

    Пример:
        m = Wav2Vec2Multimodal(model_name="facebook/wav2vec2-base-960h", device="cpu")
        text, emb = m.transcribe_and_embed(audio_np)   # один прогон
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        *,
        device: str | torch.device | None = None,
        sample_rate: int = 16000,
        # По умолчанию делаем те же шаги, что и в обучении проекта: normalize + trim_silence.
        preprocess: bool = True,
        # Из какого слоя брать эмбеддинг. -1 = последний слой (самый частый вариант).
        embedding_layer: int = -1,
        # Если strict=False, объект создаётся даже при ошибке загрузки, а ошибка будет в `load_error`.
        strict: bool = False,
    ) -> None:
        self.model_name = str(model_name)
        self.sample_rate = int(sample_rate)
        self.preprocess = bool(preprocess)
        self.embedding_layer = int(embedding_layer)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        self.processor: Any | None = None
        self.model: Any | None = None
        self.load_error: str | None = None

        try:
            self.processor, self.model = self.load_model(self.model_name)
        except Exception as e:
            if strict:
                raise
            self.processor, self.model = None, None
            self.load_error = str(e)

    def load_model(self, model_name: str) -> tuple[Any, Any]:
        """Загружает Wav2Vec2 CTC модель и processor из HuggingFace.

        Returns:
            (processor, model)

        Raises:
            RuntimeError: если transformers не установлен или модель не загрузилась.
        """

        try:
            from transformers import AutoModelForCTC, AutoProcessor
        except Exception as e:
            raise RuntimeError(
                "Не удалось импортировать transformers. Установите зависимости: pip install -r requirements.txt"
            ) from e

        try:
            processor = AutoProcessor.from_pretrained(model_name)
            model = AutoModelForCTC.from_pretrained(model_name)
        except Exception as e:
            raise RuntimeError(
                f"Не удалось загрузить Wav2Vec2 CTC модель {model_name!r}. "
                "Проверьте имя модели и наличие интернета/кэша HuggingFace."
            ) from e

        model.to(self.device)
        model.eval()
        return processor, model

    @property
    def embedding_dim(self) -> int:
        """Размерность эмбеддинга (hidden_size) базовой части Wav2Vec2."""

        if self.model is None:
            raise RuntimeError(self.load_error or "Wav2Vec2 модель не загружена.")

        base = self._base_model()
        return int(base.config.hidden_size)

    def _base_model(self) -> Any:
        """Возвращает базовую часть модели (Wav2Vec2Model), без CTC головы."""

        if self.model is None:
            raise RuntimeError(self.load_error or "Wav2Vec2 модель не загружена.")

        # Для Wav2Vec2ForCTC базовая часть лежит в `.wav2vec2`.
        if hasattr(self.model, "wav2vec2"):
            return self.model.wav2vec2
        # На всякий случай поддержим другие CTC модели, где базовая часть может называться иначе.
        for attr in ("hubert", "data2vec_audio"):
            if hasattr(self.model, attr):
                return getattr(self.model, attr)
        return self.model

    def _ensure_loaded(self) -> None:
        if self.model is None or self.processor is None:
            raise RuntimeError(self.load_error or "Wav2Vec2 модель не загружена.")

    @staticmethod
    def _as_batch(audio: Any) -> list[np.ndarray]:
        """Приводит вход к списку np.ndarray (batch)."""

        if isinstance(audio, (list, tuple)):
            items = list(audio)
        else:
            items = [audio]

        out: list[np.ndarray] = []
        for item in items:
            if isinstance(item, torch.Tensor):
                item = item.detach().cpu().numpy()
            x = np.asarray(item, dtype=np.float32).reshape(-1)
            out.append(x)
        return out

    def _preprocess_audio(self, batch: list[np.ndarray]) -> list[np.ndarray]:
        """normalize + trim_silence (как в обучении проекта)."""

        if not self.preprocess:
            return batch

        out: list[np.ndarray] = []
        for x in batch:
            y = normalize_audio(np.asarray(x, dtype=np.float32))
            y = trim_silence(np.asarray(y, dtype=np.float32))
            y = np.asarray(y, dtype=np.float32)
            out.append(y)
        return out

    def _pool_hidden(self, hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        """Пулит (усредняет) скрытые состояния по времени с учётом паддинга."""

        if attention_mask is None:
            return hidden.mean(dim=1)

        base = self._base_model()

        if not hasattr(base, "_get_feature_vector_attention_mask"):
            # Фолбэк: если по какой-то причине нет метода, усредняем без маски.
            return hidden.mean(dim=1)

        feature_mask = base._get_feature_vector_attention_mask(hidden.shape[1], attention_mask)
        feature_mask = feature_mask.unsqueeze(-1).to(dtype=hidden.dtype)

        masked_sum = (hidden * feature_mask).sum(dim=1)
        denom = feature_mask.sum(dim=1).clamp(min=1.0)
        return masked_sum / denom

    def _pick_hidden_layer(self, base_out: Any, layer: int) -> torch.Tensor:
        """Выбирает тензор скрытых состояний нужного слоя."""

        # Если слой -1, можно обойтись last_hidden_state без включения output_hidden_states.
        if int(layer) == -1:
            return base_out.last_hidden_state

        hidden_states = getattr(base_out, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError(
                "Нельзя взять промежуточный слой: модель была вызвана без output_hidden_states=True."
            )

        try:
            return hidden_states[int(layer)]
        except Exception as e:
            raise RuntimeError(f"Некорректный embedding_layer={layer}.") from e

    def _forward_once(
        self,
        audio: Any,
        *,
        return_text: bool,
        return_embedding: bool,
        embedding_layer: int | None,
    ) -> Wav2Vec2InferenceResult:
        """Один прогон Wav2Vec2: возвращает текст и эмбеддинги (по необходимости)."""

        self._ensure_loaded()

        batch = self._as_batch(audio)
        batch = self._preprocess_audio(batch)

        # Защита от полностью пустых клипов после trim_silence.
        safe_batch: list[np.ndarray] = []
        for x in batch:
            if x.size == 0:
                safe_batch.append(np.zeros(160, dtype=np.float32))
            else:
                safe_batch.append(x)

        processor = self.processor
        model = self.model
        assert processor is not None
        assert model is not None

        # Processor делает паддинг и создаёт attention_mask.
        inputs = processor(
            safe_batch,
            sampling_rate=int(self.sample_rate),
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )

        input_values = inputs.get("input_values")
        if input_values is None:
            raise RuntimeError("processor не вернул input_values (неожиданный формат).")

        attention_mask = inputs.get("attention_mask")

        input_values = input_values.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Для эмбеддинга нужен хотя бы последний слой (last_hidden_state).
        need_embedding = bool(return_embedding)
        need_text = bool(return_text)

        # Если хотим слой != -1, то нужно запросить hidden_states.
        layer = self.embedding_layer if embedding_layer is None else int(embedding_layer)
        need_hidden_states = need_embedding and int(layer) != -1

        base = self._base_model()
        with torch.inference_mode():
            base_out = base(
                input_values,
                attention_mask=attention_mask,
                output_hidden_states=need_hidden_states,
                return_dict=True,
            )

            logits = None
            if need_text:
                hidden_for_logits = base_out.last_hidden_state
                # Повторяем логику Wav2Vec2ForCTC: dropout + linear head.
                if hasattr(model, "dropout"):
                    hidden_for_logits = model.dropout(hidden_for_logits)
                if not hasattr(model, "lm_head"):
                    raise RuntimeError("Ожидалась CTC-модель с `lm_head` (Wav2Vec2ForCTC).")
                logits = model.lm_head(hidden_for_logits)

        # Эмбеддинг: выбираем слой и пуллим по времени.
        embeddings = torch.empty((len(safe_batch), 0), device=self.device)
        if need_embedding:
            hidden = self._pick_hidden_layer(base_out, layer=layer)
            embeddings = self._pool_hidden(hidden, attention_mask=attention_mask)

        # Текст: greedy decode.
        texts: list[str] = [""] * len(safe_batch)
        if need_text:
            assert logits is not None
            pred_ids = torch.argmax(logits, dim=-1)
            pred_ids = pred_ids.detach().cpu()

            # У разных processor интерфейсы одинаковые, но подстрахуемся.
            if hasattr(processor, "batch_decode"):
                decoded = processor.batch_decode(pred_ids)
            elif hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "batch_decode"):
                decoded = processor.tokenizer.batch_decode(pred_ids)
            else:
                raise RuntimeError("processor не умеет декодировать токены в текст (нет batch_decode).")

            texts = [str(t).strip() for t in decoded]

        return Wav2Vec2InferenceResult(texts=texts, embeddings=embeddings)

    def transcribe(self, audio: Any) -> str | list[str]:
        """Распознаёт текст из аудио.

        Args:
            audio: np.ndarray/torch.Tensor (1D) или список таких объектов.

        Returns:
            Строка (если один пример) или список строк (если batch).
        """

        out = self._forward_once(audio, return_text=True, return_embedding=False, embedding_layer=None)
        return out.texts[0] if len(out.texts) == 1 else out.texts

    def recognize(self, audio: Any) -> str | list[str]:
        """Алиас для совместимости с старым `SpeechRecognizer.recognize()`."""

        return self.transcribe(audio)

    def extract_embedding(self, audio: Any, *, embedding_layer: int | None = None) -> torch.Tensor:
        """Извлекает эмбеддинги из скрытых слоёв.

        Args:
            audio: np.ndarray/torch.Tensor (1D) или список таких объектов.
            embedding_layer: какой слой брать (по умолчанию `self.embedding_layer`).
                -1 = последний слой (быстрее и без хранения всех hidden_states).

        Returns:
            torch.Tensor формы (batch, hidden_size).
        """

        out = self._forward_once(audio, return_text=False, return_embedding=True, embedding_layer=embedding_layer)
        return out.embeddings

    def transcribe_and_embed(
        self,
        audio: Any,
        *,
        embedding_layer: int | None = None,
    ) -> tuple[str | list[str], torch.Tensor]:
        """Возвращает и текст, и эмбеддинг за один прогон модели.

        Это главный метод, ради которого создан класс.
        """

        out = self._forward_once(audio, return_text=True, return_embedding=True, embedding_layer=embedding_layer)
        texts: str | list[str] = out.texts[0] if len(out.texts) == 1 else out.texts
        return texts, out.embeddings


__all__ = ["Wav2Vec2Multimodal", "Wav2Vec2InferenceResult"]
