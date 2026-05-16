from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.models.wav2vec2_wrapper import Wav2Vec2Wrapper
from src.preprocessing.audio_processing import normalize_audio, trim_silence


def _as_audio_batch(audio: Any) -> list[np.ndarray]:
    """Приводит вход к списку одномерных `np.ndarray`, чтобы дальше работать батчами."""

    if isinstance(audio, (list, tuple)):
        items = list(audio)
    else:
        items = [audio]

    batch: list[np.ndarray] = []
    for item in items:
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().numpy()
        batch.append(np.asarray(item, dtype=np.float32).reshape(-1))
    return batch


def _preprocess_audio_batch(batch: list[np.ndarray], *, preprocess: bool) -> list[np.ndarray]:
    """Нормализует аудио и обрезает тишину, если это поведение не отключено явно."""

    if not preprocess:
        return [np.asarray(item, dtype=np.float32) for item in batch]

    prepared: list[np.ndarray] = []
    for item in batch:
        audio = normalize_audio(np.asarray(item, dtype=np.float32))
        audio = trim_silence(np.asarray(audio, dtype=np.float32))
        prepared.append(np.asarray(audio, dtype=np.float32))
    return prepared


def _replace_empty_clips(batch: list[np.ndarray], sample_rate: int) -> list[np.ndarray]:
    """Подставляет короткий нулевой сигнал, если после trim аудио стало полностью пустым."""

    fallback_num_samples = max(1, int(sample_rate) // 100)
    safe_batch: list[np.ndarray] = []
    for item in batch:
        safe_batch.append(item if item.size > 0 else np.zeros(fallback_num_samples, dtype=np.float32))
    return safe_batch


def _decode_predictions(wrapper: Wav2Vec2Wrapper, pred_ids: torch.Tensor) -> list[str]:
    """Преобразует предсказанные token ids в строки, не привязываясь к конкретному типу processor."""

    processor = wrapper.processor
    if hasattr(processor, "batch_decode"):
        decoded = processor.batch_decode(pred_ids)
    elif hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "batch_decode"):
        decoded = processor.tokenizer.batch_decode(pred_ids)
    else:
        raise RuntimeError("Processor не умеет декодировать токены в текст: не найден batch_decode.")
    return [str(text).strip() for text in decoded]


def _pick_hidden_layer(base_outputs: Any, layer: int) -> torch.Tensor:
    """Выбирает нужный hidden state: последний слой быстро, промежуточный через hidden_states."""

    if int(layer) == -1:
        return base_outputs.last_hidden_state

    hidden_states = getattr(base_outputs, "hidden_states", None)
    if hidden_states is None:
        raise RuntimeError(
            "Нельзя взять промежуточный слой: модель была вызвана без output_hidden_states=True."
        )

    try:
        return hidden_states[int(layer)]
    except Exception as exc:
        raise RuntimeError(f"Некорректный индекс слоя: layer={layer}.") from exc


def _feature_vector_attention_mask(
    wrapper: Wav2Vec2Wrapper,
    attention_mask: torch.Tensor | None,
    feature_length: int,
) -> torch.Tensor | None:
    """Переводит маску входных сэмплов в маску признаков после feature extractor."""

    if attention_mask is None:
        return None

    base_model = wrapper.base_model
    if not hasattr(base_model, "_get_feature_vector_attention_mask"):
        return None

    feature_mask = base_model._get_feature_vector_attention_mask(feature_length, attention_mask)
    return feature_mask.to(dtype=torch.bool)


def pool_hidden_states(
    wrapper: Wav2Vec2Wrapper,
    hidden: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    pool: str = "mean",
) -> torch.Tensor:
    """Пулит скрытые состояния по времени с учётом padding-маски.

    Поддерживаются два варианта:
    - `mean`: усреднение только по валидным временным шагам
    - `max`: максимум только по валидным временным шагам
    """

    mode = str(pool).lower()
    feature_mask = _feature_vector_attention_mask(wrapper, attention_mask, hidden.shape[1])

    if feature_mask is None:
        if mode == "mean":
            return hidden.mean(dim=1)
        if mode == "max":
            return hidden.max(dim=1).values
        raise ValueError("pool должен быть 'mean' или 'max'.")

    expanded_mask = feature_mask.unsqueeze(-1)

    if mode == "mean":
        weights = expanded_mask.to(dtype=hidden.dtype)
        masked_sum = (hidden * weights).sum(dim=1)
        denom = weights.sum(dim=1).clamp(min=1.0)
        return masked_sum / denom

    if mode == "max":
        fill_value = torch.finfo(hidden.dtype).min
        masked_hidden = hidden.masked_fill(~expanded_mask, fill_value)
        pooled = masked_hidden.max(dim=1).values
        empty_rows = ~feature_mask.any(dim=1)
        if empty_rows.any():
            pooled[empty_rows] = 0.0
        return pooled

    raise ValueError("pool должен быть 'mean' или 'max'.")


def prepare_audio_batch(
    wrapper: Wav2Vec2Wrapper,
    audio: Any,
    *,
    preprocess: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Готовит батч аудио для модели.

    Аргументы:
        wrapper: загруженная обёртка с processor и моделью.
        audio: один аудиомассив или список аудиомассивов/тензоров.
        preprocess: применять ли normalize + trim_silence.

    Возвращает:
        `input_values` и `attention_mask`, уже перенесённые на `wrapper.device`.
    """

    batch = _as_audio_batch(audio)
    batch = _preprocess_audio_batch(batch, preprocess=preprocess)
    batch = _replace_empty_clips(batch, sample_rate=wrapper.sample_rate)

    inputs = wrapper.processor(
        batch,
        sampling_rate=int(wrapper.sample_rate),
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )

    input_values = inputs.get("input_values")
    if input_values is None:
        raise RuntimeError("Processor не вернул input_values.")

    attention_mask = inputs.get("attention_mask")
    input_values = input_values.to(wrapper.device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(wrapper.device)
    return input_values, attention_mask


def transcribe(
    wrapper: Wav2Vec2Wrapper,
    audio: Any,
    *,
    preprocess: bool = True,
) -> str | list[str]:
    """Распознаёт текст из аудио через полную CTC-модель."""

    input_values, attention_mask = prepare_audio_batch(wrapper, audio, preprocess=preprocess)

    with torch.inference_mode():
        logits = wrapper.model(input_values, attention_mask=attention_mask).logits

    pred_ids = torch.argmax(logits, dim=-1).detach().cpu()
    texts = _decode_predictions(wrapper, pred_ids)
    return texts[0] if len(texts) == 1 else texts


def extract_embedding(
    wrapper: Wav2Vec2Wrapper,
    audio: Any,
    *,
    layer: int = -1,
    pool: str = "mean",
    preprocess: bool = True,
) -> torch.Tensor:
    """Извлекает speech embedding из указанного слоя базовой модели."""

    input_values, attention_mask = prepare_audio_batch(wrapper, audio, preprocess=preprocess)
    need_hidden_states = int(layer) != -1

    with torch.inference_mode():
        base_outputs = wrapper.base_model(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=need_hidden_states,
            return_dict=True,
        )

    hidden = _pick_hidden_layer(base_outputs, layer=layer)
    return pool_hidden_states(wrapper, hidden, attention_mask, pool=pool)


def transcribe_and_embed(
    wrapper: Wav2Vec2Wrapper,
    audio: Any,
    *,
    layer: int = -1,
    pool: str = "mean",
    preprocess: bool = True,
) -> tuple[str | list[str], torch.Tensor]:
    """Возвращает и текст, и embedding за один проход базовой модели."""

    input_values, attention_mask = prepare_audio_batch(wrapper, audio, preprocess=preprocess)
    need_hidden_states = int(layer) != -1

    with torch.inference_mode():
        base_outputs = wrapper.base_model(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=need_hidden_states,
            return_dict=True,
        )

        hidden = _pick_hidden_layer(base_outputs, layer=layer)
        embedding = pool_hidden_states(wrapper, hidden, attention_mask, pool=pool)

        hidden_for_logits = base_outputs.last_hidden_state
        if hasattr(wrapper.model, "dropout"):
            hidden_for_logits = wrapper.model.dropout(hidden_for_logits)
        if not hasattr(wrapper.model, "lm_head"):
            raise RuntimeError("Ожидалась CTC-модель с lm_head.")

        logits = wrapper.model.lm_head(hidden_for_logits)

    pred_ids = torch.argmax(logits, dim=-1).detach().cpu()
    texts = _decode_predictions(wrapper, pred_ids)
    return (texts[0] if len(texts) == 1 else texts), embedding


__all__ = [
    "extract_embedding",
    "pool_hidden_states",
    "prepare_audio_batch",
    "transcribe",
    "transcribe_and_embed",
]
