"""Модуль для извлечения speech embedding из аудио.

Важно: это НЕ распознавание речи (не ASR и не Whisper).
Этот код превращает аудиосигнал в вектор признаков с помощью Wav2Vec2.

Legacy:
    В проекте появился единый класс `Wav2Vec2Multimodal` (ASR + эмбеддинги).
    Этот файл оставлен для совместимости/истории и может не использоваться.
"""

from __future__ import annotations

import torch
import numpy as np
from transformers import AutoFeatureExtractor, Wav2Vec2Model
from typing import Any

# ===============================
# Класс для извлечения эмбеддингов (умных признаков) из речи с помощью Wav2Vec2.
# ===============================
class SpeechEmbeddingModel:
    """Извлекает эмбеддинги речи (speech embeddings) с помощью Wav2Vec2.

    Что делает:
    - Принимает аудио (numpy/torch, 1D) или список аудио разной длины.
    - Возвращает фиксированный вектор признаков (эмбеддинг) на каждый пример.

    Что НЕ делает:
    - Не распознаёт текст (это не Whisper / не Wav2Vec2ForCTC).

    Основной метод:
    - `extract_embedding(audio) -> torch.Tensor` формы `(batch, hidden_size)`.
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-large-xlsr-53",
        device: str | torch.device | None = None,
    ) -> None:

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)

        if device is None:
            device = "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)

        self.model.eval()

    # ===============================
    # Возвращает размерность эмбеддинга.
    # ===============================
    @property
    def embedding_dim(self) -> int:
        return int(self.model.config.hidden_size)

    # ===============================
    # Превращает аудио в эмбеддинг.
    # ===============================
    def extract_embedding(self, audio: Any, sample_rate: int = 16000) -> torch.Tensor:
        """Преобразует аудио в эмбеддинг.

        Параметры:
            audio: `np.ndarray`/`torch.Tensor` (1D) или список таких объектов.
            sample_rate: частота дискретизации входного сигнала (обычно 16000).

        Возвращает:
            `torch.Tensor` формы `(batch, hidden_size)`.

        Исключения:
            Может выбросить исключения HuggingFace/torch при проблемах с моделью.
        """

        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
        elif isinstance(audio, (list, tuple)):
            audio = [
                item.detach().cpu().numpy() if isinstance(item, torch.Tensor) else item
                for item in audio
            ]

        if isinstance(audio, np.ndarray):
            audio = audio.astype(np.float32, copy=False)
        elif isinstance(audio, (list, tuple)):
            audio = [np.asarray(item, dtype=np.float32) for item in audio]

        inputs = self.feature_extractor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )

        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)

        hidden_states = outputs.last_hidden_state

        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            embedding = torch.mean(hidden_states, dim=1)
        else:
            # Переводим attention_mask из маски "по сэмплам" в маску "по фичам"
            # (Wav2Vec2 внутри сначала делает сверточное субдискретизирование).
            feature_mask = self.model._get_feature_vector_attention_mask(
                hidden_states.shape[1],
                attention_mask,
            )
            feature_mask = feature_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
            masked_sum = (hidden_states * feature_mask).sum(dim=1)
            denom = feature_mask.sum(dim=1).clamp(min=1.0)
            embedding = masked_sum / denom

        return embedding
