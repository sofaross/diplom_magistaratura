from __future__ import annotations

"""Распознавание эмоций по аудио.

Отвечает только за:
- загрузку emotion модели из чекпоинта;
- предобработку аудио в mel-спектрограмму;
- инференс (логиты -> вероятности).
"""

from pathlib import Path

import numpy as np
import torch

from src.models.emotion_model import EmotionModel, EmotionModelImproved
from src.realtime.audio_processor import AudioProcessor
from src.realtime.models_loader import _load_emotion_map, _load_state_dict, _resolve_repo_path


class EmotionRecognizer:
    """Класс для распознавания эмоции по голосу.

    Основные методы:
    - `recognize(audio)` -> (emotion_label, probabilities)

    Пример:
        processor = AudioProcessor(sample_rate=16000)
        emo = EmotionRecognizer("emotion_model_best.pt", "emotion_map.json", audio_processor=processor)
        emotion, probs = emo.recognize(audio_np)
    """

    def __init__(
        self,
        emotion_model_path: str | Path,
        emotion_map_path: str | Path,
        *,
        device: torch.device | str | None = None,
        audio_processor: AudioProcessor | None = None,
    ):
        """Создаёт распознаватель эмоций и загружает модель.

        Args:
            emotion_model_path: путь к чекпоинту (.pt).
            emotion_map_path: путь к emotion_map.json.
            device: устройство PyTorch ("cpu", "cuda", torch.device).
            audio_processor: общий процессор аудио (если хотите переиспользовать один и тот же).

        Raises:
            ValueError: если emotion_map.json имеет неверный формат.
            RuntimeError: если чекпоинт не удалось загрузить ни в одну архитектуру.
        """

        self.emotion_model_path = _resolve_repo_path(emotion_model_path)
        self.emotion_map_path = _resolve_repo_path(emotion_map_path)

        self.emotion_to_id, self.id_to_emotion = _load_emotion_map(self.emotion_map_path)
        self.num_emotions = int(len(self.id_to_emotion))

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        self.audio_processor = audio_processor or AudioProcessor(sample_rate=16000)

        self.model: torch.nn.Module = self.load_model(self.emotion_model_path)

    def load_model(self, checkpoint_path: Path) -> torch.nn.Module:
        """Загружает emotion-модель из чекпоинта.

        Мы пробуем сначала `EmotionModelImproved`, затем `EmotionModel`, чтобы поддерживать
        разные варианты архитектур, сохранённые в проекте.

        Args:
            checkpoint_path: путь к .pt файлу.

        Returns:
            torch.nn.Module в режиме eval().

        Raises:
            RuntimeError: если ни одна архитектура не смогла загрузиться.
        """

        state = _load_state_dict(checkpoint_path)

        errors: list[str] = []
        for model_cls in (EmotionModelImproved, EmotionModel):
            try:
                model = model_cls(num_emotions=self.num_emotions)
                model.load_state_dict(state, strict=True)
                model.to(self.device)
                model.eval()
                return model
            except Exception as e:
                errors.append(f"{model_cls.__name__}: {e}")

        raise RuntimeError(
            "Не удалось загрузить emotion model. Проверьте, что чекпоинт соответствует коду модели.\n"
            + "\n".join(errors)
        )

    def _preprocess_audio(self, audio: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Превращает аудио в (mel, lengths) для emotion-модели.

        Args:
            audio: np.ndarray float32, 16 kHz.

        Returns:
            mel: torch.Tensor [B=1, 1, 128, T]
            lengths: torch.Tensor [B=1] (кол-во фреймов T до padding)

        Raises:
            ValueError: если аудио пустое после trim_silence.
        """

        x = np.asarray(audio, dtype=np.float32)
        x = self.audio_processor.normalize(x)
        x = self.audio_processor.trim_silence(x)
        if x.size == 0:
            raise ValueError("После trim_silence аудио пустое: нечего распознавать.")

        mel = self.audio_processor.extract_mel(x)  # [1, 128, T]
        mel = self.audio_processor.standardize_mel(mel)

        # EmotionModel ожидает [B, 1, 128, T]
        mel = mel.unsqueeze(0).to(self.device)
        lengths = torch.tensor([int(mel.shape[-1])], dtype=torch.long, device=self.device)
        return mel, lengths

    def recognize(self, audio: np.ndarray) -> tuple[str, dict[str, float]]:
        """Распознаёт эмоцию по аудио.

        Args:
            audio: np.ndarray float32, 16kHz.

        Returns:
            (emotion_label, prob_map)
            - emotion_label: строка из id_to_emotion (как в emotion_map.json).
            - prob_map: словарь emotion->probability.

        Raises:
            ValueError: если аудио слишком короткое/пустое после обработки.
            RuntimeError: если выход модели не совпадает с emotion_map.
        """

        mel, lengths = self._preprocess_audio(audio)

        with torch.inference_mode():
            logits = self.model(mel, lengths=lengths)
            probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

        if int(probs.shape[0]) != int(self.num_emotions):
            raise RuntimeError(
                f"Размер выхода модели ({probs.shape[0]}) не совпадает с emotion_map ({self.num_emotions}). "
                "Проверьте, что emotion_map.json соответствует чекпоинту."
            )

        top_idx = int(np.argmax(probs))
        emotion = self.id_to_emotion[top_idx]
        prob_map = {self.id_to_emotion[i]: float(probs[i]) for i in range(self.num_emotions)}
        return emotion, prob_map


__all__ = ["EmotionRecognizer"]
