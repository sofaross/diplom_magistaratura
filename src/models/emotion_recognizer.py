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
from src.preprocessing.audio_processing import AudioProcessor
from src.realtime.models_loader import _load_emotion_map, _load_state_dict, _resolve_repo_path

# ===============================
# Класс для распознавания эмоции по голосу.
# ===============================
class EmotionRecognizer:
    #Создаёт распознаватель эмоций и загружает модель.
    def __init__(
        self,
        emotion_model_path: str | Path,
        emotion_map_path: str | Path,
        *,
        device: torch.device | str | None = None,
        audio_processor: AudioProcessor | None = None,
    ):

        self.emotion_model_path = _resolve_repo_path(emotion_model_path)
        self.emotion_map_path = _resolve_repo_path(emotion_map_path)

        self.emotion_to_id, self.id_to_emotion = _load_emotion_map(self.emotion_map_path)
        self.num_emotions = int(len(self.id_to_emotion))

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        self.audio_processor = audio_processor or AudioProcessor(sample_rate=16000)

        self.model: torch.nn.Module = self.load_model(self.emotion_model_path)

    #Загружает emotion-модель из чекпоинта.
    def load_model(self, checkpoint_path: Path) -> torch.nn.Module:

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

    #Превращает аудио в (mel, lengths) для emotion-модели.
    def _preprocess_audio(self, audio: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
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

    #Распознаёт эмоцию по аудио.
    def recognize(self, audio: np.ndarray) -> tuple[str, dict[str, float]]:
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
