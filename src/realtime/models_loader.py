from __future__ import annotations

"""Чистые функции и константы для загрузки моделей/маппингов.

Этот модуль специально сделан без "состояния" (без классов и глобальных синглтонов),
чтобы его было проще тестировать и переиспользовать.
"""

import json
from pathlib import Path

import torch

from src.constants.emotions import EMOTION_RU

# Корень репозитория (нужно для удобной работы с относительными путями из CLI).
REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_repo_path(value: str | Path) -> Path:
    """Преобразует путь в абсолютный.

    Если путь относительный, он считается относительно корня репозитория.

    Args:
        value: Относительный или абсолютный путь.

    Returns:
        Абсолютный путь (Path).
    """

    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _load_emotion_map(path: Path) -> tuple[dict[str, int], list[str]]:
    """Загружает emotion_map.json и возвращает две структуры: emotion->id и id->emotion.

    Поддерживаемые форматы JSON:
    1) {"emotion_to_id": {...}, "id_to_emotion": [...]}
    2) {"angry": 0, "happy": 1, ...}  (только emotion_to_id)

    Args:
        path: Путь к JSON файлу.

    Returns:
        (emotion_to_id, id_to_emotion)

    Raises:
        ValueError: если файл имеет неверный формат/пустой.
    """

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "emotion_to_id" in payload:
        emotion_to_id = payload.get("emotion_to_id")
        id_to_emotion = payload.get("id_to_emotion")
    else:
        # На случай, если пользователь дал "голую" мапу emotion->id.
        emotion_to_id = payload
        id_to_emotion = None

    if not isinstance(emotion_to_id, dict) or not emotion_to_id:
        raise ValueError(f"Некорректный emotion_map.json: {path}")

    emotion_to_id = {str(k): int(v) for k, v in emotion_to_id.items()}

    if isinstance(id_to_emotion, list) and id_to_emotion:
        id_to_emotion = [str(x) for x in id_to_emotion]
    else:
        # Собираем id->emotion из emotion->id.
        id_to_emotion = [None] * len(emotion_to_id)
        for emo, idx in emotion_to_id.items():
            if 0 <= int(idx) < len(id_to_emotion):
                id_to_emotion[int(idx)] = emo
        if any(x is None for x in id_to_emotion):
            raise ValueError(f"emotion_to_id содержит пропуски индексов: {path}")

    return emotion_to_id, id_to_emotion


def _load_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    """Загружает чекпоинт PyTorch и возвращает state_dict.

    Поддерживает форматы:
    - {"state_dict": ...}
    - "голый" state_dict

    Args:
        checkpoint_path: путь к .pt файлу.

    Returns:
        state_dict (dict).

    Raises:
        ValueError: если формат не распознан или state_dict пустой.
    """

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state = checkpoint
    else:
        raise ValueError(f"Неподдерживаемый формат чекпоинта: {checkpoint_path}")

    if not isinstance(state, dict) or not state:
        raise ValueError(f"Пустой state_dict в чекпоинте: {checkpoint_path}")
    return state


__all__ = [
    "EMOTION_RU",
    "REPO_ROOT",
    "_load_emotion_map",
    "_load_state_dict",
    "_resolve_repo_path",
]
