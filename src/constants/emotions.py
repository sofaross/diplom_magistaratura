from __future__ import annotations

"""Константы с названиями эмоций и маппингами датасетов."""


CREMA_EMOTIONS: dict[str, str] = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}


RAVDESS_EMOTIONS: dict[str, str] = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
}


EMOTION_RU: dict[str, str] = {
    "neutral": "нейтрально",
    "happy": "радость",
    "sad": "грусть",
    "angry": "злость",
    "fear": "страх",
    "disgust": "отвращение",
    "calm": "спокойствие",
    "surprise": "удивление",
}
