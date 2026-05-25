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


RESD_EMOTION_MAP_6: dict[str, str] = {
    "anger": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happiness": "happy",
    "neutral": "neutral",
    "sadness": "sad",
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
