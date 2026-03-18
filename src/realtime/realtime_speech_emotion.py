from __future__ import annotations

"""Главный модуль "живого" распознавания: текст + эмоция.

Здесь минимальная логика: только оркестрация нескольких компонентов:
- запись с микрофона (MicrophoneCapture)
- предобработка аудио (AudioProcessor)
- распознавание эмоций (EmotionRecognizer)
- распознавание речи (Wav2Vec2Multimodal)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(REPO_ROOT))

from src.mictophone.audio_capture import MicrophoneCapture
from src.preprocessing.audio_processing import AudioProcessor
from src.models.emotion_recognizer import EmotionRecognizer
from src.realtime.models_loader import EMOTION_RU, _resolve_repo_path
from src.models.speech_model import Wav2Vec2Multimodal


class RealtimeSpeechEmotionRecognizer:
    """Распознаёт текст и эмоцию из голоса по записи с микрофона.

    Основные методы:
    - `listen(duration)` -> np.ndarray
    - `recognize_speech(audio)` -> str
    - `recognize_emotion(audio)` -> (emotion_label, prob_map)
    - `process_live(duration)` -> None (печатает результат)

    Пример использования:
        r = RealtimeSpeechEmotionRecognizer(
            emotion_model_path="data/processed/models/emotion/emotion_model_best.pt",
            emotion_map_path="data/processed/models/emotion/emotion_map.json",
            speech_model_name="facebook/wav2vec2-base-960h",
        )
        r.process_live(duration=5)
    """

    def __init__(
        self,
        emotion_model_path: str | Path,
        emotion_map_path: str | Path,
        speech_model_name: str = "facebook/wav2vec2-base-960h",
    ):
        """Инициализирует компоненты.

        Args:
            emotion_model_path: путь к чекпоинту emotion модели (.pt).
            emotion_map_path: путь к emotion_map.json.
            speech_model_name: имя Wav2Vec2 CTC модели в HuggingFace (ASR).

        Raises:
            RuntimeError/ValueError: если emotion модель или emotion_map не загрузились.
        """

        self.sample_rate: int = 16000
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Общий процессор аудио: чтобы шаги предобработки были одинаковыми везде.
        self.audio_processor = AudioProcessor(sample_rate=self.sample_rate)

        # Запись с микрофона: отдельный модуль, который не зависит от моделей.
        self.microphone = MicrophoneCapture(sample_rate=self.sample_rate)

        # Распознавание эмоций: обязательно.
        self.emotion = EmotionRecognizer(
            emotion_model_path=emotion_model_path,
            emotion_map_path=emotion_map_path,
            device=self.device,
            audio_processor=self.audio_processor,
        )

        # Распознавание речи (ASR) через Wav2Vec2 CTC: может не загрузиться (например, без интернета),
        # поэтому strict=False: объект создаётся, а ошибка хранится в self.speech.load_error.
        self.speech = Wav2Vec2Multimodal(
            model_name=speech_model_name,
            device=self.device,
            sample_rate=self.sample_rate,
            preprocess=True,
            strict=False,
        )

    def listen(self, duration: float = 5.0) -> np.ndarray:
        """Записывает аудио с микрофона.

        Args:
            duration: длительность записи в секундах.

        Returns:
            np.ndarray [N] float32.
        """

        return self.microphone.listen(duration=duration)

    def recognize_speech(self, audio: np.ndarray) -> str:
        """Распознаёт текст из аудио.

        Args:
            audio: np.ndarray float32, 16kHz.

        Returns:
            Текст (строка).

        Raises:
            RuntimeError: если ASR модель не загружена или произошла ошибка распознавания.
        """

        return self.speech.recognize(audio)

    def recognize_emotion(self, audio: np.ndarray) -> tuple[str, dict[str, float]]:
        """Распознаёт эмоцию из аудио.

        Args:
            audio: np.ndarray float32, 16kHz.

        Returns:
            (emotion_label, prob_map)
        """

        return self.emotion.recognize(audio)

    def process_live(self, duration: float = 5.0) -> None:
        """Полный цикл: запись -> текст -> эмоции. Печатает результат в консоль.

        Args:
            duration: длительность записи в секундах.
        """

        print(f"Говорите {float(duration):.1f} секунд...")
        print("Запись...")
        audio = self.listen(duration=duration)

        text = ""
        try:
            text = self.recognize_speech(audio)
        except Exception as e:
            # Не прекращаем работу: эмоцию можно распознавать даже без ASR.
            print(f"[ошибка] распознавание речи: {e}")

        try:
            emotion, probs = self.recognize_emotion(audio)
        except Exception as e:
            print(f"[ошибка] распознавание эмоции: {e}")
            return

        # Человекочитаемый вывод: перевод эмоций, сортировка вероятностей.
        emotion_label = EMOTION_RU.get(emotion, emotion)
        top_prob = float(probs.get(emotion, 0.0))
        probs_sorted = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        probs_str = ", ".join([f"{EMOTION_RU.get(k, k)}: {v*100:.0f}%" for k, v in probs_sorted])

        print(f'Распознанный текст: "{text}"')
        print(f"Эмоция: {emotion_label} ({top_prob*100:.0f}% уверенности)")
        print(f"Вероятности: {probs_str}")


def main(argv: list[str] | None = None) -> int:
    """CLI для записи с микрофона и распознавания текста+эмоции.

    Args:
        argv: список аргументов (для тестирования). Если None, читаем из sys.argv.

    Returns:
        Код выхода (0 = успех).
    """

    parser = argparse.ArgumentParser(prog="python -m src.realtime.realtime_speech_emotion")
    parser.add_argument("--duration", type=float, default=5.0, help="Длительность записи в секундах.")
    parser.add_argument(
        "--emotion-model",
        required=True,
        help="Путь к чекпоинту emotion-модели (.pt), например data/processed/models/emotion/emotion_model_best.pt",
    )
    parser.add_argument(
        "--emotion-map",
        default=None,
        help="Путь к emotion_map.json. Если не указан, пытаемся взять рядом с emotion-model.",
    )
    parser.add_argument(
        "--speech-model",
        default="facebook/wav2vec2-base-960h",
        help=(
            "Wav2Vec2 CTC ASR модель из HuggingFace. "
            "Пример для русского: jonatasgrosman/wav2vec2-large-xlsr-53-russian"
        ),
    )
    args = parser.parse_args(argv)

    emotion_model_path = _resolve_repo_path(args.emotion_model)
    emotion_map_path = (
        _resolve_repo_path(args.emotion_map)
        if args.emotion_map is not None
        else emotion_model_path.with_name("emotion_map.json")
    )

    if not emotion_map_path.exists():
        raise SystemExit(
            f"Не найден emotion_map.json: {emotion_map_path}. "
            "Передайте --emotion-map или положите emotion_map.json рядом с чекпоинтом."
        )

    try:
        recognizer = RealtimeSpeechEmotionRecognizer(
            emotion_model_path=emotion_model_path,
            emotion_map_path=emotion_map_path,
            speech_model_name=args.speech_model,
        )
        recognizer.process_live(duration=args.duration)
        return 0
    except KeyboardInterrupt:
        print("Остановлено пользователем.")
        return 130
    except Exception as e:
        print(f"[ошибка] {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
