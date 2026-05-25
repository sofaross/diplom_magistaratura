"""CLI-утилита: извлечь speech embedding и распознать текст из одного аудиофайла (.wav).

Важно:
- Этот скрипт НЕ обучает модель.
- Он использует Wav2Vec2 CTC (HuggingFace): за один прогон получает и текст, и эмбеддинг.

Как понять, для какого файла он работает:
- Вы явно передаёте путь через аргумент `--audio-path`.

Пример:
    python -m src.training.extract_speech_embedding --audio-path data/raw/test_audio/example.wav --device cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from configs.model_runtime import DEFAULT_SPEECH_MODEL

REPO_ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.wav2vec2_inference import transcribe_and_embed
from src.models.wav2vec2_wrapper import Wav2Vec2Wrapper
from src.preprocessing.audio_processing import load_audio, normalize_audio, trim_silence


def _resolve_repo_path(value: str) -> Path:
    """Преобразует относительный путь (от корня репозитория) в абсолютный."""

    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def main() -> None:
    """Точка входа CLI.

    Читает .wav, делает стандартную предобработку (normalize/trim_silence),
    извлекает эмбеддинг Wav2Vec2 и печатает его размерность.
    """

    parser = argparse.ArgumentParser(
        prog="python -m src.training.extract_speech_embedding",
        description=(
            "Извлечение speech embedding + распознавание текста из аудио с помощью Wav2Vec2 CTC."
        ),
    )
    parser.add_argument(
        "--audio-path",
        required=True,
        help="Путь к .wav файлу, для которого нужно извлечь эмбеддинг.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_SPEECH_MODEL,
        help=(
            "Wav2Vec2 CTC модель из HuggingFace. "
            "Важно: для распознавания текста нужна модель, дообученная под ASR (CTC head)."
        ),
    )
    parser.add_argument("--device", default="cpu", help="cpu или cuda, например: cpu / cuda / cuda:0")
    parser.add_argument(
        "--out",
        default=None,
        help="(опционально) Куда сохранить эмбеддинг как .pt (torch tensor).",
    )
    args = parser.parse_args()

    audio_path = _resolve_repo_path(args.audio_path)
    if not audio_path.exists():
        raise SystemExit(f"Файл не найден: {audio_path}")

    print(f"[speech-emb] audio: {audio_path}")
    print(f"[speech-emb] model: {args.model_name}")
    print(f"[speech-emb] device: {args.device}")

    try:
        audio = load_audio(audio_path, sample_rate=16000)
        audio = normalize_audio(audio)
        audio = trim_silence(audio)
    except Exception as e:
        raise SystemExit(f"Не удалось прочитать/предобработать аудио: {e}")

    try:
        wrapper = Wav2Vec2Wrapper.from_pretrained(
            model_name=args.model_name,
            device=args.device,
        )
        text, embedding = transcribe_and_embed(wrapper, audio)
    except Exception as e:
        raise SystemExit(
            "Не удалось загрузить Wav2Vec2 или извлечь эмбеддинг. "
            "Возможные причины: нет интернета для скачивания модели, "
            "не установлены зависимости, недостаточно памяти. "
            f"Ошибка: {e}"
        )

    print(f"[speech-emb] text: {text}")
    print(f"[speech-emb] embedding shape: {tuple(embedding.shape)}")

    if args.out:
        out_path = _resolve_repo_path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(embedding.detach().cpu(), out_path)
        print(f"[speech-emb] saved to: {out_path}")


if __name__ == "__main__":
    main()
