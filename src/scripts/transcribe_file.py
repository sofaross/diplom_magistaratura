from pathlib import Path

import torch

from configs.config import ProjectConfig
from src.audio_io.audio_file_manager import AudioFileManager
from src.inference.wav2vec2_inference import transcribe
from src.models.speech_model import Wav2Vec2Multimodal
from src.models.wav2vec2_wrapper import Wav2Vec2Wrapper

DEFAULT_RECORDINGS_DIR = Path(ProjectConfig().clean_recordings_dir)

"""
   Загружает аудиофайл и возвращает распознанный текст.
   Если file_path не указан, берётся самый свежий .wav файл из папки recordings_dir.
   При ошибках возвращает пустую строку и печатает сообщение.
   """


def transcribe_file(
    file_path: str | Path | None = None,
    model: Wav2Vec2Wrapper | Wav2Vec2Multimodal | None = None,
    device: str | None = None,
    recordings_dir: Path = DEFAULT_RECORDINGS_DIR,
) -> str:
    if file_path is None:
        try:
            wav_files = list(recordings_dir.glob("*.wav"))
            if not wav_files:
                print("В папке нет .wav файлов.")
                return ""
            latest_file = max(wav_files, key=lambda path: path.stat().st_mtime)
            file_path = latest_file
            print(f"Использую последнюю запись: {file_path.name}")
        except Exception as e:
            print(f"Ошибка при поиске файлов в {recordings_dir}: {e}")
            return ""
    else:
        file_path = Path(file_path)

    try:
        manager = AudioFileManager(sample_rate=16000)
        audio = manager.load(file_path, target_sample_rate=16000)
    except Exception as e:
        print(f"Не удалось загрузить файл {file_path}: {e}")
        return ""

    if model is None:
        try:
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            model = Wav2Vec2Wrapper.from_pretrained(
                model_name="facebook/wav2vec2-base-960h",
                device=device,
            )
        except Exception as e:
            print(f"Не удалось загрузить модель Wav2Vec2: {e}")
            return ""

    try:
        if isinstance(model, Wav2Vec2Multimodal):
            result = model.transcribe(audio)
        else:
            result = transcribe(model, audio, preprocess=True)
        return result if isinstance(result, str) else result[0]
    except Exception as e:
        print(f"Ошибка при распознавании: {e}")
        return ""
