import torch
from pathlib import Path
from src.models.speech_model import Wav2Vec2Multimodal
from src.mictophone.audio_file_manager import AudioFileManager

DEFAULT_RECORDINGS_DIR = Path(r"C:\Users\User\PycharmProjects\diplom_magistaratura\notebooks\withoutNoise")

"""
   Загружает аудиофайл и возвращает распознанный текст.
   Если file_path не указан, берётся самый свежий .wav файл из папки recordings_dir.
   При ошибках возвращает пустую строку и печатает сообщение.
   """
def transcribe_file(file_path: str | Path | None = None,
                    model: Wav2Vec2Multimodal = None,
                    device: str = None,
                    recordings_dir: Path = DEFAULT_RECORDINGS_DIR) -> str:

    if file_path is None:
        try:
            wav_files = list(recordings_dir.glob("*.wav"))
            if not wav_files:
                print("В папке нет .wav файлов.")
                return ""
            latest_file = max(wav_files, key=lambda p: p.stat().st_mtime)
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

    # 3. Создаём модель, если не передана
    if model is None:
        try:
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            model = Wav2Vec2Multimodal(
                model_name="facebook/wav2vec2-base-960h",
                device=device,
                audio_processor=None
            )
        except Exception as e:
            print(f"Не удалось загрузить модель Wav2Vec2: {e}")
            return ""

    # 4. Распознаём текст
    try:
        result = model.process(audio, return_embeddings=False)
        return result["text"]
    except Exception as e:
        print(f"Ошибка при распознавании: {e}")
        return ""