#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Простой тест записи с автосохранением."""

from pathlib import Path
import sys
import time

# Добавляем корень проекта в путь
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.audio_io.audio_capture import MicrophoneCapture
from src.audio_io.audio_file_manager import AudioFileManager
from configs.config import ProjectConfig


def simple_recording_test():
    """Простой тест: запись с автосохранением в withoutNoise."""

    print("\n" + "=" * 60)
    print("🎤 ТЕСТ ЗАПИСИ С АВТОСОХРАНЕНИЕМ")
    print("=" * 60)

    # Папка для сохранения (твоя папка withoutNoise)
    save_path = ProjectConfig().clean_recordings_dir

    # Создаём менеджер файлов
    manager = AudioFileManager(
        save_dir=save_path,
        sample_rate=16000
    )

    # Создаём микрофон с автосохранением
    mic = MicrophoneCapture(
        sample_rate=16000,
        file_manager=manager,
        auto_save=True  # ВКЛЮЧАЕМ автосохранение
    )

    try:
        # Записываем 3 секунды
        duration = 8
        print(f"\n🎙️ Говорите что-нибудь {duration} секунды...")
        print("⏳ Запись началась...", end="", flush=True)

        # НАЧАЛО ЗАПИСИ
        start_time = time.time()
        audio = mic.listen(duration=duration)
        end_time = time.time()

        print(" ✓ Запись завершена!")

        # Информация о записи
        print(f"\n📊 Информация о записи:")
        print(f"   - Длительность: {duration} секунд (просили)")
        print(f"   - Реальное время записи: {end_time - start_time:.2f} сек")
        print(f"   - Количество сэмплов: {len(audio)}")
        print(f"   - Частота: 16000 Гц")
        print(f"   - Громкость (RMS): {mic._calculate_rms(audio):.6f}")

        # Информация о сохранении
        if mic.last_saved_path:
            print(f"\n💾 Файл сохранён:")
            print(f"   - Путь: {mic.last_saved_path}")
            print(f"   - Имя: {mic.last_saved_path.name}")
            print(f"   - Размер: {mic.last_saved_path.stat().st_size} байт")

            # Проверяем, что файл можно загрузить
            print(f"\n🔄 Проверяем загрузку файла...")
            loaded_audio = manager.load(mic.last_saved_path)
            print(f"   - Загружено {len(loaded_audio)} сэмплов")
            print(f"   - Совпадает с оригиналом? {len(loaded_audio) == len(audio)}")

            # Воспроизводим
            print(f"\n🔊 Воспроизводим запись через 2 секунды...")
            time.sleep(2)
            print("   Слушаем...")
            manager.play(audio)
            print("   ✓ Воспроизведение завершено")

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        return

    print("\n" + "=" * 60)
    print("✅ Тест успешно завершён!")
    print(f"📁 Проверь папку: {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    simple_recording_test()
