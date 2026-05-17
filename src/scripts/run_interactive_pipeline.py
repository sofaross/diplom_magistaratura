from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
import random

REPO_ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(REPO_ROOT))

from configs.config import ProjectConfig
from src.audio_io.audio_capture import MicrophoneCapture
from src.audio_io.audio_file_manager import AudioFileManager
from src.noise.noise_manager import NoiseManager
from src.pipeline import MultimodalPipeline, ProcessingResult


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


@dataclass(slots=True)
class NoiseChoice:
    """Параметры выбранного пользователем режима шума."""

    noise_type: str | None = None
    snr_db: float | None = None
    use_random_noise: bool = False


class InteractivePipelineRunner:
    """Интерактивный консольный сценарий для ручной проверки pipeline."""

    def __init__(
        self,
        *,
        config: ProjectConfig | None = None,
        speech_model_name: str | None = None,
        emotion_model_path: str | Path | None = None,
        emotion_map_path: str | Path | None = None,
        default_speech_language: str | None = None,
    ) -> None:
        self.config = config or ProjectConfig()
        self.sample_rate = int(self.config.sample_rate)
        self.default_speech_language = str(default_speech_language or self.config.default_speech_language)

        self.clean_audio_manager = AudioFileManager(
            save_dir=self.config.clean_recordings_dir,
            sample_rate=self.sample_rate,
        )
        self.noisy_audio_manager = AudioFileManager(
            save_dir=self.config.noisy_recordings_dir,
            sample_rate=self.sample_rate,
        )
        self.noise_manager = NoiseManager(
            noise_dir=self.config.noise_dir,
            sample_rate=self.sample_rate,
        )
        self.microphone_capture = MicrophoneCapture(
            sample_rate=self.sample_rate,
            file_manager=self.clean_audio_manager,
            auto_save=True,
        )
        self.pipeline = MultimodalPipeline(
            audio_manager=self.clean_audio_manager,
            processed_audio_manager=self.noisy_audio_manager,
            noise_manager=self.noise_manager,
            speech_model_name=speech_model_name,
            emotion_model_path=_resolve_repo_path(emotion_model_path or self.config.emotion_checkpoint_path),
            emotion_map_path=_resolve_repo_path(emotion_map_path or self.config.emotion_map_path),
            default_speech_language=self.default_speech_language,
        )

    def run(self) -> int:
        self._print_header()

        source_audio_path = self._obtain_source_audio()
        print(f"\nИсходное аудио: {source_audio_path}")

        if self._ask_yes_no("Прослушать исходную запись? [y/n]: ", default=False):
            self._play_audio_file(source_audio_path)

        speech_language = self._ask_speech_language()
        noise_choice = self._ask_noise_choice()

        processed_audio_path = source_audio_path
        noise_applied = False
        applied_noise_type: str | None = None
        applied_snr_db: float | None = None

        if noise_choice is not None:
            processed_audio_path, applied_noise_type = self._build_noisy_audio(
                source_audio_path=source_audio_path,
                noise_choice=noise_choice,
            )
            noise_applied = True
            applied_snr_db = float(noise_choice.snr_db)

            print(f"Обработанное аудио сохранено: {processed_audio_path}")
            if self._ask_yes_no("Прослушать зашумлённое аудио? [y/n]: ", default=False):
                self._play_audio_file(processed_audio_path)

        print("\nВыполняется распознавание речи...")
        print("Выполняется распознавание эмоции...")
        result = self.pipeline.process_audio(
            processed_audio_path,
            speech_language=speech_language,
        )

        self._print_result(
            source_audio_path=source_audio_path,
            processed_audio_path=processed_audio_path,
            noise_applied=noise_applied,
            noise_type=applied_noise_type,
            snr_db=applied_snr_db,
            result=result,
        )
        return 0 if not result.errors else 1

    def _print_header(self) -> None:
        print("Интерактивный режим мультимодального pipeline")
        print("Запись -> шум -> распознавание речи -> распознавание эмоций\n")

    def _obtain_source_audio(self) -> Path:
        while True:
            clean_recordings = self._list_recordings(self.config.clean_recordings_dir)
            noisy_recordings = self._list_recordings(self.config.noisy_recordings_dir)
            total_recordings = len(clean_recordings) + len(noisy_recordings)

            print("Выберите источник аудио:")
            print("1 - записать новое голосовое сообщение")
            print(f"2 - выбрать из уже записанных ({total_recordings})")
            print("3 - указать путь к аудиофайлу вручную")
            choice = input("Введите номер варианта: ").strip()

            if choice == "1":
                duration = self._ask_float("Введите длительность записи в секундах: ", minimum=0.1)
                print("Идёт запись. Говорите...")
                _, saved_path = self.microphone_capture.listen_and_save(duration=duration)
                print(f"Аудио сохранено: {saved_path}")
                return saved_path.resolve()

            if choice == "2":
                if total_recordings == 0:
                    print("Сохранённых записей пока нет.")
                    continue
                selected_path = self._ask_saved_recording_path(clean_recordings, noisy_recordings)
                if selected_path is not None:
                    return selected_path
                continue

            if choice == "3":
                return self._ask_existing_audio_path()

            print("Неверный выбор. Введите 1, 2 или 3.")

    def _ask_saved_recording_path(
        self,
        clean_recordings: list[Path],
        noisy_recordings: list[Path],
    ) -> Path | None:
        while True:
            print("\nВыберите тип сохранённых записей:")
            print(f"1 - без шума ({len(clean_recordings)})")
            print(f"2 - с шумом ({len(noisy_recordings)})")
            print("3 - назад")
            choice = input("Введите номер варианта: ").strip()

            if choice == "1":
                if not clean_recordings:
                    print("Нет сохранённых записей без шума.")
                    continue
                return self._choose_recording_from_list(
                    clean_recordings,
                    title="Доступные записи без шума:",
                )

            if choice == "2":
                if not noisy_recordings:
                    print("Нет сохранённых записей с шумом.")
                    continue
                return self._choose_recording_from_list(
                    noisy_recordings,
                    title="Доступные записи с шумом:",
                )

            if choice == "3":
                return None

            print("Неверный выбор. Введите 1, 2 или 3.")

    def _choose_recording_from_list(self, recordings: list[Path], *, title: str) -> Path:
        print(f"\n{title}")
        for index, path in enumerate(recordings, start=1):
            print(f"{index} - {self._describe_audio_file(path)}")

        while True:
            raw_value = input("Выберите запись по номеру: ").strip()
            try:
                index = int(raw_value)
            except ValueError:
                print("Введите номер из списка.")
                continue

            if 1 <= index <= len(recordings):
                return recordings[index - 1]
            print("Номер вне диапазона списка.")

    @staticmethod
    def _list_recordings(directory: Path) -> list[Path]:
        return AudioFileManager.list_audio_files(directory, recursive=True)

    def _describe_audio_file(self, path: Path) -> str:
        try:
            info = self.clean_audio_manager.get_info(path)
            suffix = path.suffix.lower().lstrip(".") or "audio"
            duration = self._format_duration(float(info.get("duration", 0.0) or 0.0))
            return f"{path.name} [{suffix}, {duration}]"
        except Exception:
            return path.name

    @staticmethod
    def _format_duration(duration_seconds: float) -> str:
        total_seconds = max(0.0, float(duration_seconds))
        if total_seconds < 60.0:
            return f"{total_seconds:.1f} c"

        minutes = int(total_seconds // 60)
        seconds = total_seconds - minutes * 60
        return f"{minutes} мин {seconds:.1f} c"

    def _ask_existing_audio_path(self) -> Path:
        while True:
            raw_value = input("Введите путь к существующему аудиофайлу: ").strip()
            if not raw_value:
                print("Путь не должен быть пустым.")
                continue

            path = _resolve_repo_path(raw_value)
            if not path.exists():
                print(f"Файл не найден: {path}")
                continue
            if not path.is_file():
                print(f"Это не файл: {path}")
                continue
            if not self.clean_audio_manager.is_supported_audio_file(path):
                supported = ", ".join(self.clean_audio_manager.SUPPORTED_INPUT_EXTENSIONS)
                print(f"Поддерживаются только аудиофайлы: {supported}")
                continue
            return path

    def _ask_speech_language(self) -> str:
        while True:
            raw_value = input(
                f"Выберите язык распознавания речи [ru/en] (по умолчанию {self.default_speech_language}): "
            ).strip().lower()
            if not raw_value:
                return self.default_speech_language
            if raw_value in {"ru", "en"}:
                return raw_value
            print("Введите 'ru' или 'en'.")

    def _ask_noise_choice(self) -> NoiseChoice | None:
        if not self._ask_yes_no("Наложить шум? [y/n]: ", default=False):
            return None

        while True:
            print("\nВыберите режим шума:")
            print("1 - белый шум")
            print("2 - розовый шум")
            print("3 - коричневый шум")
            print("4 - реальный шум из папки")
            print("5 - случайный шум")
            choice = input("Введите номер варианта: ").strip()

            if choice in {"1", "2", "3", "4", "5"}:
                break
            print("Неверный выбор. Введите число от 1 до 5.")

        snr_db = self._ask_float("Введите SNR в dB (например 5, 10, 15): ")

        if choice == "1":
            return NoiseChoice(noise_type="white", snr_db=snr_db)
        if choice == "2":
            return NoiseChoice(noise_type="pink", snr_db=snr_db)
        if choice == "3":
            return NoiseChoice(noise_type="brown", snr_db=snr_db)
        if choice == "4":
            real_noise_name = self._ask_real_noise_variant()
            return NoiseChoice(noise_type=real_noise_name, snr_db=snr_db)
        return NoiseChoice(use_random_noise=True, snr_db=snr_db)

    def _ask_real_noise_variant(self) -> str:
        variants = self.noise_manager.list_available_noise_variants()
        if not variants:
            raise RuntimeError(
                "В папке с реальными шумами не найдено ни одного wav-файла. "
                f"Проверьте каталог: {self.config.noise_dir}"
            )

        print("\nДоступные реальные шумы:")
        for index, variant in enumerate(variants, start=1):
            print(f"{index} - {variant}")

        while True:
            raw_value = input("Выберите реальный шум по номеру: ").strip()
            try:
                index = int(raw_value)
            except ValueError:
                print("Введите номер из списка.")
                continue

            if 1 <= index <= len(variants):
                return variants[index - 1]
            print("Номер вне диапазона списка.")

    def _build_noisy_audio(self, *, source_audio_path: Path, noise_choice: NoiseChoice) -> tuple[Path, str]:
        if noise_choice.snr_db is None:
            raise ValueError("Для наложения шума должен быть задан SNR.")

        selected_noise_type = noise_choice.noise_type
        if noise_choice.use_random_noise:
            available = self.pipeline.list_available_noises()
            if not available:
                raise RuntimeError("В проекте не найдено доступных шумов.")
            selected_noise_type = random.choice(available)
            print(f"Случайно выбранный шум: {selected_noise_type}")

        if selected_noise_type is None:
            raise ValueError("Не удалось определить тип шума.")

        noisy_path = self.noise_manager.process_file(
            input_path=source_audio_path,
            output_dir=self.noisy_audio_manager.save_dir,
            noise_type=selected_noise_type,
            snr_db=float(noise_choice.snr_db),
        )

        if selected_noise_type in self.noise_manager.SYNTHETIC_NOISES:
            applied_noise_type = selected_noise_type
        else:
            applied_noise_type = self.noise_manager.get_real_noise_group(selected_noise_type)

        return noisy_path.resolve(), applied_noise_type

    def _play_audio_file(self, audio_path: Path) -> None:
        try:
            audio = self.clean_audio_manager.load(audio_path, target_sample_rate=self.sample_rate)
            self.clean_audio_manager.play(audio, sample_rate=self.sample_rate, wait=True)
        except Exception as exc:
            print(f"Не удалось воспроизвести аудио: {exc}")

    def _print_result(
        self,
        *,
        source_audio_path: Path,
        processed_audio_path: Path,
        noise_applied: bool,
        noise_type: str | None,
        snr_db: float | None,
        result: ProcessingResult,
    ) -> None:
        print("\nРезультат:")
        print(f"Путь к исходному аудио: {source_audio_path}")
        print(f"Путь к обработанному аудио: {processed_audio_path}")
        print(f"Шум наложен: {'да' if noise_applied else 'нет'}")
        print(f"Тип шума: {noise_type or '-'}")
        print(f"SNR: {snr_db if snr_db is not None else '-'}")
        print(f"Распознанный текст: {result.recognized_text or '-'}")
        print(f"Скорее всего текст: {result.suggested_text or result.recognized_text or '-'}")
        print(f"Распознанная эмоция: {result.predicted_emotion or '-'}")
        print("Вероятности по эмоциям:")
        if result.emotion_probabilities:
            for emotion, probability in sorted(
                result.emotion_probabilities.items(),
                key=lambda item: item[1],
                reverse=True,
            ):
                print(f"  {emotion}: {probability:.4f}")
        else:
            print("  -")

        if result.errors:
            print("Ошибки:")
            for error in result.errors:
                print(f"  - {error}")

    @staticmethod
    def _ask_yes_no(prompt: str, *, default: bool) -> bool:
        true_values = {"y", "yes", "д", "да"}
        false_values = {"n", "no", "н", "нет"}

        while True:
            raw_value = input(prompt).strip().lower()
            if not raw_value:
                return default
            if raw_value in true_values:
                return True
            if raw_value in false_values:
                return False
            print("Введите 'y' или 'n'.")

    @staticmethod
    def _ask_float(prompt: str, *, minimum: float | None = None) -> float:
        while True:
            raw_value = input(prompt).strip().replace(",", ".")
            try:
                value = float(raw_value)
            except ValueError:
                print("Введите число.")
                continue

            if minimum is not None and value < minimum:
                print(f"Значение должно быть не меньше {minimum}.")
                continue
            return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.scripts.run_interactive_pipeline")
    parser.add_argument("--speech-language", default=None, help="Язык ASR по умолчанию: ru или en.")
    parser.add_argument("--speech-model", default=None, help="Явное имя Hugging Face модели ASR.")
    parser.add_argument("--emotion-model", default=None, help="Путь к чекпоинту модели эмоций.")
    parser.add_argument("--emotion-map", default=None, help="Путь к emotion_map.json.")
    args = parser.parse_args(argv)

    runner = InteractivePipelineRunner(
        speech_model_name=args.speech_model,
        emotion_model_path=args.emotion_model,
        emotion_map_path=args.emotion_map,
        default_speech_language=args.speech_language,
    )

    try:
        return runner.run()
    except KeyboardInterrupt:
        print("\nСценарий прерван пользователем.")
        return 130
    except Exception as exc:
        print(f"\nСценарий завершился с ошибкой: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
