from __future__ import annotations

from math import ceil
from pathlib import Path

import numpy as np

from src.audio_io.audio_file_manager import AudioFileManager


class NoiseManager:
    """Менеджер шумов для генерации зашумлённых версий чистых аудиозаписей."""

    SYNTHETIC_NOISES: tuple[str, ...] = ("white", "pink", "brown")

    def __init__(self, noise_dir: str | Path = "data/noises", sample_rate: int = 16000, random_seed: int | None = None) -> None:
        """Инициализирует менеджер шумов и загружает доступные реальные шумы.

        Args:
            noise_dir: Папка с файлами реальных шумов `.wav`.
            sample_rate: Частота дискретизации для загрузки и генерации.
            random_seed: Необязательное зерно для воспроизводимых синтетических шумов
                и случайного выбора фрагментов реальных шумов.
        """

        self.noise_dir = Path(noise_dir)
        self.sample_rate = int(sample_rate)
        self.rng = np.random.default_rng(random_seed)
        self.audio_file_manager = AudioFileManager(save_dir=self.noise_dir, sample_rate=self.sample_rate)

        self.noise_dir.mkdir(parents=True, exist_ok=True)

        self.real_noises: dict[str, np.ndarray] = {}
        self.load_errors: dict[str, str] = {}

        self.reload_real_noises()

    def reload_real_noises(self) -> None:
        """Повторно сканирует `noise_dir` и загружает все доступные реальные шумы."""

        self.real_noises.clear()
        self.load_errors.clear()

        for path in sorted(self.noise_dir.rglob("*.wav")):
            try:
                noise_name = self._build_noise_name(path)
                audio = self.audio_file_manager.load(path, target_sample_rate=self.sample_rate)
                audio = np.asarray(audio, dtype=np.float32).reshape(-1)

                if audio.size == 0:
                    raise ValueError("шумовой файл пустой")

                if noise_name in self.real_noises:
                    raise ValueError(
                        f"обнаружен дубликат имени шума '{noise_name}'. "
                        "Имена wav-файлов в noise_dir должны быть уникальными."
                    )

                self.real_noises[noise_name] = audio
            except Exception as exc:
                self.load_errors[str(path)] = str(exc)

    def list_available_noises(self) -> list[str]:
        """Возвращает список доступных типов шума: синтетические и реальные."""

        available = set(self.SYNTHETIC_NOISES)
        available.update(self.real_noises.keys())
        return sorted(available)

    def generate_synthetic_noise(self, noise_type: str, duration: float) -> np.ndarray:
        """Генерирует синтетический шум заданной длительности.

        Поддерживаются:
        - `white`: белый шум
        - `pink`: розовый шум
        - `brown`: коричневый шум
        """

        noise_name = str(noise_type).strip().lower()
        num_samples = self._duration_to_samples(duration)

        if noise_name == "white":
            noise = self.rng.standard_normal(num_samples)
            return self._normalize_noise(noise)

        if noise_name == "pink":
            return self._generate_colored_noise(exponent=1.0, num_samples=num_samples)

        if noise_name in {"brown", "brownian"}:
            return self._generate_colored_noise(exponent=2.0, num_samples=num_samples)

        raise ValueError(
            f"Неизвестный синтетический шум: {noise_type!r}. "
            f"Поддерживаются: {', '.join(self.SYNTHETIC_NOISES)}."
        )

    def get_real_noise(self, noise_name: str, duration: float) -> np.ndarray:
        """Возвращает фрагмент реального шума нужной длительности.

        Если исходный шум длиннее, выбирается случайный сегмент.
        Если короче, он циклически повторяется до нужной длины.
        """

        resolved_name = self._resolve_real_noise_name(noise_name)
        if resolved_name is None:
            available_real = ", ".join(sorted(self.real_noises.keys())) or "нет доступных реальных шумов"
            raise KeyError(
                f"Реальный шум {noise_name!r} не найден. Доступные реальные шумы: {available_real}."
            )

        source = self.real_noises[resolved_name]
        target_length = self._duration_to_samples(duration)
        return self._fit_noise_to_length(source, target_length)

    def add_noise(self, clean_audio: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
        """Накладывает шум на чистый сигнал с заданным SNR.

        Формула:
            SNR = 10 * log10(P_signal / P_noise)

        Args:
            clean_audio: Чистый сигнал.
            noise: Шумовой сигнал. Если длина не совпадает, шум будет подогнан.
            snr_db: Целевой уровень SNR в децибелах.

        Returns:
            Зашумлённый сигнал той же длины, что и `clean_audio`.
        """

        clean = np.asarray(clean_audio, dtype=np.float32).reshape(-1)
        if clean.size == 0:
            raise ValueError("Нельзя накладывать шум на пустой сигнал.")

        fitted_noise = self._fit_noise_to_length(np.asarray(noise, dtype=np.float32).reshape(-1), clean.size)

        signal_power = self._power(clean)
        noise_power = self._power(fitted_noise)

        if signal_power <= 0.0:
            raise ValueError("Нельзя вычислить SNR для сигнала с нулевой мощностью.")
        if noise_power <= 0.0:
            raise ValueError("Нельзя использовать шум с нулевой мощностью.")

        target_noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
        scale = np.sqrt(target_noise_power / noise_power)
        scaled_noise = fitted_noise * np.float32(scale)

        return (clean + scaled_noise).astype(np.float32, copy=False)

    def process_file(self, input_path: str | Path, output_dir: str | Path, noise_type: str, snr_db: float, **kwargs,) -> Path:
        """Создаёт зашумлённую версию одного файла и сохраняет её."""

        input_file = Path(input_path)
        output_path = Path(output_dir)

        if not input_file.exists():
            raise FileNotFoundError(f"Чистый аудиофайл не найден: {input_file}")

        try:
            clean_audio = self.audio_file_manager.load(input_file, target_sample_rate=self.sample_rate)
            duration = float(clean_audio.shape[0]) / float(self.sample_rate)
            noise_audio = self._resolve_noise(noise_type=noise_type, duration=duration)
            noisy_audio = self.add_noise(clean_audio=clean_audio, noise=noise_audio, snr_db=snr_db)

            output_manager = AudioFileManager(save_dir=output_path, sample_rate=self.sample_rate)
            filename = self._build_output_filename(
                input_file=input_file,
                noise_type=noise_type,
                snr_db=snr_db,
                filename=kwargs.get("filename"),
            )
            return output_manager.save(noisy_audio, filename=filename)
        except Exception as exc:
            raise RuntimeError(
                f"Не удалось обработать файл {input_file} шумом {noise_type!r} при SNR={snr_db} dB."
            ) from exc

    def process_folder(self, input_dir: str | Path, output_dir: str | Path, noise_type: str, snr_db: float, ext: str = ".wav") -> list[Path]:
        """Обрабатывает все файлы с расширением `ext` в папке `input_dir`."""

        source_dir = Path(input_dir)
        if not source_dir.exists():
            raise FileNotFoundError(f"Папка с чистыми файлами не найдена: {source_dir}")

        suffix = ext if str(ext).startswith(".") else f".{ext}"
        input_files = sorted(path for path in source_dir.glob(f"*{suffix}") if path.is_file())

        saved_files: list[Path] = []
        for input_file in input_files:
            saved_files.append(
                self.process_file(
                    input_path=input_file,
                    output_dir=output_dir,
                    noise_type=noise_type,
                    snr_db=snr_db,
                )
            )
        return saved_files

    def create_noisy_dataset(self, clean_dir: str | Path, output_root: str | Path, noise_types: list[str], snr_levels: list[float], ext: str = ".wav",) -> None:
        """Создаёт набор зашумлённых версий для всех комбинаций шумов и уровней SNR."""

        clean_path = Path(clean_dir)
        output_path = Path(output_root)

        if not clean_path.exists():
            raise FileNotFoundError(f"Папка с чистыми файлами не найдена: {clean_path}")

        for noise_type in noise_types:
            safe_noise_name = self._sanitize_noise_name(noise_type)
            for snr_db in snr_levels:
                target_dir = output_path / safe_noise_name / self._format_snr_folder(snr_db)
                self.process_folder(
                    input_dir=clean_path,
                    output_dir=target_dir,
                    noise_type=noise_type,
                    snr_db=snr_db,
                    ext=ext,
                )

    def plot_audio(self, clean: np.ndarray, noisy: np.ndarray, snr: float):
        """Строит графики чистого и зашумлённого сигнала, а также их спектрограммы."""

        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            raise RuntimeError("matplotlib не установлен или не может быть импортирован.") from exc

        clean_audio = np.asarray(clean, dtype=np.float32).reshape(-1)
        noisy_audio = np.asarray(noisy, dtype=np.float32).reshape(-1)

        if clean_audio.size == 0 or noisy_audio.size == 0:
            raise ValueError("Для визуализации нужны непустые сигналы.")

        time_clean = np.arange(clean_audio.size, dtype=np.float32) / float(self.sample_rate)
        time_noisy = np.arange(noisy_audio.size, dtype=np.float32) / float(self.sample_rate)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"Сравнение сигналов, SNR={float(snr):g} dB")

        axes[0, 0].plot(time_clean, clean_audio)
        axes[0, 0].set_title("Чистый сигнал")
        axes[0, 0].set_xlabel("Время, сек")
        axes[0, 0].set_ylabel("Амплитуда")

        axes[0, 1].plot(time_noisy, noisy_audio)
        axes[0, 1].set_title("Зашумлённый сигнал")
        axes[0, 1].set_xlabel("Время, сек")
        axes[0, 1].set_ylabel("Амплитуда")

        axes[1, 0].specgram(clean_audio, Fs=self.sample_rate)
        axes[1, 0].set_title("Спектрограмма чистого сигнала")

        axes[1, 1].specgram(noisy_audio, Fs=self.sample_rate)
        axes[1, 1].set_title("Спектрограмма зашумлённого сигнала")

        fig.tight_layout()
        return fig

    @staticmethod
    def measure_snr(clean_audio: np.ndarray, noisy_audio: np.ndarray) -> float:
        """Оценивает фактический SNR по чистому и зашумлённому сигналу."""

        clean = np.asarray(clean_audio, dtype=np.float32).reshape(-1)
        noisy = np.asarray(noisy_audio, dtype=np.float32).reshape(-1)

        if clean.size == 0 or noisy.size == 0:
            raise ValueError("Нельзя измерить SNR для пустых сигналов.")
        if clean.size != noisy.size:
            raise ValueError("Для измерения SNR сигналы должны быть одной длины.")

        noise_component = noisy - clean
        signal_power = NoiseManager._power(clean)
        noise_power = NoiseManager._power(noise_component)

        if signal_power <= 0.0:
            raise ValueError("Мощность чистого сигнала равна нулю.")
        if noise_power <= 0.0:
            raise ValueError("Мощность шумовой компоненты равна нулю.")

        return float(10.0 * np.log10(signal_power / noise_power))

    def _resolve_noise(self, noise_type: str, duration: float) -> np.ndarray:
        """Возвращает шум нужного типа и длительности."""

        name = str(noise_type).strip().lower()
        if name in self.SYNTHETIC_NOISES or name == "brownian":
            return self.generate_synthetic_noise(name, duration)
        return self.get_real_noise(name, duration)

    def _resolve_real_noise_name(self, noise_name: str) -> str | None:
        """Проверяет, есть ли реальный шум с указанным именем файла."""

        normalized = str(noise_name).strip().lower()
        return normalized if normalized in self.real_noises else None

    def _fit_noise_to_length(self, noise: np.ndarray, target_length: int) -> np.ndarray:
        """Подгоняет шум под длину сигнала: обрезает или циклически повторяет."""

        source = np.asarray(noise, dtype=np.float32).reshape(-1)
        if source.size == 0:
            raise ValueError("Нельзя использовать пустой шумовой сигнал.")
        if target_length <= 0:
            raise ValueError("Целевая длина шума должна быть положительной.")

        if source.size == target_length:
            return source.copy()

        if source.size > target_length:
            max_start = source.size - target_length
            start = int(self.rng.integers(0, max_start + 1)) if max_start > 0 else 0
            return source[start : start + target_length].astype(np.float32, copy=False)

        offset = int(self.rng.integers(0, source.size)) if source.size > 1 else 0
        rolled = np.roll(source, -offset)
        repeats = ceil(target_length / rolled.size)
        tiled = np.tile(rolled, repeats)
        return tiled[:target_length].astype(np.float32, copy=False)

    def _generate_colored_noise(self, exponent: float, num_samples: int) -> np.ndarray:
        """Генерирует цветной шум в частотной области.

        `exponent=1` даёт розовый шум с PSD ~ 1/f.
        `exponent=2` даёт коричневый шум с PSD ~ 1/f^2.
        """

        if num_samples <= 0:
            raise ValueError("Количество сэмплов должно быть положительным.")

        freqs = np.fft.rfftfreq(num_samples, d=1.0 / float(self.sample_rate))
        spectrum = self.rng.standard_normal(freqs.shape[0]) + 1j * self.rng.standard_normal(freqs.shape[0])

        scale = np.zeros_like(freqs, dtype=np.float64)
        nonzero = freqs > 0.0
        scale[nonzero] = 1.0 / np.power(freqs[nonzero], exponent / 2.0)

        colored_spectrum = spectrum * scale
        colored_spectrum[0] = 0.0

        noise = np.fft.irfft(colored_spectrum, n=num_samples)
        return self._normalize_noise(noise)

    @staticmethod
    def _normalize_noise(noise: np.ndarray) -> np.ndarray:
        """Нормализует шум к нулевому среднему и единичному стандартному отклонению."""

        signal = np.asarray(noise, dtype=np.float64).reshape(-1)
        if signal.size == 0:
            raise ValueError("Нельзя нормализовать пустой шум.")

        signal = signal - np.mean(signal)
        std = float(np.std(signal))
        if not np.isfinite(std) or std <= 0.0:
            raise ValueError("Не удалось нормализовать шум: нулевая или некорректная дисперсия.")

        signal = signal / std
        return signal.astype(np.float32, copy=False)

    @staticmethod
    def _power(signal: np.ndarray) -> float:
        """Вычисляет среднюю мощность сигнала."""

        x = np.asarray(signal, dtype=np.float64).reshape(-1)
        if x.size == 0:
            return 0.0
        return float(np.mean(np.square(x)))

    def _duration_to_samples(self, duration: float) -> int:
        """Переводит длительность в секундах в количество сэмплов."""

        value = float(duration)
        if value <= 0.0:
            raise ValueError("Длительность должна быть положительной.")
        return max(1, int(round(value * float(self.sample_rate))))

    def _build_noise_name(self, path: Path) -> str:
        """Возвращает имя шума только по имени файла, без учёта папок."""

        return path.stem.strip().lower()

    def _build_output_filename(self, input_file: Path, noise_type: str, snr_db: float, filename: str | None = None,) -> str:
        """Формирует имя выходного файла."""

        if filename is not None:
            return str(filename)

        safe_noise_name = self._sanitize_noise_name(noise_type)
        snr_label = self._format_snr_value(snr_db)
        return f"{input_file.stem}_noise_{safe_noise_name}_snr{snr_label}.wav"

    @staticmethod
    def _sanitize_noise_name(noise_type: str) -> str:
        """Преобразует имя шума к безопасному для файлов/папок виду."""

        raw = str(noise_type).strip().lower()
        if not raw:
            return "noise"
        return raw.replace("/", "_").replace("\\", "_").replace(" ", "_")

    @staticmethod
    def _format_snr_value(snr_db: float) -> str:
        """Форматирует значение SNR для имени файла."""

        text = f"{float(snr_db):g}"
        text = text.replace("-", "minus")
        return text.replace(".", "_")

    def _format_snr_folder(self, snr_db: float) -> str:
        """Форматирует имя папки для конкретного уровня SNR."""

        return f"snr_{self._format_snr_value(snr_db)}db"


__all__ = ["NoiseManager"]
