from __future__ import annotations

from math import ceil
from pathlib import Path
import re

import numpy as np

from configs.config import ProjectConfig
from src.audio_io.audio_file_manager import AudioFileManager

DEFAULT_NOISE_CONFIG = ProjectConfig()


class NoiseManager:
    """Менеджер наложения синтетических и файловых шумов."""

    SYNTHETIC_NOISES: tuple[str, ...] = ("white", "pink", "brown")

    def __init__(
        self,
        noise_dir: str | Path = DEFAULT_NOISE_CONFIG.noise_dir,
        sample_rate: int = DEFAULT_NOISE_CONFIG.sample_rate,
        random_seed: int | None = None,
    ) -> None:
        self.noise_dir = Path(noise_dir)
        self.sample_rate = int(sample_rate)
        self.rng = np.random.default_rng(random_seed)
        self.audio_file_manager = AudioFileManager(save_dir=self.noise_dir, sample_rate=self.sample_rate)

        self.noise_dir.mkdir(parents=True, exist_ok=True)

        self.real_noises: dict[str, np.ndarray] = {}
        self.real_noise_groups: dict[str, list[str]] = {}
        self.load_errors: dict[str, str] = {}

        self.reload_real_noises()

    def reload_real_noises(self) -> None:
        self.real_noises.clear()
        self.real_noise_groups.clear()
        self.load_errors.clear()

        for path in sorted(self.noise_dir.rglob("*.wav")):
            try:
                noise_name = self._build_noise_name(path)
                audio = self.audio_file_manager.load(path, target_sample_rate=self.sample_rate)
                audio = np.asarray(audio, dtype=np.float32).reshape(-1)

                if audio.size == 0:
                    raise ValueError("Noise file is empty.")

                if noise_name in self.real_noises:
                    raise ValueError(f"Duplicate noise name detected: {noise_name!r}.")

                self.real_noises[noise_name] = audio
            except Exception as exc:
                self.load_errors[str(path)] = str(exc)

        for noise_name in sorted(self.real_noises.keys()):
            group_name = self._build_noise_group_name(noise_name)
            self.real_noise_groups.setdefault(group_name, []).append(noise_name)

    def list_available_noises(self) -> list[str]:
        """Возвращает синтетические шумы и сгруппированные категории реальных шумов."""

        available = set(self.SYNTHETIC_NOISES)
        available.update(self.real_noise_groups.keys())
        return sorted(available)

    def list_available_noise_variants(self) -> list[str]:
        """Возвращает точные имена файловых шумов, найденных в папке."""

        return sorted(self.real_noises.keys())

    def generate_synthetic_noise(self, noise_type: str, duration: float) -> np.ndarray:
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
            f"Unknown synthetic noise: {noise_type!r}. "
            f"Supported values: {', '.join(self.SYNTHETIC_NOISES)}."
        )

    def get_real_noise(self, noise_name: str, duration: float) -> np.ndarray:
        _, fitted_noise = self.get_real_noise_with_name(noise_name, duration)
        return fitted_noise

    def get_real_noise_with_name(self, noise_name: str, duration: float) -> tuple[str, np.ndarray]:
        resolved_name = self._resolve_real_noise_name(noise_name)
        if resolved_name is None:
            available_real = ", ".join(sorted(self.real_noise_groups.keys())) or "none"
            raise KeyError(
                f"Real noise {noise_name!r} was not found. Available real noises: {available_real}."
            )

        source = self.real_noises[resolved_name]
        target_length = self._duration_to_samples(duration)
        return resolved_name, self._fit_noise_to_length(source, target_length)

    def get_real_noise_group(self, noise_name: str) -> str:
        return self._build_noise_group_name(str(noise_name).strip().lower())

    def add_noise(self, clean_audio: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
        clean = np.asarray(clean_audio, dtype=np.float32).reshape(-1)
        if clean.size == 0:
            raise ValueError("Cannot add noise to an empty signal.")

        fitted_noise = self._fit_noise_to_length(np.asarray(noise, dtype=np.float32).reshape(-1), clean.size)

        signal_power = self._power(clean)
        noise_power = self._power(fitted_noise)

        if signal_power <= 0.0:
            raise ValueError("Signal power is zero; SNR cannot be computed.")
        if noise_power <= 0.0:
            raise ValueError("Noise power is zero; selected noise is invalid.")

        target_noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
        scale = np.sqrt(target_noise_power / noise_power)
        scaled_noise = fitted_noise * np.float32(scale)

        return (clean + scaled_noise).astype(np.float32, copy=False)

    def process_file(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        noise_type: str,
        snr_db: float,
        **kwargs,
    ) -> Path:
        input_file = Path(input_path)
        output_path = Path(output_dir)

        if not input_file.exists():
            raise FileNotFoundError(f"Clean audio file not found: {input_file}")

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
                f"Failed to process file {input_file} with noise {noise_type!r} at SNR={snr_db} dB."
            ) from exc

    def process_folder(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        noise_type: str,
        snr_db: float,
        ext: str = ".wav",
    ) -> list[Path]:
        source_dir = Path(input_dir)
        if not source_dir.exists():
            raise FileNotFoundError(f"Clean-audio directory not found: {source_dir}")

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

    def create_noisy_dataset(
        self,
        clean_dir: str | Path,
        output_root: str | Path,
        noise_types: list[str],
        snr_levels: list[float],
        ext: str = ".wav",
    ) -> None:
        clean_path = Path(clean_dir)
        output_path = Path(output_root)

        if not clean_path.exists():
            raise FileNotFoundError(f"Clean-audio directory not found: {clean_path}")

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
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            raise RuntimeError("matplotlib is not available.") from exc

        clean_audio = np.asarray(clean, dtype=np.float32).reshape(-1)
        noisy_audio = np.asarray(noisy, dtype=np.float32).reshape(-1)

        if clean_audio.size == 0 or noisy_audio.size == 0:
            raise ValueError("Signals for plotting must be non-empty.")

        time_clean = np.arange(clean_audio.size, dtype=np.float32) / float(self.sample_rate)
        time_noisy = np.arange(noisy_audio.size, dtype=np.float32) / float(self.sample_rate)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"Signal comparison, SNR={float(snr):g} dB")

        axes[0, 0].plot(time_clean, clean_audio)
        axes[0, 0].set_title("Clean signal")
        axes[0, 0].set_xlabel("Time, s")
        axes[0, 0].set_ylabel("Amplitude")

        axes[0, 1].plot(time_noisy, noisy_audio)
        axes[0, 1].set_title("Noisy signal")
        axes[0, 1].set_xlabel("Time, s")
        axes[0, 1].set_ylabel("Amplitude")

        axes[1, 0].specgram(clean_audio, Fs=self.sample_rate)
        axes[1, 0].set_title("Clean spectrogram")

        axes[1, 1].specgram(noisy_audio, Fs=self.sample_rate)
        axes[1, 1].set_title("Noisy spectrogram")

        fig.tight_layout()
        return fig

    @staticmethod
    def measure_snr(clean_audio: np.ndarray, noisy_audio: np.ndarray) -> float:
        clean = np.asarray(clean_audio, dtype=np.float32).reshape(-1)
        noisy = np.asarray(noisy_audio, dtype=np.float32).reshape(-1)

        if clean.size == 0 or noisy.size == 0:
            raise ValueError("Cannot measure SNR for empty signals.")
        if clean.size != noisy.size:
            raise ValueError("Signals must have equal length to measure SNR.")

        noise_component = noisy - clean
        signal_power = NoiseManager._power(clean)
        noise_power = NoiseManager._power(noise_component)

        if signal_power <= 0.0:
            raise ValueError("Clean-signal power is zero.")
        if noise_power <= 0.0:
            raise ValueError("Noise-component power is zero.")

        return float(10.0 * np.log10(signal_power / noise_power))

    def _resolve_noise(self, noise_type: str, duration: float) -> np.ndarray:
        name = str(noise_type).strip().lower()
        if name in self.SYNTHETIC_NOISES or name == "brownian":
            return self.generate_synthetic_noise(name, duration)
        return self.get_real_noise(name, duration)

    def _resolve_real_noise_name(self, noise_name: str) -> str | None:
        normalized = str(noise_name).strip().lower()
        if normalized in self.real_noises:
            return normalized

        variants = self.real_noise_groups.get(normalized)
        if not variants:
            return None

        if len(variants) == 1:
            return variants[0]

        index = int(self.rng.integers(0, len(variants)))
        return variants[index]

    def _fit_noise_to_length(self, noise: np.ndarray, target_length: int) -> np.ndarray:
        source = np.asarray(noise, dtype=np.float32).reshape(-1)
        if source.size == 0:
            raise ValueError("Noise signal must be non-empty.")
        if target_length <= 0:
            raise ValueError("Target noise length must be positive.")

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
        if num_samples <= 0:
            raise ValueError("Number of samples must be positive.")

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
        signal = np.asarray(noise, dtype=np.float64).reshape(-1)
        if signal.size == 0:
            raise ValueError("Noise signal must be non-empty.")

        signal = signal - np.mean(signal)
        std = float(np.std(signal))
        if not np.isfinite(std) or std <= 0.0:
            raise ValueError("Noise normalization failed due to zero or invalid standard deviation.")

        signal = signal / std
        return signal.astype(np.float32, copy=False)

    @staticmethod
    def _power(signal: np.ndarray) -> float:
        x = np.asarray(signal, dtype=np.float64).reshape(-1)
        if x.size == 0:
            return 0.0
        return float(np.mean(np.square(x)))

    def _duration_to_samples(self, duration: float) -> int:
        value = float(duration)
        if value <= 0.0:
            raise ValueError("Duration must be positive.")
        return max(1, int(round(value * float(self.sample_rate))))

    def _build_noise_name(self, path: Path) -> str:
        return path.stem.strip().lower()

    @staticmethod
    def _build_noise_group_name(noise_name: str) -> str:
        normalized = str(noise_name).strip().lower()
        match = re.match(r"^(?P<base>.+?)_(?P<index>\d+)$", normalized)
        if match is None:
            return normalized
        return str(match.group("base"))

    def _build_output_filename(
        self,
        input_file: Path,
        noise_type: str,
        snr_db: float,
        filename: str | None = None,
    ) -> str:
        if filename is not None:
            return str(filename)

        safe_noise_name = self._sanitize_noise_name(noise_type)
        snr_label = self._format_snr_value(snr_db)
        return f"{input_file.stem}_noise_{safe_noise_name}_snr{snr_label}.wav"

    @staticmethod
    def _sanitize_noise_name(noise_type: str) -> str:
        raw = str(noise_type).strip().lower()
        if not raw:
            return "noise"
        return raw.replace("/", "_").replace("\\", "_").replace(" ", "_")

    @staticmethod
    def _format_snr_value(snr_db: float) -> str:
        text = f"{float(snr_db):g}"
        text = text.replace("-", "minus")
        return text.replace(".", "_")

    def _format_snr_folder(self, snr_db: float) -> str:
        return f"snr_{self._format_snr_value(snr_db)}db"


__all__ = ["NoiseManager"]
