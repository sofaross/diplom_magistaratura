from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from configs.config import ProjectConfig
from src.audio_io.audio_file_manager import AudioFileManager
from src.dto.multimodal_result_dto import PreparedAudioDto
from src.noise.noise_manager import NoiseManager


class NoiseService:
    """Накладывает случайный или выбранный шум и сохраняет обработанный аудиофайл."""

    def __init__(
        self,
        *,
        noise_manager: NoiseManager | None = None,
        output_file_manager: AudioFileManager | None = None,
        config: ProjectConfig | None = None,
    ) -> None:
        self.config = config or ProjectConfig()
        self.sample_rate = int(self.config.sample_rate)
        self.noise_manager = noise_manager or NoiseManager(
            noise_dir=self.config.noise_dir,
            sample_rate=self.sample_rate,
        )
        self.output_file_manager = output_file_manager or AudioFileManager(
            save_dir=self.config.noisy_recordings_dir,
            sample_rate=self.sample_rate,
        )

    def list_available_noises(self) -> list[str]:
        return self.noise_manager.list_available_noises()

    def list_available_noise_variants(self) -> list[str]:
        return self.noise_manager.list_available_noise_variants()

    def apply_random_noise(
        self,
        prepared_audio: PreparedAudioDto,
        *,
        snr_db: float = 10.0,
        candidate_noise_types: list[str] | None = None,
        filename: str | None = None,
    ) -> tuple[PreparedAudioDto, dict[str, object]]:
        candidates = candidate_noise_types or self.list_available_noises()
        if not candidates:
            raise ValueError("No available noise types to choose from.")

        noise_type = random.choice(list(candidates))
        return self.apply_selected_noise(
            prepared_audio,
            noise_type=noise_type,
            snr_db=snr_db,
            filename=filename,
            noise_mode="random",
        )

    def apply_selected_noise(
        self,
        prepared_audio: PreparedAudioDto,
        *,
        noise_type: str,
        snr_db: float = 10.0,
        filename: str | None = None,
        noise_mode: str = "selected",
    ) -> tuple[PreparedAudioDto, dict[str, object]]:
        clean_audio = np.asarray(prepared_audio.audio, dtype=np.float32)
        duration = prepared_audio.duration_seconds
        resolved_noise_type, resolved_noise_variant, noise_audio = self._resolve_noise_audio(noise_type, duration)
        noisy_audio = self.noise_manager.add_noise(clean_audio, noise_audio, snr_db=snr_db)

        output_filename = filename or self._build_output_filename(prepared_audio, resolved_noise_variant, snr_db)
        output_path = self.output_file_manager.save(noisy_audio, filename=output_filename)

        payload = PreparedAudioDto(
            source_type=prepared_audio.source_type,
            original_file_path=prepared_audio.original_file_path,
            audio=noisy_audio,
            sample_rate=prepared_audio.sample_rate,
            duration_seconds=prepared_audio.duration_seconds,
            prepared_file_path=output_path.resolve(),
        )
        metadata = {
            "noise_applied": True,
            "noise_mode": str(noise_mode),
            "noise_type": str(resolved_noise_type).lower(),
            "noise_variant": str(resolved_noise_variant).lower(),
            "snr_db": float(snr_db),
        }
        return payload, metadata

    def _resolve_noise_audio(self, noise_type: str, duration: float) -> tuple[str, str, np.ndarray]:
        normalized_name = str(noise_type).strip().lower()
        try:
            normalized_name = "brown" if normalized_name == "brownian" else normalized_name
            return (
                normalized_name,
                normalized_name,
                self.noise_manager.generate_synthetic_noise(normalized_name, duration),
            )
        except ValueError:
            resolved_variant, noise_audio = self.noise_manager.get_real_noise_with_name(normalized_name, duration)
            resolved_type = self.noise_manager.get_real_noise_group(resolved_variant)
            return resolved_type, resolved_variant, noise_audio

    def _build_output_filename(self, prepared_audio: PreparedAudioDto, noise_type: str, snr_db: float) -> str:
        base_name = prepared_audio.original_file_path.stem
        safe_noise = str(noise_type).strip().lower().replace(" ", "_")
        snr_label = str(float(snr_db)).replace(".", "_").replace("-", "minus")
        return f"{base_name}_noise_{safe_noise}_snr{snr_label}.wav"


__all__ = ["NoiseService"]
