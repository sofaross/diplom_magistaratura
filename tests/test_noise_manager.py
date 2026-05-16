from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.audio_io.audio_file_manager import AudioFileManager
from src.noise.noise_manager import NoiseManager


class NoiseManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.clean_dir = self.root / "clean"
        self.noise_dir = self.root / "noises"
        self.output_dir = self.root / "output"

        self.sample_rate = 16000
        self.clean_manager = AudioFileManager(save_dir=self.clean_dir, sample_rate=self.sample_rate)
        self.noise_file_manager = AudioFileManager(save_dir=self.noise_dir, sample_rate=self.sample_rate)

        duration = 1.0
        time = np.arange(int(self.sample_rate * duration), dtype=np.float32) / float(self.sample_rate)

        self.clean_audio = 0.1 * np.sin(2.0 * np.pi * 220.0 * time).astype(np.float32)
        self.clean_file = self.clean_manager.save(self.clean_audio, "clean.wav")
        self.clean_manager.save(self.clean_audio * 0.8, "clean_2.wav")

        short_real_noise = 0.03 * np.sin(2.0 * np.pi * 37.0 * time[: self.sample_rate // 4]).astype(np.float32)
        self.noise_file_manager.save(short_real_noise, "metro.wav")
        self.noise_file_manager.save(short_real_noise * 0.8, "rain_1.wav")
        self.noise_file_manager.save(short_real_noise * 0.6, "rain_2.wav")

        self.manager = NoiseManager(
            noise_dir=self.noise_dir,
            sample_rate=self.sample_rate,
            random_seed=123,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_list_available_noises_contains_synthetic_and_real(self) -> None:
        available = self.manager.list_available_noises()

        self.assertIn("white", available)
        self.assertIn("pink", available)
        self.assertIn("brown", available)
        self.assertIn("metro", available)
        self.assertIn("rain", available)
        self.assertNotIn("rain_1", available)

    def test_get_real_noise_returns_requested_length(self) -> None:
        noise = self.manager.get_real_noise("metro", duration=0.75)

        self.assertEqual(noise.shape[0], int(0.75 * self.sample_rate))
        self.assertTrue(np.isfinite(noise).all())

    def test_grouped_real_noise_name_resolves_to_variant(self) -> None:
        variant_name, noise = self.manager.get_real_noise_with_name("rain", duration=0.75)

        self.assertIn(variant_name, {"rain_1", "rain_2"})
        self.assertEqual(self.manager.get_real_noise_group(variant_name), "rain")
        self.assertEqual(noise.shape[0], int(0.75 * self.sample_rate))

    def test_add_noise_matches_target_snr(self) -> None:
        noise = self.manager.generate_synthetic_noise("white", duration=1.0)
        noisy = self.manager.add_noise(self.clean_audio, noise, snr_db=10.0)
        measured_snr = self.manager.measure_snr(self.clean_audio, noisy)

        self.assertAlmostEqual(measured_snr, 10.0, delta=0.5)

    def test_process_folder_creates_outputs(self) -> None:
        saved = self.manager.process_folder(
            input_dir=self.clean_dir,
            output_dir=self.output_dir,
            noise_type="white",
            snr_db=5.0,
        )

        self.assertEqual(len(saved), 2)
        for path in saved:
            self.assertTrue(path.exists())
            self.assertIn("noise_white", path.name)

    def test_create_noisy_dataset_builds_expected_structure(self) -> None:
        self.manager.create_noisy_dataset(
            clean_dir=self.clean_dir,
            output_root=self.output_dir,
            noise_types=["white", "metro"],
            snr_levels=[10.0, 0.0],
        )

        expected_paths = [
            self.output_dir / "white" / "snr_10db" / "clean_noise_white_snr10.wav",
            self.output_dir / "white" / "snr_0db" / "clean_noise_white_snr0.wav",
            self.output_dir / "metro" / "snr_10db" / "clean_noise_metro_snr10.wav",
            self.output_dir / "metro" / "snr_0db" / "clean_noise_metro_snr0.wav",
        ]

        for path in expected_paths:
            self.assertTrue(path.exists(), msg=f"Не найден ожидаемый файл: {path}")


if __name__ == "__main__":
    unittest.main()
