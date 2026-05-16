from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
import numpy as np

from src.audio_io.audio_file_manager import AudioFileManager
from src.utils.dataset_loader import (
    EmotionDataset,
    WaveformNoiseAugmentConfig,
    prepare_datasets,
)


class WaveformNoiseAugmentationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.audio_dir = self.root / "audio"
        self.noise_dir = self.root / "noise"

        self.sample_rate = 16000
        self.audio_manager = AudioFileManager(save_dir=self.audio_dir, sample_rate=self.sample_rate)
        self.noise_manager = AudioFileManager(save_dir=self.noise_dir, sample_rate=self.sample_rate)

        time = np.arange(self.sample_rate, dtype=np.float32) / float(self.sample_rate)
        clean_audio = 0.15 * np.sin(2.0 * np.pi * 220.0 * time).astype(np.float32)
        real_noise = 0.05 * np.sin(2.0 * np.pi * 70.0 * time).astype(np.float32)

        self.audio_path = self.audio_manager.save(clean_audio, "sample.wav")
        self.noise_manager.save(real_noise, "rain_1.wav")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_waveform_noise_augmentation_changes_train_mel(self) -> None:
        frame = pd.DataFrame([{"path": str(self.audio_path), "label": 0}])

        clean_dataset = EmotionDataset(
            frame,
            augment=False,
            waveform_noise_augment=False,
            use_cache=False,
        )
        noisy_dataset = EmotionDataset(
            frame,
            augment=False,
            waveform_noise_augment=True,
            waveform_noise_config=WaveformNoiseAugmentConfig(
                noise_prob=1.0,
                noise_dir=self.noise_dir,
                noise_types=("white",),
                snr_min=10.0,
                snr_max=10.0,
                random_seed=123,
            ),
            use_cache=False,
        )

        random.seed(123)
        clean_mel, _ = clean_dataset[0]
        random.seed(123)
        noisy_mel, _ = noisy_dataset[0]

        self.assertFalse(torch.allclose(clean_mel, noisy_mel))

    @patch("src.utils.dataset_loader.prepare_splits")
    def test_prepare_datasets_enables_waveform_noise_only_for_train(self, mock_prepare_splits) -> None:
        train_df = pd.DataFrame([{"path": str(self.audio_path), "label": 0}])
        val_df = pd.DataFrame([{"path": str(self.audio_path), "label": 0}])
        test_df = pd.DataFrame([{"path": str(self.audio_path), "label": 0}])
        mock_prepare_splits.return_value = (train_df, val_df, test_df, {"neutral": 0})

        train_ds, val_ds, test_ds, emotion_map = prepare_datasets(
            "crema",
            "ravdess",
            augment=True,
            waveform_noise_augment=True,
            waveform_noise_config=WaveformNoiseAugmentConfig(
                noise_prob=0.5,
                noise_dir=self.noise_dir,
                noise_types=("white",),
                snr_min=5.0,
                snr_max=20.0,
            ),
            cache_dir=self.root / "cache",
        )

        self.assertEqual(emotion_map, {"neutral": 0})
        self.assertTrue(train_ds.waveform_noise_augment)
        self.assertFalse(val_ds.waveform_noise_augment)
        self.assertFalse(test_ds.waveform_noise_augment)
        self.assertFalse(train_ds.use_cache)


if __name__ == "__main__":
    unittest.main()
