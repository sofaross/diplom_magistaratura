from __future__ import annotations

import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from src.utils.dataset_loader import EmotionDataset, load_resd_hf, prepare_splits


class DatasetLoaderResdTests(unittest.TestCase):
    def test_load_resd_hf_maps_project_emotions_and_drops_enthusiasm(self) -> None:
        fake_dataset = {
            "train": [
                {
                    "name": "sample_1",
                    "emotion": "anger",
                    "speech": {"array": [0.1, -0.1, 0.05], "sampling_rate": 16000},
                },
                {
                    "name": "sample_2",
                    "emotion": "enthusiasm",
                    "speech": {"array": [0.2, -0.2], "sampling_rate": 16000},
                },
                {
                    "name": "sample_3",
                    "emotion": "sadness",
                    "speech": {"array": [0.3, -0.3, 0.1], "sampling_rate": 16000},
                },
            ]
        }

        fake_module = SimpleNamespace(load_dataset=lambda _: fake_dataset)
        with patch("src.utils.dataset_loader.importlib.import_module", return_value=fake_module):
            df = load_resd_hf("Aniemore/resd", split_names=("train",), verbose=False)

        self.assertEqual(len(df), 2)
        self.assertEqual(sorted(df["emotion"].tolist()), ["angry", "sad"])
        self.assertTrue(all(str(path).startswith("hf://Aniemore__resd/train/") for path in df["path"].tolist()))

    def test_prepare_splits_appends_resd_only_to_train(self) -> None:
        base_df = pd.DataFrame(
            [
                {"path": f"base_{idx}.wav", "emotion": emotion}
                for idx, emotion in enumerate(["angry", "disgust", "fear", "happy", "neutral", "sad"], start=1)
            ]
        )
        resd_df = pd.DataFrame(
            [
                {
                    "path": "hf://Aniemore__resd/train/sample_a",
                    "emotion": "angry",
                    "hf_audio": np.zeros(16, dtype=np.float32),
                    "hf_sampling_rate": 16000,
                },
                {
                    "path": "hf://Aniemore__resd/train/sample_b",
                    "emotion": "sad",
                    "hf_audio": np.zeros(16, dtype=np.float32),
                    "hf_sampling_rate": 16000,
                },
            ]
        )

        def fake_split(df: pd.DataFrame):
            return df.iloc[:2].copy(), df.iloc[2:4].copy(), df.iloc[4:6].copy()

        with patch("src.utils.dataset_loader.load_crema_d", return_value=base_df):
            with patch("src.utils.dataset_loader.load_ravdess", return_value=base_df.iloc[0:0].copy()):
                with patch("src.utils.dataset_loader.load_resd_hf", return_value=resd_df):
                    with patch("src.utils.dataset_loader.split_dataset", side_effect=fake_split):
                        train_df, val_df, test_df, emotion_map = prepare_splits(
                            "crema",
                            "ravdess",
                            emotion_set=6,
                            verbose=False,
                            use_resd=True,
                        )

        self.assertEqual(len(train_df), 4)
        self.assertEqual(len(val_df), 2)
        self.assertEqual(len(test_df), 2)
        self.assertEqual(sum(str(path).startswith("hf://") for path in train_df["path"].tolist()), 2)
        self.assertEqual(sum(str(path).startswith("hf://") for path in val_df["path"].tolist()), 0)
        self.assertEqual(sum(str(path).startswith("hf://") for path in test_df["path"].tolist()), 0)
        self.assertEqual(set(emotion_map.keys()), {"angry", "disgust", "fear", "happy", "neutral", "sad"})

    def test_emotion_dataset_supports_embedded_hf_audio(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "path": "hf://Aniemore__resd/train/sample_1",
                    "emotion": "angry",
                    "label": 0,
                    "hf_audio": np.linspace(-0.5, 0.5, 800, dtype=np.float32),
                    "hf_sampling_rate": 8000,
                }
            ]
        )

        dataset = EmotionDataset(df, augment=False, waveform_noise_augment=False, use_cache=False)
        mel, label = dataset[0]

        self.assertIsInstance(mel, torch.Tensor)
        self.assertEqual(int(label.item()), 0)
        self.assertEqual(int(mel.shape[0]), 1)
        self.assertEqual(int(mel.shape[1]), 128)

    def test_emotion_dataset_ignores_nan_hf_columns_for_regular_rows(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "path": "base_sample.wav",
                    "emotion": "angry",
                    "label": 0,
                    "hf_audio": np.nan,
                    "hf_sampling_rate": np.nan,
                }
            ]
        )

        dataset = EmotionDataset(df, augment=False, waveform_noise_augment=False, use_cache=False)
        with patch("src.utils.dataset_loader.load_audio", return_value=np.linspace(-0.3, 0.3, 1600, dtype=np.float32)) as mock_load:
            mel, label = dataset[0]

        self.assertTrue(mock_load.called)
        self.assertIsInstance(mel, torch.Tensor)
        self.assertEqual(int(label.item()), 0)

    def test_load_resd_hf_reads_audio_from_bytes_when_path_is_not_local(self) -> None:
        buffer = io.BytesIO()
        sf.write(buffer, np.linspace(-0.2, 0.2, 1600, dtype=np.float32), 16000, format="WAV")
        audio_bytes = buffer.getvalue()

        fake_dataset = {
            "train": [
                {
                    "name": "sample_bytes",
                    "emotion": "anger",
                    "speech": {"bytes": audio_bytes, "path": "missing.wav"},
                }
            ]
        }

        class FakeSplit(list):
            def cast_column(self, *_args, **_kwargs):
                return self

        fake_module = SimpleNamespace(
            load_dataset=lambda _: {"train": FakeSplit(fake_dataset["train"])},
            Audio=lambda **_kwargs: None,
        )

        with patch("src.utils.dataset_loader.importlib.import_module", return_value=fake_module):
            df = load_resd_hf("Aniemore/resd", split_names=("train",), verbose=False)

        self.assertEqual(len(df), 1)
        self.assertIsInstance(df.iloc[0]["hf_audio"], np.ndarray)
        self.assertGreater(int(df.iloc[0]["hf_audio"].shape[0]), 0)


if __name__ == "__main__":
    unittest.main()
