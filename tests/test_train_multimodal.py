from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from src.models.emotion_model import EmotionModel, EmotionModelImproved
from src.models.multimodal_model import FusionModel
from src.training.train_multimodal import (
    _infer_emotion_set,
    _load_emotion_model,
    train_fusion_model,
)


class DummyEmotionModel(torch.nn.Module):
    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.last_lengths: torch.Tensor | None = None

    def extract_embedding(self, mels: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        self.last_lengths = None if lengths is None else lengths.detach().cpu().clone()
        return mels.new_zeros((mels.shape[0], self.embedding_dim))


class TrainMultimodalTests(unittest.TestCase):
    def test_infer_emotion_set_accepts_only_supported_sizes(self) -> None:
        self.assertEqual(_infer_emotion_set({"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5}), 6)
        self.assertEqual(_infer_emotion_set({str(i): i for i in range(8)}), 8)

        with self.assertRaises(ValueError):
            _infer_emotion_set({"a": 0, "b": 1})

    def test_load_emotion_model_supports_improved_and_baseline_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            improved_path = temp_root / "improved.pt"
            baseline_path = temp_root / "baseline.pt"

            torch.save({"state_dict": EmotionModelImproved(num_emotions=6).state_dict()}, improved_path)
            torch.save({"state_dict": EmotionModel(num_emotions=6).state_dict()}, baseline_path)

            improved = _load_emotion_model(improved_path, num_emotions=6, device=torch.device("cpu"))
            baseline = _load_emotion_model(baseline_path, num_emotions=6, device=torch.device("cpu"))

            self.assertIsInstance(improved, EmotionModelImproved)
            self.assertIsInstance(baseline, EmotionModel)

    def test_train_fusion_model_passes_lengths_to_emotion_model(self) -> None:
        fusion_model = FusionModel(speech_dim=4, emotion_dim=128, num_classes=2)
        emotion_model = DummyEmotionModel()

        batch = (
            [np.zeros(32, dtype=np.float32), np.zeros(24, dtype=np.float32)],
            torch.zeros(2, 1, 128, 8),
            torch.tensor([8, 5], dtype=torch.long),
            torch.tensor([0, 1], dtype=torch.long),
        )

        with patch("src.training.train_multimodal.extract_embedding", return_value=torch.zeros(2, 4)):
            with patch(
                "src.training.train_multimodal.evaluate_fusion_model",
                return_value={"accuracy": 1.0, "f1_macro": 1.0},
            ):
                _, summary = train_fusion_model(
                    fusion_model,
                    speech_wrapper=object(),
                    emotion_model=emotion_model,
                    train_loader=[batch],
                    val_loader=[],
                    epochs=1,
                    lr=1e-3,
                    device="cpu",
                    out_dir=None,
                )

        self.assertIsNotNone(emotion_model.last_lengths)
        self.assertEqual(emotion_model.last_lengths.tolist(), [8, 5])
        self.assertEqual(summary["selection_metric"], "val_f1_macro")

    def test_train_fusion_model_uses_requested_speech_embedding_settings(self) -> None:
        fusion_model = FusionModel(speech_dim=4, emotion_dim=128, num_classes=2)
        emotion_model = DummyEmotionModel()

        batch = (
            [np.zeros(32, dtype=np.float32), np.zeros(24, dtype=np.float32)],
            torch.zeros(2, 1, 128, 8),
            torch.tensor([8, 5], dtype=torch.long),
            torch.tensor([0, 1], dtype=torch.long),
        )

        with patch("src.training.train_multimodal.extract_embedding", return_value=torch.zeros(2, 4)) as mock_extract:
            with patch(
                "src.training.train_multimodal.evaluate_fusion_model",
                return_value={"accuracy": 0.5, "f1_macro": 0.7},
            ) as mock_eval:
                _, summary = train_fusion_model(
                    fusion_model,
                    speech_wrapper=object(),
                    emotion_model=emotion_model,
                    train_loader=[batch],
                    val_loader=[],
                    epochs=1,
                    lr=1e-3,
                    speech_layer=8,
                    speech_pool="max",
                    device="cpu",
                    out_dir=None,
                )

        self.assertEqual(mock_extract.call_args.kwargs["layer"], 8)
        self.assertEqual(mock_extract.call_args.kwargs["pool"], "max")
        self.assertEqual(mock_eval.call_args.kwargs["speech_layer"], 8)
        self.assertEqual(mock_eval.call_args.kwargs["speech_pool"], "max")
        self.assertEqual(summary["speech_layer"], 8)
        self.assertEqual(summary["speech_pool"], "max")

    def test_fusion_model_forward_returns_expected_shape(self) -> None:
        model = FusionModel(speech_dim=16, emotion_dim=8, num_classes=6)
        logits = model(torch.randn(3, 16), torch.randn(3, 8))
        self.assertEqual(tuple(logits.shape), (3, 6))

    def test_fusion_model_forward_supports_modality_dropout(self) -> None:
        model = FusionModel(
            speech_dim=16,
            emotion_dim=8,
            num_classes=6,
            modality_dropout_prob=1.0,
        )
        model.train()
        logits = model(torch.randn(3, 16), torch.randn(3, 8))
        self.assertEqual(tuple(logits.shape), (3, 6))

    def test_train_fusion_model_supports_focal_loss(self) -> None:
        fusion_model = FusionModel(speech_dim=4, emotion_dim=128, num_classes=2)
        emotion_model = DummyEmotionModel()

        batch = (
            [np.zeros(32, dtype=np.float32), np.zeros(24, dtype=np.float32)],
            torch.zeros(2, 1, 128, 8),
            torch.tensor([8, 5], dtype=torch.long),
            torch.tensor([0, 1], dtype=torch.long),
        )

        with patch("src.training.train_multimodal.extract_embedding", return_value=torch.zeros(2, 4)):
            with patch(
                "src.training.train_multimodal.evaluate_fusion_model",
                return_value={"accuracy": 0.5, "f1_macro": 0.6},
            ):
                _, summary = train_fusion_model(
                    fusion_model,
                    speech_wrapper=object(),
                    emotion_model=emotion_model,
                    train_loader=[batch],
                    val_loader=[],
                    epochs=1,
                    lr=1e-3,
                    loss_name="focal",
                    focal_gamma=1.5,
                    device="cpu",
                    out_dir=None,
                )

        self.assertEqual(summary["loss_name"], "focal")
        self.assertAlmostEqual(summary["focal_gamma"], 1.5, places=6)


if __name__ == "__main__":
    unittest.main()
