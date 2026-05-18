from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from src.training.train_emotion import train_emotion_model


class DummyEmotionClassifier(torch.nn.Module):
    def __init__(self, num_emotions: int = 2) -> None:
        super().__init__()
        self.head = torch.nn.Linear(1, num_emotions)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        features = x.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(-1)
        return self.head(features)


class TrainEmotionTests(unittest.TestCase):
    def test_train_emotion_model_selects_best_epoch_by_val_f1_macro(self) -> None:
        model = DummyEmotionClassifier(num_emotions=2)

        batch = (
            torch.zeros(2, 1, 128, 8),
            torch.tensor([8, 8], dtype=torch.long),
            torch.tensor([0, 1], dtype=torch.long),
        )

        metrics_sequence = [
            {"accuracy": 0.80, "f1_macro": 0.40},
            {"accuracy": 0.70, "f1_macro": 0.60},
        ]

        with patch("src.training.train_emotion.evaluate_emotion_model", side_effect=metrics_sequence):
            _, summary = train_emotion_model(
                model,
                train_loader=[batch],
                val_loader=[batch],
                epochs=2,
                lr=1e-3,
                scheduler_name="plateau",
                early_stopping_patience=10,
                device="cpu",
                out_dir=None,
            )

        self.assertEqual(summary["best_epoch"], 2)
        self.assertAlmostEqual(summary["best_val_accuracy"], 0.70, places=6)
        self.assertAlmostEqual(summary["best_val_f1_macro"], 0.60, places=6)
        self.assertEqual(summary["selection_metric"], "val_f1_macro")

    def test_train_emotion_model_supports_focal_loss(self) -> None:
        model = DummyEmotionClassifier(num_emotions=2)

        batch = (
            torch.zeros(2, 1, 128, 8),
            torch.tensor([8, 8], dtype=torch.long),
            torch.tensor([0, 1], dtype=torch.long),
        )

        with patch(
            "src.training.train_emotion.evaluate_emotion_model",
            return_value={"accuracy": 0.5, "f1_macro": 0.5},
        ):
            _, summary = train_emotion_model(
                model,
                train_loader=[batch],
                val_loader=[batch],
                epochs=1,
                lr=1e-3,
                loss_name="focal",
                focal_gamma=1.5,
                scheduler_name="plateau",
                device="cpu",
                out_dir=None,
            )

        self.assertEqual(summary["loss_name"], "focal")
        self.assertAlmostEqual(summary["focal_gamma"], 1.5, places=6)


if __name__ == "__main__":
    unittest.main()
