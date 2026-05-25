from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_inverse_frequency_class_weights(labels: Sequence[int]) -> torch.Tensor | None:
    """Строит веса классов по обратной частоте.

    Это мягкий способ поддержать редкие и трудные классы без переписывания
    всей логики обучения. Для пустого или одноклассового набора веса не нужны.
    """

    labels = [int(label) for label in labels]
    if not labels:
        return None

    num_classes = int(max(labels) + 1)
    if num_classes <= 1:
        return None

    counts = [0] * num_classes
    for label in labels:
        counts[int(label)] += 1

    total = float(sum(counts))
    weights = [total / max(1.0, num_classes * float(count)) for count in counts]
    return torch.tensor(weights, dtype=torch.float32)


class FocalCrossEntropyLoss(nn.Module):
    """Focal loss поверх cross-entropy.

    Нужна там, где модель уже неплохо учит простые классы, но продолжает путать
    более трудные эмоции вроде fear/sad/disgust. В текущем проекте это ровно тот случай.
    """

    def __init__(
        self,
        *,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if float(gamma) < 0:
            raise ValueError("gamma для focal loss должен быть >= 0.")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction должен быть 'none', 'mean' или 'sum'.")

        self.gamma = float(gamma)
        self.reduction = str(reduction)
        self.label_smoothing = float(label_smoothing)
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        base_ce = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            label_smoothing=float(self.label_smoothing),
        )
        weighted_ce = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none",
            label_smoothing=float(self.label_smoothing),
        )

        pt = torch.exp(-base_ce)
        focal_factor = torch.pow(1.0 - pt, self.gamma)
        loss = focal_factor * weighted_ce

        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()

