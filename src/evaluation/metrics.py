from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# ===============================
# Считает все метрики классификации:
#   accuracy:         сколько процентов угадано правильно
#   f1_macro:         средняя точность по всем эмоциям (учитывает дисбаланс)
#   f1_weighted:      взвешенная точность (по количеству примеров)
#   confusion_matrix: матрица ошибок (кто с кем путается)
# ===============================
def compute_classification_metrics(y_true, y_pred) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

# ===============================
# Создает красивый текстовый отчет с точностью по каждой эмоции
# ===============================
def build_classification_report(y_true, y_pred, target_names=None) -> str:
    return classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
