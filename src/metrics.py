"""Метрики бинарной классификации empty / value (метка 1 = empty, 0 = value)."""
import numpy as np


def binary_report(pred, labels) -> dict:
    """accuracy, а также recall по каждому классу и их размеры."""
    pred = np.asarray(pred)
    labels = np.asarray(labels)
    n_empty = int((labels == 1).sum())
    n_value = int((labels == 0).sum())
    return {
        "acc": float((pred == labels).mean()),
        "value_recall": float(((pred == 0) & (labels == 0)).sum() / n_value) if n_value else float("nan"),
        "empty_recall": float(((pred == 1) & (labels == 1)).sum() / n_empty) if n_empty else float("nan"),
        "n_value": n_value,
        "n_empty": n_empty,
    }
