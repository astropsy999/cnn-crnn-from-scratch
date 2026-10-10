"""Критерий выбора лучшей эпохи: не accuracy, а то, что нужно в проде."""
import numpy as np

from choose_threshold import choose_threshold


def binary_log_loss(p_empty, labels, eps: float = 1e-7) -> float:
    """Логарифмические потери по вероятностям P(empty) (для разрешения ничьих)."""
    p = np.clip(np.asarray(p_empty, dtype=np.float64), eps, 1 - eps)
    y = np.asarray(labels)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def epoch_score(p_empty, labels, min_value_recall: float = 0.99) -> dict:
    """Оценка эпохи: какой empty_recall достижим при value_recall >= цели (порог подбирается на этих же данных)."""
    r = choose_threshold(p_empty, labels, min_value_recall)
    return {"empty_recall": r["empty_recall"], "value_recall": r["value_recall"], "threshold": r["threshold"]}
