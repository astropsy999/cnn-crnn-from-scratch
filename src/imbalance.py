"""Дисбаланс классов: веса для функции потерь и веса для сэмплера (метка 1 = empty, 0 = value)."""
import numpy as np


def _counts(labels) -> np.ndarray:
    labels = np.asarray(labels)
    counts = np.array([(labels == 0).sum(), (labels == 1).sum()], dtype=np.float64)
    if (counts == 0).any():
        raise ValueError("нужны кропы обоих классов")
    return counts


def class_weights(labels) -> np.ndarray:
    """Вес класса для функции потерь: N / (2 · n_класса). Редкий класс получает вес больше 1."""
    counts = _counts(labels)
    return counts.sum() / (2.0 * counts)


def sample_weights(labels) -> np.ndarray:
    """Вес каждого кропа для сэмплера: 1 / n_его_класса. Суммарный вес обоих классов одинаков,
    значит, в батчах классы встречаются поровну."""
    labels = np.asarray(labels)
    counts = _counts(labels)
    return 1.0 / counts[labels]


def parse_imbalance(mode: str) -> tuple[bool, bool]:
    """Режим → (использовать веса в loss, использовать сэмплер)."""
    table = {"none": (False, False), "weights": (True, False), "sampler": (False, True), "both": (True, True)}
    if mode not in table:
        raise ValueError(f"imbalance: допустимо {sorted(table)}, получено {mode!r}")
    return table[mode]
