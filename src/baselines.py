"""Простые базовые модели: с ними сравниваем всё остальное.

Метки классификации: 1 = empty, 0 = value (как в Dataset).
"""
from pathlib import Path

import numpy as np

from image_to_tensor import preprocess_array


def load_tensors(records: list[dict], image_root) -> np.ndarray:
    """Кропы → массив [N, 1, 48, 160] (препроцессинг урока 2, без аугментации)."""
    return np.stack([preprocess_array(Path(image_root) / r["image"]) for r in records]).astype(np.float32)


def load_features(records: list[dict], image_root, pool: int = 2) -> np.ndarray:
    """Кропы → матрица [N, D]: препроцессинг урока 2 и усреднение блоками pool×pool."""
    rows = []
    for r in records:
        arr = preprocess_array(Path(image_root) / r["image"])[0]      # [48, 160]
        if pool > 1:
            h, w = arr.shape
            arr = arr.reshape(h // pool, pool, w // pool, pool).mean(axis=(1, 3))
        rows.append(arr.reshape(-1))
    return np.stack(rows).astype(np.float32)


class OneNN:
    """Один ближайший сосед по евклидову расстоянию между пикселями."""

    def fit(self, X: np.ndarray, y) -> "OneNN":
        self.X = np.asarray(X, dtype=np.float32)
        self.y = list(y)
        self._sq = (self.X ** 2).sum(axis=1)
        return self

    def nearest(self, Q: np.ndarray, chunk: int = 256) -> np.ndarray:
        """Индекс ближайшего обучающего объекта для каждой строки Q."""
        Q = np.asarray(Q, dtype=np.float32)
        out = []
        for i in range(0, len(Q), chunk):
            part = Q[i:i + chunk]
            # |q - x|^2 = |q|^2 - 2 q·x + |x|^2; |q|^2 для argmin не нужен
            dist = self._sq[None, :] - 2.0 * part @ self.X.T
            out.append(dist.argmin(axis=1))
        return np.concatenate(out)

    def predict(self, Q: np.ndarray) -> list:
        return [self.y[i] for i in self.nearest(Q)]


class MajorityClass:
    """Всегда отвечает самым частым классом обучающей выборки."""

    def fit(self, X, y) -> "MajorityClass":
        values, counts = np.unique(np.asarray(y), return_counts=True)
        self.answer = int(values[counts.argmax()])
        return self

    def predict(self, X) -> np.ndarray:
        return np.full(len(X), self.answer, dtype=np.int64)


def ink_mass(X: np.ndarray) -> np.ndarray:
    """Сумма всех пикселей кропа (первая идея «сколько чернил»)."""
    return X.reshape(len(X), -1).sum(axis=1)


def bright_pixels(X: np.ndarray, level: float = 0.3) -> np.ndarray:
    """Число «ярких» пикселей: ярче level (вторая, более честная идея «сколько чернил»)."""
    return (X.reshape(len(X), -1) > level).sum(axis=1)


class InkThreshold:
    """Правило «мало чернил — пусто»: порог подбирается на train по точности."""

    def __init__(self, feature=bright_pixels) -> None:
        self.feature = feature

    def fit(self, X: np.ndarray, y) -> "InkThreshold":
        f = self.feature(X)
        y = np.asarray(y)
        candidates = np.unique(np.quantile(f, np.linspace(0.005, 0.995, 500)))
        self.threshold = float(max(candidates, key=lambda t: ((f < t).astype(int) == y).mean()))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.feature(X) < self.threshold).astype(np.int64)
