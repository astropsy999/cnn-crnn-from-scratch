"""Простые базовые модели: с ними сравниваем всё остальное (в уроке 7 добавятся ещё)."""
from pathlib import Path

import numpy as np

from image_to_tensor import preprocess_array


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
