"""Жадное декодирование CTC и уверенность: берём самый вероятный символ на каждом шаге и схлопываем."""
from dataclasses import dataclass

import numpy as np

from alphabet import decode_indices
from ctc_from_scratch import collapse


@dataclass
class Decoded:
    text: str
    indices: list
    confidence: float


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=axis, keepdims=True)


def greedy_decode(probs: np.ndarray, task: str, blank: int = 0) -> Decoded:
    """probs: [T, C] вероятности (после softmax). Возвращает текст, индексы и уверенность.

    Уверенность: для каждого выданного символа берём наибольшую вероятность, достигнутую на его серии
    шагов, и усредняем по символам. Если выдано пусто, уверенность — средняя вероятность blank по всем шагам
    (насколько модель уверена, что в клетке ничего нет).
    """
    probs = np.asarray(probs)
    path = probs.argmax(axis=1)
    indices = collapse(path.tolist(), blank=blank)
    if not indices:
        return Decoded("", [], float(probs[:, blank].mean()))
    run_scores, previous = [], None
    for t, symbol in enumerate(path):
        if symbol == blank:
            previous = symbol
            continue
        if symbol != previous:
            run_scores.append(probs[t, symbol])               # начало новой серии
        else:
            run_scores[-1] = max(run_scores[-1], probs[t, symbol])
        previous = symbol
    return Decoded(decode_indices(indices, task), indices, float(np.mean(run_scores)))


def greedy_decode_batch(probs: np.ndarray, task: str, blank: int = 0) -> list[Decoded]:
    """probs: [B, T, C]."""
    return [greedy_decode(p, task, blank) for p in probs]


def main() -> None:
    """Демо: путь урока, уверенность по сериям и пустая клетка."""
    C = 12                                                  # 11 символов measurement + blank

    def probs_for(path, maxima):
        p = np.zeros((len(path), C))
        for t, (c, m) in enumerate(zip(path, maxima)):
            p[t, :] = (1 - m) / (C - 1)
            p[t, c] = m
        return p

    path = [8, 0, 8, 8, 0, 11, 0, 4, 4]                      # «7-77-,-33»: '7' = 8, ',' = 11, '3' = 4
    maxima = [0.90, 0.90, 0.70, 0.80, 0.90, 0.95, 0.90, 0.60, 0.85]
    r = greedy_decode(probs_for(path, maxima), "measurement")
    print(f"путь 7-77-,-33 → текст {r.text!r}, индексы {r.indices}, уверенность {r.confidence:.3f}")
    print("  уверенность по сериям: 7 → 0.90, 7 → max(0.70, 0.80) = 0.80, ',' → 0.95, 3 → max(0.60, 0.85) = 0.85")
    empty = greedy_decode(probs_for([0, 0, 0, 0], [0.9, 0.7, 0.8, 0.9]), "measurement")
    print(f"пустая клетка → текст {empty.text!r}, уверенность {empty.confidence:.3f}  (среднее по blank)")


if __name__ == "__main__":
    main()
