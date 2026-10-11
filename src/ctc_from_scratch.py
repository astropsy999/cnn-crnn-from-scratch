"""Урок 10 — «напиши сам»: схлопывание пути CTC и минимальное число шагов (остальное готово).

Задача: реализовать collapse и min_time_steps (ниже). Запуск тестов: python -m pytest tests/test_ctc.py -q

CTC руками: схлопывание пути, минимальное число шагов и прямой проход (alpha-рекурсия).

Путь — это ответ модели на каждом временном шаге (символ или blank). CTC учит модель так,
чтобы СУММА вероятностей всех путей, которые схлопываются в нужную строку, была как можно больше.

Схлопывание: сначала склеиваем подряд идущие одинаковые символы, затем выбрасываем blank.
Поэтому blank между двумя одинаковыми символами сохраняет оба: "7-7" -> "77", а "77" -> "7".
"""
import itertools
import math

import numpy as np

NEG_INF = -math.inf


# --- TODO: напиши сам ------------------------------------------------------------

def collapse(path, blank="-"):
    """Схлопывает путь CTC: склеить подряд идущие одинаковые символы, затем убрать blank.

    path: строка или список (индексов); blank: символ-пустышка ('-' для строк, 0 для индексов).

    Требования (их проверяют тесты tests/test_ctc.py):
      1. "-aa-b-" -> "ab";  "7-,-33" -> "7,3";  "7-77-,-33" -> "77,3".
      2. blank между двумя одинаковыми символами сохраняет оба: "7-7" -> "77", а "77" -> "7".
         Порядок действий важен: если сначала выбросить blank, а потом склеить повторы,
         получится неверно (проверьте на "7-77-,-33").
      3. Работает и со строками, и со списками; возвращает объект того же типа
         (строку для строки, список для списка).
      4. Параметр blank задаёт, какой символ считать пустышкой.
    """
    raise NotImplementedError


def min_time_steps(text) -> int:
    """Минимум временных шагов, при котором из модели вообще может получиться строка text.

    Подсказка: каждому символу нужен свой шаг, а между двумя ОДИНАКОВЫМИ соседними символами
    нужен ещё blank.  "7,33" -> 5, "77" -> 3, "7" -> 1, "" -> 0.
    Тесты сверяют формулу с перебором всех путей (она должна быть точной).
    """
    raise NotImplementedError


# --- Дальше всё готово: прямой проход CTC и проверка перебором ------------------------

def logsumexp(values) -> float:
    """log(sum(exp(v))) без переполнения; все -inf дают -inf."""
    values = [v for v in values if v != NEG_INF]
    if not values:
        return NEG_INF
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


def ctc_neg_log_likelihood(log_probs: np.ndarray, target, blank: int = 0) -> float:
    """-log P(target | модель) прямым проходом CTC (alpha-рекурсия в логарифмах).

    log_probs: [T, C] — логарифмы вероятностей классов на каждом шаге (после log_softmax).
    target: список индексов без blank. Если шагов меньше, чем min_time_steps(target), результат inf.

    Расширенная цель: blank, y1, blank, y2, ..., yL, blank (длина 2L+1). alpha[t, s] — суммарная
    вероятность всех путей длины t+1, закончившихся в позиции s расширенной цели.
    """
    T = log_probs.shape[0]
    ext = [blank]
    for y in target:
        ext += [y, blank]
    S = len(ext)
    alpha = np.full((T, S), NEG_INF)
    alpha[0, 0] = log_probs[0, blank]
    if S > 1:
        alpha[0, 1] = log_probs[0, ext[1]]
    for t in range(1, T):
        for s in range(S):
            candidates = [alpha[t - 1, s]]                       # остались на месте
            if s >= 1:
                candidates.append(alpha[t - 1, s - 1])           # шагнули на соседнюю позицию
            if s >= 2 and ext[s] != blank and ext[s] != ext[s - 2]:
                candidates.append(alpha[t - 1, s - 2])           # перепрыгнули blank (нельзя, если символы равны)
            alpha[t, s] = logsumexp(candidates) + log_probs[t, ext[s]]
    ends = [alpha[T - 1, S - 1]] + ([alpha[T - 1, S - 2]] if S > 1 else [])
    return -logsumexp(ends)


def bruteforce_neg_log_likelihood(log_probs: np.ndarray, target, blank: int = 0) -> float:
    """То же самое «в лоб»: перебрать ВСЕ пути, схлопнуть и сложить вероятности подходящих.
    Работает только для крошечных T и C, зато очевидно верно: эталон для проверки alpha-рекурсии."""
    T, C = log_probs.shape
    target = list(target)
    total = 0.0
    for path in itertools.product(range(C), repeat=T):
        if collapse(list(path), blank=blank) == target:
            total += math.exp(sum(log_probs[t, c] for t, c in enumerate(path)))
    return -math.log(total) if total > 0 else math.inf


def log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - logits.max(axis=axis, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))


def main() -> None:
    print("схлопывание пути (blank = '-'):")
    for path in ["-aa-b-", "7-,-33", "7-77-,-33"]:
        note = "   # blank разделяет повторы: так записывается «77»" if path == "7-77-,-33" else ""
        print(f"  {path:<10} → {collapse(path)}{note}")
    print(f"  min_time_steps(\"7,33\") = {min_time_steps('7,33')}")

    rng = np.random.default_rng(0)
    log_probs = log_softmax(rng.normal(size=(5, 3)))
    target = [1, 1, 2]                                   # «aab» с повтором: нужно хотя бы 4 шага
    fast = ctc_neg_log_likelihood(log_probs, target)
    slow = bruteforce_neg_log_likelihood(log_probs, target)
    print(f"\nпрямой проход vs перебор всех путей (T=5, C=3, цель {target}):")
    print(f"  alpha-рекурсия: {fast:.6f}   перебор: {slow:.6f}   разница {abs(fast - slow):.1e}")

    try:
        import torch
        import torch.nn.functional as F
        lp = torch.from_numpy(log_probs).float()[:, None, :]              # [T, B=1, C]
        ref = F.ctc_loss(lp, torch.tensor([target]), torch.tensor([5]), torch.tensor([len(target)]),
                         blank=0, reduction="none")[0].item()
        print(f"  torch nn.CTCLoss: {ref:.6f}   разница с alpha {abs(fast - ref):.1e}")
    except ImportError:
        print("  torch не установлен: сравнение с nn.CTCLoss пропущено")


if __name__ == "__main__":
    main()
