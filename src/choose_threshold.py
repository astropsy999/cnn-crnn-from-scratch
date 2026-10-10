"""Урок 8 — «напиши сам»: порог решения для empty-детектора.

Задача: реализовать choose_threshold (ниже). Запуск тестов: python -m pytest tests/test_choose_threshold.py -q


Модель отдаёт P(empty). Кроп объявляется пустым, если P(empty) >= порога.
Пропустить настоящее значение дороже, чем прочитать лишнюю пустую клетку, поэтому порог
выбирается так: вынуть из выборки как можно больше пустых клеток (empty_recall),
НЕ уронив value_recall ниже заданной планки (по умолчанию 0.99).

Порог выбирается на val, а итоговые числа смотрим на test.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from stats import wilson_interval


def check_inputs(p_empty, labels, min_value_recall: float) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p_empty, dtype=np.float64)
    y = np.asarray(labels)
    if p.shape != y.shape or p.ndim != 1:
        raise ValueError("p_empty и labels должны быть одномерными массивами одной длины")
    if not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
        raise ValueError("p_empty должны быть числами из [0, 1]")
    if not set(np.unique(y)) <= {0, 1}:
        raise ValueError("labels должны содержать только 0 (value) и 1 (empty)")
    if (y == 0).sum() == 0 or (y == 1).sum() == 0:
        raise ValueError("нужны кропы обоих классов")
    if not 0 < min_value_recall <= 1:
        raise ValueError("min_value_recall должен быть в (0, 1]")
    return p, y


# --- TODO: напиши сам ------------------------------------------------------------

def choose_threshold(p_empty, labels, min_value_recall: float = 0.99) -> dict:
    """Порог по P(empty): максимум empty_recall при value_recall >= min_value_recall.

    Кроп объявляется пустым, если p_empty >= threshold.

    Требования (их проверяют тесты tests/test_choose_threshold.py):
      1. Среди порогов, для которых value_recall >= min_value_recall, выбирается тот,
         где empty_recall максимален. Кандидаты: все различные значения p_empty и ещё одно число
         чуть выше максимума (порог, при котором пустым не объявляется ничего).
      2. Из нескольких порогов с одинаковым empty_recall берётся НАИБОЛЬШИЙ:
         он безопаснее для value_recall.
      3. Возвращаемый порог лежит посередине между выбранным кандидатом и ближайшим меньшим
         значением p_empty (чтобы не «прилипать» к конкретному кропу). Если пустым не объявляется
         ничего, возвращается сам кандидат «чуть выше максимума».
      4. Возвращаемые value_recall и empty_recall посчитаны по ВОЗВРАЩЁННОМУ порогу
         (value: p_empty < порога; empty: p_empty >= порога).
      5. Некорректные входы дают ValueError: для этого уже есть готовая функция check_inputs.

    Возвращает {"threshold", "value_recall", "empty_recall", "n_value", "n_empty", "min_value_recall"}.
    Подсказка: np.searchsorted(отсортированные_оценки, кандидаты, side="left") считает,
    сколько оценок строго меньше каждого кандидата.
    """
    raise NotImplementedError


# --- Дальше всё готово: оценка порога, чтение оценок, командная строка ------------------

def evaluate_threshold(p_empty, labels, threshold: float) -> dict:
    """Метрики при заданном пороге, с интервалами Уилсона для обоих recall."""
    p = np.asarray(p_empty, dtype=np.float64)
    y = np.asarray(labels)
    pred_empty = p >= threshold
    n_value, n_empty = int((y == 0).sum()), int((y == 1).sum())
    value_ok = int((~pred_empty & (y == 0)).sum())
    empty_ok = int((pred_empty & (y == 1)).sum())
    return {
        "threshold": float(threshold),
        "value_recall": value_ok / n_value if n_value else float("nan"),
        "value_ci": wilson_interval(value_ok, n_value),
        "n_value": n_value,
        "empty_recall": empty_ok / n_empty if n_empty else float("nan"),
        "empty_ci": wilson_interval(empty_ok, n_empty),
        "n_empty": n_empty,
        "acc": float(((pred_empty) == (y == 1)).mean()),
    }


def load_scores(path) -> dict:
    data = np.load(path)
    return {k: data[k] for k in ("val_p_empty", "val_labels", "test_p_empty", "test_labels")}


def format_report(name: str, r: dict) -> str:
    v, e = r["value_ci"], r["empty_ci"]
    return (f"{name:<5} value_recall {r['value_recall']:.4f} [{v[0]:.3f}–{v[1]:.3f}] (n={r['n_value']})   "
            f"empty_recall {r['empty_recall']:.4f} [{e[0]:.3f}–{e[1]:.3f}] (n={r['n_empty']})")


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="чекпоинт tiny CNN (нужен torch)")
    src.add_argument("--scores", help="готовые оценки .npz (val_p_empty, val_labels, test_p_empty, test_labels)")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--min-value-recall", type=float, default=0.99)
    ap.add_argument("--save-scores", help="сохранить оценки модели в .npz (для plot_threshold_curve.py)")
    ap.add_argument("--write", action="store_true", help="записать выбранный порог в чекпоинт")
    args = ap.parse_args()

    if args.scores:
        s = load_scores(args.scores)
    else:
        from inference import scores_from_checkpoint  # импорт здесь: нужен torch
        s = scores_from_checkpoint(args.checkpoint, args.data_dir)
    if args.save_scores:
        np.savez(args.save_scores, **s)
        print(f"оценки сохранены: {args.save_scores}")

    chosen = choose_threshold(s["val_p_empty"], s["val_labels"], args.min_value_recall)
    thr = chosen["threshold"]
    print(f"порог выбран на val: {thr:.4f}  (ограничение: value_recall >= {args.min_value_recall})")
    val = evaluate_threshold(s["val_p_empty"], s["val_labels"], thr)
    test = evaluate_threshold(s["test_p_empty"], s["test_labels"], thr)
    print(format_report("val", val))
    print(format_report("test", test))
    if test["value_ci"][0] < args.min_value_recall:
        print(f"ВНИМАНИЕ: нижняя граница интервала value_recall на test "
              f"({test['value_ci'][0]:.3f}) ниже цели {args.min_value_recall}.")

    if args.write:
        if not args.checkpoint:
            sys.exit("--write требует --checkpoint")
        from inference import write_threshold  # импорт здесь: нужен torch
        write_threshold(args.checkpoint, chosen, val, test)
        print(f"порог записан в {args.checkpoint}")


if __name__ == "__main__":
    main()
