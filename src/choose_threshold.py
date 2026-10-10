"""Порог решения для empty-детектора.

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


def choose_threshold(p_empty, labels, min_value_recall: float = 0.99) -> dict:
    """Порог по P(empty): максимум empty_recall при value_recall >= min_value_recall.

    Кроп объявляется пустым, если p_empty >= threshold.
    Из нескольких порогов с одинаковым empty_recall берётся НАИБОЛЬШИЙ (он безопаснее для value).
    Возвращаемый порог лежит посередине между двумя соседними значениями p_empty,
    чтобы не «прилипать» к конкретному кропу.
    Возвращает {"threshold", "value_recall", "empty_recall", "n_value", "n_empty", "min_value_recall"}.
    """
    p, y = check_inputs(p_empty, labels, min_value_recall)
    value_scores = np.sort(p[y == 0])
    empty_scores = np.sort(p[y == 1])
    n_value, n_empty = len(value_scores), len(empty_scores)

    scores = np.unique(p)
    above_max = np.nextafter(scores[-1], np.inf)        # порог, при котором пустым не объявляется ничего
    candidates = np.append(scores, above_max)
    value_recall = np.searchsorted(value_scores, candidates, side="left") / n_value
    empty_recall = (n_empty - np.searchsorted(empty_scores, candidates, side="left")) / n_empty

    feasible = value_recall >= min_value_recall - 1e-9
    best_empty = empty_recall[feasible].max()
    best = int(np.flatnonzero(feasible & (empty_recall == best_empty))[-1])

    chosen = float(candidates[best])
    if best == len(candidates) - 1:
        threshold = chosen
    else:
        lower = scores[scores < chosen]
        threshold = float((chosen + (lower.max() if len(lower) else 0.0)) / 2)

    return {
        "threshold": threshold,
        "value_recall": float((value_scores < threshold).mean()),
        "empty_recall": float((empty_scores >= threshold).mean()),
        "n_value": int(n_value),
        "n_empty": int(n_empty),
        "min_value_recall": float(min_value_recall),
    }


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
