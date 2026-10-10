import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from choose_threshold import choose_threshold  # noqa: E402

# 5 value (метка 0) и 4 empty (метка 1)
P = [0.1, 0.2, 0.3, 0.4, 0.8, 0.5, 0.7, 0.9, 0.95]
Y = [0] * 5 + [1] * 4


def recalls(p, y, thr):
    p, y = np.asarray(p), np.asarray(y)
    return float((p[y == 0] < thr).mean()), float((p[y == 1] >= thr).mean())


def brute_force(p, y, target):
    """Эталон для сверки: перебор всех порогов-кандидатов в лоб."""
    p, y = np.asarray(p), np.asarray(y)
    best = -1.0
    for t in list(np.unique(p)) + [np.nextafter(p.max(), np.inf)]:
        v, e = recalls(p, y, t)
        if v >= target - 1e-9:
            best = max(best, e)
    return best


def test_hand_example():
    r = choose_threshold(P, Y, 0.8)
    assert r["threshold"] == pytest.approx(0.45)      # середина между 0.4 (value) и 0.5 (empty)
    assert r["value_recall"] == 0.8
    assert r["empty_recall"] == 1.0
    assert (r["n_value"], r["n_empty"]) == (5, 4)


def test_perfectly_separated_classes_give_threshold_in_the_gap():
    r = choose_threshold([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1], 0.99)
    assert r["threshold"] == pytest.approx(0.5)
    assert r["value_recall"] == 1.0 and r["empty_recall"] == 1.0


def test_tie_goes_to_the_largest_threshold():
    # Пороги 0.2, 0.3 и 0.5 дают одинаковый empty_recall = 1.0; безопаснее для value самый большой.
    r = choose_threshold([0.1, 0.2, 0.3, 0.5, 0.9], [0, 0, 0, 1, 1], 0.3)
    assert r["threshold"] == pytest.approx(0.4)       # середина между 0.3 и 0.5
    assert r["value_recall"] == 1.0 and r["empty_recall"] == 1.0


def test_strict_target_costs_empty_recall():
    r = choose_threshold(P, Y, 1.0)
    assert r["value_recall"] == 1.0
    assert r["threshold"] == pytest.approx(0.85)      # между 0.8 (value) и 0.9 (empty)
    assert r["empty_recall"] == 0.5


def test_inverted_model_declares_nothing_empty():
    # Модель «наоборот»: у value вероятности выше, чем у empty. Ограничение выполнимо только при пороге выше всех.
    r = choose_threshold([0.9, 0.95, 0.1, 0.2], [0, 0, 1, 1], 0.99)
    assert r["value_recall"] == 1.0 and r["empty_recall"] == 0.0
    assert r["threshold"] > 0.95


def test_returned_threshold_reproduces_reported_recalls_and_meets_target():
    rng = np.random.default_rng(0)
    for trial in range(40):
        n_value, n_empty = int(rng.integers(20, 120)), int(rng.integers(20, 120))
        p = np.concatenate([rng.beta(1.5, 4, n_value), rng.beta(4, 1.5, n_empty)])
        y = np.array([0] * n_value + [1] * n_empty)
        target = float(rng.choice([0.9, 0.95, 0.99, 1.0]))
        r = choose_threshold(p, y, target)
        v, e = recalls(p, y, r["threshold"])
        assert v == pytest.approx(r["value_recall"]) and e == pytest.approx(r["empty_recall"])
        assert v >= target - 1e-9, (trial, target, v)


def test_matches_brute_force_optimum():
    rng = np.random.default_rng(1)
    for trial in range(40):
        n_value, n_empty = int(rng.integers(10, 80)), int(rng.integers(10, 80))
        # округление создаёт повторяющиеся значения: проверяем и случай совпадающих оценок
        p = np.round(np.concatenate([rng.beta(2, 3, n_value), rng.beta(3, 2, n_empty)]), 2)
        y = np.array([0] * n_value + [1] * n_empty)
        for target in (0.8, 0.95, 0.99):
            assert choose_threshold(p, y, target)["empty_recall"] == pytest.approx(brute_force(p, y, target))


def test_higher_target_never_improves_empty_recall():
    rng = np.random.default_rng(2)
    p = np.concatenate([rng.beta(2, 5, 200), rng.beta(5, 2, 150)])
    y = np.array([0] * 200 + [1] * 150)
    recalls_by_target = [choose_threshold(p, y, t)["empty_recall"] for t in (0.8, 0.9, 0.95, 0.99, 1.0)]
    assert recalls_by_target == sorted(recalls_by_target, reverse=True)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        choose_threshold([0.1, 0.2], [0, 0], 0.99)                 # нет класса empty
    with pytest.raises(ValueError):
        choose_threshold([0.1, 0.2], [1, 1], 0.99)                 # нет класса value
    with pytest.raises(ValueError):
        choose_threshold([0.1, 1.5], [0, 1], 0.99)                 # вероятность вне [0, 1]
    with pytest.raises(ValueError):
        choose_threshold([0.1, 0.2, 0.3], [0, 1], 0.99)            # разная длина
    with pytest.raises(ValueError):
        choose_threshold([0.1, 0.9], [0, 1], 1.5)                  # цель вне (0, 1]
