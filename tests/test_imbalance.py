import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from imbalance import class_weights, parse_imbalance, sample_weights  # noqa: E402

LABELS = np.array([0] * 60 + [1] * 40)            # 60 value, 40 empty


def test_class_weights_by_hand():
    w = class_weights(LABELS)
    assert w[0] == pytest.approx(100 / (2 * 60)) and w[1] == pytest.approx(100 / (2 * 40))
    assert w[1] > 1 > w[0]                        # редкий класс получает вес больше 1
    assert 60 * w[0] == pytest.approx(40 * w[1])  # суммарный вклад классов в loss одинаков


def test_sample_weights_give_both_classes_the_same_total_weight():
    w = sample_weights(LABELS)
    assert w.shape == LABELS.shape
    assert w[LABELS == 0].sum() == pytest.approx(1.0) and w[LABELS == 1].sum() == pytest.approx(1.0)
    assert w[LABELS == 1][0] > w[LABELS == 0][0]  # у кропа редкого класса вес больше


def test_sampler_weights_balance_classes_in_simulation():
    for empty_share in (0.4, 0.1, 0.9):
        labels = (np.random.default_rng(0).random(5000) < empty_share).astype(int)
        w = sample_weights(labels)
        drawn = np.random.default_rng(1).choice(len(labels), size=40000, replace=True, p=w / w.sum())
        assert abs(labels[drawn].mean() - 0.5) < 0.02, empty_share


def test_both_classes_are_required():
    with pytest.raises(ValueError):
        class_weights([0, 0, 0])
    with pytest.raises(ValueError):
        sample_weights([1, 1])


def test_parse_imbalance():
    assert parse_imbalance("none") == (False, False)
    assert parse_imbalance("weights") == (True, False)
    assert parse_imbalance("sampler") == (False, True)
    assert parse_imbalance("both") == (True, True)
    with pytest.raises(ValueError):
        parse_imbalance("oversample")
