import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from greedy_decode import greedy_decode, greedy_decode_batch, softmax  # noqa: E402

C = 12                                                   # 11 символов measurement + blank


def probs_for(path, maxima):
    """Матрица вероятностей: на каждом шаге заданный класс получает заданную вероятность, остальное поровну."""
    p = np.zeros((len(path), C))
    for t, (c, m) in enumerate(zip(path, maxima)):
        p[t, :] = (1 - m) / (C - 1)
        p[t, c] = m
    return p


def test_softmax_rows_sum_to_one_and_ignore_shift():
    x = np.random.default_rng(0).normal(size=(5, C)) * 30
    assert np.allclose(softmax(x).sum(axis=1), 1.0)
    assert np.allclose(softmax(x), softmax(x + 123.0))


def test_decodes_the_lesson_path_into_77_3():
    # индексы measurement: '7' = 8, ',' = 11, '3' = 4;  путь «7-77-,-33»
    path = [8, 0, 8, 8, 0, 11, 0, 4, 4]
    r = greedy_decode(probs_for(path, [1.0] * len(path)), "measurement")
    assert r.text == "77,3" and r.indices == [8, 8, 11, 4] and r.confidence == pytest.approx(1.0)


def test_confidence_is_mean_of_the_best_probability_inside_each_run():
    path = [0, 8, 8, 0, 11, 0, 4]
    maxima = [0.9, 0.8, 0.6, 0.9, 0.7, 0.9, 0.5]
    r = greedy_decode(probs_for(path, maxima), "measurement")
    assert r.text == "7,3"
    assert r.confidence == pytest.approx((0.8 + 0.7 + 0.5) / 3)     # серия «77» даёт max(0.8, 0.6) = 0.8


def test_same_symbol_after_blank_is_a_new_token_with_its_own_confidence():
    r = greedy_decode(probs_for([8, 0, 8], [0.9, 0.9, 0.5]), "measurement")
    assert r.text == "77"
    assert r.confidence == pytest.approx((0.9 + 0.5) / 2)


def test_empty_output_confidence_is_mean_blank_probability():
    r = greedy_decode(probs_for([0, 0, 0], [0.9, 0.7, 0.8]), "measurement")
    assert r.text == "" and r.indices == [] and r.confidence == pytest.approx(0.8)


def test_confidence_is_lower_for_a_less_sure_prediction():
    path = [0, 8, 0, 4]
    sure = greedy_decode(probs_for(path, [0.95] * 4), "measurement").confidence
    unsure = greedy_decode(probs_for(path, [0.55] * 4), "measurement").confidence
    assert unsure < sure


def test_batch_decoding():
    a = probs_for([8, 0, 4], [0.9] * 3)
    b = probs_for([0, 0, 0], [0.9] * 3)
    out = greedy_decode_batch(np.stack([a, b]), "measurement")
    assert [d.text for d in out] == ["73", ""]
