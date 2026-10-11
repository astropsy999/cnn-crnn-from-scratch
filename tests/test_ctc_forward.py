import itertools
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ctc_from_scratch import (  # noqa: E402
    bruteforce_neg_log_likelihood, collapse, ctc_neg_log_likelihood, log_softmax, logsumexp, min_time_steps,
)


def random_log_probs(rng, T, C):
    return log_softmax(rng.normal(size=(T, C)))


def test_logsumexp():
    assert logsumexp([0.0, 0.0]) == math.log(2)
    assert math.isinf(logsumexp([-math.inf, -math.inf]))
    assert logsumexp([-math.inf, -2.0]) == -2.0
    assert math.isfinite(logsumexp([1000.0, 1000.0]))              # без переполнения


def test_log_softmax_rows_are_normalised():
    lp = log_softmax(np.random.default_rng(0).normal(size=(4, 7)) * 50)
    assert np.allclose(np.exp(lp).sum(axis=1), 1.0)


def test_alpha_recursion_matches_enumeration_of_all_paths():
    rng = np.random.default_rng(0)
    targets = [[], [1], [1, 2], [1, 1], [1, 2, 1], [2, 2, 2], [1, 2, 3]]
    checked = 0
    for T in (3, 4, 5, 6):
        C = 3 if T == 6 else 4
        for target in targets:
            lp = random_log_probs(rng, T, C)
            fast = ctc_neg_log_likelihood(lp, [x for x in target if x < C])
            slow = bruteforce_neg_log_likelihood(lp, [x for x in target if x < C])
            if math.isinf(slow):
                assert math.isinf(fast)
            else:
                assert abs(fast - slow) < 1e-9, (T, target, fast, slow)
            checked += 1
    assert checked == 28


def test_probabilities_of_all_possible_targets_sum_to_one():
    # Каждый путь схлопывается ровно в одну строку, поэтому сумма вероятностей всех строк равна 1.
    rng = np.random.default_rng(1)
    T, C = 4, 3
    lp = random_log_probs(rng, T, C)
    targets = {tuple(collapse(list(p), blank=0)) for p in itertools.product(range(C), repeat=T)}
    total = sum(math.exp(-ctc_neg_log_likelihood(lp, list(t))) for t in targets)
    assert abs(total - 1.0) < 1e-9


def test_too_few_steps_is_impossible_and_exact_minimum_is_possible():
    rng = np.random.default_rng(2)
    for text, target in (("7,3", [1, 2, 3]), ("33", [1, 1]), ("7,33", [1, 2, 3, 3])):
        need = min_time_steps(text)
        assert math.isinf(ctc_neg_log_likelihood(random_log_probs(rng, need - 1, 5), target))
        assert math.isfinite(ctc_neg_log_likelihood(random_log_probs(rng, need, 5), target))


def test_repeated_symbol_needs_a_blank_between():
    rng = np.random.default_rng(3)
    assert math.isinf(ctc_neg_log_likelihood(random_log_probs(rng, 2, 4), [1, 1]))
    assert math.isfinite(ctc_neg_log_likelihood(random_log_probs(rng, 3, 4), [1, 1]))


def test_empty_target_means_all_blank_path():
    lp = random_log_probs(np.random.default_rng(4), 5, 4)
    assert abs(ctc_neg_log_likelihood(lp, []) - (-lp[:, 0].sum())) < 1e-12


def test_prediction_matching_the_target_beats_blank_only():
    C, T = 6, 8
    def probs(path):
        p = np.full((T, C), 0.1 / (C - 1))
        for t, c in enumerate(path):
            p[t, c] = 0.9
        return np.log(p)
    target = [2, 4, 3]
    good = ctc_neg_log_likelihood(probs([0, 2, 2, 0, 4, 0, 3, 3]), target)
    blank = ctc_neg_log_likelihood(probs([0] * T), target)
    assert good < 1.0 < 5.0 < blank


def test_moving_probability_toward_the_target_lowers_the_loss():
    target = [1, 2]
    losses = []
    for sharp in (0.4, 0.6, 0.8, 0.95):
        p = np.full((4, 3), (1 - sharp) / 2)
        for t, c in enumerate([1, 1, 2, 2]):
            p[t, c] = sharp
        losses.append(ctc_neg_log_likelihood(np.log(p), target))
    assert losses == sorted(losses, reverse=True)
