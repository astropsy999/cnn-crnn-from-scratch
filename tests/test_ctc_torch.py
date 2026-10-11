import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ctc_from_scratch import ctc_neg_log_likelihood, log_softmax  # noqa: E402


def torch_loss(torch, log_probs, target, input_length=None, zero_infinity=False):
    F = torch.nn.functional
    lp = torch.from_numpy(log_probs).float()[:, None, :]                      # [T, B=1, C]
    return F.ctc_loss(lp, torch.tensor([target]), torch.tensor([input_length or log_probs.shape[0]]),
                      torch.tensor([len(target)]), blank=0, reduction="none", zero_infinity=zero_infinity)[0].item()


def test_alpha_matches_nn_ctc_loss():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    for target in ([1, 2, 3], [1, 1, 2], [3, 3, 3], [2]):
        lp = log_softmax(rng.normal(size=(10, 5)))
        assert abs(ctc_neg_log_likelihood(lp, target) - torch_loss(torch, lp, target)) < 1e-4, target


def test_impossible_target_gives_inf_and_zero_infinity_hides_it():
    torch = pytest.importorskip("torch")
    lp = log_softmax(np.random.default_rng(1).normal(size=(2, 5)))
    assert math.isinf(torch_loss(torch, lp, [1, 2, 3]))                       # шагов меньше, чем нужно
    assert torch_loss(torch, lp, [1, 2, 3], zero_infinity=True) == 0.0        # молча выброшено из обучения


def test_forgetting_log_softmax_changes_the_number_without_an_error():
    torch = pytest.importorskip("torch")
    lp = log_softmax(np.random.default_rng(2).normal(size=(8, 5)))
    right = torch_loss(torch, lp, [1, 2])
    wrong = torch_loss(torch, np.exp(lp), [1, 2])                             # подали вероятности вместо логарифмов
    assert abs(right - wrong) > 1e-3
