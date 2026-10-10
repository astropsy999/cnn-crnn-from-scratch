import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from choose_threshold import choose_threshold  # noqa: E402
from selection import binary_log_loss, epoch_score  # noqa: E402


def test_log_loss_by_hand():
    assert binary_log_loss([0.5, 0.5], [0, 1]) == pytest.approx(math.log(2))
    assert binary_log_loss([0.9], [1]) == pytest.approx(-math.log(0.9))
    assert binary_log_loss([0.9], [0]) == pytest.approx(-math.log(0.1))


def test_log_loss_survives_extreme_probabilities():
    assert math.isfinite(binary_log_loss([0.0, 1.0], [1, 0]))      # без обрезки было бы бесконечно
    assert binary_log_loss([1.0, 0.0], [1, 0]) < 1e-5


def test_better_ranking_gives_lower_loss():
    y = [0, 0, 1, 1]
    assert binary_log_loss([0.1, 0.2, 0.8, 0.9], y) < binary_log_loss([0.4, 0.5, 0.5, 0.6], y)


def test_epoch_score_matches_choose_threshold():
    rng = np.random.default_rng(0)
    p = np.concatenate([rng.beta(2, 5, 100), rng.beta(5, 2, 80)])
    y = np.array([0] * 100 + [1] * 80)
    s, r = epoch_score(p, y, 0.95), choose_threshold(p, y, 0.95)
    assert s == {"empty_recall": r["empty_recall"], "value_recall": r["value_recall"], "threshold": r["threshold"]}
    assert s["value_recall"] >= 0.95


def test_epoch_score_rewards_a_better_model_even_when_accuracy_is_equal():
    # Две модели ошибаются на одном и том же числе кропов при пороге 0.5, но ранжируют по-разному.
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    sharp = np.array([0.1, 0.2, 0.3, 0.6, 0.4, 0.8, 0.9, 0.95])      # value 0.6 выше empty 0.4
    clean = np.array([0.1, 0.2, 0.3, 0.45, 0.55, 0.8, 0.9, 0.95])    # классы разделены
    assert epoch_score(clean, y, 0.99)["empty_recall"] > epoch_score(sharp, y, 0.99)["empty_recall"]
