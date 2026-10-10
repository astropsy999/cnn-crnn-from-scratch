import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ablation import best_variant, format_table, summarize, verdict  # noqa: E402


def row(variant, seed, val_empty, half=0.97, test_value=0.99, test_empty=0.95):
    return {"variant": variant, "seed": seed, "val_empty_recall": val_empty, "val_value_recall_at_half": half,
            "test_value_recall": test_value, "test_empty_recall": test_empty}


def make(none, sampler):
    results = [row("none", i, v) for i, v in enumerate(none)] + [row("sampler", i, v) for i, v in enumerate(sampler)]
    return summarize(results)


def test_summarize_by_hand():
    s = make([0.90, 0.92], [0.94, 0.96])
    assert s["none"]["n"] == 2
    assert s["none"]["val_empty_recall"]["mean"] == pytest.approx(0.91)
    assert s["none"]["val_empty_recall"]["std"] == pytest.approx(0.0141421356, rel=1e-6)      # ddof = 1
    assert list(s) == ["none", "sampler"]                                                      # порядок «от простого к сложному»


def test_single_seed_has_nan_std():
    s = summarize([row("none", 0, 0.9)])
    assert math.isnan(s["none"]["val_empty_recall"]["std"])


def test_best_variant_prefers_the_simpler_one_on_a_tie():
    results = [row("both", 0, 0.9), row("weights", 0, 0.9), row("none", 0, 0.9), row("sampler", 0, 0.9)]
    assert best_variant(summarize(results)) == "none"


def test_best_variant_is_chosen_by_val_not_by_test():
    results = [row("none", 0, 0.90, test_empty=0.99), row("sampler", 0, 0.95, test_empty=0.10)]
    assert best_variant(summarize(results)) == "sampler"


def test_verdict_when_none_wins():
    assert "не нужна" in verdict(make([0.95, 0.95], [0.90, 0.91]))


def test_verdict_when_gain_is_within_noise():
    text = verdict(make([0.90, 0.94], [0.91, 0.95]))           # выигрыш 1 п.п., разброс ≈ 2.8 п.п.
    assert "эффекта не видно" in text and "sampler" in text


def test_verdict_when_gain_exceeds_noise():
    text = verdict(make([0.900, 0.901], [0.950, 0.951]))        # выигрыш 5 п.п., разброс ≈ 0.07 п.п.
    assert "эффект есть" in text and "sampler" in text


def test_verdict_with_single_seed_does_not_crash():
    s = summarize([row("none", 0, 0.9), row("sampler", 0, 0.95)])
    assert "sampler" in verdict(s)


def test_format_table_lists_all_variants():
    table = format_table(make([0.9, 0.92], [0.94, 0.96]))
    assert "none" in table and "sampler" in table and "±" in table and "0.9100" in table
