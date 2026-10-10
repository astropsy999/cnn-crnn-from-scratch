import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from choose_threshold import evaluate_threshold, format_report  # noqa: E402


def test_evaluate_threshold_by_hand():
    p = [0.1, 0.2, 0.6, 0.3, 0.7, 0.8]          # три value и три empty
    y = [0, 0, 0, 1, 1, 1]
    r = evaluate_threshold(p, y, 0.5)
    assert r["n_value"] == 3 and r["n_empty"] == 3
    assert r["value_recall"] == 2 / 3           # 0.6 >= 0.5: value объявлен пустым
    assert r["empty_recall"] == 2 / 3           # 0.3 < 0.5: empty пропущен
    assert abs(r["acc"] - 4 / 6) < 1e-12


def test_threshold_is_inclusive_for_empty():
    r = evaluate_threshold([0.5, 0.5], [0, 1], 0.5)       # p >= порога → empty
    assert r["value_recall"] == 0.0 and r["empty_recall"] == 1.0


def test_ci_is_returned_and_wide_for_small_n():
    r = evaluate_threshold([0.1, 0.9] * 5, [0, 1] * 5, 0.5)
    assert r["value_recall"] == 1.0
    assert r["value_ci"][0] < 0.8 < r["value_ci"][1] <= 1.0      # 5 кропов: интервал широкий


def test_format_report_contains_numbers():
    text = format_report("val", evaluate_threshold([0.1, 0.9], [0, 1], 0.5))
    assert "val" in text and "value_recall 1.0000" in text and "n=1" in text
