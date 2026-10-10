import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metrics import binary_report  # noqa: E402


def test_binary_report_by_hand():
    labels = [0, 0, 0, 0, 1, 1]           # 4 value, 2 empty
    pred = [0, 0, 0, 1, 1, 0]             # value: 3 из 4; empty: 1 из 2
    r = binary_report(pred, labels)
    assert r["n_value"] == 4 and r["n_empty"] == 2
    assert r["value_recall"] == 0.75
    assert r["empty_recall"] == 0.5
    assert abs(r["acc"] - 4 / 6) < 1e-12


def test_accuracy_can_hide_missed_values():
    labels = [0] * 10 + [1] * 90                   # 90 % пустых
    pred = [1] * 100                               # модель «всегда empty»
    r = binary_report(pred, labels)
    assert r["acc"] == 0.9 and r["value_recall"] == 0.0


def test_missing_class_gives_nan():
    r = binary_report([1, 1], [1, 1])
    assert math.isnan(r["value_recall"]) and r["empty_recall"] == 1.0
