import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from early_stopping import EarlyStopping  # noqa: E402


def test_first_epoch_is_always_the_best():
    es = EarlyStopping(patience=3)
    assert es.step(0.5, 1.0) is True and not es.should_stop


def test_stops_after_patience_epochs_without_improvement():
    es = EarlyStopping(patience=2)
    flags = [es.step(s, loss) for s, loss in [(0.50, 1.0), (0.60, 0.9), (0.60, 0.95), (0.59, 0.8)]]
    assert flags == [True, True, False, False]       # 3-я: ничья по метрике, loss хуже; 4-я: метрика хуже
    assert es.should_stop and es.bad_epochs == 2


def test_improvement_resets_the_counter():
    es = EarlyStopping(patience=2)
    es.step(0.5, 1.0)
    es.step(0.4, 1.0)
    assert es.bad_epochs == 1 and not es.should_stop
    assert es.step(0.7, 1.0) is True
    assert es.bad_epochs == 0


def test_tie_on_metric_is_broken_by_lower_loss():
    es = EarlyStopping(patience=5)
    es.step(0.9, 0.50)
    assert es.step(0.9, 0.40) is True                # метрика та же, loss меньше: это улучшение
    assert es.step(0.9, 0.45) is False               # loss снова больше
    assert es.best_loss == 0.40


def test_patience_must_be_positive():
    with pytest.raises(ValueError):
        EarlyStopping(patience=0)


def test_stops_exactly_at_patience():
    es = EarlyStopping(patience=3)
    es.step(0.8, 1.0)
    stops = []
    for _ in range(4):
        es.step(0.7, 1.0)
        stops.append(es.should_stop)
    assert stops == [False, False, True, True]
