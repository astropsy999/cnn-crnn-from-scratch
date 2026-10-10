import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stats import wilson_interval  # noqa: E402
from verify_split import min_value_crops  # noqa: E402


def test_zero_errors_lower_bound_matches_closed_form():
    # при 0 ошибок нижняя граница = n / (n + z²): та самая арифметика из урока 5
    for n in (50, 200, 381, 439):
        low, high = wilson_interval(n, n)
        assert math.isclose(low, n / (n + 1.96 ** 2), rel_tol=1e-9)
        assert math.isclose(high, 1.0)


def test_381_crops_are_enough_for_0_99():
    n = min_value_crops()
    assert wilson_interval(n, n)[0] >= 0.99 > wilson_interval(n - 1, n - 1)[0]


def test_interval_contains_the_estimate_and_stays_in_bounds():
    for s, n in ((0, 10), (3, 10), (50, 100), (99, 100), (1000, 1000)):
        low, high = wilson_interval(s, n)
        assert 0.0 <= low <= s / n <= high <= 1.0


def test_known_value():
    low, high = wilson_interval(50, 100)
    assert abs(low - 0.4038) < 1e-3 and abs(high - 0.5962) < 1e-3


def test_more_data_narrows_the_interval():
    w = lambda s, n: wilson_interval(s, n)[1] - wilson_interval(s, n)[0]
    assert w(500, 1000) < w(50, 100) < w(5, 10)


def test_empty_sample_is_nan():
    low, high = wilson_interval(0, 0)
    assert math.isnan(low) and math.isnan(high)
