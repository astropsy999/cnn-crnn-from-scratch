import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baselines import InkThreshold, MajorityClass, bright_pixels, ink_mass  # noqa: E402


def make_images(n_each: int = 20):
    """«Пустые» кропы: бумага чуть темнее (фон 0.12), ярких пикселей нет.
    «Value»-кропы: фон 0, но есть яркий штрих. Масса пикселей у пустых БОЛЬШЕ, чем у value."""
    rng = np.random.default_rng(0)
    empty = 0.12 + rng.normal(0, 0.01, size=(n_each, 1, 48, 160))
    value = rng.normal(0, 0.005, size=(n_each, 1, 48, 160)).clip(0, 1)
    value[:, :, 10:38, 20:26] = 1.0
    X = np.concatenate([empty, value]).astype(np.float32)
    y = np.array([1] * n_each + [0] * n_each)         # 1 = empty
    return X, y


def test_majority_class():
    m = MajorityClass().fit(None, [0, 0, 1])
    assert m.predict(range(4)).tolist() == [0, 0, 0, 0]
    assert MajorityClass().fit(None, [1, 1, 1, 0]).answer == 1


def test_background_brightness_fools_ink_mass_but_not_bright_pixels():
    X, y = make_images()
    assert ink_mass(X)[:20].mean() > ink_mass(X)[20:].mean()      # у пустых «масса» больше, чем у value
    by_mass = (InkThreshold(ink_mass).fit(X, y).predict(X) == y).mean()
    by_bright = (InkThreshold(bright_pixels).fit(X, y).predict(X) == y).mean()
    assert by_mass <= 0.55          # масса не умеет отличать: признак измеряет фон, а не чернила
    assert by_bright == 1.0


def test_ink_threshold_predicts_empty_below_threshold():
    X, y = make_images()
    model = InkThreshold(bright_pixels).fit(X, y)
    assert model.predict(X[:20]).tolist() == [1] * 20
    assert model.predict(X[20:]).tolist() == [0] * 20


def test_features_shapes_and_values():
    X = np.zeros((3, 1, 4, 4), dtype=np.float32)
    X[1, 0, :2, :] = 0.5
    X[2, 0, :, :] = 1.0
    assert ink_mass(X).tolist() == [0.0, 4.0, 16.0]
    assert bright_pixels(X, level=0.3).tolist() == [0, 8, 16]
    assert bright_pixels(X, level=0.7).tolist() == [0, 0, 16]
