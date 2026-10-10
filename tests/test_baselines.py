import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baselines import OneNN, load_features  # noqa: E402


def test_one_nn_picks_nearest_point():
    X = np.array([[0.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)
    model = OneNN().fit(X, ["a", "b", "c"])
    Q = np.array([[1.0, 1.0], [9.0, 9.0], [1.0, 9.0]], dtype=np.float32)
    assert model.predict(Q) == ["a", "b", "c"]


def test_one_nn_is_exact_on_training_points():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 8)).astype(np.float32)
    labels = [str(i) for i in range(50)]
    assert OneNN().fit(X, labels).predict(X) == labels


def test_chunking_does_not_change_result():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 6)).astype(np.float32)
    Q = rng.normal(size=(30, 6)).astype(np.float32)
    model = OneNN().fit(X, range(40))
    assert (model.nearest(Q, chunk=7) == model.nearest(Q, chunk=1000)).all()


def test_load_features_shape():
    with tempfile.TemporaryDirectory() as d:
        Image.new("L", (100, 40), 230).save(Path(d) / "a.png")
        records = [{"image": "a.png"}, {"image": "a.png"}]
        assert load_features(records, d, pool=2).shape == (2, 24 * 80)
        assert load_features(records, d, pool=1).shape == (2, 48 * 160)
