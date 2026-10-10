import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from conv_from_scratch import (  # noqa: E402
    TINY_CNN_LAYERS, conv2d_fast, conv2d_naive, output_size, receptive_field, tiny_cnn_summary,
)


def test_output_size():
    assert output_size(48, 3, 1, 1) == 48       # «same»-свёртка
    assert output_size(48, 2, 2, 0) == 24       # пулинг
    assert output_size(5, 3, 2, 0) == 2
    assert output_size(7, 7, 1, 0) == 1


def test_hand_computed_example():
    x = np.arange(1, 10, dtype=np.float64).reshape(1, 3, 3)       # 1 2 3 / 4 5 6 / 7 8 9
    w = np.array([[[[1.0, 0.0], [0.0, -1.0]]]])                   # x[i,j] - x[i+1,j+1]
    out = conv2d_naive(x, w, b=np.array([10.0]))
    assert out.shape == (1, 2, 2)
    assert (out == 6.0).all()                                     # 1-5=-4, плюс bias 10
    # ядро не переворачивается: это кросс-корреляция; при развороте получилось бы другое число
    flipped = conv2d_naive(x, w[:, :, ::-1, ::-1], b=np.array([10.0]))
    assert not (flipped == out).all()


def test_identity_kernel_keeps_image():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(1, 8, 9))
    w = np.zeros((1, 1, 3, 3))
    w[0, 0, 1, 1] = 1.0
    assert np.allclose(conv2d_naive(x, w, padding=1), x)
    assert np.allclose(conv2d_fast(x, w, padding=1), x)


def test_naive_equals_fast_for_many_configs():
    rng = np.random.default_rng(1)
    for c_in in (1, 3):
        for stride in (1, 2):
            for padding in (0, 1, 2):
                x = rng.normal(size=(c_in, 11, 14))
                w = rng.normal(size=(4, c_in, 3, 3))
                b = rng.normal(size=4)
                a = conv2d_naive(x, w, b, stride, padding)
                f = conv2d_fast(x, w, b, stride, padding)
                assert a.shape == f.shape
                assert np.allclose(a, f, atol=1e-9), (c_in, stride, padding)


def test_output_shape_follows_output_size():
    x = np.zeros((2, 12, 20))
    w = np.zeros((5, 2, 3, 3))
    out = conv2d_fast(x, w, stride=2, padding=1)
    assert out.shape == (5, output_size(12, 3, 2, 1), output_size(20, 3, 2, 1))


def test_matches_scipy_correlate():
    signal = pytest.importorskip("scipy.signal")
    rng = np.random.default_rng(2)
    x = rng.normal(size=(1, 10, 13))
    w = rng.normal(size=(1, 1, 3, 4))
    ref = signal.correlate2d(x[0], w[0, 0], mode="valid")
    assert np.allclose(conv2d_fast(x, w)[0], ref)
    assert np.allclose(conv2d_naive(x, w)[0], ref)


def test_matches_torch_conv2d():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F
    rng = np.random.default_rng(3)
    x = rng.normal(size=(3, 12, 20)).astype(np.float32)
    w = rng.normal(size=(4, 3, 3, 3)).astype(np.float32)
    b = rng.normal(size=4).astype(np.float32)
    for stride, padding in ((1, 1), (2, 0), (1, 0)):
        ref = F.conv2d(torch.from_numpy(x)[None], torch.from_numpy(w), torch.from_numpy(b),
                       stride=stride, padding=padding)[0].numpy()
        mine = conv2d_naive(x, w, b, stride, padding)
        assert np.abs(mine - ref).max() < 1e-4


def test_receptive_field_of_single_layers():
    assert receptive_field([("c", 3, 1)]) == [("c", 3, 1)]
    assert receptive_field([("c", 3, 1), ("c", 3, 1)])[-1][1] == 5      # две 3×3 дают поле 5


def test_receptive_field_of_tiny_cnn():
    assert receptive_field(TINY_CNN_LAYERS) == [
        ("conv1", 3, 1), ("pool1", 4, 2), ("conv2", 8, 2),
        ("pool2", 10, 4), ("conv3", 18, 4), ("pool3", 22, 8)]


def test_tiny_cnn_summary():
    shapes, params = tiny_cnn_summary()
    assert params == 23538
    assert shapes[0][1] == (1, 48, 160)
    assert shapes[-3][1] == (64, 6, 20)       # после третьего пулинга
    assert shapes[-1][1] == (2,)
