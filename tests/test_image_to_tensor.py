import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from image_to_tensor import CANVAS_H, CANVAS_W, preprocess_array  # noqa: E402


def text_like(w: int, h: int, bg: int = 240, ink: int = 30) -> Image.Image:
    """Светлый фон и тёмный «штрих» посередине."""
    img = Image.new("L", (w, h), bg)
    img.paste(ink, (w // 4, h // 4, w // 2, 3 * h // 4))
    return img


def test_shape_dtype_range():
    arr = preprocess_array(text_like(200, 50))
    assert arr.shape == (1, CANVAS_H, CANVAS_W)
    assert arr.dtype == np.float32
    assert arr.min() >= 0.0 and arr.max() <= 1.0


def test_ink_is_bright():
    arr = preprocess_array(text_like(200, 50))
    assert arr.max() > 0.9


def test_horizontal_padding_is_zero_for_narrow_image():
    arr = preprocess_array(text_like(40, 50))  # после масштаба ширина ≈ 38
    assert np.all(arr[0, :, 40:] == 0.0)


def test_vertical_padding_is_zero_for_wide_image():
    arr = preprocess_array(text_like(400, 30))  # после масштаба высота ≈ 12
    assert np.all(arr[0, :10, :] == 0.0)
    assert np.all(arr[0, -10:, :] == 0.0)


def test_empty_crop_noise_is_not_amplified():
    rng = np.random.default_rng(0)
    noise = rng.normal(240, 3, size=(45, 120)).clip(0, 255).astype(np.uint8)
    arr = preprocess_array(Image.fromarray(noise, mode="L"))
    assert arr.max() < 0.2


def test_transparent_background_becomes_white():
    img = Image.new("RGBA", (100, 40), (0, 0, 0, 0))  # полностью прозрачная картинка
    arr = preprocess_array(img)
    assert arr.max() < 0.01


def test_zero_size_image_raises():
    with pytest.raises(ValueError):
        preprocess_array(Image.new("L", (0, 10)))


def test_tensor_wrapper():
    torch = pytest.importorskip("torch")
    from image_to_tensor import image_to_tensor
    t = image_to_tensor(text_like(200, 50))
    assert tuple(t.shape) == (1, CANVAS_H, CANVAS_W)
    assert t.dtype == torch.float32
