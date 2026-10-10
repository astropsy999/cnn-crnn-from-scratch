"""Свёртка «руками»: что на самом деле делает nn.Conv2d, и откуда берётся receptive field.

Всё на NumPy. Если установлен PyTorch, скрипт дополнительно сверяет результат с nn.Conv2d.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# слои tiny CNN из урока 7: (имя, размер окна, шаг)
TINY_CNN_LAYERS = [("conv1", 3, 1), ("pool1", 2, 2), ("conv2", 3, 1),
                   ("pool2", 2, 2), ("conv3", 3, 1), ("pool3", 2, 2)]


def output_size(n: int, k: int, stride: int = 1, padding: int = 0) -> int:
    """Размер выхода свёртки или пулинга по одной оси."""
    return (n + 2 * padding - k) // stride + 1


def conv2d_naive(x: np.ndarray, w: np.ndarray, b=None, stride: int = 1, padding: int = 0) -> np.ndarray:
    """Свёртка в лоб, четырьмя вложенными циклами. x: [C_in, H, W], w: [C_out, C_in, kh, kw].

    Как и nn.Conv2d, это на самом деле кросс-корреляция: ядро не переворачивается.
    """
    c_out, c_in, kh, kw = w.shape
    xp = np.pad(x, ((0, 0), (padding, padding), (padding, padding)))
    h_out = output_size(x.shape[1], kh, stride, padding)
    w_out = output_size(x.shape[2], kw, stride, padding)
    out = np.zeros((c_out, h_out, w_out), dtype=np.float64)
    for o in range(c_out):
        for i in range(h_out):
            for j in range(w_out):
                patch = xp[:, i * stride:i * stride + kh, j * stride:j * stride + kw]
                out[o, i, j] = (patch * w[o]).sum() + (0.0 if b is None else b[o])
    return out


def conv2d_fast(x: np.ndarray, w: np.ndarray, b=None, stride: int = 1, padding: int = 0) -> np.ndarray:
    """То же без циклов: окна вырезаются одним представлением, умножение одним einsum."""
    _, _, kh, kw = w.shape
    xp = np.pad(x, ((0, 0), (padding, padding), (padding, padding)))
    windows = sliding_window_view(xp, (kh, kw), axis=(1, 2))[:, ::stride, ::stride]
    out = np.einsum("chwij,ocij->ohw", windows, w)
    if b is not None:
        out = out + np.asarray(b)[:, None, None]
    return out


def receptive_field(layers) -> list[tuple[str, int, int]]:
    """Рецептивное поле после каждого слоя: сколько пикселей входа «видит» одна клетка выхода.

    r — размер поля, j — «прыжок» (расстояние между соседними клетками в пикселях входа).
    """
    r, j = 1, 1
    rows = []
    for name, k, s in layers:
        r += (k - 1) * j
        j *= s
        rows.append((name, r, j))
    return rows


def tiny_cnn_summary(h: int = 48, w: int = 160, channels=(16, 32, 64), num_classes: int = 2):
    """Формы и число параметров tiny CNN без PyTorch (для сверки с настоящей моделью)."""
    shapes = [("вход", (1, h, w))]
    params = 0
    c_in = 1
    for n, c in enumerate(channels, 1):
        params += c * c_in * 3 * 3 + 2 * c            # свёртка без bias + BatchNorm (вес и сдвиг)
        shapes.append((f"conv{n}+BN+ReLU", (c, h, w)))
        h, w = output_size(h, 2, 2), output_size(w, 2, 2)
        shapes.append((f"pool{n}", (c, h, w)))
        c_in = c
    shapes.append(("global avg pool", (c_in,)))
    params += c_in * num_classes + num_classes         # линейная голова
    shapes.append(("голова", (num_classes,)))
    return shapes, params


def main() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 12, 20)).astype(np.float32)
    w = rng.normal(size=(4, 3, 3, 3)).astype(np.float32)
    b = rng.normal(size=4).astype(np.float32)

    naive = conv2d_naive(x, w, b, stride=1, padding=1)
    fast = conv2d_fast(x, w, b, stride=1, padding=1)
    print(f"форма результата: {naive.shape}")
    print(f"naive vs fast: max |разница| = {np.abs(naive - fast).max():.2e}")

    try:
        import torch
        import torch.nn.functional as F
        ref = F.conv2d(torch.from_numpy(x)[None], torch.from_numpy(w), torch.from_numpy(b),
                       stride=1, padding=1)[0].numpy()
        print(f"naive vs torch nn.Conv2d: max |разница| = {np.abs(naive - ref).max():.2e}")
    except ImportError:
        print("torch не установлен: сравнение с nn.Conv2d пропущено")

    print("\nrecept. поле tiny CNN (слой: поле в пикселях входа, прыжок):")
    for name, r, j in receptive_field(TINY_CNN_LAYERS):
        print(f"  {name:<6} поле {r:>3} px, прыжок {j}")

    shapes, params = tiny_cnn_summary()
    print("\nформы tiny CNN:")
    for name, shape in shapes:
        print(f"  {name:<16} {shape}")
    print(f"параметров: {params}")


if __name__ == "__main__":
    main()
