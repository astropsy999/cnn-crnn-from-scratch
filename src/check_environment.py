"""Проверка окружения: версии, устройство, один тензорный расчёт."""
import platform
from importlib.metadata import version

import numpy as np
import torch

from device import pick_device


def main() -> None:
    device = pick_device()  # cuda → mps → cpu
    print(f"Python      : {platform.python_version()}")
    print(f"Platform    : {platform.platform()}")
    print(f"torch       : {torch.__version__}")
    print(f"numpy       : {np.__version__}")
    print(f"Pillow      : {version('Pillow')}")
    print(f"Device      : {device}")
    print(f"CPU threads : {torch.get_num_threads()}")

    x = torch.randn(2, 3, device=device)  # маленький тензор на выбранном устройстве
    y = x @ x.T  # матричное умножение: дымовой тест «устройство считает»
    assert y.shape == (2, 2), "неожиданная форма результата"
    print(f"Tensor OK   : shape={tuple(x.shape)}, device={x.device}")


if __name__ == "__main__":
    main()
