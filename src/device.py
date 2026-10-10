"""Выбор устройства для вычислений: cuda → mps → cpu."""
import torch


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")  # NVIDIA + работающий драйвер
    mps = getattr(torch.backends, "mps", None)  # Apple Silicon, если модуль есть
    if mps is not None and mps.is_available():
        return torch.device("mps")  # ускорение Metal на Mac
    return torch.device("cpu")  # запасной путь: весь курс рассчитан на CPU
