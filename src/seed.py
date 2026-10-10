"""Воспроизводимость: фиксируем все источники случайности."""
import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Фиксирует seed в Python, NumPy и PyTorch (CPU и CUDA).

    deterministic=True просит PyTorch использовать только детерминированные
    алгоритмы. На GPU это может замедлить обучение или бросить ошибку для
    операций без детерминированной версии.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # без CUDA вызов безвреден
    if deterministic:
        # Переменная должна быть задана до первой операции на CUDA.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)


def make_generator(seed: int) -> torch.Generator:
    """Отдельный генератор для DataLoader(shuffle=True, generator=...)."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker(worker_id: int) -> None:
    """worker_init_fn для DataLoader: у каждого воркера свой, но воспроизводимый seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
