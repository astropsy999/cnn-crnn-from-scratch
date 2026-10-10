"""Небольшая статистика для честной отчётности."""
import math


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Интервал Уилсона для доли successes / n (по умолчанию 95 %).

    Лучше обычного «p ± 2σ» работает у границ 0 и 1: именно там мы живём,
    когда recall близок к 0.99.
    """
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)
