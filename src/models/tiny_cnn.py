"""Tiny CNN для бинарной классификации empty / value.

Вход [B, 1, 48, 160]. Три блока Conv→BatchNorm→ReLU→MaxPool, затем усреднение
по всей карте признаков и линейная голова на 2 класса (0 = value, 1 = empty).
"""
import torch
from torch import nn


class TinyCNN(nn.Module):
    def __init__(self, channels=(16, 32, 64), num_classes: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c_in = 1
        for c in channels:
            layers += [
                nn.Conv2d(c_in, c, kernel_size=3, padding=1, bias=False),  # bias не нужен: за ним BatchNorm
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            c_in = c
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)      # усредняем по всей карте: ширина входа не важна
        self.head = nn.Linear(c_in, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)
