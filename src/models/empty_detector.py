"""Empty-детектор: tiny CNN из урока 7 + Dropout перед головой."""
import torch
from torch import nn

from models.tiny_cnn import TinyCNN


class EmptyDetector(TinyCNN):
    def __init__(self, channels=(16, 32, 64), dropout: float = 0.2, num_classes: int = 2) -> None:
        super().__init__(channels, num_classes)
        self.dropout = nn.Dropout(dropout)       # при обучении «выключает» часть признаков; параметров нет

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(self.dropout(x))
