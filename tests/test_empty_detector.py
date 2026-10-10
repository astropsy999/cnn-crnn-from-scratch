import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_shapes_and_parameter_count_equal_tiny_cnn():
    torch = pytest.importorskip("torch")
    from models.empty_detector import EmptyDetector
    from models.tiny_cnn import TinyCNN
    model = EmptyDetector().eval()
    assert tuple(model(torch.zeros(3, 1, 48, 160)).shape) == (3, 2)
    assert isinstance(model, TinyCNN)
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in TinyCNN().parameters()) == 23538


def test_dropout_is_random_in_train_and_off_in_eval():
    torch = pytest.importorskip("torch")
    from models.empty_detector import EmptyDetector
    torch.manual_seed(0)
    x = torch.rand(4, 1, 48, 160)
    model = EmptyDetector(dropout=0.5)
    model.train()
    assert not torch.equal(model(x), model(x))          # при обучении признаки случайно «выключаются»
    model.eval()
    assert torch.equal(model(x), model(x))              # при оценке результат детерминирован
