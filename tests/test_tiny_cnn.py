import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_forward_shapes_and_width_independence():
    torch = pytest.importorskip("torch")
    from models.tiny_cnn import TinyCNN
    model = TinyCNN().eval()
    assert tuple(model(torch.zeros(4, 1, 48, 160)).shape) == (4, 2)
    assert tuple(model(torch.zeros(2, 1, 48, 200)).shape) == (2, 2)     # усреднение по карте: ширина не важна


def test_parameter_count_matches_numpy_summary():
    pytest.importorskip("torch")
    from conv_from_scratch import tiny_cnn_summary
    from models.tiny_cnn import TinyCNN
    assert sum(p.numel() for p in TinyCNN().parameters()) == tiny_cnn_summary()[1] == 23538


def test_can_overfit_a_tiny_batch():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F
    from models.tiny_cnn import TinyCNN
    torch.manual_seed(0)
    x = torch.rand(8, 1, 48, 160)
    y = torch.tensor([0, 1, 0, 1, 1, 0, 0, 1])
    model = TinyCNN()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = None
    for _ in range(150):
        loss = F.cross_entropy(model(x), y)
        first = loss.item() if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1 and loss.item() < first * 0.2
