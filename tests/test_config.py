import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import DEFAULTS, load_config  # noqa: E402


def write_yaml(text: str) -> str:
    d = tempfile.mkdtemp()
    path = Path(d) / "c.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_shipped_yaml_matches_defaults():
    assert load_config(str(ROOT / "configs" / "empty-detector.yaml")) == DEFAULTS


def test_defaults_are_valid_and_result_is_a_copy():
    cfg = load_config()
    cfg["channels"].append(99)
    assert DEFAULTS["channels"] == [16, 32, 64]


def test_yaml_values_override_defaults_and_overrides_override_yaml():
    path = write_yaml("epochs: 5\nimbalance: both\n")
    cfg = load_config(path, {"epochs": 7, "seed": None})
    assert cfg["epochs"] == 7 and cfg["imbalance"] == "both" and cfg["seed"] == DEFAULTS["seed"]


def test_unknown_keys_are_rejected():
    with pytest.raises(ValueError):
        load_config(write_yaml("epochz: 5\n"))                  # опечатка в ключе
    with pytest.raises(ValueError):
        load_config(None, {"learning_rate": 0.1})


def test_invalid_values_are_rejected():
    bad = [
        {"epochs": 0}, {"epochs": 2.5}, {"batch_size": -1}, {"patience": 0}, {"seed": True},
        {"lr": 0}, {"lr": "fast"}, {"weight_decay": -1}, {"dropout": 1.0}, {"dropout": -0.1},
        {"min_value_recall": 0}, {"min_value_recall": 1.5}, {"imbalance": "balanced"},
        {"augment": "strong"}, {"channels": []}, {"channels": [16, 0]}, {"channels": "16,32"},
        {"max_batches": 0}, {"max_batches": 1.5},
    ]
    for overrides in bad:
        with pytest.raises(ValueError):
            load_config(None, overrides)


def test_max_batches_may_be_none():
    assert load_config(None, {"max_batches": 3})["max_batches"] == 3
    assert load_config()["max_batches"] is None
