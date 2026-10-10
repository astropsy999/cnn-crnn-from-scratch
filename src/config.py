"""Конфиги обучения (yaml) с проверкой: опечатка в ключе не должна молча игнорироваться."""
import copy

import yaml

IMBALANCE_MODES = ("none", "weights", "sampler", "both")
AUGMENT_MODES = ("none", "default", "printed")

DEFAULTS = {
    "data_dir": "data/processed",
    "out": "checkpoints/empty-detector.pt",
    "seed": 42,
    "epochs": 25,
    "batch_size": 32,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "patience": 8,
    "min_value_recall": 0.99,
    "imbalance": "none",
    "augment": "default",
    "channels": [16, 32, 64],
    "dropout": 0.2,
    "workers": 0,
    "max_batches": None,
}


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate_config(cfg: dict) -> dict:
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"неизвестные ключи конфига: {sorted(unknown)}; допустимы {sorted(DEFAULTS)}")
    missing = set(DEFAULTS) - set(cfg)
    if missing:
        raise ValueError(f"в конфиге нет ключей: {sorted(missing)}")
    for key in ("seed", "epochs", "batch_size", "patience", "workers"):
        if not _is_int(cfg[key]) or cfg[key] < (0 if key in ("seed", "workers") else 1):
            raise ValueError(f"{key}: ожидалось целое число, получено {cfg[key]!r}")
    for key in ("lr", "weight_decay"):
        if not _is_number(cfg[key]) or cfg[key] < 0 or (key == "lr" and cfg[key] == 0):
            raise ValueError(f"{key}: ожидалось положительное число, получено {cfg[key]!r}")
    if not _is_number(cfg["min_value_recall"]) or not 0 < cfg["min_value_recall"] <= 1:
        raise ValueError("min_value_recall должен быть числом из (0, 1]")
    if not _is_number(cfg["dropout"]) or not 0 <= cfg["dropout"] < 1:
        raise ValueError("dropout должен быть числом из [0, 1)")
    if cfg["imbalance"] not in IMBALANCE_MODES:
        raise ValueError(f"imbalance: допустимо {IMBALANCE_MODES}, получено {cfg['imbalance']!r}")
    if cfg["augment"] not in AUGMENT_MODES:
        raise ValueError(f"augment: допустимо {AUGMENT_MODES}, получено {cfg['augment']!r}")
    ch = cfg["channels"]
    if not isinstance(ch, (list, tuple)) or not ch or not all(_is_int(c) and c > 0 for c in ch):
        raise ValueError("channels должен быть непустым списком положительных целых")
    if cfg["max_batches"] is not None and (not _is_int(cfg["max_batches"]) or cfg["max_batches"] < 1):
        raise ValueError("max_batches: None или целое число >= 1")
    return cfg


def load_config(path=None, overrides: dict | None = None) -> dict:
    """Значения по умолчанию ← файл yaml ← overrides (значения None в overrides игнорируются)."""
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        unknown = set(loaded) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"{path}: неизвестные ключи {sorted(unknown)}; допустимы {sorted(DEFAULTS)}")
        cfg.update(loaded)
    for key, value in (overrides or {}).items():
        if key not in DEFAULTS:
            raise ValueError(f"неизвестный ключ переопределения {key!r}")
        if value is not None:
            cfg[key] = value
    return validate_config(cfg)
