"""Dataset поверх train/val/test.jsonl (по split.json) и функции сборки батча.

DataLoader не требует наследования от torch.utils.data.Dataset: достаточно __len__ и __getitem__.
Поэтому сам Dataset и сборка батча на NumPy не зависят от PyTorch; в тензоры батч превращает
тонкая обёртка to_torch().
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from alphabet import encode_text
from image_to_tensor import preprocess_array
from split_groups import read_jsonl

VALUE_CLASS = 0
EMPTY_CLASS = 1   # P(empty) в уроках 7–9 — это вероятность класса 1


class CropDataset:
    """Один сплит. augment: функция (PIL-образ, rng) -> массив [1, 48, 160] или None."""

    def __init__(self, data_dir, split: str, augment=None, seed: int = 0) -> None:
        data_dir = Path(data_dir)
        manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
        self.task = manifest["task"]
        self.image_root = Path(manifest["image_root"])
        self.records = read_jsonl(data_dir / f"{split}.jsonl")
        self.augment = augment
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        """Аугментация меняется от эпохи к эпохе, но воспроизводима."""
        self.epoch = epoch

    def __getitem__(self, i: int) -> dict:
        rec = self.records[i]
        path = self.image_root / rec["image"]
        if self.augment is None:
            image = preprocess_array(path)
        else:
            # у каждого (seed, эпоха, индекс) свой генератор: результат не зависит от воркеров DataLoader
            rng = np.random.default_rng([self.seed, self.epoch, i])
            img = Image.open(path)
            img.load()
            image = self.augment(img, rng)
        return {
            "id": rec["id"],
            "image": image,
            "text": rec["text"],
            "target": np.array(encode_text(rec["text"], self.task), dtype=np.int64),
            "label": EMPTY_CLASS if rec["is_empty"] else VALUE_CLASS,
            "is_empty": bool(rec["is_empty"]),
        }


def collate_ctc_np(batch: list[dict]) -> dict:
    """Батч для CTC: цели склеены в один вектор + длины целей (пустая клетка = длина 0)."""
    return {
        "images": np.stack([b["image"] for b in batch]).astype(np.float32),
        "targets": np.concatenate([b["target"] for b in batch]).astype(np.int64),
        "target_lengths": np.array([len(b["target"]) for b in batch], dtype=np.int64),
        "texts": [b["text"] for b in batch],
        "ids": [b["id"] for b in batch],
    }


def collate_classify_np(batch: list[dict]) -> dict:
    """Батч для бинарной классификации empty / value."""
    return {
        "images": np.stack([b["image"] for b in batch]).astype(np.float32),
        "labels": np.array([b["label"] for b in batch], dtype=np.int64),
        "ids": [b["id"] for b in batch],
    }


def to_torch(batch: dict) -> dict:
    import torch  # импорт здесь: всё остальное в файле не требует torch
    return {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in batch.items()}


def collate_ctc(batch: list[dict]) -> dict:
    return to_torch(collate_ctc_np(batch))


def collate_classify(batch: list[dict]) -> dict:
    return to_torch(collate_classify_np(batch))
