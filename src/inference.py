"""Вывод модели: вероятности P(empty) и работа с порогом в чекпоинте (нужен torch)."""
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.crop_dataset import EMPTY_CLASS, CropDataset, collate_classify
from device import pick_device
from models.tiny_cnn import TinyCNN


def predict_p_empty(model, ds, device, batch_size: int = 64, workers: int = 0):
    """P(empty) и метки для всего датасета (без перемешивания и без аугментации)."""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_classify,
                        num_workers=workers)
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["images"].to(device))
            probs.append(torch.softmax(logits, dim=1)[:, EMPTY_CLASS].cpu().numpy())
            labels.append(batch["labels"].numpy())
    return np.concatenate(probs).astype(np.float64), np.concatenate(labels)


def load_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location="cpu")
    model = TinyCNN()
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), ckpt


def scores_from_checkpoint(checkpoint, data_dir) -> dict:
    """Оценки модели на val и test в формате, который понимают choose_threshold и plot_threshold_curve."""
    device = pick_device()
    model, _ = load_model(checkpoint, device)
    out = {}
    for split in ("val", "test"):
        p, y = predict_p_empty(model, CropDataset(data_dir, split), device)
        out[f"{split}_p_empty"], out[f"{split}_labels"] = p, y
    return out


def write_threshold(checkpoint, chosen: dict, val: dict, test: dict) -> None:
    """Дописывает выбранный порог в чекпоинт (только обычные числа и списки)."""
    ckpt = torch.load(checkpoint, map_location="cpu")

    def plain(report: dict) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in report.items()}

    ckpt["threshold"] = {"value": chosen["threshold"], "min_value_recall": chosen["min_value_recall"],
                         "chosen_on": "val", "val": plain(val), "test": plain(test)}
    torch.save(ckpt, checkpoint)
