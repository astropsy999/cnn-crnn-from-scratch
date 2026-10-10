"""Обучение empty-детектора по конфигу: баланс классов, ранняя остановка, порог на val."""
import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from augment import augment_default, augment_printed
from choose_threshold import choose_threshold, evaluate_threshold, format_report
from config import load_config
from datasets.crop_dataset import CropDataset, collate_classify
from device import pick_device
from early_stopping import EarlyStopping
from imbalance import class_weights, parse_imbalance, sample_weights
from inference import predict_p_empty, write_threshold
from models.empty_detector import EmptyDetector
from seed import make_generator, seed_worker, set_seed
from selection import binary_log_loss, epoch_score

AUGMENTS = {"none": None, "default": augment_default, "printed": augment_printed}


def make_train_loader(ds, labels: np.ndarray, cfg: dict) -> DataLoader:
    """Загрузчик train. Режим sampler/both: кропы выбираются с вероятностью 1/n_класса, батчи сбалансированы."""
    _, use_sampler = parse_imbalance(cfg["imbalance"])
    common = dict(batch_size=cfg["batch_size"], collate_fn=collate_classify,
                  worker_init_fn=seed_worker, num_workers=cfg["workers"])
    if use_sampler:
        sampler = WeightedRandomSampler(torch.from_numpy(sample_weights(labels)), num_samples=len(ds),
                                        replacement=True, generator=make_generator(cfg["seed"]))
        return DataLoader(ds, sampler=sampler, **common)
    return DataLoader(ds, shuffle=True, generator=make_generator(cfg["seed"]), **common)


def train_epoch(model, loader, device, optimizer, loss_weight, max_batches) -> float:
    model.train()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = batch["images"].to(device)
        y = batch["labels"].to(device)
        loss = F.cross_entropy(model(x), y, weight=loss_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / n


def train(cfg: dict, verbose: bool = True) -> dict:
    """Полный цикл по конфигу. Возвращает метрики; веса и порог пишет в cfg['out']."""
    say = print if verbose else (lambda *a, **k: None)
    set_seed(cfg["seed"])
    device = pick_device()
    train_ds = CropDataset(cfg["data_dir"], "train", augment=AUGMENTS[cfg["augment"]], seed=cfg["seed"])
    val_ds = CropDataset(cfg["data_dir"], "val")
    test_ds = CropDataset(cfg["data_dir"], "test")
    labels = np.array([int(r["is_empty"]) for r in train_ds.records])
    use_weights, _ = parse_imbalance(cfg["imbalance"])
    loss_weight = (torch.tensor(class_weights(labels), dtype=torch.float32, device=device)
                   if use_weights else None)
    say(f"Device: {device}; imbalance={cfg['imbalance']}; train: {len(labels)} кропов, "
        f"доля empty {labels.mean():.3f}")

    model = EmptyDetector(tuple(cfg["channels"]), cfg["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loader = make_train_loader(train_ds, labels, cfg)
    stopper = EarlyStopping(cfg["patience"])

    best_state, best = None, None
    epoch = 0
    for epoch in range(1, cfg["epochs"] + 1):
        train_ds.set_epoch(epoch)
        train_loss = train_epoch(model, loader, device, optimizer, loss_weight, cfg["max_batches"])
        p, y = predict_p_empty(model, val_ds, device)
        score = epoch_score(p, y, cfg["min_value_recall"])
        val_loss = binary_log_loss(p, y)
        improved = stopper.step(score["empty_recall"], val_loss)
        if improved:
            best = {"epoch": epoch, **score, "val_loss": val_loss}
            best_state = copy.deepcopy(model.state_dict())
        say(f"эпоха {epoch:>2}: train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
            f"empty_recall при value_recall >= {cfg['min_value_recall']}: {score['empty_recall']:.4f}"
            f"{'  <- лучшая' if improved else ''}")
        if stopper.should_stop:
            say(f"ранняя остановка: {cfg['patience']} эпох без улучшения")
            break

    model.load_state_dict(best_state)
    val_p, val_y = predict_p_empty(model, val_ds, device)
    test_p, test_y = predict_p_empty(model, test_ds, device)
    chosen = choose_threshold(val_p, val_y, cfg["min_value_recall"])
    val_rep = evaluate_threshold(val_p, val_y, chosen["threshold"])
    test_rep = evaluate_threshold(test_p, test_y, chosen["threshold"])

    out = Path(cfg["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": cfg, "classes": ["value", "empty"],
                "seed": cfg["seed"], "epoch": best["epoch"]}, out)
    write_threshold(out, chosen, val_rep, test_rep)
    say(f"\nчекпоинт: {out} (лучшая эпоха {best['epoch']}, остановились на {epoch})")
    say(f"порог выбран на val: {chosen['threshold']:.4f}")
    say(format_report("val", val_rep))
    say(format_report("test", test_rep))
    half = evaluate_threshold(val_p, val_y, 0.5)                    # решение «по умолчанию», без подбора порога
    return {"threshold": chosen["threshold"], "best_epoch": best["epoch"], "epochs_run": epoch,
            "val_empty_recall": val_rep["empty_recall"], "val_value_recall": val_rep["value_recall"],
            "val_value_recall_at_half": half["value_recall"],
            "test_value_recall": test_rep["value_recall"], "test_empty_recall": test_rep["empty_recall"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/empty-detector.yaml")
    ap.add_argument("--data-dir")
    ap.add_argument("--out")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--imbalance", choices=["none", "weights", "sampler", "both"])
    ap.add_argument("--max-batches", type=int)
    args = ap.parse_args()
    overrides = {"data_dir": args.data_dir, "out": args.out, "seed": args.seed, "epochs": args.epochs,
                 "imbalance": args.imbalance, "max_batches": args.max_batches}
    train(load_config(args.config, overrides))


if __name__ == "__main__":
    main()
