"""Обучение tiny CNN (empty / value), выбор порога и сравнение с базовыми моделями."""
import argparse
import copy
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from augment import augment_default
from choose_threshold import choose_threshold, evaluate_threshold, format_report
from compare_baselines import evaluate_baselines, print_table
from datasets.crop_dataset import CropDataset, collate_classify
from device import pick_device
from inference import predict_p_empty, write_threshold
from metrics import binary_report
from models.tiny_cnn import TinyCNN
from seed import make_generator, seed_worker, set_seed


def make_loader(ds, batch_size: int, shuffle: bool, seed: int, workers: int) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_classify,
                      generator=make_generator(seed) if shuffle else None,
                      worker_init_fn=seed_worker, num_workers=workers)


def run_epoch(model, loader, device, optimizer=None, max_batches=None) -> dict:
    """Один проход по данным. С optimizer — обучение, без него — оценка."""
    training = optimizer is not None
    model.train(training)
    preds, labels, loss_sum, n = [], [], 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = batch["images"].to(device)
        y = batch["labels"].to(device)
        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_sum += loss.item() * len(y)
        n += len(y)
        preds.append(logits.argmax(dim=1).cpu().numpy())
        labels.append(y.cpu().numpy())
    report = binary_report(np.concatenate(preds), np.concatenate(labels))
    report["loss"] = loss_sum / n
    return report


def check_initial_loss(model, batch, device) -> float:
    """У необученной модели на двух классах loss должен быть около ln 2 ≈ 0.693."""
    model.eval()
    with torch.no_grad():
        loss = F.cross_entropy(model(batch["images"].to(device)), batch["labels"].to(device)).item()
    flag = "OK" if abs(loss - math.log(2)) < 0.2 else "ПОДОЗРИТЕЛЬНО: ждали около 0.69"
    print(f"стартовый loss: {loss:.3f} (ln 2 = {math.log(2):.3f})  {flag}")
    return loss


def overfit_one_batch(model, batch, device, steps: int = 200, lr: float = 1e-3) -> bool:
    """Модель обязана переобучиться на ОДНОМ батче: иначе в коде или данных ошибка."""
    x = batch["images"].to(device)
    y = batch["labels"].to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for step in range(steps + 1):
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        if step % 25 == 0:
            print(f"  шаг {step:>3}: loss {loss.item():.4f}")
        if step == steps:
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    acc = (logits.argmax(dim=1) == y).float().mean().item()
    ok = loss.item() < 0.05 and acc == 1.0
    print(f"overfit одного батча: loss {loss.item():.4f}, accuracy {acc:.3f}  {'OK' if ok else 'НЕ ПОЛУЧИЛОСЬ'}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--augment", action="store_true", help="аугментация default на train")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--min-value-recall", type=float, default=0.99,
                    help="ограничение на value_recall при выборе порога (на val)")
    ap.add_argument("--max-batches", type=int, default=None, help="ограничить число батчей за эпоху (smoke)")
    ap.add_argument("--overfit-batch", action="store_true", help="только проверка: переобучить один батч")
    ap.add_argument("--out", default="checkpoints/tiny_cnn.pt")
    args = ap.parse_args()

    set_seed(args.seed)
    device = pick_device()
    print(f"Device: {device}")

    train_ds = CropDataset(args.data_dir, "train", augment=augment_default if args.augment else None,
                           seed=args.seed)
    val_ds = CropDataset(args.data_dir, "val")
    model = TinyCNN().to(device)
    print(f"параметров: {sum(p.numel() for p in model.parameters())}")

    first_batch = next(iter(make_loader(train_ds, args.batch_size, False, args.seed, 0)))
    check_initial_loss(model, first_batch, device)
    if args.overfit_batch:
        sys.exit(0 if overfit_one_batch(model, first_batch, device) else 1)

    train_loader = make_loader(train_ds, args.batch_size, True, args.seed, args.workers)
    val_loader = make_loader(val_ds, 64, False, args.seed, args.workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best, best_state = None, None
    for epoch in range(1, args.epochs + 1):
        train_ds.set_epoch(epoch)
        tr = run_epoch(model, train_loader, device, optimizer, args.max_batches)
        va = run_epoch(model, val_loader, device, None, args.max_batches)
        mark = ""
        if best is None or va["acc"] > best["acc"]:
            best = {**va, "epoch": epoch}
            best_state = copy.deepcopy(model.state_dict())
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": model.state_dict(), "classes": ["value", "empty"],
                        "config": {k: v for k, v in vars(args).items() if k != "out"},
                        "epoch": epoch, "val": va, "seed": args.seed}, out)
            mark = "  <- лучшая"
        print(f"эпоха {epoch:>2}: train loss {tr['loss']:.4f} acc {tr['acc']:.3f} | "
              f"val loss {va['loss']:.4f} acc {va['acc']:.3f} "
              f"(value_recall {va['value_recall']:.3f}, empty_recall {va['empty_recall']:.3f}){mark}")

    print(f"\nчекпоинт: {args.out} (эпоха {best['epoch']})\n")
    rows = evaluate_baselines(args.data_dir, "val")
    rows.append({"model": f"tiny CNN (эпоха {best['epoch']})", **{k: best[k] for k in ("acc", "value_recall", "empty_recall")}})
    print_table(rows, "val")

    # Порог: выбираем на val, итоговые числа смотрим на test.
    model.load_state_dict(best_state)
    val_p, val_y = predict_p_empty(model, val_ds, device)
    test_p, test_y = predict_p_empty(model, CropDataset(args.data_dir, "test"), device)
    chosen = choose_threshold(val_p, val_y, args.min_value_recall)
    val_rep = evaluate_threshold(val_p, val_y, chosen["threshold"])
    test_rep = evaluate_threshold(test_p, test_y, chosen["threshold"])
    write_threshold(args.out, chosen, val_rep, test_rep)
    print(f"\nпорог выбран на val: {chosen['threshold']:.4f} (ограничение: value_recall >= {args.min_value_recall})")
    print(format_report("val", val_rep))
    print(format_report("test", test_rep))
    print(f"порог записан в {args.out}")


if __name__ == "__main__":
    main()
