"""Демо утечки: одна и та же модель (1-NN по пикселям), два способа сделать val.

LEAKY — val выбран случайно среди кропов (листы перемешаны между train и val).
SHEET — val из split.json: листы val модель не видела.
Размеры train и val в обоих случаях одинаковы, меняется только способ деления.
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baselines import OneNN, load_features  # noqa: E402
from split_groups import read_jsonl  # noqa: E402


def evaluate(train: list[dict], val: list[dict], image_root, pool: int) -> dict:
    X_train = load_features(train, image_root, pool)
    X_val = load_features(val, image_root, pool)
    model = OneNN().fit(X_train, [r["text"] for r in train])
    idx = model.nearest(X_val)
    pred = [model.y[i] for i in idx]
    exact = float(np.mean([p == r["text"] for p, r in zip(pred, val)]))
    same_sheet = float(np.mean([train[i]["group_id"] == r["group_id"] for i, r in zip(idx, val)]))
    seen = {r["text"] for r in train}
    ceiling = float(np.mean([r["text"] in seen for r in val]))
    return {"exact": exact, "same_sheet": same_sheet, "ceiling": ceiling}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool", type=int, default=2, help="усреднение пикселей блоками (ускоряет)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    image_root = manifest["image_root"]
    train = [r for r in read_jsonl(data_dir / "train.jsonl") if not r["is_empty"]]
    val = [r for r in read_jsonl(data_dir / "val.jsonl") if not r["is_empty"]]

    # честный вариант: val из split.json
    honest = evaluate(train, val, image_root, args.pool)

    # утечка: тот же пул кропов, но val выбран случайно и того же размера
    pool_records = train + val
    order = list(range(len(pool_records)))
    random.Random(args.seed).shuffle(order)
    leaky_val = [pool_records[i] for i in order[:len(val)]]
    leaky_train = [pool_records[i] for i in order[len(val):]]
    leaky = evaluate(leaky_train, leaky_val, image_root, args.pool)

    top = Counter(r["text"] for r in train).most_common(1)[0]
    chance = float(np.mean([r["text"] == top[0] for r in val]))
    print(f"train {len(train)} кропов, val {len(val)} кропов (только value)")
    print(f"наивный ответ «всегда {top[0]!r}»: exact {chance:.3f}\n")
    print(f"{'':<8}{'exact':>8}{'сосед с того же листа':>26}{'потолок*':>11}")
    print(f"{'LEAKY':<8}{leaky['exact']:>8.3f}{leaky['same_sheet']:>26.3f}{leaky['ceiling']:>11.3f}")
    print(f"{'SHEET':<8}{honest['exact']:>8.3f}{honest['same_sheet']:>26.3f}{honest['ceiling']:>11.3f}")
    print("* потолок: доля val-кропов, значение которых вообще встречается в train")


if __name__ == "__main__":
    main()
