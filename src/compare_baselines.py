"""Базовые модели для empty / value и таблица сравнения (в уроке 7 к ней добавится CNN)."""
import argparse
import json
from pathlib import Path

import numpy as np

from baselines import InkThreshold, MajorityClass, OneNN, bright_pixels, ink_mass, load_features, load_tensors
from metrics import binary_report
from split_groups import read_jsonl


def labels_of(records: list[dict]) -> np.ndarray:
    return np.array([int(r["is_empty"]) for r in records], dtype=np.int64)


def evaluate_baselines(data_dir, split: str = "val") -> list[dict]:
    """Обучает базовые модели на train, считает метрики на split. Возвращает строки таблицы."""
    data_dir = Path(data_dir)
    root = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))["image_root"]
    train = read_jsonl(data_dir / "train.jsonl")
    test = read_jsonl(data_dir / f"{split}.jsonl")
    y_train, y_test = labels_of(train), labels_of(test)
    X_train, X_test = load_tensors(train, root), load_tensors(test, root)

    rows = []

    def add(name: str, pred) -> None:
        rows.append({"model": name, **binary_report(pred, y_test)})

    add("всегда самый частый класс", MajorityClass().fit(X_train, y_train).predict(X_test))
    add("порог по сумме пикселей", InkThreshold(ink_mass).fit(X_train, y_train).predict(X_test))
    add("порог по числу ярких пикселей", InkThreshold(bright_pixels).fit(X_train, y_train).predict(X_test))
    f_train, f_test = load_features(train, root), load_features(test, root)
    add("1-NN по пикселям", np.array(OneNN().fit(f_train, y_train).predict(f_test)))
    return rows


def print_table(rows: list[dict], split: str = "val") -> None:
    print(f"{'модель':<34}{'accuracy':>10}{'value_recall':>14}{'empty_recall':>14}   ({split})")
    for r in rows:
        print(f"{r['model']:<34}{r['acc']:>10.3f}{r['value_recall']:>14.3f}{r['empty_recall']:>14.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    args = ap.parse_args()
    print_table(evaluate_baselines(args.data_dir, args.split), args.split)


if __name__ == "__main__":
    main()
