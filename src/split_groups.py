"""Сплит train/val/test ПО ГРУППАМ (листам), а не по кропам.

Зачем: кропы одного листа похожи друг на друга (перо, бумага, сканер) и часто повторяют
значения. Если часть листа попала в train, а часть в val, модель «узнаёт» лист, а не учится
читать, и оценка получается завышенной (утечка).

Результат: split.json (списки групп, один на все модели курса) и train/val/test.jsonl.
"""
import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

from patterns import label_pattern

SPLITS = ("train", "val", "test")


def check_ratios(ratios: dict) -> None:
    if set(ratios) != set(SPLITS):
        raise ValueError(f"ratios должен содержать ключи {SPLITS}, получено {sorted(ratios)}")
    if any(v < 0 for v in ratios.values()):
        raise ValueError("доли не могут быть отрицательными")
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError(f"доли должны давать в сумме 1, получено {sum(ratios.values())}")


def split_groups(strata: dict[str, str], ratios: dict[str, float], seed: int = 42) -> dict[str, list[str]]:
    """Делит группы на train/val/test, сохраняя пропорции страт.

    strata: {group_id: страта}; ratios: {"train": .., "val": .., "test": ..}.
    Внутри каждой страты группы перемешиваются отдельным генератором
    random.Random(f"{seed}:{страта}"), затем val получает round(n * доля_val) групп,
    test — round(n * доля_test), остальное — train.
    Возвращает {"train": [...], "val": [...], "test": [...]}, списки отсортированы.
    """
    check_ratios(ratios)
    result: dict[str, list[str]] = {name: [] for name in SPLITS}
    for stratum in sorted(set(strata.values())):
        groups = sorted(g for g, s in strata.items() if s == stratum)
        random.Random(f"{seed}:{stratum}").shuffle(groups)
        n = len(groups)
        n_val = round(n * ratios["val"])
        n_test = round(n * ratios["test"])
        result["val"] += groups[:n_val]
        result["test"] += groups[n_val:n_val + n_test]
        result["train"] += groups[n_val + n_test:]
    return {name: sorted(ids) for name, ids in result.items()}


# --- Дальше всё готово: чтение данных, страты, запись файлов ------------------------

def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def group_strata(records: list[dict], rare_threshold: float = 0.10) -> dict[str, str]:
    """Страта группы = какие редкие паттерны в ней встречаются (редкий: < 10 % value-кропов)."""
    counts = Counter(label_pattern(r["text"], r["task"]) for r in records if not r["is_empty"])
    total = sum(counts.values())
    rare = {p for p, c in counts.items() if c / total < rare_threshold}
    found: dict[str, set] = {}
    for r in records:
        found.setdefault(r["group_id"], set())
        if not r["is_empty"]:
            pattern = label_pattern(r["text"], r["task"])
            if pattern in rare:
                found[r["group_id"]].add(pattern)
    return {g: "+".join(sorted(s)) or "none" for g, s in found.items()}


def split_hash(split: dict) -> str:
    """Отпечаток разбиения: по нему позже проверим, что две модели учились на одном сплите."""
    payload = json.dumps({name: split[name] for name in SPLITS}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/processed", help="папка с all.jsonl")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--test-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    records = read_jsonl(Path(args.src) / "all.jsonl")
    ratios = {"train": round(1 - args.val_ratio - args.test_ratio, 6),
              "val": args.val_ratio, "test": args.test_ratio}
    strata = group_strata(records)
    split = split_groups(strata, ratios, args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"version": 1, "seed": args.seed, "ratios": ratios,
            "strata": dict(sorted(Counter(strata.values()).items())),
            "hash": split_hash(split), **split}
    (out / "split.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")

    owner = {g: name for name in SPLITS for g in split[name]}
    for name in SPLITS:
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in records:
                if owner[r["group_id"]] == name:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"split.json: hash={meta['hash']}, seed={args.seed}")
    for name in SPLITS:
        crops = [r for r in records if owner[r["group_id"]] == name]
        value = sum(not r["is_empty"] for r in crops)
        print(f"  {name:<5} групп {len(split[name]):>3}  кропов {len(crops):>4}  "
              f"(value {value}, empty {len(crops) - value})")
    print(f"страты (групп): {meta['strata']}")


if __name__ == "__main__":
    sys.exit(main())
