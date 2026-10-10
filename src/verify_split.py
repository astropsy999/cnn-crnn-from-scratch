"""Проверка разбиения: нет пересечений групп, ничего не потеряно, test достаточно велик."""
import argparse
import json
import math
import sys
from pathlib import Path

from split_groups import SPLITS, read_jsonl, split_hash

Z = 1.96           # 95 %-й доверительный интервал
TARGET_RECALL = 0.99


def min_value_crops(target: float = TARGET_RECALL, z: float = Z) -> int:
    """Сколько value-кропов нужно, чтобы при 0 ошибок нижняя граница Уилсона достигла target.

    Нижняя граница при 0 ошибок на n объектах: n / (n + z^2).
    """
    return math.ceil(target * z * z / (1 - target))


def verify(data_dir) -> tuple[list[str], list[str], list[dict]]:
    """Возвращает (ошибки, предупреждения, строки таблицы)."""
    data_dir = Path(data_dir)
    errors: list[str] = []
    warnings: list[str] = []
    all_records = read_jsonl(data_dir / "all.jsonl")
    part = {name: read_jsonl(data_dir / f"{name}.jsonl") for name in SPLITS}
    groups = {name: {r["group_id"] for r in part[name]} for name in SPLITS}

    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            common = groups[a] & groups[b]
            if common:
                sample = ", ".join(sorted(common)[:3])
                errors.append(f"УТЕЧКА: {len(common)} групп есть и в {a}, и в {b} (например, {sample})")

    ids = [r["id"] for name in SPLITS for r in part[name]]
    if len(ids) != len(set(ids)):
        errors.append("есть кропы, попавшие в несколько сплитов")
    missing = {r["id"] for r in all_records} - set(ids)
    if missing:
        errors.append(f"потеряно кропов: {len(missing)} (есть в all.jsonl, нет ни в одном сплите)")

    split_path = data_dir / "split.json"
    if split_path.exists():
        meta = json.loads(split_path.read_text(encoding="utf-8"))
        for name in SPLITS:
            if set(meta[name]) != groups[name]:
                errors.append(f"split.json не совпадает с {name}.jsonl")
        if meta.get("hash") != split_hash(meta):
            errors.append("hash в split.json не соответствует спискам групп")
    else:
        errors.append("нет split.json")

    table = []
    for name in SPLITS:
        value = sum(not r["is_empty"] for r in part[name])
        table.append({"split": name, "groups": len(groups[name]), "crops": len(part[name]),
                      "value": value, "empty": len(part[name]) - value})
    need = min_value_crops()
    test_value = next(row["value"] for row in table if row["split"] == "test")
    if test_value < need:
        warnings.append(f"в test {test_value} value-кропов; для проверки recall ≥ {TARGET_RECALL} "
                        f"нужно хотя бы {need} (иначе даже 0 ошибок не докажет цель)")
    return errors, warnings, table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    args = ap.parse_args()
    errors, warnings, table = verify(args.data_dir)

    print(f"{'split':<6}{'groups':>8}{'crops':>8}{'value':>8}{'empty':>8}")
    for row in table:
        print(f"{row['split']:<6}{row['groups']:>8}{row['crops']:>8}{row['value']:>8}{row['empty']:>8}")
    for w in warnings:
        print("WARN:", w)
    if errors:
        for e in errors:
            print("ERROR:", e)
        print("verify_split: FAIL")
        sys.exit(1)
    test_value = next(row["value"] for row in table if row["split"] == "test")
    print(f"пересечения групп: нет; test value-кропов: {test_value} (нужно ≥ {min_value_crops()})")
    print("verify_split: OK")


if __name__ == "__main__":
    main()
