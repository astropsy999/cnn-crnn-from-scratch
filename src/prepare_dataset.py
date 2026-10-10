"""Источник данных → data/processed/all.jsonl + manifest.json (единая схема для всех уроков)."""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from alphabet import num_classes
from data_sources.base import check_meta, check_records
from data_sources.synthetic import SyntheticSource
from patterns import label_pattern

SCHEMA = ["id", "image", "text", "group_id", "task", "is_empty"]


def build_source(name: str, src: str):
    if name == "synthetic":
        if not (Path(src) / "index.jsonl").is_file():
            sys.exit(f"В {src} нет index.jsonl. Сначала: python src/make_synthetic_crops.py --out {src}")
        return SyntheticSource(src)
    sys.exit(f"Источник {name!r} подключается позже: public — приложение A2, real — приложение A4.")


def repeat_share(records: list[dict]) -> float:
    """Доля value-кропов, чьё значение встречается на том же листе ещё раз."""
    by_group = defaultdict(Counter)
    for r in records:
        if not r["is_empty"]:
            by_group[r["group_id"]][r["text"]] += 1
    value_total = sum(sum(c.values()) for c in by_group.values())
    repeated = sum(n for c in by_group.values() for n in c.values() if n > 1)
    return repeated / value_total if value_total else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "public", "real"], required=True)
    ap.add_argument("--src", required=True, help="папка источника")
    ap.add_argument("--out", default="data/processed")
    args = ap.parse_args()

    source = build_source(args.source, args.src)
    meta = source.meta()
    problems = check_meta(meta)
    records = sorted(source.iter_records(), key=lambda r: (r["group_id"], r["id"]))
    problems += check_records(records, meta["image_root"], meta["task"])
    if problems:
        print("Источник не соответствует контракту:")
        for p in problems:
            print("  -", p)
        sys.exit(1)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "all.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({k: r[k] for k in SCHEMA}, ensure_ascii=False) + "\n")

    per_group = Counter(r["group_id"] for r in records)
    empty = sum(r["is_empty"] for r in records)
    manifest = {
        "schema_version": 1,
        "source": meta["source"],
        "task": meta["task"],
        "alphabet": meta["alphabet"],
        "num_classes": num_classes(meta["task"]),
        "image_root": meta["image_root"],
        "license": meta["license"],
        "fields": SCHEMA,
        "counts": {"total": len(records), "value": len(records) - empty, "empty": empty},
        "empty_share": round(empty / len(records), 4),
        "groups": len(per_group),
        "crops_per_group": {"min": min(per_group.values()), "max": max(per_group.values())},
        "patterns": dict(sorted(Counter(label_pattern(r["text"], r["task"]) for r in records).items())),
        "repeated_on_sheet_share": round(repeat_share(records), 4),
        "generator": meta.get("generator"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"записано {len(records)} кропов ({manifest['counts']['value']} value, {empty} empty), "
          f"{len(per_group)} групп -> {out / 'all.jsonl'}")
    print(f"манифест: {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
