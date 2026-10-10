"""Намеренно портит разбиение: делит по КРОПАМ, а не по группам (для проверки verify_split)."""
import argparse
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from split_groups import SPLITS, read_jsonl, split_hash  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/processed")
    ap.add_argument("--out", default="data/processed-leaky")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "manifest.json", out / "manifest.json")
    shutil.copy(src / "all.jsonl", out / "all.jsonl")

    records = read_jsonl(src / "all.jsonl")
    random.Random(args.seed).shuffle(records)                 # перемешиваем кропы, а не листы
    n = len(records)
    cuts = {"train": records[: int(n * 0.70)],
            "val": records[int(n * 0.70): int(n * 0.85)],
            "test": records[int(n * 0.85):]}
    meta = {"version": 1, "seed": args.seed}
    for name in SPLITS:
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in cuts[name]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        meta[name] = sorted({r["group_id"] for r in cuts[name]})
    meta["hash"] = split_hash(meta)
    (out / "split.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"испорченный сплит записан в {out}")


if __name__ == "__main__":
    main()
