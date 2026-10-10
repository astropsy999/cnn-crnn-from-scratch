"""Сетка образцов: 8 value и 8 empty из разных листов (видно, как различаются стили листов)."""
import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402


def pick(records: list[dict], n: int, rng: random.Random) -> list[dict]:
    """По одному кропу с n разных листов."""
    by_group: dict[str, list[dict]] = {}
    for r in records:
        by_group.setdefault(r["group_id"], []).append(r)
    groups = rng.sample(sorted(by_group), min(n, len(by_group)))
    return [rng.choice(by_group[g]) for g in groups]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out", default="experiments/samples_grid.png")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in open(data_dir / "all.jsonl", encoding="utf-8")]
    rng = random.Random(args.seed)
    values = pick([r for r in records if not r["is_empty"]], 8, rng)
    empties = pick([r for r in records if r["is_empty"]], 8, rng)
    if len(values) < 8 or len(empties) < 8:
        sys.exit("Мало листов для сетки 8 + 8: сгенерируйте больше листов.")

    fig, axes = plt.subplots(4, 4, figsize=(11, 6.5))
    for ax, rec, kind in zip(axes.flat, values + empties, ["value"] * 8 + ["empty"] * 8):
        ax.imshow(Image.open(Path(manifest["image_root"]) / rec["image"]), cmap="gray", vmin=0, vmax=255)
        label = rec["text"] if kind == "value" else "(пусто)"
        ax.set_title(f"{label}   {rec['group_id']}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Верхние два ряда: value; нижние два: empty (включая трудные). Каждый кроп с другого листа.",
                 fontsize=9)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"сохранено: {out}")


if __name__ == "__main__":
    main()
