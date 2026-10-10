"""Сетка 4×4: исходный кроп и 15 его аугментаций."""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from augment import DEFAULT, PRINTED, augment, validate_config, with_overrides  # noqa: E402
from image_to_tensor import load_image, preprocess_array  # noqa: E402


def parse_overrides(items: list[str]) -> dict:
    out = {}
    for item in items:
        key, _, value = item.partition("=")
        out[key] = float(value)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="путь к кропу")
    ap.add_argument("--find-text", help="взять первый кроп из train.jsonl с такой меткой, например 6")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--style", choices=["default", "printed"], default="default")
    ap.add_argument("--override", action="append", default=[], metavar="ключ=значение",
                    help="переопределить параметр, например rot_deg=40 или noise=0.1")
    ap.add_argument("--unsafe", action="store_true", help="не проверять правило «поворот ≤ 5°»")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/aug_grid.png")
    args = ap.parse_args()

    if args.find_text:
        data_dir = Path(args.data_dir)
        manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
        with open(data_dir / "train.jsonl", encoding="utf-8") as f:
            rec = next((r for r in map(json.loads, f) if r["text"] == args.find_text), None)
        if rec is None:
            sys.exit(f"В {data_dir / 'train.jsonl'} нет кропа с меткой {args.find_text!r}: попробуйте другую метку.")
        image_path = Path(manifest["image_root"]) / rec["image"]
    elif args.image:
        image_path = Path(args.image)
    else:
        sys.exit("Укажите --image или --find-text")

    cfg = with_overrides(DEFAULT if args.style == "default" else PRINTED, **parse_overrides(args.override))
    img = load_image(image_path)
    rng = np.random.default_rng(args.seed)

    try:
        validate_config(cfg)
    except ValueError as e:
        if not args.unsafe:
            sys.exit(f"Параметры отклонены: {e}.\nДля эксперимента добавьте --unsafe (в обучении так делать нельзя).")

    fig, axes = plt.subplots(4, 4, figsize=(10, 4.2))
    for k, ax in enumerate(axes.flat):
        arr = preprocess_array(img) if k == 0 else augment(img, rng, cfg, validate=False)
        ax.imshow(arr[0], cmap="gray", vmin=0, vmax=1)
        ax.set_title("исходный" if k == 0 else f"#{k}", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    extra = f"  {' '.join(args.override)}" if args.override else ""
    fig.suptitle(f"{image_path.name}: стиль {args.style}{extra}", fontsize=9)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"сохранено: {out}")


if __name__ == "__main__":
    main()
