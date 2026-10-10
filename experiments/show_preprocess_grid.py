"""Сетка «исходник | холст до инверсии | тензор» для всех кропов в папке."""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from image_to_tensor import (  # noqa: E402
    autocontrast_safe, letterbox, load_image, preprocess_array, to_gray_on_white,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/synthetic/raw")
    ap.add_argument("--out", default="experiments/preprocess_grid.png")
    args = ap.parse_args()

    paths = sorted(Path(args.dir).glob("*.png"))
    if not paths:
        sys.exit(f"В {args.dir} нет PNG. Сначала: python src/make_demo_crops.py")

    fig, axes = plt.subplots(len(paths), 3, figsize=(10, 1.25 * len(paths) + 0.6))
    for row, path in enumerate(paths):
        raw = load_image(path)
        gray = to_gray_on_white(raw)
        canvas = letterbox(autocontrast_safe(gray))
        tensor = preprocess_array(path)[0]

        axes[row, 0].imshow(np.asarray(gray), cmap="gray", vmin=0, vmax=255)
        axes[row, 1].imshow(np.asarray(canvas), cmap="gray", vmin=0, vmax=255)
        axes[row, 2].imshow(tensor, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_ylabel(path.stem.replace("sample_", ""), rotation=0,
                                ha="right", va="center", fontsize=8)
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    for col, title in enumerate(["исходник", "холст 48×160 (до инверсии)", "тензор (штрих = светлый)"]):
        axes[0, col].set_title(title, fontsize=9)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"сохранено: {out}")


if __name__ == "__main__":
    main()
