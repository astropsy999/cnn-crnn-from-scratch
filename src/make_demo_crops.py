"""Несколько искусственных кропов для уроков 2–3 (полный генератор будет в уроке 4)."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# (текст, часть имени файла, ширина, высота)
SAMPLES = [
    ("7,3", "7_3", 120, 40),
    ("12/14", "12_14", 200, 50),
    ("0,85", "0_85", 150, 60),
    ("105", "105", 90, 45),
    ("1-2", "1-2", 100, 40),
    ("123,45", "123_45", 380, 70),   # очень широкий
    ("7", "7", 40, 50),               # очень узкий
    ("", "empty", 120, 45),           # пустая клетка
    ("", "empty_noisy", 140, 45),     # пустая клетка с мусором
]


def get_font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except Exception:  # старый Pillow или нет FreeType
        return ImageFont.load_default()


def draw_crop(text: str, w: int, h: int, rng: np.random.Generator, noisy: bool) -> Image.Image:
    paper = rng.integers(225, 256)
    arr = rng.normal(paper, 3.0, size=(h, w)).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    draw = ImageDraw.Draw(img)

    if text:
        pad = int(rng.integers(6, 13))
        size = int(h * 0.6)
        font = get_font(size)
        bbox = draw.textbbox((0, 0), text, font=font)
        while bbox[2] - bbox[0] > w - 2 * pad and size > 8:
            size -= 2
            font = get_font(size)
            bbox = draw.textbbox((0, 0), text, font=font)
        ink = int(rng.integers(20, 70))
        x = pad - bbox[0]
        y = (h - (bbox[3] - bbox[1])) // 2 - bbox[1]
        draw.text((x, y), text, fill=ink, font=font)

    if noisy:
        draw.line([(0, h - 4), (w, h - 4)], fill=150, width=1)  # край клетки
        for _ in range(6):                                       # пятнышки
            cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
            draw.ellipse([cx, cy, cx + 2, cy + 2], fill=int(rng.integers(90, 160)))
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic/raw")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for text, name, w, h in SAMPLES:
        img = draw_crop(text, w, h, rng, noisy=name.endswith("noisy"))
        path = out / f"sample_{name}.png"
        img.save(path)
        print(f"{path}  ({w}x{h})  text={text!r}")


if __name__ == "__main__":
    main()
