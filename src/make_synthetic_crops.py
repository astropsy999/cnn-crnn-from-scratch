"""Генератор синтетических кропов.

Что моделируем (это то, на чём потом ловятся ошибки):
  * у каждого листа свой стиль: размер клетки, толщина пера, наклон, шум, фон;
  * значения внутри листа повторяются (как в реальных бланках);
  * пустые клетки бывают трудными: пятна, линии, штриховка, царапины.

Результат в папке --out:
  images/<sheet_id>/<n>.png   кропы
  index.jsonl                 одна строка на кроп (родной формат генератора)
  sheets.json                 стиль каждого листа
  generator.json              параметры запуска и версия генератора
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from alphabet import normalize_label, validate_format

GENERATOR_VERSION = "1"

MEASUREMENT_KINDS = {"dec1": 0.50, "dec2": 0.25, "int": 0.20, "lead0": 0.05}
DIAMETER_KINDS = {"single": 0.60, "fraction": 0.40}
THICKNESS_KINDS = {"single": 0.45, "range": 0.55}
EMPTY_KINDS = {"clean": 0.30, "stain": 0.15, "line": 0.20, "hatch": 0.15, "scratch": 0.10, "specks": 0.10}
TASKS = ("measurement", "diameter", "thickness")


# ----------------------------------------------------------------------------- значения

def choice_weighted(rng: np.random.Generator, weights: dict):
    keys = list(weights)
    p = np.array([weights[k] for k in keys], dtype=float)
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


def rand_int_part(rng: np.random.Generator) -> int:
    n_digits = choice_weighted(rng, {1: 0.45, 2: 0.40, 3: 0.15})
    lo = 1 if n_digits == 1 else 10 ** (n_digits - 1)
    return int(rng.integers(lo, 10 ** n_digits))


def rand_number(rng: np.random.Generator) -> str:
    """Число для thickness: 1–2 цифры, иногда с десятичной запятой."""
    base = int(rng.integers(1, 31))
    return f"{base},{int(rng.integers(1, 10))}" if rng.random() < 0.5 else str(base)


def make_value(task: str, rng: np.random.Generator) -> str:
    if task == "measurement":
        kind = choice_weighted(rng, MEASUREMENT_KINDS)
        if kind == "int":
            return str(rand_int_part(rng))
        if kind == "dec1":
            return f"{rand_int_part(rng)},{int(rng.integers(0, 10))}"
        if kind == "dec2":
            return f"{rand_int_part(rng)},{int(rng.integers(0, 100)):02d}"
        if rng.random() < 0.5:
            return f"0,{int(rng.integers(1, 10))}"
        return f"0,{int(rng.integers(1, 100)):02d}"
    if task == "diameter":
        if choice_weighted(rng, DIAMETER_KINDS) == "single":
            return str(int(rng.integers(10, 1000)))
        return f"{int(rng.integers(10, 100))}/{int(rng.integers(10, 100))}"
    if task == "thickness":
        if choice_weighted(rng, THICKNESS_KINDS) == "single":
            return rand_number(rng)
        return f"{rand_number(rng)}-{rand_number(rng)}"
    raise ValueError(f"неизвестная задача {task!r}")


# ----------------------------------------------------------------------------- стиль листа

def sample_style(rng: np.random.Generator) -> dict:
    return {
        "cell_w": int(rng.integers(110, 201)),
        "cell_h": int(rng.integers(40, 65)),
        "font_ratio": round(float(rng.uniform(0.50, 0.70)), 3),
        "stroke": int(rng.choice([0, 0, 1, 1, 2])),
        "slant": round(float(rng.uniform(-0.25, 0.25)), 3),
        "jitter_dy": round(float(rng.uniform(0.0, 1.8)), 2),
        "jitter_rot": round(float(rng.uniform(0.0, 5.0)), 2),
        "ink_gray": int(rng.integers(10, 91)),
        "paper_gray": int(rng.integers(215, 253)),
        "paper_noise": round(float(rng.uniform(1.0, 3.0)), 2),
        "shade": round(float(rng.uniform(0.0, 5.0)), 2),
        "blur": float(rng.choice([0.0, 0.5, 0.8])),
        "left_pad": int(rng.integers(5, 15)),
        "border": bool(rng.random() < 0.5),
        "border_gray": int(rng.integers(120, 190)),
    }


# ----------------------------------------------------------------------------- отрисовка

def get_font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except Exception:  # старый Pillow или нет FreeType
        return ImageFont.load_default()


def paste_lighter(dst: Image.Image, src: Image.Image, x: int, y: int) -> None:
    """Накладывает src на dst (берём более яркий пиксель), с обрезкой по границам."""
    w, h = dst.size
    sx0, sy0 = max(0, -x), max(0, -y)
    dx0, dy0 = max(0, x), max(0, y)
    cw = min(src.width - sx0, w - dx0)
    ch = min(src.height - sy0, h - dy0)
    if cw <= 0 or ch <= 0:
        return
    region = dst.crop((dx0, dy0, dx0 + cw, dy0 + ch))
    patch = src.crop((sx0, sy0, sx0 + cw, sy0 + ch))
    dst.paste(ImageChops.lighter(region, patch), (dx0, dy0))


def draw_text_mask(mask: Image.Image, text: str, style: dict, rng: np.random.Generator) -> None:
    """Рисует текст посимвольно (у каждого символа свой сдвиг и поворот) на маске чернил."""
    w, h = mask.size
    stroke = style["stroke"]
    size = max(10, round(style["font_ratio"] * h))
    pad = style["left_pad"] + int(rng.integers(-2, 3))
    while True:
        font = get_font(size)
        advances = [font.getlength(ch) + 1 + stroke for ch in text]
        if sum(advances) <= w - pad - 4 or size <= 10:
            break
        size -= 2
    bbox = font.getbbox("0")
    y0 = (h - (bbox[3] - bbox[1])) // 2 - bbox[1]
    x = float(pad)
    for ch, adv in zip(text, advances):
        dy = float(rng.normal(0.0, style["jitter_dy"]))
        rot = float(rng.normal(0.0, style["jitter_rot"]))
        tile = Image.new("L", (int(adv) + size, size * 2), 0)
        ImageDraw.Draw(tile).text((size // 2, size // 2), ch, font=font, fill=255,
                                  stroke_width=stroke, stroke_fill=255)
        if abs(rot) > 0.01:
            tile = tile.rotate(rot, resample=Image.Resampling.BICUBIC)
        paste_lighter(mask, tile, int(round(x)) - size // 2, int(round(y0 + dy)) - size // 2)
        x += adv


def shear(mask: Image.Image, slant: float) -> Image.Image:
    if abs(slant) < 1e-3:
        return mask
    w, h = mask.size
    return mask.transform((w, h), Image.Transform.AFFINE, (1, slant, -slant * h / 2, 0, 1, 0),
                          resample=Image.Resampling.BICUBIC)


def draw_empty_marks(mask: Image.Image, kind: str, rng: np.random.Generator) -> None:
    """Трудные пустые клетки: пятна, линии, штриховка, царапины, точки (kind='clean' — ничего)."""
    w, h = mask.size
    draw = ImageDraw.Draw(mask)
    if kind == "stain":
        layer = Image.new("L", (w, h), 0)
        ld = ImageDraw.Draw(layer)
        for _ in range(int(rng.integers(1, 4))):
            cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
            rx, ry = int(rng.integers(6, 22)), int(rng.integers(4, 14))
            ld.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=int(rng.integers(50, 130)))
        layer = layer.filter(ImageFilter.GaussianBlur(float(rng.uniform(2.5, 5.0))))
        paste_lighter(mask, layer, 0, 0)
    elif kind == "line":
        y = int(rng.integers(h // 2, h - 3))
        x1 = int(rng.integers(0, w // 3))
        x2 = int(rng.integers(2 * w // 3, w))
        draw.line([(x1, y), (x2, y)], fill=int(rng.integers(140, 230)), width=int(rng.integers(1, 3)))
        if rng.random() < 0.3:
            xv = int(rng.integers(0, w))
            draw.line([(xv, 0), (xv, h)], fill=int(rng.integers(110, 190)), width=1)
    elif kind == "hatch":
        step = int(rng.integers(7, 14))
        fill = int(rng.integers(120, 220))
        for x in range(-h, w, step):
            draw.line([(x, h), (x + h, 0)], fill=fill, width=1)
    elif kind == "scratch":
        for _ in range(int(rng.integers(1, 3))):
            pts = [(int(rng.integers(0, w)), int(rng.integers(0, h))) for _ in range(int(rng.integers(3, 7)))]
            draw.line(pts, fill=int(rng.integers(100, 200)), width=1)
    elif kind == "specks":
        for _ in range(int(rng.integers(3, 11))):
            cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
            r = int(rng.integers(1, 3))
            draw.ellipse([cx, cy, cx + r, cy + r], fill=int(rng.integers(90, 200)))


def paper_array(style: dict, rng: np.random.Generator, w: int, h: int) -> np.ndarray:
    arr = rng.normal(float(style["paper_gray"]), style["paper_noise"], size=(h, w)).astype(np.float32)
    angle = float(rng.uniform(0, 2 * math.pi))
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    return arr + style["shade"] * (math.cos(angle) * xs + math.sin(angle) * ys) / 2.0


def render_crop(text: str, empty_kind: str, style: dict, rng: np.random.Generator) -> Image.Image:
    w = max(60, style["cell_w"] + int(rng.integers(-4, 5)))
    h = max(30, style["cell_h"] + int(rng.integers(-3, 4)))
    mask = Image.new("L", (w, h), 0)
    if text:
        draw_text_mask(mask, text, style, rng)
        mask = shear(mask, style["slant"])
    else:
        draw_empty_marks(mask, empty_kind, rng)
    if style["blur"] > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(style["blur"]))
    alpha = np.asarray(mask, dtype=np.float32) / 255.0
    img = paper_array(style, rng, w, h) * (1.0 - alpha) + style["ink_gray"] * alpha
    if style["border"]:
        img[h - 2:, :] = np.minimum(img[h - 2:, :], style["border_gray"])
    return Image.fromarray(img.clip(0, 255).astype(np.uint8), mode="L")


# ----------------------------------------------------------------------------- листы

def generate(out, sheets: int, per_sheet: int, seed: int = 42, task: str = "measurement",
             empty_share: float = 0.4, repeat_share: float = 0.3) -> dict:
    if task not in TASKS:
        raise ValueError(f"неизвестная задача {task!r}")
    out = Path(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    styles, lines = {}, []
    for s in range(sheets):
        sheet_id = f"sheet_{s:04d}"
        rng = np.random.default_rng([seed, s])      # у каждого листа свой независимый поток
        style = sample_style(rng)
        styles[sheet_id] = style
        (out / "images" / sheet_id).mkdir(exist_ok=True)
        values = []                                  # значения, уже встречавшиеся на листе
        for n in range(per_sheet):
            if rng.random() < empty_share:
                text, kind, empty_kind = "", "empty", choice_weighted(rng, EMPTY_KINDS)
            else:
                if values and rng.random() < repeat_share:
                    text = values[int(rng.integers(0, len(values)))]
                else:
                    text = make_value(task, rng)
                values.append(text)
                kind, empty_kind = "value", ""
            assert normalize_label(text) == text and validate_format(text, task)[0], text
            rel = f"images/{sheet_id}/{n:03d}.png"
            render_crop(text, empty_kind, style, rng).save(out / rel)
            lines.append({"file": rel, "text": text, "sheet": sheet_id, "task": task,
                          "kind": kind, "empty_kind": empty_kind})
    with open(out / "index.jsonl", "w", encoding="utf-8") as f:
        for rec in lines:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (out / "sheets.json").write_text(json.dumps(styles, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = {"version": GENERATOR_VERSION, "seed": seed, "task": task, "sheets": sheets,
            "per_sheet": per_sheet, "empty_share": empty_share, "repeat_share": repeat_share}
    (out / "generator.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--sheets", type=int, default=250)
    ap.add_argument("--per-sheet", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--task", choices=TASKS, default="measurement")
    ap.add_argument("--empty-share", type=float, default=0.4)
    ap.add_argument("--repeat-share", type=float, default=0.3)
    args = ap.parse_args()
    meta = generate(args.out, args.sheets, args.per_sheet, args.seed, args.task,
                    args.empty_share, args.repeat_share)
    print(f"готово: {args.sheets} листов × {args.per_sheet} кропов = {args.sheets * args.per_sheet} "
          f"в {args.out} (task={meta['task']}, seed={meta['seed']})")


if __name__ == "__main__":
    main()
