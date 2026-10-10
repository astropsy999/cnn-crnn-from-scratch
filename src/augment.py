"""Аугментации, СОХРАНЯЮЩИЕ МЕТКУ.

Правила (их проверяют тесты):
  * без зеркал и переворотов; поворот не больше ±5° (иначе 6 превращается в 9);
  * геометрия не режет цифры: если после преобразования чернила вышли бы за край кропа,
    преобразование отклоняется и пробуется заново;
  * паддинг холста остаётся ровно 0 (как при проверке на val и в проде).

Аугментация идёт в два этапа:
  1) геометрия на исходном кропе (поворот, масштаб, наклон, сдвиг);
  2) фотометрия на готовом холсте 48×160 (бледность чернил, размытие, JPEG, шум):
     шум добавляется ПОСЛЕ растяжения контраста, поэтому защита из урока 2 не нужна.
"""
import io
import math
from dataclasses import dataclass, replace

import numpy as np
from PIL import Image, ImageFilter

from image_to_tensor import CANVAS_H, CANVAS_W, load_image, preprocess_array, to_gray_on_white

ROT_CAP_DEG = 5.0       # больше нельзя: 6 и 9 начинают путаться
MIN_INK_CONTRAST = 25   # слабее — в кропе нет заметных чернил
INK_LEVEL = 0.35        # чернилами считаем пиксели темнее бумаги на 35 % контраста
LINE_FRAC = 0.85        # строка/столбец, занятые чернилами на 85 %+, — линия-рамка, а не цифры
GEOMETRY_TRIES = 8


@dataclass(frozen=True)
class AugmentConfig:
    rot_deg: float = 3.0
    scale: tuple = (0.93, 1.07)
    aspect: tuple = (0.96, 1.04)
    shear: float = 0.05
    shift_x: float = 0.03        # доли ширины кропа
    shift_y: float = 0.04        # доли высоты кропа
    gain: tuple = (0.85, 1.0)    # множитель яркости чернил (меньше = бледнее)
    blur_prob: float = 0.3
    blur_radius: tuple = (0.4, 0.9)
    jpeg_prob: float = 0.0
    jpeg_quality: tuple = (60, 95)
    noise: float = 0.02          # максимум σ гауссова шума (в долях 0..1)


DEFAULT = AugmentConfig()
PRINTED = AugmentConfig(rot_deg=5.0, scale=(0.85, 1.15), aspect=(0.90, 1.10), shear=0.10,
                        shift_x=0.05, shift_y=0.06, gain=(0.55, 1.0), blur_prob=0.7,
                        blur_radius=(0.5, 1.4), jpeg_prob=0.5, jpeg_quality=(30, 85), noise=0.06)


def validate_config(cfg: AugmentConfig) -> None:
    if abs(cfg.rot_deg) > ROT_CAP_DEG:
        raise ValueError(f"rot_deg={cfg.rot_deg}: поворот больше ±{ROT_CAP_DEG}° не сохраняет метку")


def content_box(w0: int, h0: int, h: int = CANVAS_H, w: int = CANVAS_W) -> tuple[int, int, int, int]:
    """Где на холсте лежит кроп размера w0×h0 после letterbox: (x0, y0, x1, y1)."""
    scale = min(w / w0, h / h0)
    new_w = max(1, round(w0 * scale))
    new_h = max(1, round(h0 * scale))
    top = (h - new_h) // 2
    return 0, top, new_w, top + new_h


def ink_mask(gray: np.ndarray) -> np.ndarray | None:
    """Маска «цифр» на исходном кропе (без линий-рамок); None, если чернил нет."""
    g = gray.astype(np.float32)
    paper = float(np.median(g))
    contrast = paper - float(g.min())
    if contrast < MIN_INK_CONTRAST:
        return None
    mask = (paper - g) > INK_LEVEL * contrast
    rows_full = mask.mean(axis=1) > LINE_FRAC
    cols_full = mask.mean(axis=0) > LINE_FRAC
    mask[rows_full, :] = False
    mask[:, cols_full] = False
    return mask if mask.any() else None


def ink_bbox(gray: np.ndarray):
    """(x0, y0, x1, y1) рамка чернил (x1, y1 не включены) или None."""
    mask = ink_mask(gray)
    if mask is None:
        return None
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _translation(tx: float, ty: float) -> np.ndarray:
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)


def affine_forward(w: int, h: int, rot_deg: float, sx: float, sy: float,
                   shear: float, tx: float, ty: float) -> np.ndarray:
    """Матрица 3×3: поворот, наклон и масштаб вокруг центра кропа, затем сдвиг."""
    th = math.radians(rot_deg)
    rot = np.array([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]])
    sh = np.array([[1, shear, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    sc = np.diag([sx, sy, 1.0])
    return _translation(w / 2 + tx, h / 2 + ty) @ rot @ sh @ sc @ _translation(-w / 2, -h / 2)


def sample_matrix(w: int, h: int, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    """Случайное преобразование кропа w×h. Масштабы положительные, поворот ≤ rot_deg: отражений не бывает."""
    rot = float(rng.uniform(-cfg.rot_deg, cfg.rot_deg))
    s = float(rng.uniform(*cfg.scale))
    a = float(rng.uniform(*cfg.aspect))
    shear = float(rng.uniform(-cfg.shear, cfg.shear))
    tx = float(rng.uniform(-cfg.shift_x, cfg.shift_x)) * w
    ty = float(rng.uniform(-cfg.shift_y, cfg.shift_y)) * h
    return affine_forward(w, h, rot, s * math.sqrt(a), s / math.sqrt(a), shear, tx, ty)


def geometric(img: Image.Image, rng: np.random.Generator, cfg: AugmentConfig) -> tuple[Image.Image, bool]:
    """Геометрия на исходном кропе. Возвращает (картинка, применилась ли геометрия)."""
    gray = to_gray_on_white(img)
    w, h = gray.size
    arr = np.asarray(gray)
    bbox = ink_bbox(arr)
    paper = int(np.median(arr))
    for _ in range(GEOMETRY_TRIES):
        m = sample_matrix(w, h, rng, cfg)
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            corners = m @ np.array([[x0, x1, x1, x0], [y0, y0, y1, y1], [1, 1, 1, 1]], dtype=np.float64)
            if (corners[0].min() < 1 or corners[0].max() > w - 1
                    or corners[1].min() < 1 or corners[1].max() > h - 1):
                continue                      # цифры вышли бы за край: пробуем другие параметры
        inv = np.linalg.inv(m)
        coeffs = (inv[0, 0], inv[0, 1], inv[0, 2], inv[1, 0], inv[1, 1], inv[1, 2])
        out = gray.transform((w, h), Image.Transform.AFFINE, coeffs,
                             resample=Image.Resampling.BILINEAR, fillcolor=paper)
        return out, True
    return gray, False


def photometric(arr: np.ndarray, box: tuple, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    """Фотометрия на холсте [1, 48, 160]; паддинг в конце снова обнуляется."""
    a = arr[0].copy()
    x0, y0, x1, y1 = box
    a[y0:y1, x0:x1] *= float(rng.uniform(*cfg.gain))
    if rng.random() < cfg.blur_prob:
        im = Image.fromarray((a * 255).round().astype(np.uint8), mode="L")
        a = np.asarray(im.filter(ImageFilter.GaussianBlur(float(rng.uniform(*cfg.blur_radius)))),
                       dtype=np.float32) / 255.0
    if rng.random() < cfg.jpeg_prob:
        buf = io.BytesIO()
        Image.fromarray((a * 255).round().astype(np.uint8), mode="L").save(
            buf, "JPEG", quality=int(rng.integers(cfg.jpeg_quality[0], cfg.jpeg_quality[1] + 1)))
        buf.seek(0)
        a = np.asarray(Image.open(buf).convert("L"), dtype=np.float32) / 255.0
    sigma = float(rng.uniform(0.0, cfg.noise))
    if sigma > 0:
        a = a + rng.normal(0.0, sigma, size=a.shape).astype(np.float32)
    a = np.clip(a, 0.0, 1.0)
    keep = np.zeros(a.shape, dtype=bool)
    keep[y0:y1, x0:x1] = True
    a[~keep] = 0.0                       # паддинг остаётся ровно 0
    return a[None, :, :].astype(np.float32)


def augment(source, rng: np.random.Generator, cfg: AugmentConfig, validate: bool = True) -> np.ndarray:
    """Путь или PIL-образ → аугментированный массив [1, 48, 160]."""
    if validate:
        validate_config(cfg)
    img, _ = geometric(load_image(source), rng, cfg)
    arr = preprocess_array(img)
    return photometric(arr, content_box(*img.size), rng, cfg)


def augment_default(source, rng: np.random.Generator) -> np.ndarray:
    return augment(source, rng, DEFAULT)


def augment_printed(source, rng: np.random.Generator) -> np.ndarray:
    return augment(source, rng, PRINTED)


def with_overrides(cfg: AugmentConfig, **kwargs) -> AugmentConfig:
    return replace(cfg, **kwargs)
