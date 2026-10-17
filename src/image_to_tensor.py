"""Картинка → тензор: единый контракт входа для моделей курса (1×48×160).

Ядро написано на NumPy, чтобы его можно было проверять без PyTorch;
в тензор превращает тонкая обёртка image_to_tensor().
"""
import argparse

import numpy as np
from PIL import Image, ImageOps

CANVAS_H = 48
CANVAS_W = 160
PAD_FILL = 255           # фон холста: белая бумага (после инверсии станет 0)
MIN_CONTRAST_RANGE = 30  # если в кропе меньше 30 уровней яркости, контраст не растягиваем


def load_image(source) -> Image.Image:
    """Принимает путь или готовый PIL-образ."""
    if isinstance(source, Image.Image):
        return source  # уже открытый образ (тесты, Dataset)
    return Image.open(source)  # путь к файлу


def to_gray_on_white(img: Image.Image) -> Image.Image:
    """В оттенки серого; прозрачность подкладывается на белый фон."""
    has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
    if has_alpha:
        rgba = img.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))  # белая подложка
        img = Image.alpha_composite(white, rgba)  # иначе прозрачность станет чёрной
    return img.convert("L")  # один канал яркости


def autocontrast_safe(img: Image.Image, min_range: int = MIN_CONTRAST_RANGE) -> Image.Image:
    """Растягивает контраст, но не трогает почти однородные кропы.

    Пустая клетка почти однородна. Если растянуть её контраст до 0..255,
    шум бумаги превратится в «текст».
    """
    lo, hi = img.getextrema()  # мин/макс яркости в кропе
    # stubs Pillow: для RGB extrema — кортеж пар; для L — (lo, hi).
    if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
        raise TypeError(f"нужен одноканальный L, режим={img.mode!r}")
    if hi - lo < min_range:
        return img  # почти однородно: не раздуваем шум бумаги
    return ImageOps.autocontrast(img)  # растягиваем контраст до 0..255


def letterbox(img: Image.Image, h: int = CANVAS_H, w: int = CANVAS_W,
              fill: int = PAD_FILL) -> Image.Image:
    """Вписывает картинку в холст h×w без искажения пропорций.

    По горизонтали прижимаем влево (текст начинается всегда с одного места),
    по вертикали центрируем. Остаток холста заливаем фоном.
    """
    w0, h0 = img.size
    scale = min(w / w0, h / h0)  # вписать целиком, без растяжения
    new_w = max(1, round(w0 * scale))
    new_h = max(1, round(h0 * scale))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (w, h), fill)  # белый холст 48×160
    canvas.paste(resized, (0, (h - new_h) // 2))  # влево + по вертикали центр
    return canvas


def preprocess_array(source) -> np.ndarray:
    """Путь или PIL-образ → float32-массив [1, 48, 160]: фон ≈ 0, штрих ≈ 1."""
    img = load_image(source)
    if min(img.size) == 0:
        raise ValueError("пустое изображение (нулевой размер)")
    gray = to_gray_on_white(img)
    gray = autocontrast_safe(gray)
    canvas = letterbox(gray)
    arr = np.asarray(canvas, dtype=np.float32) / 255.0  # [0, 1], бумага ≈ 1
    return (1.0 - arr)[None, :, :]  # инверсия: штрих ≈ 1, паддинг = 0; канал впереди


def image_to_tensor(source):
    """То же, но в виде torch.Tensor."""
    import torch  # импорт здесь: ядро выше не требует torch
    return torch.from_numpy(preprocess_array(source))


def main() -> None:
    ap = argparse.ArgumentParser(description="Показать результат препроцессинга одного кропа")
    ap.add_argument("--image", required=True)
    args = ap.parse_args()

    arr = preprocess_array(args.image)
    zero_cols = int((arr[0].max(axis=0) == 0).sum())  # столбцы чистого паддинга
    print(f"shape={arr.shape} dtype={arr.dtype} "
          f"min={arr.min():.3f} max={arr.max():.3f} mean={arr.mean():.3f}")
    print(f"полностью нулевых столбцов (паддинг): {zero_cols} из {arr.shape[2]}")


if __name__ == "__main__":
    main()
