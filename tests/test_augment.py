import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from augment import (  # noqa: E402
    DEFAULT, PRINTED, ROT_CAP_DEG, augment, content_box, geometric, ink_bbox, ink_mask, sample_matrix,
    validate_config,
)
from image_to_tensor import CANVAS_H, CANVAS_W, preprocess_array, to_gray_on_white  # noqa: E402
from make_synthetic_crops import generate  # noqa: E402

import json  # noqa: E402
import math  # noqa: E402
import tempfile  # noqa: E402

_cache: dict = {}
_tmp: list = []


def sample_crops() -> list:
    """Value-кропы двух листов (один с рамкой, другой без): (PIL-образ, метка)."""
    if "crops" not in _cache:
        d = tempfile.TemporaryDirectory()
        _tmp.append(d)
        generate(d.name, sheets=8, per_sheet=10, seed=11)
        styles = json.loads((Path(d.name) / "sheets.json").read_text(encoding="utf-8"))
        crops = []
        for line in open(Path(d.name) / "index.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if r["kind"] == "value":
                img = Image.open(Path(d.name) / r["file"])
                img.load()
                crops.append((img, r["text"], styles[r["sheet"]]["border"]))
        _cache["crops"] = crops
    return _cache["crops"]


def text_like(w: int, h: int) -> Image.Image:
    img = Image.new("L", (w, h), 235)
    img.paste(40, (w // 5, h // 4, w // 2, 3 * h // 4))
    img.paste(40, (w // 2 + 4, h // 4, 7 * w // 10, 3 * h // 4))
    return img


def test_content_box_matches_letterbox():
    for w0, h0 in [(40, 50), (100, 40), (200, 50), (380, 70), (160, 48), (30, 30)]:
        dark = preprocess_array(Image.new("L", (w0, h0), 0))[0]       # однородно тёмный → единицы только в содержимом
        x0, y0, x1, y1 = content_box(w0, h0)
        expected = np.zeros((CANVAS_H, CANVAS_W), dtype=bool)
        expected[y0:y1, x0:x1] = True
        assert ((dark > 0.5) == expected).all(), (w0, h0)


def test_shape_dtype_range_for_both_styles():
    for cfg in (DEFAULT, PRINTED):
        arr = augment(text_like(120, 40), np.random.default_rng(0), cfg)
        assert arr.shape == (1, CANVAS_H, CANVAS_W)
        assert arr.dtype == np.float32
        assert arr.min() >= 0.0 and arr.max() <= 1.0


def test_deterministic_for_same_rng_and_different_for_other():
    img = text_like(120, 40)
    a = augment(img, np.random.default_rng([1, 2, 3]), PRINTED)
    b = augment(img, np.random.default_rng([1, 2, 3]), PRINTED)
    c = augment(img, np.random.default_rng([1, 2, 4]), PRINTED)
    assert (a == b).all()
    assert not (a == c).all()


def test_padding_stays_exactly_zero():
    img = text_like(40, 50)                      # узкий кроп: справа много паддинга
    x0, y0, x1, y1 = content_box(40, 50)
    for seed in range(20):
        arr = augment(img, np.random.default_rng(seed), PRINTED)[0]
        assert np.all(arr[:, x1:] == 0.0)
        assert np.all(arr[:y0, :] == 0.0) and np.all(arr[y1:, :] == 0.0)


def test_geometry_never_cuts_digits():
    # Рамку клетки после поворота нельзя отличить от чернил, поэтому границы проверяем на листах без рамки;
    # потерю чернил (резать цифры нельзя) проверяем на всех.
    checked_bbox = checked_mass = 0
    for img, text, border in sample_crops():
        gray = np.asarray(to_gray_on_white(img))
        before = ink_mask(gray)
        for seed in range(15):
            out, applied = geometric(img, np.random.default_rng(seed), PRINTED)
            if not applied:
                continue
            out_gray = np.asarray(out)
            assert ink_mask(out_gray).sum() >= 0.6 * before.sum(), text
            checked_mass += 1
            if border:
                continue
            after = ink_bbox(out_gray)
            h, w = out_gray.shape
            assert after is not None, text
            assert after[0] >= 1 and after[1] >= 1 and after[2] <= w - 1 and after[3] <= h - 1, (text, after)
            checked_bbox += 1
    assert checked_mass > 200 and checked_bbox > 80, (checked_mass, checked_bbox)


def test_borders_do_not_disable_geometry():
    # Линия-рамка клетки не должна считаться «цифрами»: иначе на листах с рамкой геометрия не применялась бы.
    applied = {True: 0, False: 0}
    total = {True: 0, False: 0}
    for img, text, border in sample_crops():
        for seed in range(5):
            _, ok = geometric(img, np.random.default_rng(seed), PRINTED)
            applied[border] += ok
            total[border] += 1
    for border in (True, False):
        assert total[border] > 0
        assert applied[border] / total[border] > 0.8, (border, applied[border] / total[border])


def test_rotation_cap():
    validate_config(DEFAULT)
    validate_config(PRINTED)
    with pytest.raises(ValueError):
        validate_config(replace(DEFAULT, rot_deg=ROT_CAP_DEG + 1))
    with pytest.raises(ValueError):
        augment(text_like(100, 40), np.random.default_rng(0), replace(DEFAULT, rot_deg=30))
    augment(text_like(100, 40), np.random.default_rng(0), replace(DEFAULT, rot_deg=30), validate=False)


def pooled(a: np.ndarray) -> np.ndarray:
    return a.reshape(CANVAS_H // 2, 2, CANVAS_W // 2, 2).mean(axis=(1, 3))


def best_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Лучшая корреляция по небольшим сдвигам: устойчива к сдвигу текста, чувствительна к зеркалу."""
    a, b = pooled(a), pooled(b)
    best = -1.0
    for dy in range(-3, 4):
        for dx in range(-5, 6):
            shifted = np.roll(np.roll(b, dy, axis=0), dx, axis=1)
            best = max(best, float(np.corrcoef(a.ravel(), shifted.ravel())[0, 1]))
    return best


def test_transform_is_never_a_reflection_and_respects_rotation_cap():
    # Переворот по вертикали по корреляции не проверить (цифры почти симметричны), поэтому проверяем матрицу:
    # определитель > 0 и положительный масштаб по x исключают отражения, угол ограничен.
    for cfg in (DEFAULT, PRINTED):
        rng = np.random.default_rng(0)
        for _ in range(2000):
            m = sample_matrix(160, 48, rng, cfg)
            assert np.linalg.det(m[:2, :2]) > 0
            assert m[0, 0] > 0
            assert abs(math.degrees(math.atan2(m[1, 0], m[0, 0]))) <= min(cfg.rot_deg, ROT_CAP_DEG) + 1e-9


def test_no_horizontal_mirror():
    # Аугментированный кроп (с поправкой на сдвиг) в среднем похож на исходный сильнее, чем на его зеркало.
    # Почти симметричные кропы пропускаем: у них зеркало ≈ оригинал.
    checked = 0
    for img, text, _ in sample_crops()[:30]:
        clean = preprocess_array(img)[0]
        if clean.std() < 1e-3:
            continue
        mirrored = np.ascontiguousarray(clean[:, ::-1])
        if best_corr(clean, mirrored) > 0.5:
            continue
        augs = [augment(img, np.random.default_rng(seed), PRINTED)[0] for seed in range(5)]
        same = np.mean([best_corr(a, clean) for a in augs])
        other = np.mean([best_corr(a, mirrored) for a in augs])
        assert same > other, (text, same, other)
        checked += 1
    assert checked >= 10, checked


def test_printed_is_stronger_than_default():
    img = text_like(150, 45)
    clean = preprocess_array(img)[0]
    def mean_change(cfg):
        return np.mean([np.abs(augment(img, np.random.default_rng(s), cfg)[0] - clean).mean() for s in range(60)])
    assert mean_change(PRINTED) > 1.5 * mean_change(DEFAULT)
