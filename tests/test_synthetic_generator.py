import atexit
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alphabet import validate_format  # noqa: E402
from image_to_tensor import MIN_CONTRAST_RANGE  # noqa: E402
from make_synthetic_crops import EMPTY_KINDS, generate  # noqa: E402
from patterns import label_pattern  # noqa: E402
from PIL import Image  # noqa: E402

_cache: dict = {}
_tmp: list = []


def read_index(root):
    with open(Path(root) / "index.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def big_dataset() -> Path:
    """60 листов × 20 кропов, seed 42; создаётся один раз на все тесты."""
    if "big" not in _cache:
        d = tempfile.TemporaryDirectory()
        _tmp.append(d)
        generate(d.name, sheets=60, per_sheet=20, seed=42)
        _cache["big"] = Path(d.name)
    return _cache["big"]


atexit.register(lambda: [d.cleanup() for d in _tmp])


def test_same_seed_gives_identical_files():
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        generate(a, sheets=3, per_sheet=8, seed=7)
        generate(b, sheets=3, per_sheet=8, seed=7)
        assert (Path(a) / "index.jsonl").read_bytes() == (Path(b) / "index.jsonl").read_bytes()
        for rec in read_index(a):
            assert (Path(a) / rec["file"]).read_bytes() == (Path(b) / rec["file"]).read_bytes()


def test_sheets_are_independent_of_total_count():
    # Листы строятся независимыми потоками: первые 3 листа не зависят от того, сколько листов всего.
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        generate(a, sheets=3, per_sheet=8, seed=7)
        generate(b, sheets=6, per_sheet=8, seed=7)
        first_a = read_index(a)
        first_b = [r for r in read_index(b) if r["sheet"] in {r2["sheet"] for r2 in first_a}]
        assert first_a == first_b
        for rec in first_a:
            assert (Path(a) / rec["file"]).read_bytes() == (Path(b) / rec["file"]).read_bytes()


def test_labels_valid_for_all_tasks_and_patterns_present():
    expected = {"measurement": {"dec1", "dec2", "int", "lead0"},
                "diameter": {"single", "fraction"},
                "thickness": {"single", "range"}}
    for task, patterns in expected.items():
        with tempfile.TemporaryDirectory() as d:
            generate(d, sheets=10, per_sheet=20, seed=3, task=task)
            texts = [r["text"] for r in read_index(d) if r["kind"] == "value"]
            for t in texts:
                ok, reason = validate_format(t, task)
                assert ok, f"{task}: {t!r}: {reason}"
            assert {label_pattern(t, task) for t in texts} >= patterns, task


def test_empty_share_close_to_requested():
    rows = read_index(big_dataset())
    share = sum(r["kind"] == "empty" for r in rows) / len(rows)
    assert 0.33 <= share <= 0.47, share


def test_rare_patterns_are_at_least_three_percent():
    rows = read_index(big_dataset())
    values = [r["text"] for r in rows if r["kind"] == "value"]
    for pattern in ("dec1", "dec2", "int", "lead0"):
        share = sum(label_pattern(t, "measurement") == pattern for t in values) / len(values)
        assert share >= 0.03, f"{pattern}: {share:.3f}"


def test_all_hard_negative_kinds_occur():
    rows = read_index(big_dataset())
    kinds = {r["empty_kind"] for r in rows if r["kind"] == "empty"}
    assert kinds == set(EMPTY_KINDS), kinds


def repeated_share(root) -> float:
    by_sheet: dict = {}
    for r in read_index(root):
        if r["kind"] == "value":
            by_sheet.setdefault(r["sheet"], []).append(r["text"])
    total = sum(len(v) for v in by_sheet.values())
    repeated = sum(1 for v in by_sheet.values() for t in v if v.count(t) > 1)
    return repeated / total


def test_repeat_share_increases_repeats_on_sheet():
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        generate(a, sheets=30, per_sheet=20, seed=5, repeat_share=0.0)
        generate(b, sheets=30, per_sheet=20, seed=5, repeat_share=0.6)
        assert repeated_share(b) > repeated_share(a) + 0.2


def test_sheet_styles_differ():
    styles = json.loads((big_dataset() / "sheets.json").read_text(encoding="utf-8"))
    assert len({s["cell_w"] for s in styles.values()}) >= 20
    assert len({s["ink_gray"] for s in styles.values()}) >= 20
    assert {s["border"] for s in styles.values()} == {True, False}


def test_clean_empty_crops_are_left_alone_by_contrast_guard():
    # Связь с уроком 2: чистая пустая клетка (без рамки) не должна растягиваться по контрасту.
    root = big_dataset()
    styles = json.loads((root / "sheets.json").read_text(encoding="utf-8"))
    checked = 0
    for r in read_index(root):
        if r["kind"] == "empty" and r["empty_kind"] == "clean" and not styles[r["sheet"]]["border"]:
            lo, hi = Image.open(root / r["file"]).getextrema()
            assert hi - lo < MIN_CONTRAST_RANGE, (r["file"], hi - lo)
            checked += 1
    assert checked >= 20
