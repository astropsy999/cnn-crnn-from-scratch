import sys
import tempfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_sources.base import REQUIRED_META, check_meta, check_records  # noqa: E402
from data_sources.synthetic import SyntheticSource  # noqa: E402
from make_synthetic_crops import generate  # noqa: E402


def assert_contract(source) -> None:
    """Контракт-тест: его должен проходить ЛЮБОЙ адаптер (synthetic, public, real)."""
    meta = source.meta()
    assert check_meta(meta) == [], check_meta(meta)
    records = list(source.iter_records())
    assert records, "источник не отдал ни одной записи"
    errors = check_records(records, meta["image_root"], meta["task"])
    assert errors == [], errors
    assert any(r["is_empty"] for r in records) and any(not r["is_empty"] for r in records)


def test_synthetic_source_passes_contract():
    with tempfile.TemporaryDirectory() as d:
        generate(d, sheets=4, per_sheet=10, seed=1)
        assert_contract(SyntheticSource(d))


def test_meta_has_required_keys():
    with tempfile.TemporaryDirectory() as d:
        generate(d, sheets=1, per_sheet=2, seed=1)
        meta = SyntheticSource(d).meta()
        for key in REQUIRED_META:
            assert key in meta


# --- контракт должен ловить типовые нарушения --------------------------------

def make_record(image: str, **over) -> dict:
    rec = {"id": "g1_000", "image": image, "text": "7,3", "group_id": "g1",
           "task": "measurement", "is_empty": False}
    rec.update(over)
    return rec


def with_png(fn):
    with tempfile.TemporaryDirectory() as d:
        Image.new("L", (10, 10), 255).save(Path(d) / "a.png")
        fn(d)


def test_check_records_accepts_good_record():
    def run(d):
        assert check_records([make_record("a.png")], d, "measurement") == []
    with_png(run)


def test_check_records_catches_violations():
    def run(d):
        cases = {
            "нет файла": make_record("missing.png"),
            "пустой group_id": make_record("a.png", group_id=""),
            "не нормализована": make_record("a.png", text="7.3"),
            "is_empty не согласовано": make_record("a.png", text="", is_empty=False),
            "формат": make_record("a.png", text="01,02,24"),
            "чужая задача": make_record("a.png", task="diameter"),
        }
        for name, rec in cases.items():
            assert check_records([rec], d, "measurement"), f"нарушение не поймано: {name}"
        dup = [make_record("a.png"), make_record("a.png")]
        assert check_records(dup, d, "measurement"), "повторяющийся id не пойман"
    with_png(run)
