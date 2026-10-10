import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from split_groups import split_hash  # noqa: E402
from verify_split import min_value_crops, verify  # noqa: E402


def make_records() -> list[dict]:
    records = []
    for g in range(6):
        for n in range(4):
            empty = n % 2 == 1
            records.append({"id": f"g{g}_{n}", "image": f"g{g}/{n}.png", "text": "" if empty else "7,3",
                            "group_id": f"g{g}", "task": "measurement", "is_empty": empty})
    return records


def write_dir(d, parts: dict, split_lists: dict | None = None, tamper_hash: bool = False) -> None:
    d = Path(d)
    records = make_records()
    write = lambda name, rows: (d / name).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    write("all.jsonl", records)
    for name, groups in parts.items():
        write(f"{name}.jsonl", [r for r in records if r["group_id"] in groups])
    lists = split_lists or {name: sorted(groups) for name, groups in parts.items()}
    meta = dict(lists)
    meta["hash"] = "bad" if tamper_hash else split_hash(meta)
    (d / "split.json").write_text(json.dumps(meta), encoding="utf-8")


GOOD = {"train": {"g0", "g1", "g2", "g3"}, "val": {"g4"}, "test": {"g5"}}


def test_min_value_crops_matches_wilson_arithmetic():
    n = min_value_crops()
    assert n == 381
    assert n / (n + 1.96 ** 2) >= 0.99 > (n - 1) / (n - 1 + 1.96 ** 2)


def test_good_split_has_no_errors():
    with tempfile.TemporaryDirectory() as d:
        write_dir(d, GOOD)
        errors, warnings, table = verify(d)
        assert errors == []
        assert warnings, "test слишком мал: ожидали предупреждение"
        assert [row["split"] for row in table] == ["train", "val", "test"]


def test_group_in_two_splits_is_reported_as_leak():
    parts = {"train": {"g0", "g1", "g2", "g3"}, "val": {"g3", "g4"}, "test": {"g5"}}
    with tempfile.TemporaryDirectory() as d:
        write_dir(d, parts)
        errors, _, _ = verify(d)
        assert any("УТЕЧКА" in e and "train" in e and "val" in e for e in errors), errors


def test_lost_crops_are_reported():
    parts = {"train": {"g0", "g1", "g2"}, "val": {"g4"}, "test": {"g5"}}      # g3 нигде нет
    with tempfile.TemporaryDirectory() as d:
        write_dir(d, parts)
        errors, _, _ = verify(d)
        assert any("потеряно" in e for e in errors), errors


def test_tampered_split_json_is_reported():
    with tempfile.TemporaryDirectory() as d:
        write_dir(d, GOOD, tamper_hash=True)
        errors, _, _ = verify(d)
        assert any("hash" in e for e in errors), errors
    with tempfile.TemporaryDirectory() as d:
        lists = {"train": ["g0", "g1", "g2"], "val": ["g3", "g4"], "test": ["g5"]}   # не совпадает с jsonl
        write_dir(d, GOOD, split_lists=lists)
        errors, _, _ = verify(d)
        assert any("не совпадает" in e for e in errors), errors
