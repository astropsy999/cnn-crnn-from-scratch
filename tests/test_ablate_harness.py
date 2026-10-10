import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, capture_output=True, text=True)


def test_logreg_backend_end_to_end():
    pytest.importorskip("sklearn")
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        for cmd in (
            ["src/make_synthetic_crops.py", "--out", base / "syn", "--sheets", 40, "--per-sheet", 12, "--seed", 5],
            ["src/prepare_dataset.py", "--source", "synthetic", "--src", base / "syn", "--out", base / "proc"],
            ["src/split_groups.py", "--src", base / "proc", "--out", base / "proc"],
        ):
            assert run(*cmd).returncode == 0, cmd
        r = run("experiments/ablate_imbalance.py", "--backend", "logreg", "--seeds", 2,
                "--data-dir", base / "proc", "--out", base / "abl.json")
        assert r.returncode == 0, r.stderr
        data = json.loads((base / "abl.json").read_text(encoding="utf-8"))
        assert len(data["results"]) == 8                                  # 4 варианта × 2 seed
        assert set(data["summary"]) == {"none", "weights", "sampler", "both"}
        assert "вариант" in r.stdout and "Победитель по val" in r.stdout
        for res in data["results"]:
            assert 0.0 <= res["val_empty_recall"] <= 1.0 and 0.0 <= res["test_value_recall"] <= 1.0
