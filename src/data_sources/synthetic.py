"""Адаптер синтетического источника: читает папку, созданную make_synthetic_crops.py."""
import json
from pathlib import Path
from typing import Iterator

from alphabet import get_alphabet
from data_sources.base import DataSource, Record


class SyntheticSource(DataSource):
    def __init__(self, root) -> None:
        self.root = Path(root)

    def iter_records(self) -> Iterator[Record]:
        counters: dict[str, int] = {}
        with open(self.root / "index.jsonl", encoding="utf-8") as f:
            for line in f:
                raw = json.loads(line)
                n = counters.get(raw["sheet"], 0)
                counters[raw["sheet"]] = n + 1
                yield Record(
                    id=f"{raw['sheet']}_{n:03d}",
                    image=raw["file"],
                    text=raw["text"],
                    group_id=raw["sheet"],
                    task=raw["task"],
                    is_empty=raw["kind"] == "empty",
                )

    def meta(self) -> dict:
        gen = json.loads((self.root / "generator.json").read_text(encoding="utf-8"))
        return {
            "source": "synthetic",
            "task": gen["task"],
            "alphabet": get_alphabet(gen["task"]),
            "license": "сгенерировано этим репозиторием, ограничений нет",
            "image_root": str(self.root),
            "generator": gen,
        }
