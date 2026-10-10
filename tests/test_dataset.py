import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from alphabet import decode_indices  # noqa: E402
from augment import augment_default, augment_printed, content_box  # noqa: E402
from datasets.crop_dataset import (  # noqa: E402
    EMPTY_CLASS, VALUE_CLASS, CropDataset, collate_classify_np, collate_ctc_np,
)
from image_to_tensor import preprocess_array  # noqa: E402

_cache: dict = {}
_tmp: list = []


def run(*args) -> None:
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True, capture_output=True)


def data_dir() -> Path:
    """Маленький датасет (16 листов × 10 кропов), собранный теми же командами, что в уроках 4–5."""
    if "dir" not in _cache:
        d = tempfile.TemporaryDirectory()
        _tmp.append(d)
        base = Path(d.name)
        run("src/make_synthetic_crops.py", "--out", base / "syn", "--sheets", 16, "--per-sheet", 10, "--seed", 3)
        run("src/prepare_dataset.py", "--source", "synthetic", "--src", base / "syn", "--out", base / "proc")
        run("src/split_groups.py", "--src", base / "proc", "--out", base / "proc")
        _cache["dir"] = base / "proc"
    return _cache["dir"]


def test_lengths_match_split_files():
    d = data_dir()
    for split in ("train", "val", "test"):
        n_lines = sum(1 for _ in open(d / f"{split}.jsonl", encoding="utf-8"))
        assert len(CropDataset(d, split)) == n_lines


def test_no_group_overlap_through_dataset():
    d = data_dir()
    groups = {s: {r["group_id"] for r in CropDataset(d, s).records} for s in ("train", "val", "test")}
    assert groups["train"].isdisjoint(groups["val"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["val"].isdisjoint(groups["test"])


def test_item_fields_and_shapes():
    ds = CropDataset(data_dir(), "train")
    for i in range(min(10, len(ds))):
        item = ds[i]
        assert item["image"].shape == (1, 48, 160) and item["image"].dtype == np.float32
        assert item["target"].dtype == np.int64
        assert decode_indices(item["target"].tolist(), ds.task) == item["text"]
        assert item["label"] == (EMPTY_CLASS if item["is_empty"] else VALUE_CLASS)
        assert item["is_empty"] == (item["text"] == "")


def test_without_augment_equals_plain_preprocess():
    ds = CropDataset(data_dir(), "val")
    rec = ds.records[0]
    assert (ds[0]["image"] == preprocess_array(ds.image_root / rec["image"])).all()


def test_augmentation_is_reproducible_and_changes_with_epoch():
    d = data_dir()
    a = CropDataset(d, "train", augment=augment_printed, seed=5)
    b = CropDataset(d, "train", augment=augment_printed, seed=5)
    assert (a[3]["image"] == b[3]["image"]).all()                 # тот же seed, эпоха, индекс
    first = a[3]["image"].copy()
    a.set_epoch(1)
    assert not (a[3]["image"] == first).all()                     # другая эпоха — другая аугментация
    c = CropDataset(d, "train", augment=augment_printed, seed=6)
    assert not (c[3]["image"] == first).all()                     # другой seed


def test_augmented_padding_is_zero_and_label_unchanged():
    d = data_dir()
    ds = CropDataset(d, "train", augment=augment_default, seed=0)
    plain = CropDataset(d, "train")
    for i in range(20):
        assert ds[i]["text"] == plain[i]["text"]
        assert ds[i]["target"].tolist() == plain[i]["target"].tolist()
        with Image.open(ds.image_root / ds.records[i]["image"]) as raw:
            x0, y0, x1, y1 = content_box(*raw.size)
        image = ds[i]["image"][0]
        assert np.all(image[:, x1:] == 0.0) and np.all(image[:y0, :] == 0.0) and np.all(image[y1:, :] == 0.0)


def test_collate_ctc_np():
    ds = CropDataset(data_dir(), "train")
    batch = [ds[i] for i in range(12)]
    out = collate_ctc_np(batch)
    assert out["images"].shape == (12, 1, 48, 160)
    assert out["targets"].dtype == np.int64
    assert out["target_lengths"].sum() == len(out["targets"])
    for b, length in zip(batch, out["target_lengths"]):
        assert length == len(b["text"])
    assert any(length == 0 for length in out["target_lengths"]), "в батче нет пустых клеток"


def test_collate_classify_np():
    ds = CropDataset(data_dir(), "train")
    batch = [ds[i] for i in range(12)]
    out = collate_classify_np(batch)
    assert out["labels"].tolist() == [int(b["is_empty"]) for b in batch]
    assert out["images"].shape == (12, 1, 48, 160)


def test_torch_collate_and_dataloader():
    torch = pytest.importorskip("torch")
    from datasets.crop_dataset import collate_ctc
    from seed import make_generator, seed_worker
    ds = CropDataset(data_dir(), "train", augment=augment_default, seed=0)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_ctc,
                                         generator=make_generator(0), worker_init_fn=seed_worker)
    batch = next(iter(loader))
    assert tuple(batch["images"].shape) == (8, 1, 48, 160)
    assert batch["images"].dtype == torch.float32
    assert batch["targets"].dtype == torch.int64
    assert int(batch["target_lengths"].sum()) == len(batch["targets"])
