import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from split_groups import split_groups  # noqa: E402

RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def make_strata(n_rare: int = 100, n_common: int = 150) -> dict[str, str]:
    strata = {f"g{i:03d}": "rare" for i in range(n_rare)}
    strata.update({f"g{n_rare + i:03d}": "common" for i in range(n_common)})
    return strata


def test_partition_no_overlap_nothing_lost():
    strata = make_strata()
    result = split_groups(strata, RATIOS, seed=42)
    assert set(result) == {"train", "val", "test"}
    all_ids = result["train"] + result["val"] + result["test"]
    assert len(all_ids) == len(set(all_ids)), "группа попала в несколько сплитов"
    assert set(all_ids) == set(strata), "часть групп потеряна или появилась лишняя"


def test_deterministic_and_independent_of_input_order():
    strata = make_strata(30, 40)
    shuffled_items = list(strata.items())
    random.Random(1).shuffle(shuffled_items)
    shuffled = dict(shuffled_items)
    assert split_groups(strata, RATIOS, seed=7) == split_groups(strata, RATIOS, seed=7)
    assert split_groups(strata, RATIOS, seed=7) == split_groups(shuffled, RATIOS, seed=7)


def test_different_seed_gives_different_split():
    strata = make_strata()
    assert split_groups(strata, RATIOS, seed=1) != split_groups(strata, RATIOS, seed=2)


def test_overall_proportions():
    result = split_groups(make_strata(), RATIOS, seed=42)   # 250 групп
    assert abs(len(result["val"]) - 37.5) <= 3
    assert abs(len(result["test"]) - 37.5) <= 3
    assert abs(len(result["train"]) - 175) <= 6


def test_stratified_each_stratum_keeps_proportions():
    strata = make_strata(100, 150)
    result = split_groups(strata, RATIOS, seed=42)
    for stratum, n in (("rare", 100), ("common", 150)):
        for name, ratio in RATIOS.items():
            count = sum(1 for g in result[name] if strata[g] == stratum)
            assert abs(count - n * ratio) <= 1, f"{stratum}/{name}: {count} вместо ≈{n * ratio:.1f}"


def test_tiny_stratum_goes_to_train():
    strata = make_strata(1, 20)       # в страте rare одна группа: round(0.15) = 0
    result = split_groups(strata, RATIOS, seed=42)
    assert "g000" in result["train"]


def test_lists_are_sorted():
    result = split_groups(make_strata(), RATIOS, seed=42)
    for ids in result.values():
        assert ids == sorted(ids)


def test_zero_test_ratio_is_allowed():
    ratios = {"train": 0.85, "val": 0.15, "test": 0.0}
    result = split_groups(make_strata(), ratios, seed=42)
    assert result["test"] == []
    assert len(result["train"]) + len(result["val"]) == 250


def test_bad_ratios_raise():
    strata = make_strata(5, 5)
    with pytest.raises(ValueError):
        split_groups(strata, {"train": 0.5, "val": 0.2, "test": 0.2}, seed=1)   # сумма ≠ 1
    with pytest.raises(ValueError):
        split_groups(strata, {"train": 0.8, "val": 0.2}, seed=1)                # нет test
    with pytest.raises(ValueError):
        split_groups(strata, {"train": 1.2, "val": -0.1, "test": -0.1}, seed=1) # отрицательные
