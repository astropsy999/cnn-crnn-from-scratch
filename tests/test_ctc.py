import itertools
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ctc_from_scratch import collapse, min_time_steps  # noqa: E402


def test_collapse_examples_from_the_lesson():
    assert collapse("-aa-b-") == "ab"
    assert collapse("7-,-33") == "7,3"
    assert collapse("7-77-,-33") == "77,3"          # blank между семёрками сохраняет обе


def test_collapse_edge_cases():
    assert collapse("") == ""
    assert collapse("---") == ""
    assert collapse("aaa") == "a"
    assert collapse("ab") == "ab"
    assert collapse("a-a") == "aa"
    assert collapse("-a-a-") == "aa"
    assert collapse("aa-aa") == "aa"


def test_blank_separates_repeats():
    assert collapse("77") == "7"
    assert collapse("7-7") == "77"


def test_order_matters_blanks_are_not_removed_first():
    # Если сначала выбросить blank, а потом склеить повторы, получится "7,3": это ошибка.
    assert collapse("7-77-,-33") != "7,3"


def test_works_with_index_lists_and_keeps_the_type():
    assert collapse([0, 1, 1, 0, 1, 2, 2, 0], blank=0) == [1, 1, 2]
    assert isinstance(collapse([0, 1, 0], blank=0), list)
    assert isinstance(collapse("-a-"), str)


def test_custom_blank_symbol():
    assert collapse("_aa_b_", blank="_") == "ab"
    assert collapse("a-a", blank="_") == "a-a"      # '-' больше не blank


def test_min_time_steps_examples():
    assert min_time_steps("") == 0
    assert min_time_steps("7") == 1
    assert min_time_steps("73") == 2
    assert min_time_steps("77") == 3
    assert min_time_steps("7,33") == 5
    assert min_time_steps("777") == 5
    assert min_time_steps("1,11") == 5
    assert min_time_steps("12/14") == 5
    assert min_time_steps("0,00") == 5


def test_min_time_steps_is_exact_by_brute_force():
    # Эталон: перебрать все пути над алфавитом {a, b, blank} и найти самую короткую запись нужной строки.
    def shortest(text: str) -> int:
        for T in range(0, 2 * len(text) + 2):
            for path in itertools.product("ab-", repeat=T):
                if collapse("".join(path)) == text:
                    return T
        raise AssertionError("строка недостижима")

    for length in range(0, 5):
        for chars in itertools.product("ab", repeat=length):
            text = "".join(chars)
            assert min_time_steps(text) == shortest(text), text


def test_a_path_of_min_length_exists_and_collapses_to_the_text():
    for text in ["7,33", "77", "1,11", "0,00", "123", "11,22"]:
        path = ""
        for i, ch in enumerate(text):
            path += ch
            if i + 1 < len(text) and text[i + 1] == ch:
                path += "-"                        # blank только между одинаковыми соседями
        assert len(path) == min_time_steps(text)
        assert collapse(path) == text
