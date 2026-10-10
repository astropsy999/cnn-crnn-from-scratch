import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alphabet import (  # noqa: E402
    BLANK_INDEX, decode_indices, encode_text, get_alphabet,
    normalize_label, num_classes, validate_format,
)

ACCEPT = {
    "measurement": ["", "7", "7,3", "105", "0,85", "123,45", "007"],
    "diameter": ["", "57", "12/14", "120/80"],
    "thickness": ["", "10", "1-2", "1,5-2", "1,5-2,5", "12-15"],
}
REJECT_FORMAT = {
    "measurement": ["1234", "7,", ",5", "7,345", "7,3,1", "01,02,24", "1,2,3"],
    "diameter": ["12/", "/14", "12/14/16", "1234", "12/1234"],
    "thickness": ["-2", "1-", "1--2", "1-2-3", "123", "1,", "1,5,5"],
}
REJECT_ALPHABET = {
    "measurement": ["12/14", "7.3", "abc", "7 3"],
    "diameter": ["7,3", "7.3"],
    "thickness": ["7/3", "1_2"],
}


def test_blank_is_zero():
    assert BLANK_INDEX == 0
    assert encode_text("0", "measurement") == [1]


def test_num_classes():
    assert num_classes("measurement") == 12
    assert num_classes("diameter") == 12
    assert num_classes("thickness") == 13


def test_encode_examples():
    assert encode_text("7,3", "measurement") == [8, 11, 4]
    assert encode_text("12/14", "diameter") == [2, 3, 11, 2, 5]
    assert encode_text("1,5-2", "thickness") == [2, 11, 6, 12, 3]
    assert encode_text("", "measurement") == []


def test_encode_rejects_foreign_char():
    with pytest.raises(ValueError):
        encode_text("7.3", "measurement")      # точку нормализует вызывающий код
    with pytest.raises(ValueError):
        encode_text("12/14", "measurement")


def test_decode_roundtrip():
    for task, texts in ACCEPT.items():
        for text in texts:
            assert decode_indices(encode_text(text, task), task) == text


def test_decode_rejects_blank_and_out_of_range():
    with pytest.raises(ValueError):
        decode_indices([0], "measurement")      # blank — не символ
    with pytest.raises(ValueError):
        decode_indices([12], "measurement")     # в алфавите всего 11 символов
    with pytest.raises(ValueError):
        decode_indices([-1], "measurement")


def test_normalize_label():
    assert normalize_label("7.3") == "7,3"
    assert normalize_label(" 7, 3 ") == "7,3"
    assert normalize_label("1\u20132") == "1-2"
    assert normalize_label("01.02.24") == "01,02,24"
    assert normalize_label("") == ""


def test_validate_accepts():
    for task, texts in ACCEPT.items():
        for text in texts:
            ok, reason = validate_format(text, task)
            assert ok, f"{task}: {text!r} должна приниматься, а отклонена: {reason}"


def test_validate_rejects_by_format():
    for task, texts in REJECT_FORMAT.items():
        for text in texts:
            ok, reason = validate_format(text, task)
            assert not ok, f"{task}: {text!r} должна отклоняться"
            assert "формат" in reason, f"{task}: {text!r}: причина должна содержать «формат»"


def test_validate_rejects_by_alphabet():
    for task, texts in REJECT_ALPHABET.items():
        for text in texts:
            ok, reason = validate_format(text, task)
            assert not ok, f"{task}: {text!r} должна отклоняться"
            assert "алфавит" in reason, f"{task}: {text!r}: причина должна содержать «алфавит»"


def test_trap_passes_alphabet_but_not_format():
    # Дата после normalize_label целиком состоит из допустимых символов.
    text = normalize_label("01.02.24")
    assert all(ch in get_alphabet("measurement") for ch in text)
    ok, reason = validate_format(text, "measurement")
    assert not ok and "формат" in reason


def test_unknown_task():
    with pytest.raises(ValueError):
        get_alphabet("nope")
    with pytest.raises(ValueError):
        encode_text("1", "nope")
