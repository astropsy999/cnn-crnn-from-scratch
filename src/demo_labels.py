"""Демо: алфавиты, нормализация, кодирование и проверка формата."""
from alphabet import (
    ALPHABETS, BLANK_INDEX, decode_indices, encode_text, get_alphabet,
    normalize_label, num_classes, validate_format,
)

CASES = [
    ("7.3", "measurement"),
    ("12/14", "diameter"),
    ("12/14", "measurement"),
    ("01.02.24", "measurement"),
    ("7.345", "measurement"),
    ("1\u20132", "thickness"),
    ("", "measurement"),
]


def main() -> None:
    print(f"blank = {BLANK_INDEX}, символы нумеруются с 1\n")
    for task in ALPHABETS:
        print(f"{task:<12} алфавит {get_alphabet(task)!r:<16} символов: {len(get_alphabet(task)):>2}"
              f"  классов с blank: {num_classes(task)}")
    print()
    for raw, task in CASES:
        norm = normalize_label(raw)
        try:
            ok, reason = validate_format(norm, task)
        except NotImplementedError:
            print(
                "TODO в src/alphabet.py ещё не дописаны "
                "(validate_format / encode_text / FORMAT_PATTERNS).\n"
                "Сначала доведите тесты до зелёных:\n"
                "  python -m pytest tests/test_alphabet.py -q\n"
                "Демо — после 12 passed (или смотрите эталон lesson-03)."
            )
            raise SystemExit(1) from None
        head = f"{raw!r:<11} {task:<12} → {norm!r:<10}"
        if ok:
            try:
                idx = encode_text(norm, task)
                back = decode_indices(idx, task)
            except NotImplementedError:
                print(
                    "TODO в src/alphabet.py ещё не дописаны (encode_text / decode_indices).\n"
                    "Сначала: python -m pytest tests/test_alphabet.py -q"
                )
                raise SystemExit(1) from None
            assert back == norm
            print(f"{head} → {idx} → {back!r}  OK")
        else:
            print(f"{head} → REJECT: {reason}")


if __name__ == "__main__":
    main()
