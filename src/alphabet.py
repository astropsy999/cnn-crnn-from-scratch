"""Алфавиты задач, контракт индексов и проверка формата меток.

Контракт индексов (общий для всего курса):
    blank = 0, символы алфавита нумеруются с 1 в порядке следования в строке.
"""
import re

BLANK_INDEX = 0

ALPHABETS = {
    "measurement": "0123456789,",
    "diameter": "0123456789/",
    "thickness": "0123456789,-",
}

# Формат метки поверх алфавита (пустая строка = пустая клетка, допустима всегда).
# Пишем [0-9], а не \d: \d в Python совпадает и с цифрами других письменностей.
FORMAT_PATTERNS = {
    "measurement": re.compile(r"[0-9]{1,3}(,[0-9]{1,2})?"),
    "diameter": re.compile(r"[0-9]{1,3}(/[0-9]{1,3})?"),
    "thickness": re.compile(r"[0-9]{1,2}(,[0-9]{1,2})?(-[0-9]{1,2}(,[0-9]{1,2})?)?"),
}

_DASHES = "\u2010\u2011\u2012\u2013\u2014\u2212"  # разные «тире» и минус


def get_alphabet(task: str) -> str:
    try:
        return ALPHABETS[task]
    except KeyError:
        raise ValueError(f"неизвестная задача {task!r}; доступны: {sorted(ALPHABETS)}") from None


def num_classes(task: str) -> int:
    """Размер выходного слоя модели: символы алфавита + blank."""
    return len(get_alphabet(task)) + 1


def normalize_label(text: str) -> str:
    """Приводит сырую метку к каноническому виду: без пробелов, ',' вместо '.', обычный '-'."""
    text = "".join(text.split())
    text = text.replace(".", ",")
    for dash in _DASHES:
        text = text.replace(dash, "-")
    return text


def encode_text(text: str, task: str) -> list[int]:
    """Строка → список индексов (blank = 0, символы с 1).

    Строку заранее нормализует вызывающий код: символ вне алфавита — ошибка.
    """
    alphabet = get_alphabet(task)
    indices = []
    for ch in text:
        pos = alphabet.find(ch)  # -1, если символа нет
        if pos < 0:  # иначе pos+1 даст 0 = blank — коварная ошибка
            raise ValueError(f"символ {ch!r} вне алфавита задачи {task!r}")
        indices.append(pos + 1)  # символы с 1, blank = 0
    return indices


def decode_indices(indices, task: str) -> str:
    """Список индексов → строка. Индекс 0 (blank) и выход за алфавит — ошибка."""
    alphabet = get_alphabet(task)
    chars = []
    for i in indices:
        if not 1 <= i <= len(alphabet):  # blank и «мусор» сюда не входят
            raise ValueError(f"индекс {i} недопустим для задачи {task!r} (допустимо 1..{len(alphabet)})")
        chars.append(alphabet[i - 1])  # индекс 1 → первый символ алфавита
    return "".join(chars)


def validate_format(text: str, task: str) -> tuple[bool, str]:
    """Проверяет метку: сначала алфавит, потом формат. Возвращает (ok, причина)."""
    alphabet = get_alphabet(task)
    for ch in text:
        if ch not in alphabet:  # сначала алфавит, потом формат
            return False, f"символ {ch!r} вне алфавита"
    if text == "":
        return True, ""
    if FORMAT_PATTERNS[task].fullmatch(text) is None:  # match() оставил бы хвост
        return False, "формат не соответствует задаче"
    return True, ""
