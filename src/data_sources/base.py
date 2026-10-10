"""Контракт адаптера источника данных.

Любой источник (синтетика, публичные датасеты, реальные листы) приводится к одной схеме.
Остальной курс читает только all.jsonl и split.json и про источник ничего не знает.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, TypedDict

from alphabet import normalize_label, validate_format

REQUIRED_META = ("source", "task", "alphabet", "license", "image_root")


class Record(TypedDict):
    id: str          # уникальный идентификатор кропа
    image: str       # путь к картинке относительно image_root
    text: str        # нормализованная метка; '' для пустой клетки
    group_id: str    # единица сплита (лист, виртуальный лист, официальный сплит)
    task: str        # measurement | diameter | thickness
    is_empty: bool


class DataSource(ABC):
    @abstractmethod
    def iter_records(self) -> Iterator[Record]:
        """Отдаёт записи схемы Record."""

    @abstractmethod
    def meta(self) -> dict:
        """Словарь с ключами REQUIRED_META (+ любые дополнительные сведения об источнике)."""


def check_meta(meta: dict) -> list[str]:
    return [f"meta: нет ключа {k!r}" for k in REQUIRED_META if k not in meta]


def check_records(records, image_root, task: str, max_errors: int = 20) -> list[str]:
    """Проверяет записи на соответствие контракту. Возвращает список ошибок (пустой = всё хорошо)."""
    image_root = Path(image_root)
    errors: list[str] = []
    seen: set[str] = set()
    total_errors = 0

    def add(msg: str) -> None:
        nonlocal total_errors
        total_errors += 1
        if len(errors) < max_errors:
            errors.append(msg)

    for rec in records:
        rid = rec.get("id", "<без id>")
        missing = [k for k in ("id", "image", "text", "group_id", "task", "is_empty") if k not in rec]
        if missing:
            add(f"{rid}: нет полей {missing}")
            continue
        if rec["id"] in seen:
            add(f"{rid}: повторяющийся id")
        seen.add(rec["id"])
        if not rec["group_id"]:
            add(f"{rid}: пустой group_id")
        if rec["task"] != task:
            add(f"{rid}: task={rec['task']!r}, ожидалось {task!r}")
        if not isinstance(rec["is_empty"], bool):
            add(f"{rid}: is_empty должно быть bool")
        if rec["is_empty"] != (rec["text"] == ""):
            add(f"{rid}: is_empty не согласовано с text={rec['text']!r}")
        if rec["text"] != normalize_label(rec["text"]):
            add(f"{rid}: метка не нормализована: {rec['text']!r}")
        ok, reason = validate_format(rec["text"], rec["task"])
        if not ok:
            add(f"{rid}: {reason}: {rec['text']!r}")
        if not (image_root / rec["image"]).is_file():
            add(f"{rid}: нет файла {image_root / rec['image']}")
    if total_errors > len(errors):
        errors.append(f"... и ещё {total_errors - len(errors)} ошибок")
    return errors
