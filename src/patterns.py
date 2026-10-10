"""Класс «паттерна» метки: нужен для статистики датасета и (в уроке 5) для стратификации листов."""


def label_pattern(text: str, task: str) -> str:
    if text == "":
        return "empty"
    if task == "measurement":
        if text.startswith("0,"):
            return "lead0"
        if "," not in text:
            return "int"
        return "dec1" if len(text.split(",")[1]) == 1 else "dec2"
    if task == "diameter":
        return "fraction" if "/" in text else "single"
    if task == "thickness":
        return "range" if "-" in text else "single"
    raise ValueError(f"неизвестная задача {task!r}")
