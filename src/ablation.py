"""Сводка абляции: mean ± std по seed, выбор варианта по val и честный вердикт."""
import math

import numpy as np

VARIANTS = ("none", "weights", "sampler", "both")      # порядок = «от простого к сложному»
METRICS = ("val_empty_recall", "val_value_recall_at_half", "test_value_recall", "test_empty_recall")


def summarize(results: list[dict]) -> dict:
    """results: строки {variant, seed, <метрики>}. Возвращает {variant: {n, метрика: {mean, std}}}."""
    summary: dict = {}
    for variant in sorted({r["variant"] for r in results}, key=lambda v: VARIANTS.index(v) if v in VARIANTS else 99):
        rows = [r for r in results if r["variant"] == variant]
        entry = {"n": len(rows)}
        for m in METRICS:
            values = np.array([r[m] for r in rows], dtype=float)
            entry[m] = {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else float("nan")}
        summary[variant] = entry
    return summary


def best_variant(summary: dict, metric: str = "val_empty_recall") -> str:
    """Лучший вариант по val. При ничьей берётся более простой (none < weights < sampler < both)."""
    best, best_value = None, -math.inf
    for variant in sorted(summary, key=lambda v: VARIANTS.index(v) if v in VARIANTS else 99):
        value = summary[variant][metric]["mean"]
        if value > best_value + 1e-12:
            best, best_value = variant, value
    return best


def verdict(summary: dict, metric: str = "val_empty_recall") -> str:
    best = best_variant(summary, metric)
    if best == "none":
        return "Лучший по val вариант — none: балансировка классов не нужна."
    if "none" not in summary:
        return f"Лучший по val вариант — {best} (варианта none в сравнении нет)."
    diff = summary[best][metric]["mean"] - summary["none"][metric]["mean"]
    stds = [s for s in (summary[best][metric]["std"], summary["none"][metric]["std"]) if not math.isnan(s)]
    noise = max(stds) if stds else 0.0
    if diff <= noise:
        return (f"Лучший по val вариант — {best}, но выигрыш у none {diff * 100:.2f} п.п. не больше "
                f"разброса между seed ({noise * 100:.2f} п.п.): эффекта не видно, берём более простой none.")
    return (f"Лучший по val вариант — {best}: выигрыш у none {diff * 100:.2f} п.п. больше "
            f"разброса между seed ({noise * 100:.2f} п.п.): эффект есть.")


def format_table(summary: dict) -> str:
    def cell(d: dict) -> str:
        std = "—" if math.isnan(d["std"]) else f"{d['std']:.4f}"
        return f"{d['mean']:.4f} ± {std}"
    lines = [f"{'вариант':<9}{'seed':>5}{'val value_recall@0.5':>23}{'val empty_recall':>20}"
             f"{'test value_recall':>22}{'test empty_recall':>22}"]
    for variant, e in summary.items():
        lines.append(f"{variant:<9}{e['n']:>5}{cell(e['val_value_recall_at_half']):>23}{cell(e['val_empty_recall']):>20}"
                     f"{cell(e['test_value_recall']):>22}{cell(e['test_empty_recall']):>22}")
    return "\n".join(lines)
