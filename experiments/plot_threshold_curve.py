"""Кривые выбора порога: recall по классам против порога и PR-кривая для класса empty."""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from choose_threshold import choose_threshold, load_scores  # noqa: E402


def curves(p: np.ndarray, y: np.ndarray, grid: np.ndarray) -> dict:
    value_recall, empty_recall, precision = [], [], []
    for t in grid:
        pred_empty = p >= t
        value_recall.append(float((~pred_empty[y == 0]).mean()))
        empty_recall.append(float(pred_empty[y == 1].mean()))
        declared = pred_empty.sum()
        precision.append(float((pred_empty & (y == 1)).sum() / declared) if declared else 1.0)
    return {"value_recall": np.array(value_recall), "empty_recall": np.array(empty_recall),
            "precision": np.array(precision)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help=".npz из choose_threshold.py --save-scores")
    ap.add_argument("--min-value-recall", type=float, default=0.99)
    ap.add_argument("--out", default="experiments/threshold_curve.png")
    ap.add_argument("--title", default="")
    ap.add_argument("--ymin", type=float, default=0.9,
                    help="нижняя граница осей recall/precision (0 — показать всё)")
    args = ap.parse_args()

    s = load_scores(args.scores)
    thr = choose_threshold(s["val_p_empty"], s["val_labels"], args.min_value_recall)["threshold"]
    grid = np.linspace(0, 1, 1001)
    val = curves(s["val_p_empty"], s["val_labels"], grid)
    test = curves(s["test_p_empty"], s["test_labels"], grid)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(grid, val["value_recall"], color="tab:blue", label="value_recall, val")
    a1.plot(grid, val["empty_recall"], color="tab:orange", label="empty_recall, val")
    a1.plot(grid, test["value_recall"], "--", color="tab:blue", alpha=0.7, label="value_recall, test")
    a1.plot(grid, test["empty_recall"], "--", color="tab:orange", alpha=0.7, label="empty_recall, test")
    a1.axvline(thr, color="k", lw=1, label=f"порог {thr:.3f} (выбран на val)")
    a1.axhline(args.min_value_recall, color="gray", ls=":", lw=1)
    a1.set_xlabel("порог по P(empty): пустым считаем, если P(empty) ≥ порога")
    a1.set_ylabel("recall")
    a1.set_ylim(args.ymin, 1.003)
    a1.legend(fontsize=7, loc="lower center")
    a1.set_title("Выбор порога (линия: порог с val; пунктир: те же кривые на test)", fontsize=8)

    for name, c, style in (("val", val, "-"), ("test", test, "--")):
        a2.plot(c["empty_recall"], c["precision"], style, label=f"{name}")
        i = int(np.argmin(np.abs(grid - thr)))
        a2.plot(c["empty_recall"][i], c["precision"][i], "ko")
    a2.set_xlabel("recall класса empty")
    a2.set_ylabel("precision класса empty")
    a2.set_ylim(args.ymin, 1.003)
    a2.set_xlim(args.ymin, 1.003)
    a2.legend(fontsize=8)
    a2.set_title("PR-кривая для empty (точка: выбранный порог)", fontsize=8)
    if args.title:
        fig.suptitle(args.title, fontsize=9)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"сохранено: {out}")


if __name__ == "__main__":
    main()
