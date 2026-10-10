"""ОПЦИОНАЛЬНО: оценки P(empty) от логистической регрессии вместо CNN (нужен scikit-learn).

Зачем: чтобы пройти урок 8 целиком, не дожидаясь обучения сети, и воспроизвести пример из статьи.
Это НЕ часть курса: настоящие оценки даёт CNN (choose_threshold.py --checkpoint ...).
    pip install scikit-learn
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baselines import load_features  # noqa: E402
from split_groups import read_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out", default="experiments/surrogate_scores.npz")
    args = ap.parse_args()
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        sys.exit("Нужен scikit-learn: pip install scikit-learn (или используйте оценки настоящей CNN)")

    data_dir = Path(args.data_dir)
    root = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))["image_root"]
    splits = {name: read_jsonl(data_dir / f"{name}.jsonl") for name in ("train", "val", "test")}
    X = {name: load_features(recs, root) for name, recs in splits.items()}
    y = {name: np.array([int(r["is_empty"]) for r in recs]) for name, recs in splits.items()}
    model = LogisticRegression(max_iter=300).fit(X["train"], y["train"])
    out = {}
    for name in ("val", "test"):
        out[f"{name}_p_empty"] = model.predict_proba(X[name])[:, 1]
        out[f"{name}_labels"] = y[name]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    print(f"оценки логистической регрессии сохранены: {args.out}")


if __name__ == "__main__":
    main()
