"""Абляция: нужна ли балансировка классов? Варианты none / weights / sampler / both × несколько seed.

Выбор варианта делается по val, test только отчёт. Бэкенды:
  cnn     — настоящий empty-детектор (нужен torch; долго);
  logreg  — логистическая регрессия на пикселях (нужен scikit-learn; секунды): тот же стенд без сети.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ablation import VARIANTS, best_variant, format_table, summarize, verdict  # noqa: E402
from choose_threshold import choose_threshold, evaluate_threshold  # noqa: E402
from config import load_config  # noqa: E402
from imbalance import class_weights, parse_imbalance, sample_weights  # noqa: E402

_features: dict = {}


def run_cnn(cfg: dict, variant: str, seed: int) -> dict:
    from train_empty_detector import train  # импорт здесь: нужен torch
    run_cfg = dict(cfg, imbalance=variant, seed=seed,
                   out=str(Path("experiments/ablation") / f"{variant}_seed{seed}.pt"))
    return train(run_cfg, verbose=False)


def run_logreg(cfg: dict, variant: str, seed: int) -> dict:
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        sys.exit("Для --backend logreg нужен scikit-learn: pip install scikit-learn")
    from baselines import load_features
    from split_groups import read_jsonl

    data_dir = Path(cfg["data_dir"])
    if str(data_dir) not in _features:
        root = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))["image_root"]
        data = {}
        for split in ("train", "val", "test"):
            recs = read_jsonl(data_dir / f"{split}.jsonl")
            data[split] = (load_features(recs, root), np.array([int(r["is_empty"]) for r in recs]))
        _features[str(data_dir)] = data
    data = _features[str(data_dir)]

    X, y = data["train"]
    use_weights, use_sampler = parse_imbalance(variant)
    if use_sampler:                                   # те же веса, что у WeightedRandomSampler в CNN
        w = sample_weights(y)
        idx = np.random.default_rng(seed).choice(len(y), size=len(y), replace=True, p=w / w.sum())
        X, y = X[idx], y[idx]
    cw = class_weights(y) if use_weights else None
    clf = LogisticRegression(max_iter=300, class_weight=None if cw is None else {0: cw[0], 1: cw[1]}).fit(X, y)
    pv = clf.predict_proba(data["val"][0])[:, 1]
    pt = clf.predict_proba(data["test"][0])[:, 1]
    chosen = choose_threshold(pv, data["val"][1], cfg["min_value_recall"])
    val = evaluate_threshold(pv, data["val"][1], chosen["threshold"])
    test = evaluate_threshold(pt, data["test"][1], chosen["threshold"])
    half = evaluate_threshold(pv, data["val"][1], 0.5)             # решение «по умолчанию», без подбора порога
    return {"threshold": chosen["threshold"], "val_empty_recall": val["empty_recall"],
            "val_value_recall": val["value_recall"], "val_value_recall_at_half": half["value_recall"],
            "test_value_recall": test["value_recall"], "test_empty_recall": test["empty_recall"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/empty-detector.yaml")
    ap.add_argument("--seeds", type=int, default=3, help="сколько seed (от seed из конфига)")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--backend", choices=["cnn", "logreg"], default="cnn")
    ap.add_argument("--data-dir")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--max-batches", type=int)
    ap.add_argument("--out", default="experiments/ablation_imbalance.json")
    args = ap.parse_args()

    cfg = load_config(args.config, {"data_dir": args.data_dir, "epochs": args.epochs,
                                    "max_batches": args.max_batches})
    variants = [v.strip() for v in args.variants.split(",")]
    runner = run_cnn if args.backend == "cnn" else run_logreg
    results = []
    for variant in variants:
        for k in range(args.seeds):
            seed = cfg["seed"] + k
            r = runner(cfg, variant, seed)
            results.append({"variant": variant, "seed": seed, **r})
            print(f"{variant:<8} seed {seed}: порог {r['threshold']:.3f}  val value_recall@0.5 {r['val_value_recall_at_half']:.4f}  "
                  f"val empty_recall {r['val_empty_recall']:.4f}  test value_recall {r['test_value_recall']:.4f}  "
                  f"test empty_recall {r['test_empty_recall']:.4f}")

    summary = summarize(results)
    print(f"\nцель: value_recall >= {cfg['min_value_recall']} (порог выбран на val); данные: {cfg['data_dir']}; "
          f"бэкенд: {args.backend}\n")
    print(format_table(summary))
    print("\n" + verdict(summary))
    print(f"Победитель по val: {best_variant(summary)} (итоговые числа смотрим на test).")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": cfg, "backend": args.backend, "results": results, "summary": summary},
                              indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"результаты: {out}")


if __name__ == "__main__":
    main()
