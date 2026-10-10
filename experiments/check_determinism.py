"""Проверка воспроизводимости: один seed → одни и те же веса.

Считаем на CPU, чтобы не смешивать нашу проверку с недетерминизмом GPU.
"""
import argparse
import hashlib
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from seed import set_seed  # noqa: E402


def weights_hash(model: nn.Module) -> str:
    """Короткий «отпечаток» всех весов модели."""
    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.detach().cpu().numpy().tobytes())  # байты весов → в хэш
    return h.hexdigest()[:12]  # короткий отпечаток для печати


def train_tiny(steps: int, seed: int | None) -> tuple[str, float]:
    """Мини-обучение. seed=None: set_seed сознательно НЕ вызываем («забыл seed»)."""
    if seed is not None:
        set_seed(seed)  # фиксируем все источники случайности
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 2))  # крошечная сеть
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.randn(256, 16)  # синтетические признаки
    y = (x[:, 0] > 0).long()  # метка из знака первого признака
    loss = torch.tensor(0.0)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)  # ошибка классификации
        loss.backward()
        opt.step()
    return weights_hash(model), float(loss.detach())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=None,
                    help="число потоков CPU (1 помогает, если хэши не совпали)")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)  # на части CPU влияет на хэш

    h1, l1 = train_tiny(args.steps, args.seed)  # первый запуск с seed
    h2, l2 = train_tiny(args.steps, args.seed)  # тот же seed → отпечатки должны совпасть
    h3, _ = train_tiny(args.steps, args.seed + 1)  # другой seed → отпечаток другой

    set_seed(args.seed)                    # один раз в начале...
    h4, _ = train_tiny(args.steps, None)   # ...и дальше «забываем» переустанавливать
    h5, _ = train_tiny(args.steps, None)   # генератор уже сдвинут → другой результат

    same_ok = h1 == h2 and l1 == l2  # одинаковый seed → одинаковые веса и loss
    diff_ok = h1 != h3  # разный seed → проверка не «мёртвая»
    print(f"same seed ({args.seed}, {args.seed})      : {h1} {'==' if h1 == h2 else '!='} {h2}   "
          f"{'OK' if same_ok else 'FAIL'}")
    print(f"different seed ({args.seed}, {args.seed + 1}) : {h1} {'!=' if diff_ok else '=='} {h3}   "
          f"{'OK' if diff_ok else 'FAIL'}")
    print(f"forgot to reseed        : {h4} {'!=' if h4 != h5 else '=='} {h5}   (демо, не проверка)")

    if same_ok and diff_ok:
        print("determinism: OK (hash match)")
        return 0
    print("determinism: FAIL (см. раздел «Если не так» в уроке)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
