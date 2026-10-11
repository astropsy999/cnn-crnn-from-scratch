"""Как ведёт себя CTC loss: хорошее предсказание, «всё blank», слишком короткая запись, типовые ошибки."""
import math

import numpy as np

from alphabet import encode_text
from ctc_from_scratch import ctc_neg_log_likelihood, log_softmax, min_time_steps

TASK = "measurement"
C = 12                                    # 11 символов + blank
TARGET_TEXT = "7,3"


def probs_from_path(path: list[int], sharpness: float = 0.9) -> np.ndarray:
    """Предсказание, уверенное в заданном пути: sharpness на выбранном классе, остальное поровну."""
    p = np.full((len(path), C), (1 - sharpness) / (C - 1))
    for t, c in enumerate(path):
        p[t, c] = sharpness
    return np.log(p)


def main() -> None:
    target = encode_text(TARGET_TEXT, TASK)                          # [8, 11, 4]
    print(f"цель {TARGET_TEXT!r} → индексы {target}; минимум шагов: {min_time_steps(TARGET_TEXT)}\n")

    aligned = probs_from_path([0, 8, 8, 0, 11, 0, 4, 4])             # «-77-,-33» схлопывается в «7,3»
    blank_only = probs_from_path([0] * 8)
    print("loss (отрицательный логарифм вероятности цели), T = 8:")
    print(f"  предсказание, совпадающее с целью: {ctc_neg_log_likelihood(aligned, target):7.3f}")
    print(f"  предсказание «всё blank»:          {ctc_neg_log_likelihood(blank_only, target):7.3f}   # намного больше")

    print("\nслишком короткая запись:")
    for T in (2, 3):
        lp = probs_from_path([0] * T)
        loss = ctc_neg_log_likelihood(lp, target)
        print(f"  T = {T}: loss = {loss}   ({'невозможно: шагов меньше min_time_steps' if math.isinf(loss) else 'возможно'})")

    repeated = encode_text("33", TASK)
    print(f"\nповтор требует blank: для цели '33' минимум {min_time_steps('33')} шага, а не 2:")
    for T in (2, 3):
        print(f"  T = {T}: loss = {ctc_neg_log_likelihood(probs_from_path([4] * T), repeated)}")

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("\ntorch не установлен: часть с nn.CTCLoss пропущена")
        return

    print("\nтиповые ошибки с nn.CTCLoss (результат разных вариантов на одном и том же примере):")
    T = aligned.shape[0]
    log_probs = torch.from_numpy(aligned).float()[:, None, :]                      # [T, B=1, C]
    targets = torch.tensor([target])
    in_len, tg_len = torch.tensor([T]), torch.tensor([len(target)])
    ctc = nn.CTCLoss(blank=0, reduction="none", zero_infinity=False)
    right = ctc(log_probs, targets, in_len, tg_len).item()
    print(f"  правильно:                          {right:8.3f}")
    raw = torch.from_numpy(np.exp(aligned)).float()[:, None, :]                    # забыли log_softmax: подали вероятности
    print(f"  без log_softmax (подали вероятности): {ctc(raw, targets, in_len, tg_len).item():8.3f}   # другое число, ошибки нет!")
    short = torch.tensor([2])
    print(f"  input_lengths меньше нужного:       {ctc(log_probs, targets, short, tg_len).item():8.3f}")
    ctc0 = nn.CTCLoss(blank=0, reduction="none", zero_infinity=True)
    print(f"  невозможная цель, zero_infinity=True: {ctc0(log_probs, targets, short, tg_len).item():8.3f}   # молча 0: образец выброшен из обучения")
    print("  форма [B, T, C] вместо [T, B, C]: переставьте оси (log_probs.transpose(0, 1)), иначе T и B перепутаются")


if __name__ == "__main__":
    main()
