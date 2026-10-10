# Обучаем CNN и CRNN с нуля

Код к практической серии статей: от лаборатории PyTorch до детектора клеток, чтения строк и каскада.

- Статьи: [stuzhuk.page](https://stuzhuk.page/ru/blog/) / [stuzhuklab.ru](https://stuzhuklab.ru/blog/) (серия `cnn-crnn-from-scratch`)
- Состояние после каждого урока — **git-тег** `lesson-01`, `lesson-02`, …
- Актуальный снимок: [`lesson-01`](https://github.com/astropsy999/cnn-crnn-from-scratch/tree/lesson-01)

## Быстрый старт

```bash
git clone https://github.com/astropsy999/cnn-crnn-from-scratch.git
cd cnn-crnn-from-scratch
git checkout lesson-01

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Сборку `torch` при необходимости берите со [страницы PyTorch](https://pytorch.org/get-started/locally/). Курс рассчитан на CPU.

### Урок 1 — лаборатория

```bash
python src/check_environment.py
python experiments/check_determinism.py
```

Подробнее: [`lessons/01.md`](lessons/01.md).
