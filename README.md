# Обучаем CNN и CRNN с нуля

Код к практической серии статей: от лаборатории PyTorch до детектора клеток, чтения строк и каскада.

- Статьи: [stuzhuk.page](https://stuzhuk.page/ru/blog/) / [stuzhuklab.ru](https://stuzhuklab.ru/blog/) (серия `cnn-crnn-from-scratch`)
- Состояние после каждого урока — **git-тег** `lesson-01`, `lesson-02`, …
- Актуальный снимок: [`lesson-02`](https://github.com/astropsy999/cnn-crnn-from-scratch/tree/lesson-02)

## Быстрый старт

```bash
git clone https://github.com/astropsy999/cnn-crnn-from-scratch.git
cd cnn-crnn-from-scratch
git checkout lesson-02

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

### Урок 2 — картинка → тензор

```bash
python src/make_demo_crops.py --out data/synthetic/raw
python src/image_to_tensor.py --image data/synthetic/raw/sample_7_3.png
python experiments/show_preprocess_grid.py --out experiments/preprocess_grid.png
python -m pytest tests/test_image_to_tensor.py -q
```

Подробнее: [`lessons/02.md`](lessons/02.md).
