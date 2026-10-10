# Обучаем CNN и CRNN с нуля

Код к практической серии статей: от лаборатории PyTorch до детектора клеток, чтения строк и каскада.

- Статьи: [stuzhuk.page](https://stuzhuk.page/ru/blog/) / [stuzhuklab.ru](https://stuzhuklab.ru/blog/) (серия `cnn-crnn-from-scratch`)
- Состояние после каждого урока — **git-тег** `lesson-01`, `lesson-02`, …
- Актуальный снимок: [`lesson-08`](https://github.com/astropsy999/cnn-crnn-from-scratch/tree/lesson-08)

## Быстрый старт

```bash
git clone https://github.com/astropsy999/cnn-crnn-from-scratch.git
cd cnn-crnn-from-scratch
git checkout lesson-08

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

### Урок 3 — алфавит, лейблы, формат

```bash
git checkout lesson-03-start   # упражнение «напиши сам»
python -m pytest tests/test_alphabet.py -q

git checkout lesson-03
python src/demo_labels.py
python -m pytest tests/test_alphabet.py -q
```

Подробнее: [`lessons/03.md`](lessons/03.md).

### Урок 4 — синтетический генератор

```bash
python src/make_synthetic_crops.py --out data/synthetic --sheets 250 --per-sheet 20 --seed 42
python src/prepare_dataset.py --source synthetic --src data/synthetic --out data/processed
python experiments/show_samples_grid.py --out experiments/samples_grid.png
python -m pytest tests/test_data_source_contract.py tests/test_synthetic_generator.py -q
```

Подробнее: [`lessons/04.md`](lessons/04.md).

### Урок 5 — сплит по листам

```bash
# нужны data/processed из урока 4
python src/split_groups.py --src data/processed --out data/processed --val-ratio 0.15 --test-ratio 0.15
python src/verify_split.py --data-dir data/processed
python experiments/leakage_demo.py
python -m pytest tests/test_split_groups.py tests/test_baselines.py tests/test_verify_split.py -q
```

Подробнее: [`lessons/05.md`](lessons/05.md).

### Урок 6 — Dataset и аугментации

```bash
python src/make_demo_crops.py --out data/synthetic/raw
python experiments/show_augment_grid.py --image data/synthetic/raw/sample_7_3.png --style default --out experiments/aug_default.png
python experiments/show_augment_grid.py --image data/synthetic/raw/sample_12_14.png --style printed --out experiments/aug_printed.png
python -m pytest tests/test_augment.py tests/test_dataset.py -q
```

Подробнее: [`lessons/06.md`](lessons/06.md).

### Урок 7 — Tiny CNN

```bash
python src/conv_from_scratch.py
python src/compare_baselines.py --data-dir data/processed
python src/train_tiny_cnn.py --data-dir data/processed --overfit-batch
python src/train_tiny_cnn.py --data-dir data/processed --epochs 10 --out checkpoints/tiny_cnn.pt
python -m pytest tests/test_conv_from_scratch.py tests/test_baselines_simple.py tests/test_metrics.py tests/test_tiny_cnn.py -q
```

Подробнее: [`lessons/07.md`](lessons/07.md).

### Урок 8 — порог решения

```bash
# упражнение
git checkout lesson-08-start
python -m pytest tests/test_choose_threshold.py -q

# решение
git checkout lesson-08
python src/train_tiny_cnn.py --data-dir data/processed --epochs 10 --out checkpoints/tiny_cnn.pt
python src/choose_threshold.py --checkpoint checkpoints/tiny_cnn.pt --data-dir data/processed --save-scores experiments/tiny_cnn_scores.npz
python -m pytest tests/test_choose_threshold.py tests/test_stats.py tests/test_threshold_report.py -q
```

Подробнее: [`lessons/08.md`](lessons/08.md).

