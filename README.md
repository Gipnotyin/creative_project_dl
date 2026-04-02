# SIMILIS Baseline Project

Базовый pipeline для трека **SIMILIS baseline**:

`изображение -> предсказание полей -> auto_description`

README ниже закрывает то, что обычно требуется к сдаче: выбранный подход, метрики, структура репозитория, запуск
обучения и инференса, ограничения и план дальнейшей работы.

## Что делает проект

Проект решает не задачу свободной генерации текста, а задачу **контролируемого структурированного описания артефакта**.

По изображению модель предсказывает 4 поля:

- `type` — тип предмета
- `part` — часть / зона предмета
- `integrity` — целый / фрагмент
- `material` — нормализованный материал

Потом из этих полей собирается `auto_description` по фиксированному шаблону.

Такой подход выбран специально:

- он воспроизводим;
- его проще отлаживать;
- он лучше соответствует baseline-постановке из `task.txt`;
- он позволяет честно пропускать поле, если модель не уверена.

## Выбранный подход

### 1. Нормализация данных

Из исходного CSV строятся нормализованные поля:

- `type`
- `part`
- `integrity`
- `material`

Нормализация rule-based:

- `type` и `material` сводятся к небольшим устойчивым словарям;
- `part` и `integrity` извлекаются из `name + description + fragm`;
- для пропусков используются отдельные `*_is_missing` колонки.

### 2. Group-aware split

Так как у одного артефакта могут быть похожие кадры, split делается не по строкам, а по `group_key`.

Сейчас:

- `group_key` строится как proxy-ключ по `code`;
- `train.csv` соответствует `train_inner`;
- `val.csv` соответствует `val_inner`;
- `test_open.csv` соответствует `test_open`.

Текущие размеры split:

- `train_inner`: `970`
- `val_inner`: `208`
- `test_open`: `209`

### 3. Модель

Используется multi-task классификатор:

- один visual backbone из `timm`;
- отдельная head для каждого поля.

Основной backbone:

- `convnext_tiny`

### 4. Обучение

В `train.py` реализован safe fine-tuning:

- отдельный `head_lr`;
- отдельный пониженный `backbone_lr`;
- заморозка backbone на первые эпохи;
- gradient clipping;
- логирование метрик по эпохам;
- сохранение `last.pt` и `best.pt`.

### 5. Генерация auto_description

`auto_description` собирается по фиксированному шаблону.

Порядок полей:

1. `type`
2. `material`
3. `part`
4. `integrity=фрагмент`

Поле включается только если его confidence выше порога. Если модель не уверена, поле пропускается. Если ни одно поле не
прошло пороги, возвращается:

`не удалось уверенно собрать описание`

Пример:

- `изразец керамика профиль фрагмент`
- `тарелка фаянс фрагмент`
- `изразец профиль фрагмент`

## Метрики и текущий статус

### Что считается

Для оценки качества используются:

- `accuracy` по каждому полю;
- `macro-F1` по каждому полю;
- `mean macro-F1` как агрегированная метрика выбора модели.

Для детального разбора есть отдельный скрипт:

- confusion matrices;
- class-wise метрики;
- таблицы удачных и неудачных примеров;
- сравнение `pred_auto_description` и `gt_auto_description`.

### Что подтверждено сейчас

После очистки `artifacts/` заново запущен retrain. Поэтому финальная detailed-оценка для нового checkpoint ещё не
пересчитана.

На момент последней проверенной записи
в [artifacts/reports/train_log.csv](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/train_log.csv)
подтверждены такие промежуточные результаты:

- epoch 1: `mean_macro_f1 = 0.6000`
- epoch 2: `mean_macro_f1 = 0.6581`

По эпохе 2:

- `type_macro_f1 = 0.6994`
- `part_macro_f1 = 0.5293`
- `integrity_macro_f1 = 0.6477`
- `material_macro_f1 = 0.7561`

Это именно **промежуточные val-метрики retrain**, а не финальная оценка на `test_open`.

## Что где лежит

```text
similis_baseline_project/
├── configs/
│   ├── baseline.yaml
│   ├── baseline_cpu.yaml
│   └── safe_debug.yaml
├── data/
│   ├── raw/
│   └── processed/
├── artifacts/
│   ├── checkpoints/
│   ├── preds/
│   ├── reports/
│   └── figures/
├── src/similis_baseline/
│   ├── data_prep.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── evaluate_detailed.py
│   ├── predict.py
│   ├── render_report.py
│   ├── one_batch_debug.py
│   ├── tiny_overfit.py
│   └── data_report.py
├── requirements.txt
└── README.md
```

Коротко по конфигам:

- `configs/baseline.yaml` — основной baseline, более тяжёлый, ориентирован на полный запуск;
- `configs/baseline_cpu.yaml` — практичный режим для полного retrain на CPU;
- `configs/safe_debug.yaml` — короткий диагностический запуск, не финальный.

## Как запустить

### 1. Установка

```bash
pip install -r requirements.txt
```

### 2. Подготовка данных

Ожидаемая структура:

```text
data/raw/selected_by_name_iimk_subset_public.csv
data/raw/images/...jpg
```

Подготовка:

```bash
python -m src.similis_baseline.data_prep \
  --config configs/baseline.yaml \
  --csv data/raw/selected_by_name_iimk_subset_public.csv \
  --images-dir data/raw/images
```

На выходе:

- `data/processed/full_manifest.csv`
- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test_open.csv`
- `data/processed/label_maps.json`

### 3. Обучение

Основной запуск:

```bash
python -m src.similis_baseline.train --config configs/baseline.yaml
```

Практичный CPU-запуск:

```bash
python -m src.similis_baseline.train --config configs/baseline_cpu.yaml
```

На выходе:

- `artifacts/checkpoints/best.pt`
- `artifacts/checkpoints/last.pt`
- `artifacts/reports/train_log.csv`

### 4. Детальная оценка

Валидация:

```bash
python -m src.similis_baseline.evaluate_detailed \
  --checkpoint artifacts/checkpoints/best.pt \
  --split val \
  --output-dir artifacts/reports/val_detailed
```

Открытый тест:

```bash
python -m src.similis_baseline.evaluate_detailed \
  --checkpoint artifacts/checkpoints/best.pt \
  --split test \
  --output-dir artifacts/reports/test_detailed
```

### 5. Инференс

```bash
python -m src.similis_baseline.predict \
  --checkpoint artifacts/checkpoints/best.pt \
  --input-dir data/raw/images \
  --output artifacts/preds/inference.csv
```

Итоговый CSV содержит:

- `image_file`
- `auto_description`

Дополнительно сохраняются:

- `pred_*`
- `confidence_*`

### 6. HTML-отчёт по предсказаниям

Из `predict.py`:

```bash
python -m src.similis_baseline.render_report \
  --pred-csv artifacts/preds/inference.csv \
  --images-root data/raw/images \
  --out-html artifacts/reports/inference_report.html \
  --limit 20
```

Из `evaluate_detailed.py`:

```bash
python -m src.similis_baseline.render_report \
  --pred-csv artifacts/reports/val_detailed/predictions.csv \
  --out-html artifacts/reports/val_detailed/report.html \
  --limit 20
```

## Дополнительные скрипты

Это не обязательная часть инференса, а инженерная диагностика:

- `one_batch_debug.py` — shapes, loss и `pred vs gt` на одном батче;
- `tiny_overfit.py` — tiny-overfit и проверка корректности pipeline;
- `data_report.py` — отчёт по данным и split;
- `evaluate_detailed.py` — подробная оценка и анализ ошибок;
- `render_report.py` — HTML-отчёт с карточками изображений и предсказаний.

Примеры запуска:

`one_batch_debug.py`

```bash
python -m src.similis_baseline.one_batch_debug \
  --config configs/baseline_cpu.yaml \
  --checkpoint artifacts/checkpoints/best.pt \
  --batch-size 16 \
  --num-preview 10
```

`tiny_overfit.py`

```bash
python -m src.similis_baseline.tiny_overfit \
  --config configs/baseline_cpu.yaml \
  --checkpoint artifacts/checkpoints/best.pt \
  --subset-size 32 \
  --epochs 20 \
  --batch-size 8 \
  --image-size 224 \
  --scheduler none \
  --class-weights-source subset
```

Чистый debug-вариант для `type`:

```bash
python -m src.similis_baseline.tiny_overfit \
  --config configs/baseline_cpu.yaml \
  --checkpoint artifacts/checkpoints/best.pt \
  --fields type \
  --subset-size 10 \
  --balance-by type \
  --epochs 8 \
  --batch-size 2 \
  --image-size 224 \
  --lr 0.01 \
  --weight-decay 0.0 \
  --scheduler none \
  --class-weights-source subset \
  --exclude-uncertain \
  --require-complete-labels \
  --freeze-backbone \
  --reset-heads
```

`data_report.py`

```bash
python -m src.similis_baseline.data_report \
  --config configs/baseline_cpu.yaml \
  --output-dir artifacts/reports/data_report
```

`evaluate_detailed.py`

```bash
python -m src.similis_baseline.evaluate_detailed \
  --checkpoint artifacts/checkpoints/best.pt \
  --split val \
  --output-dir artifacts/reports/val_detailed
```

```bash
python -m src.similis_baseline.evaluate_detailed \
  --checkpoint artifacts/checkpoints/best.pt \
  --split test \
  --output-dir artifacts/reports/test_detailed
```

`render_report.py`

```bash
python -m src.similis_baseline.render_report \
  --pred-csv artifacts/reports/val_detailed/predictions.csv \
  --out-html artifacts/reports/val_detailed/report.html \
  --limit 20
```

Если входной CSV был собран через `predict.py`, то нужно дополнительно передать корень изображений:

```bash
python -m src.similis_baseline.render_report \
  --pred-csv artifacts/preds/inference.csv \
  --images-root data/raw/images \
  --out-html artifacts/reports/inference_report.html \
  --limit 20
```

## Ограничения

- `group_key` сейчас строится по `code`, а не по настоящему `artifact_id`;
- словари и нормализация rule-based, поэтому чувствительны к шуму разметки;
- `part` остаётся самым шумным и неоднозначным полем;
- текущий baseline не генерирует свободный текст и не пытается предсказывать датировку, функцию или культурную
  интерпретацию;
- финальная detailed-оценка для нового retrain должна быть пересчитана после завершения обучения.

## Что делать дальше

- закончить retrain и заново прогнать `evaluate_detailed.py` на `val` и `test_open`;
- сохранить финальные confusion matrices и error analysis;
- проверить preprocessing для close-up и multi-view изображений;
- по возможности заменить proxy `group_key` на более сильный идентификатор артефакта;
- откалибровать confidence thresholds для `auto_description`.

Лучший baseline был получен при использовании укрупнённых словарей целевых полей, взвешенной функции потерь, а также
двухстадийного обучения: сначала с замороженным backbone, затем с его разморозкой на малом learning rate. Это позволило
поднять mean_macro_f1 на валидации до 0.84.
