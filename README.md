# SIMILIS Baseline Project

Базовый pipeline для задачи:

`изображение -> предсказание полей -> auto_description`

Проект решает baseline-постановку из `task.txt` как **multi-task классификацию нескольких визуально наблюдаемых полей**
с последующей **шаблонной сборкой описания**.

Подробный воспроизводимый отчёт собран в [REPORT.md](/Users/gipnotyin/Downloads/similis_baseline_project/REPORT.md).

## Что предсказывает модель

Из изображения предсказываются 4 нормализованных поля:

- `type`
- `part`
- `integrity`
- `material`

Потом из них собирается `auto_description`.

Правило сборки сейчас такое:

1. `type`
2. `material`
3. `part`
4. `integrity=фрагмент`

Поле включается только если confidence выше порога. Если ни одно поле не прошло порог, возвращается:

`не удалось уверенно собрать описание`

Примеры:

- `изразец керамика профиль фрагмент`
- `тарелка фаянс фрагмент`
- `изразец профиль фрагмент`

## Данные и split

Сплит делается через `GroupShuffleSplit` по `group_key`, чтобы при появлении повторов одного артефакта они не разъезжались между train/val/test.

Текущее соответствие:

- `data/processed/train.csv` -> `train_inner`
- `data/processed/val.csv` -> `val_inner`
- `data/processed/test_open.csv` -> `test_open`

Размеры:

- `train_inner`: `970`
- `val_inner`: `208`
- `test_open`: `209`

В этом открытом корпусе **повторов одного и того же артефакта нет**: все 1387 строк имеют уникальный `code`, и никакая
комбинация других колонок (`name`, `description`, `cultlayer+execorg+survyear`) не даёт реальных артефакт-уровневых
повторов — совпадения в `description` оказываются разными находками с одинаковой обобщённой формулировкой
(например, "Тарелки фаянсовой профиль" встречается у 11 разных артефактов из разных раскопок).

Поэтому `group_key=code` фактически эквивалентен row-level split: leakage по повторам невозможен, потому что повторов
нет. Это ограничение **самих данных**, а не split-логики. Когда в корпус попадут предметы с настоящим `artifact_id`,
group-aware pipeline сработает без изменений в коде.

Воспроизводимость:

- повторный `data_prep` с тем же `seed=42` даёт те же split’ы;
- проверка пересечения групп в `artifacts/reports/data_report/report.json`: `train ∩ val = 0`, `train ∩ test = 0`,
  `val ∩ test = 0`, `repeated_group_key_count = 0`;
- в `splits/` лежат alias-файлы `train_inner.csv`, `val_inner.csv`, `test_open.csv`.

## Модель и обучение

Модель:

- backbone: `convnext_tiny`
- heads: отдельная head на каждое поле

Сводка по параметрам лежит в
[artifacts/reports/model_summary.json](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/model_summary.json):

- всего параметров: `27,834,739`
- backbone: `27,820,128`
- heads: `14,611`

В `train.py` реализован safe fine-tuning:

- отдельный `head_lr`
- пониженный `backbone_lr`
- заморозка backbone на первые эпохи
- gradient clipping
- cosine schedule по двум param groups
- сохранение `last.pt` и `best.pt`

## Подтверждённые метрики

Текущий основной checkpoint:

- `artifacts/checkpoints/best.pt` (~334 MB, ConvNeXt-Tiny @384). Файл не хранится в git из-за размера; ссылка на скачивание — см. ниже секцию **Веса модели**.

Он получен из чистого 9-эпохного retrain на `configs/baseline.yaml` (image_size=384, ConvNeXt-Tiny, MPS); выбран по
`val mean_macro_f1` и соответствует `epoch=7`. Полный лог обучения лежит в
[artifacts/reports/train_log.csv](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/train_log.csv)
и **синхронизирован с этим checkpoint** (предыдущий рассинхронизированный лог сохранён в
`artifacts/archive_pre_clean_retrain/`).

Валидация:

- `mean_macro_f1 = 0.8191`
- `type_macro_f1 = 0.8419`
- `part_macro_f1 = 0.7710`
- `integrity_macro_f1 = 0.7518`
- `material_macro_f1 = 0.9117`

Источник:
[artifacts/reports/val_detailed/metrics.json](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/val_detailed/metrics.json)

Открытый тест:

- `mean_macro_f1 = 0.7023`
- `type_macro_f1 = 0.7556`
- `part_macro_f1 = 0.5890`
- `integrity_macro_f1 = 0.6240`
- `material_macro_f1 = 0.8404`

Источник:
[artifacts/reports/test_detailed/metrics.json](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/test_detailed/metrics.json)

Графики train_log:

- [artifacts/reports/train_log_report/loss_curve.png](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/train_log_report/loss_curve.png)
- [artifacts/reports/train_log_report/macro_f1_curve.png](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/train_log_report/macro_f1_curve.png)
- [artifacts/reports/train_log_report/lr_curve.png](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/train_log_report/lr_curve.png)

## Ablation Study

Для короткого сравнимого CPU-бюджета были прогнаны 3 эксперимента с одинаковым backbone, `image_size=224`, `epochs=4`,
`freeze_backbone_epochs=1`.

Итоговая таблица лежит в:

- [artifacts/ablations/report/ablation_results.csv](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/ablations/report/ablation_results.csv)
- [artifacts/ablations/report/ablation_summary.md](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/ablations/report/ablation_summary.md)

Результаты:

- `pad + class_weights`: `val=0.7021`, `test=0.7010`
- `pad + no_class_weights`: `val=0.7331`, `test=0.6163`
- `stretch + class_weights`: `val=0.7676`, `test=0.6712`

Короткий вывод:

- на коротком `val`-бюджете лучшим оказался `stretch`;
- на `test_open` лучшую переносимость показал `pad + class_weights`;
- поэтому финальный основной baseline в проекте оставлен в safe-варианте `pad`, а результаты `stretch` трактуются как
  полезный, но пока нестабильный сигнал.

## Проверки воспроизводимости

Есть отдельный артефакт, который проверяет `reload checkpoint -> same prediction`:

- [artifacts/reports/checkpoint_roundtrip.json](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/checkpoint_roundtrip.json)

В нём показано, что после повторной загрузки checkpoint logits совпадают по всем полям.

## Data-Centric Артефакты

Расширенный отчёт по данным лежит в:

- [artifacts/reports/data_report/report.json](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/data_report/report.json)
- [artifacts/reports/data_report/image_heuristics.csv](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/data_report/image_heuristics.csv)
- [artifacts/reports/data_report/field_candidates.csv](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/data_report/field_candidates.csv)
- [artifacts/reports/data_report/normalization_examples.csv](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/data_report/normalization_examples.csv)
- [artifacts/reports/data_report/label_policy.csv](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/data_report/label_policy.csv)

Примеры:

- `row + image`: [row_image_examples.png](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/data_report/row_image_examples.png)
- проблемные изображения: [problematic_images.png](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/data_report/problematic_images.png)
- visual modes: [layout_mode_examples.png](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/data_report/layout_mode_examples.png)

Что видно по отчёту:

- `label_is_uncertain`: `68` строк
- `layout_mode`: `single_object 75.5%`, `multi_view 21.8%`, `close_up 2.7%`
- `bg_type`: `white_uniform 74.7%`, `light_photo 23.2%`, `dark_or_complex 2.1%`
- `has_overlay_text`: около `9.8%`
- `has_scale_bar`: почти не встречается в открытом наборе

Факторный error analysis лежит в:

- [artifacts/reports/val_error_factors/factor_breakdown.csv](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/val_error_factors/factor_breakdown.csv)
- [artifacts/reports/test_error_factors/factor_breakdown.csv](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/test_error_factors/factor_breakdown.csv)
- [artifacts/reports/val_error_factors/error_reason_breakdown.csv](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/val_error_factors/error_reason_breakdown.csv)
- [artifacts/reports/test_error_factors/error_reason_breakdown.csv](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/test_error_factors/error_reason_breakdown.csv)

Короткий вывод:

- на `val` ошибки заметно чаще на `light_photo` и особенно `dark_or_complex` фоне;
- `multi_view` сильнее бьёт по `auto_description_match`, чем `single_object`;
- по heuristic taxonomy среди ошибок отдельно видны `layout_complexity`, `uncertain_or_label_noise` и
  `visual_ambiguity_low_confidence`.

Примеры осторожного поведения модели лежат в:

- [artifacts/reports/val_cautious_examples/cautious_examples.csv](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/val_cautious_examples/cautious_examples.csv)
- [artifacts/reports/test_cautious_examples/cautious_examples.csv](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/test_cautious_examples/cautious_examples.csv)

## Структура репозитория

```text
similis_baseline_project/
├── configs/
│   ├── baseline.yaml
│   ├── baseline_cpu.yaml
│   ├── safe_debug.yaml
│   ├── ablation_cpu_base.yaml
│   ├── ablation_cpu_no_class_weights.yaml
│   └── ablation_cpu_stretch.yaml
├── data/
│   ├── raw/
│   └── processed/
├── artifacts/
│   ├── checkpoints/
│   ├── preds/
│   ├── reports/
│   ├── figures/
│   └── ablations/
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
│   ├── data_report.py
│   ├── transform_report.py
│   ├── safe_crop_demo.py
│   ├── error_factor_report.py
│   ├── train_log_report.py
│   ├── cautious_examples.py
│   ├── image_analysis.py
│   └── ablation_report.py
├── scripts/
│   ├── build_baseline_notebook.py
│   ├── post_retrain.sh
│   └── regen_checkpoint_artifacts.py
├── baseline_report.ipynb
├── requirements.txt
├── README.md
└── REPORT.md
```

Коротко по конфигам:

- `configs/baseline.yaml` — основной baseline, `image_size=384`
- `configs/baseline_cpu.yaml` — более практичный запуск для CPU
- `configs/safe_debug.yaml` — короткий диагностический запуск
- `configs/ablation_cpu_*.yaml` — сравнимые CPU-эксперименты для ablation study

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

Короткая ablation-study:

```bash
python -m src.similis_baseline.train --config configs/ablation_cpu_base.yaml
python -m src.similis_baseline.train --config configs/ablation_cpu_no_class_weights.yaml
python -m src.similis_baseline.train --config configs/ablation_cpu_stretch.yaml
python -m src.similis_baseline.ablation_report \
  --configs configs/ablation_cpu_base.yaml configs/ablation_cpu_no_class_weights.yaml configs/ablation_cpu_stretch.yaml \
  --output-dir artifacts/ablations/report
```

### 4. Оценка

Быстрая оценка:

```bash
python -m src.similis_baseline.evaluate \
  --checkpoint artifacts/checkpoints/best.pt \
  --split val \
  --output artifacts/reports/val_metrics.json
```

Подробная оценка:

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
- `pred_*`
- `confidence_*`

### 6. HTML-отчёт по предсказаниям

Из `predict.py`:

```bash
python -m src.similis_baseline.render_report \
  --pred-csv artifacts/preds/inference.csv \
  --images-root data/raw/images \
  --out-html artifacts/reports/inference_report.html \
  --image-mode auto \
  --limit 20
```

Из `evaluate_detailed.py`:

```bash
python -m src.similis_baseline.render_report \
  --pred-csv artifacts/reports/val_detailed/predictions.csv \
  --out-html artifacts/reports/val_detailed/report.html \
  --image-mode auto \
  --limit 20
```

Для больших отчётов лучше использовать `--image-mode link`.

Примеры:

Пагинация:

```bash
python -m src.similis_baseline.render_report \
  --pred-csv artifacts/reports/val_detailed/predictions.csv \
  --out-html artifacts/reports/val_detailed/report_page_2.html \
  --image-mode link \
  --offset 200 \
  --limit 200
```

Только ошибки:

```bash
python -m src.similis_baseline.render_report \
  --pred-csv artifacts/reports/val_detailed/predictions.csv \
  --out-html artifacts/reports/val_detailed/errors.html \
  --image-mode link \
  --filter-col any_error \
  --filter-value 1 \
  --sort-by num_field_errors \
  --sort-desc \
  --limit 200
```

Сортировка по confidence:

```bash
python -m src.similis_baseline.render_report \
  --pred-csv artifacts/reports/val_detailed/predictions.csv \
  --out-html artifacts/reports/val_detailed/low_confidence.html \
  --image-mode link \
  --sort-by mean_confidence \
  --limit 200
```

## Дополнительные диагностические скрипты

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

`transform_report.py`

```bash
python -m src.similis_baseline.transform_report \
  --config configs/baseline.yaml \
  --output-dir artifacts/reports/transform_report
```

`error_factor_report.py`

```bash
python -m src.similis_baseline.error_factor_report \
  --pred-csv artifacts/reports/val_detailed/predictions.csv \
  --split-name val \
  --output-dir artifacts/reports/val_error_factors
```

```bash
python -m src.similis_baseline.error_factor_report \
  --pred-csv artifacts/reports/test_detailed/predictions.csv \
  --split-name test \
  --output-dir artifacts/reports/test_error_factors
```

`train_log_report.py`

```bash
python -m src.similis_baseline.train_log_report \
  --train-log artifacts/ablations/base/reports/train_log.csv \
  --output-dir artifacts/ablations/base/reports/train_log_report
```

`cautious_examples.py`

```bash
python -m src.similis_baseline.cautious_examples \
  --checkpoint artifacts/checkpoints/best.pt \
  --pred-csv artifacts/reports/val_detailed/predictions.csv \
  --output-dir artifacts/reports/val_cautious_examples
```

`ablation_report.py`

```bash
python -m src.similis_baseline.ablation_report \
  --configs configs/ablation_cpu_base.yaml configs/ablation_cpu_no_class_weights.yaml configs/ablation_cpu_stretch.yaml \
  --output-dir artifacts/ablations/report
```

## Веса модели

`artifacts/checkpoints/best.pt` (~334 MB) и `last.pt` не включены в git из-за лимита GitHub на размер файлов.
Скачать всю папку `checkpoints/` с Yandex Disk:

**https://disk.yandex.ru/d/s_DG-gyUyLocCQ**

После скачивания положить файлы в `artifacts/checkpoints/`, чтобы получилось:

```
artifacts/checkpoints/best.pt
artifacts/checkpoints/last.pt
```

Параметры и проверка checkpoint сохранены в
[artifacts/reports/model_summary.json](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/model_summary.json)
(total params, backbone, fields, epoch, best_metric) и
[artifacts/reports/checkpoint_roundtrip.json](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/checkpoint_roundtrip.json)
(`reload checkpoint -> same logits` round-trip test). После скачивания можно проверить целостность:

```bash
python scripts/regen_checkpoint_artifacts.py
```

Скрипт перезапишет оба JSON-файла и покажет, что параметры совпадают с теми, что приведены в репо.

## Ограничения

- в открытом корпусе нет повторов одного артефакта, поэтому group-aware split фактически совпадает с row-level split
  (это ограничение данных, не pipeline)
- нормализация словарей rule-based и чувствительна к шуму разметки
- `part` остаётся самым шумным полем (visual + label noise)
- baseline не пытается генерировать свободный текст и не предсказывает интерпретационные поля
- preprocessing-вывод по `stretch` пока нельзя считать окончательным: на коротком CPU-study он выиграл `val`, но проиграл
  `test_open`

## Что ещё стоит сделать

- при появлении данных с настоящим `artifact_id` подключить его в `data_prep` без изменений в split-логике
- вручную проверить качество heuristic-разметки `layout_mode/bg_type/overlay_text` на небольшой подвыборке
- довести интегрированную data-centric часть (трек 1.2) — отдельная работа, не входит в baseline
