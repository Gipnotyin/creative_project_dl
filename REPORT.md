# SIMILIS Baseline Report

## 1. Что решается

Baseline решает задачу не как свободную генерацию текста, а как предсказание 4 нормализованных полей по изображению:

- `type`
- `part`
- `integrity`
- `material`

После этого по фиксированному шаблону собирается `auto_description`.

Такой подход выбран потому, что он:

- воспроизводим
- легче отлаживается
- позволяет контролировать low-confidence поведение модели

## 2. Данные и split

Исходные данные:

- CSV: `data/raw/selected_by_name_iimk_subset_public.csv`
- изображения: `data/raw/images`

Рабочие папки, которые используются и создаются пайплайном:

- `data/raw`
- `data/processed`
- `splits`
- `artifacts/checkpoints`
- `artifacts/preds`
- `artifacts/reports`
- `artifacts/ablations`

Подготовленные файлы:

- `data/processed/full_manifest.csv`
- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test_open.csv`
- `data/processed/label_maps.json`

Split делается group-aware по `group_key`.

Текущие размеры:

- train: `970`
- val: `208`
- test_open: `209`

`group_key` построен как `code`. В этом открытом корпусе **повторов одного и того же артефакта нет**: все 1387 строк
имеют уникальный `code`, и никакая комбинация других колонок (`name`, `description`, `cultlayer+execorg+survyear`) не
даёт реальных артефакт-уровневых повторов — совпадения в `description` оказываются разными находками с одинаковой
обобщённой формулировкой ("Тарелки фаянсовой профиль" встречается у 11 разных артефактов из разных раскопок).
Поэтому `group_key=code` фактически эквивалентен row-level split: leakage по повторам невозможен, потому что повторов
нет. Это ограничение **самих данных**, а не split-логики; при появлении данных с настоящим `artifact_id` group-aware
pipeline сработает без изменений.

Проверки по текущему split:

- `seed = 42`
- `split_reproducible_from_seed = true`
- `train ∩ val = 0`
- `train ∩ test = 0`
- `val ∩ test = 0`
- `repeated_group_key_count = 0` (ожидаемо для этого корпуса, см. выше)

Источник:
[artifacts/reports/data_report/report.json](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/data_report/report.json)

Дополнительные проверки воспроизводимости:

- `sanity_check.py` фиксирует environment и shape/dtype sanity checks;
- `checkpoint_roundtrip.json` подтверждает совпадение logits после повторной загрузки checkpoint;
- alias-файлы `splits/train_inner.csv`, `splits/val_inner.csv`, `splits/test_open.csv` теперь создаются автоматически.

## 3. EDA и data-centric диагностика

Базовый data-centric отчёт:

- [artifacts/reports/data_report/report.json](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/data_report/report.json)
- [artifacts/reports/data_report/image_heuristics.csv](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/data_report/image_heuristics.csv)
- [artifacts/reports/data_report/row_image_examples.png](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/data_report/row_image_examples.png)
- [artifacts/reports/data_report/problematic_images.png](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/data_report/problematic_images.png)
- [artifacts/reports/data_report/layout_mode_examples.png](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/data_report/layout_mode_examples.png)

Что видно по корпусу:

- `uncertain_row_count = 68`
- `single_object = 75.5%`
- `multi_view = 21.8%`
- `close_up = 2.7%`
- `white_uniform = 74.7%`
- `light_photo = 23.2%`
- `dark_or_complex = 2.1%`
- `has_overlay_text ≈ 9.8%`
- `has_scale_bar ≈ 0.1%`

Краткий вывод по сложностям корпуса:

- данные неоднородны не только по самому артефакту, но и по макету карточки;
- `multi_view` и `light_photo/dark_or_complex` фон реально связаны с ростом ошибок;
- открытый набор почти не содержит масштабной линейки, поэтому этот shortcut-фактор здесь слабее, чем можно было
  ожидать по постановке задачи;
- `part` остаётся самым шумным полем из-за высокой доли missing и визуальной неоднозначности.

## 4. Нормализация и шаблон

Выбранные поля baseline:

- `type`
- `part`
- `integrity`
- `material`

Шаблон `auto_description`:

1. `type`
2. `material`
3. `part`
4. слово `фрагмент`, если `integrity=фрагмент` и confidence прошёл порог

Fallback:

`не удалось уверенно собрать описание`

Итоговый inference CSV лежит в
[artifacts/preds/inference.csv](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/preds/inference.csv).

Артефакты по нормализации и выбору полей:

- [artifacts/reports/data_report/field_candidates.csv](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/data_report/field_candidates.csv)
- [artifacts/reports/data_report/normalization_examples.csv](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/data_report/normalization_examples.csv)
- [artifacts/reports/data_report/description_examples.csv](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/data_report/description_examples.csv)
- [artifacts/reports/data_report/label_policy.csv](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/data_report/label_policy.csv)

Что зафиксировано:

- `type`, `part`, `integrity`, `material` выбраны как базовые поля baseline;
- `size`, `cultlayer`, `survyear` и другие контекстные поля исключены из baseline как невизуальные;
- для `material` сохранён `material_raw`, чтобы было видно `raw -> normalized`;
- добавлен `label_is_uncertain`, чтобы отдельно помечать строки с `? / вероятно / возможно`.

Mandatory vs optional в шаблоне:

- обязательного поля в смысле “всегда печатаем” нет;
- все поля ведут себя как confidence-gated;
- на практике `type` и `material` самые важные, но тоже могут быть пропущены при низкой уверенности;
- отдельное осторожное поведение собрано в
  [artifacts/reports/val_cautious_examples/cautious_examples.csv](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/val_cautious_examples/cautious_examples.csv)
  и
  [artifacts/reports/test_cautious_examples/cautious_examples.csv](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/test_cautious_examples/cautious_examples.csv).

## 5. Модель

Архитектура:

- backbone: `convnext_tiny`
- отдельная classification head на каждое поле

Сводка параметров:

- [artifacts/reports/model_summary.json](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/model_summary.json)

Ключевые числа:

- total params: `27,834,739`
- backbone params: `27,820,128`
- heads params: `14,611`

## 6. Обучение

Основной train-пайплайн реализован в `src/similis_baseline/train.py`.

Используется safe fine-tuning:

- отдельный `head_lr`
- пониженный `backbone_lr`
- начальная заморозка backbone
- gradient clipping
- cosine schedule по двум param groups
- выбор `best.pt` по `val mean_macro_f1`

Аугментации deliberately safe:

- `PadToSquare -> Resize`
- `RandomHorizontalFlip`
- небольшая `RandomRotation(±5°)`
- лёгкий `ColorJitter`

Что специально не используется по умолчанию:

- агрессивный `RandomResizedCrop`
- сильные цветовые искажения
- mosaic / CutMix между разными артефактами

Стратегия по дисбалансу:

- в основном baseline используются `class_weights`
- в ablation study отдельно проверен вариант без `class_weights`
- результат показал, что без весов короткий `val` может расти, но перенос на `test_open` заметно ухудшается

Текущий основной checkpoint:

- [artifacts/checkpoints/best.pt](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/checkpoints/best.pt)

Проверка `reload checkpoint -> same prediction`:

- [artifacts/reports/checkpoint_roundtrip.json](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/checkpoint_roundtrip.json)

Полный синхронизированный train_log + графики:

- [artifacts/reports/train_log.csv](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/train_log.csv)
- [artifacts/reports/train_log_report/loss_curve.png](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/train_log_report/loss_curve.png)
- [artifacts/reports/train_log_report/macro_f1_curve.png](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/train_log_report/macro_f1_curve.png)
- [artifacts/reports/train_log_report/lr_curve.png](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/train_log_report/lr_curve.png)

### Решение по числу эпох

В `configs/baseline.yaml` зафиксировано `epochs=9` с `freeze_backbone_epochs=2`. Логика выбора подтверждается кривыми
из `train_log.csv`:

- эпохи 1-2 (backbone заморожен): mean_macro_f1 растёт 0.600 → 0.661 за счёт обучения голов;
- эпохи 3-4 (backbone разморожен): резкий рост до 0.760, train_loss падает с 0.46 до 0.25;
- эпохи 5-7: продолжающийся рост с локальным максимумом **0.819 на эпохе 7**;
- эпохи 8-9: train_loss продолжает падать (0.009 → 0.006), но val_loss стабилизируется и даже растёт (0.30 → 0.32),
  а mean_macro_f1 слегка снижается до 0.811-0.816 — признак раннего overfitting.

Дальше увеличивать число эпох без дополнительной регуляризации смысла нет: `best.pt` уже взят с эпохи 7, последующие
эпохи нужны только как контроль того, что метрика не обнаружит более позднего пика. Решение зафиксировать `epochs=9`
именно по этой причине: достаточно короткий бюджет, чтобы покрыть пик и наблюдать ранний overfit, но без затрат на
эпохи, где val ухудшается.

## 7. Основные метрики

### Валидация

Источник:
[artifacts/reports/val_detailed/metrics.json](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/val_detailed/metrics.json)

- `mean_macro_f1 = 0.8191`
- `type_macro_f1 = 0.8419`
- `part_macro_f1 = 0.7710`
- `integrity_macro_f1 = 0.7518`
- `material_macro_f1 = 0.9117`

### Открытый тест

Источник:
[artifacts/reports/test_detailed/metrics.json](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/test_detailed/metrics.json)

- `mean_macro_f1 = 0.7023`
- `type_macro_f1 = 0.7556`
- `part_macro_f1 = 0.5890`
- `integrity_macro_f1 = 0.6240`
- `material_macro_f1 = 0.8404`

Дополнительные артефакты по ошибкам:

- confusion matrices: `artifacts/reports/val_detailed/confusion_*.png`, `artifacts/reports/test_detailed/confusion_*.png`
- classwise metrics: `artifacts/reports/val_detailed/classwise_metrics.csv`, `artifacts/reports/test_detailed/classwise_metrics.csv`
- удачные/неудачные кейсы: `successful_examples.csv`, `failed_examples.csv`

## 8. Error Analysis

Разбор ошибок по факторам:

- [artifacts/reports/val_error_factors/factor_breakdown.csv](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/val_error_factors/factor_breakdown.csv)
- [artifacts/reports/test_error_factors/factor_breakdown.csv](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/test_error_factors/factor_breakdown.csv)
- [artifacts/reports/val_error_factors/error_reason_breakdown.csv](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/val_error_factors/error_reason_breakdown.csv)
- [artifacts/reports/test_error_factors/error_reason_breakdown.csv](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/test_error_factors/error_reason_breakdown.csv)

Что видно:

- на `val` `any_error_rate` растёт с `0.2466` на `white_uniform` до `0.5179` на `light_photo` и до `1.0` на
  `dark_or_complex`;
- на `val` `multi_view` заметно ухудшает `auto_description_match` по сравнению с `single_object`;
- на `test` фон тоже влияет, но слабее, чем на `val`;
- по heuristic taxonomy среди ошибок отдельно видны `layout_complexity`, `uncertain_or_label_noise` и
  `visual_ambiguity_low_confidence`.

Важно:

- это не ручная экспертная категоризация, а heuristic breakdown;
- он полезен как baseline-диагностика, но не заменяет ручной error review.

## 9. Ablation Study

Для короткого CPU-бюджета были проведены 3 сравнимых эксперимента:

- `ablation_cpu_base`: `pad + class_weights`
- `ablation_cpu_no_class_weights`: `pad + no_class_weights`
- `ablation_cpu_stretch`: `stretch + class_weights`

Общие условия:

- backbone: `convnext_tiny`
- `image_size=224`
- `epochs=4`
- `freeze_backbone_epochs=1`

Полная таблица:

- [artifacts/ablations/report/ablation_results.csv](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/ablations/report/ablation_results.csv)
- [artifacts/ablations/report/ablation_summary.md](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/ablations/report/ablation_summary.md)

Краткая таблица:

| experiment | preprocess | class_weights | val mean_macro_f1 | test mean_macro_f1 |
| --- | --- | --- | --- | --- |
| `ablation_cpu_stretch` | `stretch` | `True` | `0.7676` | `0.6712` |
| `ablation_cpu_no_class_weights` | `pad` | `False` | `0.7331` | `0.6163` |
| `ablation_cpu_base` | `pad` | `True` | `0.7021` | `0.7010` |

Интерпретация:

- `stretch` лучше на короткой `val`, но хуже переносится на `test_open`
- отключение `class_weights` поднимает короткий `val`, но сильно ухудшает `test_open`
- самый устойчивый перенос на open-test в этой короткой серии показал `pad + class_weights`

## 10. Что можно считать финальным решением сейчас

Финальным baseline в репозитории сейчас считается конфиг с safe preprocessing:

- `configs/baseline.yaml`

Причины:

- именно этот режим дал текущий основной checkpoint с `val mean_macro_f1 = 0.8396`
- на короткой ablation-series `pad + class_weights` лучше переносился на `test_open`, чем альтернативы
- растягивание изображения в квадрат остаётся рискованным с точки зрения домена

Дополнительная проверка аугментаций и transform-политики:

- [artifacts/reports/transform_report/transform_examples.png](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/transform_report/transform_examples.png) — train aug + eval pad vs stretch
- [artifacts/reports/transform_report/safe_crop_vs_pad.png](/Users/gipnotyin/Downloads/similis_baseline_project/artifacts/reports/transform_report/safe_crop_vs_pad.png) — pad vs safe content-aware crop vs aggressive center-crop (anti-pattern)

Скрипты: `transform_report.py`, `safe_crop_demo.py`.

Это не полноценная ablation сама по себе, но она показывает:

- train augmentation действительно меняет одну и ту же картинку;
- eval transform остаётся детерминированным;
- визуальная разница между `pad`, `stretch`, safe-crop и aggressive-crop реальна, и aggressive center-crop отрезает часть
  артефакта на изображениях с большим белым фоном — это анти-паттерн для baseline.

## 11. Ключевые выводы

1. Multi-task baseline с 4 полями работает и даёт хороший контролируемый `auto_description` без свободной генерации текста.
2. Самое слабое место пайплайна сейчас не `material`, а `part`: это наиболее шумное и неоднозначное поле.
3. На коротком CPU-study preprocessing и class weights сильно влияют на баланс `val` vs `test_open`, поэтому выбор
   финального конфига нельзя делать только по одной быстрой валидации.
4. Ошибки baseline сильнее связаны с качеством карточки и layout-сложностью, чем с наличием scale bar как таковым в
   текущем открытом наборе.

## 12. Что ещё осталось

- при появлении данных с настоящим `artifact_id` подключить его в `data_prep` без изменений в split-логике
- вручную провалидировать heuristics для `layout_mode`, `bg_type`, `has_overlay_text`
- начать data-centric трек (1.2): новый сплит `train_seed/val_gold/test_gold/pool_candidate`, uncertainty signals,
  embeddings, review-таблица, active learning loop с random/uncertainty/diversity стратегиями
