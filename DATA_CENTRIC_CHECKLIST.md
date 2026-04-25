# SIMILIS Data-Centric Implementation Checklist

Рабочий план доведения проекта до полного соответствия треку **1.2 SIMILIS — data-centric**
(`Творческое задание/data_centric.md`, 100 баллов, 21 подзадание).

Этот трек **не повторяет** baseline; основной акцент не на архитектуре, а на:

- честном экспериментальном протоколе с пулами `train_seed / val_gold / test_gold / pool_candidate`,
- сигналах неопределённости и поиске шумных меток,
- сравнении стратегий active learning (random vs uncertainty vs diversity),
- учёте визуальных nuisance-факторов (`layout_mode`, `bg_type`, `foreground_ratio`, `has_scale_bar`, `has_overlay_text`).

## Статусы

- `done` — реализовано и есть чем это показать
- `partial` — что-то есть, но пункт не закрыт полностью (часто — переиспользование baseline без data-centric проверок)
- `missing` — реализации или обязательных артефактов нет
- `failed` — есть критичный дефект, из-за которого пункт нельзя считать выполненным

## Критичные блокеры

- Подзадания **#1, #2, #7 закрыты**: новый сплит `train_seed / val_gold / test_gold / pool_candidate` готов
  ([data/processed/data_centric/](data/processed/data_centric/),
  отчёт [split_report.json](artifacts/reports/data_centric/split_report.json),
  раздел в [REPORT_DATA_CENTRIC.md](REPORT_DATA_CENTRIC.md)).
- baseline-инфраструктура (`SimilisDataset`, `train.py`, `evaluate_detailed.py`, `predict.py`) переиспользуется,
  но через отдельный config `configs/data_centric.yaml`, чтобы не сломать baseline-репродуктивность.
- Дальше нужны: data-centric-специфичные доделки EDA (#3, #4, #6), выбор полей и гипотез (#5), baseline на
  train_seed (#10, #11), uncertainty signals (#12), эмбеддинги (#13), review-таблица (#14), три стратегии
  query (#15-#17), AL-симуляция (#18), сравнение и learning curves (#19), финальный отчёт (#20-#21).

## Рекомендуемый порядок реализации

1. Подготовить папки и зафиксировать data-centric окружение (#1).
2. Спроектировать и нарезать пулы `train_seed / val_gold / test_gold / pool_candidate` (#2, #7) — это самое
   важное для честности всего трека.
3. Переиспользовать EDA + добавить data-centric специфику: подозрительные записи, конфликтующие группы,
   nuisance-таблица (#3, #4).
4. Выбрать 1-3 поля + сформулировать ≥2 гипотезы (#5).
5. Нормализовать разметку с флагами качества (#6) — поверх baseline-нормализации добавить `quality_flag` и
   `noise_score`.
6. Подключить image pipeline + Dataset под новые сплиты (#8).
7. Sanity-check протокола: пересечения групп, скрытые метки `pool_candidate`, детерминизм (#9).
8. Зафиксировать baseline-архитектуру и метрику (#10), обучить на `train_seed`, оценить на `val_gold` (#11).
9. Реализовать ≥2 сигнала неопределённости (#12).
10. Извлечь эмбеддинги из предпоследнего слоя + nearest neighbors (#13).
11. Построить review-таблицу на ≥30 объектов с `reason_flag` (#14).
12. Реализовать random sampling как контрольную стратегию (#15).
13. Реализовать uncertainty sampling (#16).
14. Реализовать diversity / гибрид (#17).
15. Прогнать AL-симуляцию (#18).
16. Сравнить стратегии в одной таблице + learning curves (#19).
17. Финальный data-centric отчёт + рекомендации куратору + финальный inference (#20).
18. Свести в финальный data-centric чек-лист (#21).

## Чеклист по заданиям

### 1. Подготовить окружение и рабочие папки (2 балла)
Status: `done`

Уже есть:
- `requirements.txt`, `configs/baseline.yaml`, основные `data/raw`, `data/processed`, `splits`, `artifacts/*`
- фиксация seed в `utils.py`
- созданы папки `artifacts/review/`, `artifacts/active_learning/`, `artifacts/embeddings/`,
  `data/processed/data_centric/`, `artifacts/reports/data_centric/`, `artifacts/checkpoints/data_centric/`
- `configs/data_centric.yaml` создан и наследует архитектуру baseline для честного AL-сравнения
- seed зафиксирован через `seed_everything()` во всех data-centric скриптах
- описание окружения и папок зафиксировано в [REPORT_DATA_CENTRIC.md](REPORT_DATA_CENTRIC.md) раздел 1

Нужно доделать:
- [x] создать data-centric папки
- [x] добавить `configs/data_centric.yaml` с путями пулов и AL-параметрами
- [x] зафиксировать seed для всего data-centric пайплайна
- [x] описать создание папок и окружение в `REPORT_DATA_CENTRIC.md`

### 2. Загрузить данные и определить пулы (2 балла)
Status: `done`

Уже есть:
- 4 пула описаны таблично в [REPORT_DATA_CENTRIC.md](REPORT_DATA_CENTRIC.md) раздел 2
- размеры: train_seed=350, val_gold=150, test_gold=209, pool_candidate=678; всего 1387 (full corpus)
- val_gold спроектирован чистым (label_is_uncertain=0)
- test_gold = baseline test_open для прямого сравнения с baseline-метриками
- pool_candidate с **скрытыми метками** (`__HIDDEN__`, `*_is_missing=1`); правда в `pool_candidate_oracle.csv`
- защита от leakage: двухуровневая (кодовая + архитектурная), см. REPORT_DATA_CENTRIC.md раздел 3

Нужно доделать:
- [x] описать роль каждого пула
- [x] показать число строк, доля пропусков и `label_is_uncertain` в `split_report.json` и REPORT
- [x] прокомментировать механизм leakage-protection

### 3. Проверить структуру таблицы и связать изображения с метаданными (3 балла)
Status: `done`

Уже есть:
- матчинг `code -> image_file`, `group_key`, проверка битых путей, дублей `image_file` (из baseline)
- 3-5 примеров `строка + изображение`
  ([row_image_examples.png](artifacts/reports/data_report/row_image_examples.png))
- честное обоснование выбора `group_key=code` (см. baseline REPORT)
- per-pool структура в [pool_structure.json](artifacts/reports/data_centric/pool_structure.json):
  4 пула × `rows`, `unique_codes`, `unique_image_files`, `unique_group_keys`, `broken_image_rows=0`,
  `repeated_group_keys=0`
- список колонок в data-centric пулах (включая новые `pool`, `quality_flag`, `noise_score`) описан в REPORT_DATA_CENTRIC.md раздел 4

Нужно доделать:
- [x] явно перечислить колонки в каждом пуле (REPORT_DATA_CENTRIC.md раздел 4)
- [x] для каждого пула посчитать число валидных image, дубли, повторы по group_key

### 4. EDA по полям и качеству разметки (4 балла)
Status: `done`

Уже есть:
- baseline-heuristics: `layout_mode`, `bg_type`, `has_scale_bar`, `has_overlay_text`, `foreground_ratio`
- скрипт [data_centric_eda.py](src/similis_baseline/data_centric_eda.py)
- 40 label noise candidates в [label_noise_candidates.csv](artifacts/reports/data_centric/label_noise_candidates.csv)
  с reason flags (`flag_uncertain`, `flag_incomplete`, `flag_conflict`, `flag_rare`, человекочитаемая колонка `noise_reasons`)
- nuisance breakdown отдельно для каждого поля:
  [material](artifacts/reports/data_centric/nuisance_breakdown_material.csv),
  [type](artifacts/reports/data_centric/nuisance_breakdown_type.csv),
  [part](artifacts/reports/data_centric/nuisance_breakdown_part.csv),
  [integrity](artifacts/reports/data_centric/nuisance_breakdown_integrity.csv)
- noise_score распределения по пулам в `noise_score_distribution.csv`
- ручная классификация топ-10 кандидатов с пометкой `genuinely_hard / incomplete / label_noise_suspect` в
  REPORT_DATA_CENTRIC.md раздел 5

Нужно доделать:
- [x] ≥10 примеров подозрительных записей с reason
- [x] nuisance breakdown отдельно по каждому полю
- [x] вывод какие факторы могут маскироваться под шум разметки (REPORT раздел 5: layout/bg могут "красть"
      uncertainty-сигнал у настоящего label noise)
- [x] label noise candidates: description-vs-norm conflicts через keyword-roots детектор
- [ ] финальная разметка `reason_flag ∈ {label_noise_suspect, visual_nuisance, genuinely_hard}` для топ-30 —
      переходит в #14 (нужны model-side сигналы после обучения baseline на train_seed)

### 5. Выбрать 1-3 целевых поля и сформулировать гипотезы (4 балла)
Status: `done`

Уже есть:
- 2 поля выбраны: **`material`** (primary) и **`part`** (secondary). `integrity`, `type` остаются как наблюдаемые
  в multi-task, но не primary
- таблица обоснования полей в REPORT_DATA_CENTRIC.md раздел 6 (классы, missing rate, baseline test метрика,
  источники шума)
- 3 гипотезы зафиксированы **до запуска AL-симуляций**:
  - H1: uncertainty_least_confidence > random по `material_macro_f1` при B=50
  - H2: diversity_coreset даёт меньше дублирования query, чем pure uncertainty (выше средняя попарная дистанция)
  - H3 (бонус): ≥30% строк в review-таблице с несогласием модели/метки окажутся label_noise_suspect, а не genuine errors

Нужно доделать:
- [x] выбрать 1-3 поля
- [x] таблица обоснования полей
- [x] ≥2 гипотезы зафиксированы до запуска экспериментов

### 6. Нормализовать разметку и подготовить рабочие таблицы (3 балла)
Status: `done`

Уже есть:
- baseline rule-based normalization + `label_maps.json` + `*_is_missing` + `label_is_uncertain` (без изменений)
- словарь нормализации `raw → norm` минимум для одного поля (`material`) виден в
  [normalization_examples.csv](artifacts/reports/data_report/normalization_examples.csv)
  и [label_policy.csv](artifacts/reports/data_report/label_policy.csv)
- в `manifest_data_centric.csv` добавлены: `pool`, `noise_score`, `quality_flag` ∈ `{clean, incomplete, rare,
  conflict, uncertain, hidden}`, `flag_uncertain`, `flag_incomplete`, `flag_conflict`, `flag_rare`,
  `noise_reasons`
- `pool_candidate` помечен `quality_flag=hidden` чтобы AL-стратегии не могли использовать noise_score как утечку
- распределение `quality_flag` по пулам приведено в REPORT_DATA_CENTRIC.md раздел 7

Нужно доделать:
- [x] добавить `quality_flag` (clean / incomplete / rare / conflict / uncertain / hidden)
- [x] добавить `noise_score` ∈ [0, 1]
- [x] показать словарь нормализации raw → norm минимум для одного поля
- [x] сохранить рабочую таблицу `manifest_data_centric.csv` с полным набором флагов

### 7. Честный экспериментальный протокол и split'ы (4 балла)
Status: `done`

Уже есть:
- скрипт [src/similis_baseline/data_centric_split.py](src/similis_baseline/data_centric_split.py)
  нарезает 4 пула group-aware через `GroupShuffleSplit` (sub-seed `seed+1` для декорреляции carve-outs)
- бюджеты зафиксированы в `configs/data_centric.yaml`: `[50, 100]`, 1 round, 3 стратегии
- метки pool_candidate маскированы в `pool_candidate.csv` (`__HIDDEN__`, `*_is_missing=1`,
  `pool_candidate_labels_hidden=1`); truth в `pool_candidate_oracle.csv`
- проверки в [split_report.json](artifacts/reports/data_centric/split_report.json):
  `all_intersections_zero=true`, `covers_full_corpus=true`, первые 5 codes для воспроизводимости
- группы: 6 пар пулов × `group_intersection = 0`
- класс-распределения по пулам приведены в REPORT_DATA_CENTRIC.md раздел 3

Нужно доделать:
- [x] реализовать `data_centric_split.py` с 4 пулами
- [x] group-aware split (GroupShuffleSplit по `group_key`)
- [x] скрыть метки pool_candidate (`__HIDDEN__` + `*_is_missing=1` + флаг)
- [x] зафиксировать бюджеты в конфиге (`[50, 100]`)
- [x] показать размеры, пересечения, воспроизводимость в `split_report.json` и REPORT

### 8. Image pipeline + Dataset + DataLoader (4 балла)
Status: `done`

Уже есть:
- `SimilisDataset`, `build_transforms` с `pad+resize+norm`, train/val разделение из baseline переиспользуется
  без изменений
- проверено: при чтении `pool_candidate.csv` (HIDDEN labels + `*_is_missing=1`) датасет автоматически
  возвращает `target_mask=0` для всех 4 полей (sanity-check #9)
- shapes/dtypes одного батча из всех 4 пулов задокументированы в REPORT раздел 8 и в
  `data_centric_sanity_check.json`
- train transform стохастичен, eval transform детерминирован (проверено в sanity)

Нужно доделать:
- [x] `SimilisDataset` корректно работает с pool_candidate (target_mask=0)
- [x] Helper для итерации по pool — не нужен, существующий `DataLoader(SimilisDataset(pool_candidate.csv, train=False))` достаточен
- [x] Показать shapes/dtypes/декодирование/отсутствие train aug в eval (sanity_check.json)

### 9. Финальный sanity-check протокола (3 балла)
Status: `done`

Уже есть:
- скрипт [src/similis_baseline/data_centric_sanity_check.py](src/similis_baseline/data_centric_sanity_check.py)
  с 8 проверками
- отчёт [artifacts/reports/data_centric/data_centric_sanity_check.json](artifacts/reports/data_centric/data_centric_sanity_check.json) — `all_checks_pass: true`
- проверки: пересечения = 0 по `code` и `group_key` (6 пар × 2 ключа), full-manifest coverage,
  pool_candidate masked, oracle truth, val_gold uncertain=0, dataset shapes/dtypes,
  train stochastic / eval deterministic, воспроизводимость через `--rerun-split`

Нужно доделать:
- [x] sanity-script с 8 проверками
- [x] JSON-отчёт

### 10. Фиксированный baseline + главная метрика (4 балла)
Status: `done`

Уже есть:
- архитектура зафиксирована в `configs/data_centric.yaml` (идентична baseline: `convnext_tiny @384`,
  multi-task heads, AdamW, cosine, etc.)
- главная метрика зафиксирована: `material_macro_f1` на `val_gold` (config: `primary_field: material`),
  вторичная — `mean_macro_f1` по 4 полям
- параметры: 27,834,739 total (backbone 27,820,128 + heads 14,611), сохранены в
  [baseline_seed_model_summary.json](artifacts/reports/data_centric/baseline_seed_model_summary.json)
- описание архитектуры и метрики в REPORT_DATA_CENTRIC.md раздел 10

Нужно доделать:
- [x] зафиксировать ту же multi-task архитектуру в data_centric.yaml
- [x] зафиксировать главную метрику до запусков сравнений (`material_macro_f1` на `val_gold`)
- [x] записать выбор в REPORT и конфиге
- [x] показать число параметров и schema модели (model_summary.json)

### 11. Обучить baseline и зафиксировать стартовую точку (8 баллов)
Status: `done`

Уже есть:
- baseline на train_seed обучен (9 эпох, MPS, ~7 мин) через `train.py --config configs/data_centric.yaml`
- best чекпоинт на эпохе 8: [baseline_seed_best.pt](artifacts/checkpoints/data_centric/baseline_seed_best.pt)
  (mean_macro_f1=0.683, **material_macro_f1=0.703**)
- кривые обучения: [baseline_seed_train_log_report/](artifacts/reports/data_centric/baseline_seed_train_log_report/)
  (loss_curve.png, macro_f1_curve.png, lr_curve.png)
- detailed eval: [baseline_seed_val_detailed/](artifacts/reports/data_centric/baseline_seed_val_detailed/) и
  [baseline_seed_test_detailed/](artifacts/reports/data_centric/baseline_seed_test_detailed/)
  (confusion matrices для всех 4 полей, classwise metrics, predictions, успешные/неуспешные примеры)
- стартовые метрики зафиксированы в REPORT_DATA_CENTRIC.md раздел 11
- сравнение с full-corpus baseline: разрыв `material` 0.840 → 0.582 на test_gold (-31%) — это запас, который AL должен восстанавливать
- worst class по `material`: `стекло` (4 примера в val_gold) → довод за diversity-стратегию

Нужно доделать:
- [x] прогнать train.py на train_seed
- [x] сохранить baseline_seed_best.pt
- [x] кривые обучения
- [x] confusion matrix + classwise breakdown для main field
- [x] стартовая метрика как точка отсчёта (`material_macro_f1=0.703` на val_gold)

### 12. Сигналы неопределённости (4 балла)
Status: `done`

Уже есть:
- скрипт [src/similis_baseline/uncertainty_and_embeddings.py](src/similis_baseline/uncertainty_and_embeddings.py)
  считает 3 сигнала за один forward pass: `max_prob` (1-max softmax), `entropy`, `margin` для каждого поля
- сохраняет `artifacts/active_learning/{pool}_uncertainty.csv` для всех 4 пулов
- Spearman корреляции между сигналами 0.99+ для всех пулов (модель откалибрована)
- распределения uncertainty в pool_candidate: `material_max_prob` p50=0.04, p90=0.36 — топ-10% даёт реальный AL-сигнал
- описание в REPORT_DATA_CENTRIC.md раздел 12

Нужно доделать:
- [x] реализовать ≥2 сигнала (сделано 3: max_prob, entropy, margin)
- [x] сохранить uncertainty CSV для каждого пула
- [x] показать формулы, распределения, корреляции (Spearman)

### 13. Эмбеддинги + nearest neighbors / UMAP (5 баллов)
Status: `done`

Уже есть:
- backbone-features 768-d извлекаются прямо в `uncertainty_and_embeddings.py` через `model.backbone(x)`
  (ConvNeXt-Tiny с `global_pool=avg, num_classes=0` уже возвращает feature vector)
- `artifacts/embeddings/{pool}_embeddings.npy` для всех 4 пулов: `(350, 768)`, `(150, 768)`, `(209, 768)`, `(678, 768)`
- индекс `image_file ↔ row` в `{pool}_image_files.csv`
- скрипт [src/similis_baseline/nearest_neighbors_viz.py](src/similis_baseline/nearest_neighbors_viz.py):
  для топ-5 неуверенных pool_candidate находит 3 NN в train_seed по cosine
- визуализация [artifacts/embeddings/nearest_neighbors/nearest_neighbors.png](artifacts/embeddings/nearest_neighbors/nearest_neighbors.png) +
  CSV пар query↔neighbor с cosine_similarity
- описание в REPORT_DATA_CENTRIC.md раздел 13

Нужно доделать:
- [x] метод/хук для извлечения эмбеддингов (через `model.backbone(x)`)
- [x] скрипт извлечения эмбеддингов для 4 пулов
- [x] показать форму матрицы эмбеддингов (768-d)
- [x] для 5 объектов pool_candidate — 3 NN в train_seed с картинками (PNG)
- [ ] UMAP/t-SNE 2D-проекция (опционально, не критично — соседи и так видны на NN-сетке)
- [x] вывод: эмбеддинги дают основу для diversity sampling в #17

### 14. Review-таблица подозрительных примеров (8 баллов)
Status: `done`

Уже есть:
- скрипт [src/similis_baseline/review_table.py](src/similis_baseline/review_table.py)
- ранжирующая функция: взвешенная сумма 4 сигналов (disagreement 0.5, uncertainty 0.4, noise_score 0.3,
  embedding_outlier 0.2)
- [artifacts/review/review_table.csv](artifacts/review/review_table.csv) — **40 объектов** (≥30 требуемых)
  с полным набором колонок: image_file, group_key, code, raw_label, norm_label, pred_label, confidence,
  uncertainty_score, noise_score, embedding_outlier, disagreement, combined_score, quality_flag, flag_*,
  noise_reasons, layout_mode, bg_type, foreground_ratio, has_scale_bar, has_overlay_text, reason_flag
- auto-classification reason_flag по правилам из REPORT раздел 14
- визуальная сетка [artifacts/review/review_top10_grid.png](artifacts/review/review_top10_grid.png) — 10 топовых кейсов
- статистика [artifacts/review/summary.json](artifacts/review/summary.json):
  100% disagreements, 80% genuinely_hard, 10% label_noise_suspect, 10% visual_nuisance
- разбор top-10 в REPORT раздел 14, преимущественные паттерны: путаница `фаянс↔фарфор` и редкий класс `стекло`

Нужно доделать:
- [x] объединить сигналы: disagreement, uncertainty, noise, embedding_outlier
- [x] ранжирующая функция (взвешенная сумма)
- [x] review_table.csv ≥30 строк с полным набором колонок
- [x] 10 примеров на картинках (review_top10_grid.png)
- [x] reason_flag для каждой строки (auto)
- [ ] **ручная** перепроверка топ-30 — будет в #20 после AL-симуляции, когда станет видно эффект новой
      разметки на тех же объектах

### 15. Random sampling — контрольная стратегия (4 балла)
Status: `missing`

Нужно сделать:
- [ ] реализовать в `src/similis_baseline/al_strategies.py` функцию
      `random_query(pool_df, B, seed) -> queried_ids`
- [ ] querying только из `pool_candidate`, без пересечений по `group_key`
- [ ] показать: список `image_file` или `group_key` для одного запуска, что при том же seed список тот же,
      распределение query по классам (после раскрытия меток)

### 16. Uncertainty sampling (5 баллов)
Status: `missing`

Нужно сделать:
- [ ] реализовать в `al_strategies.py` функции
      `least_confidence_query`, `entropy_query`, `smallest_margin_query`
- [ ] фильтрация дубликатов и tie-break: при равенстве score выбирать по `group_key` лексикографически
- [ ] показать: 10 объектов с наибольшей uncertainty, сравнение с random выбором — сколько объектов
      пересекается, сколько новых
- [ ] комментарий: не сводится ли uncertainty к близким снимкам / маленькому foreground / overlay text
      (проверить через nuisance-таблицу)

### 17. Diversity / гибридная стратегия (7 баллов)
Status: `missing`

Нужно сделать:
- [ ] реализовать в `al_strategies.py` минимум один из вариантов:
  - **k-center / coreset** по эмбеддингам: greedy выбор B точек, максимизирующих минимальное расстояние
    до уже выбранных
  - кластеризация (KMeans) с выбором представителя из каждого кластера
  - **гибрид uncertainty + diversity**: top-k uncertain, потом coreset внутри них
- [ ] использовать эмбеддинги из задания 13
- [ ] показать: 10 выбранных объектов, визуализацию или таблицу, что они менее избыточны, чем чистый
      uncertainty (например, средняя попарная дистанция выбранных объектов больше)

### 18. Симуляция доразметки / round active learning (5 баллов)
Status: `missing`

Нужно сделать:
- [ ] реализовать `src/similis_baseline/al_loop.py`, который:
  1. загружает `baseline_seed_best.pt`
  2. для каждой стратегии (`random`, `uncertainty`, `diversity_or_hybrid`) выбирает B из `pool_candidate`
  3. "раскрывает" метки из `pool_candidate_oracle.csv` (или вручную проверенных)
  4. добавляет их к `train_seed` → новый `train_seed_plus_query.csv`
  5. дообучает модель ровно по тому же recipe (тот же config, те же epochs, тот же seed)
  6. оценивает на `val_gold` → сохраняет метрики
- [ ] сравнить либо 2 бюджета (B=50, B=100), либо 2 раунда по B=50
- [ ] **критично**: tот же training recipe, тот же seed, тот же val_gold для всех стратегий
- [ ] сохранить `artifacts/active_learning/{strategy}/{budget}/` с queried_ids, метриками, чекпоинтом

### 19. Сравнение стратегий + learning curves (8 баллов)
Status: `missing`

Нужно сделать:
- [ ] свести в `artifacts/active_learning/comparison/results.csv` колонки:
      `strategy`, `budget`, `val_metric`, `delta_vs_baseline`, `delta_vs_random`,
      `noise_rate_among_queried`, `notes`
- [ ] построить **learning curves** или barplot: x — budget, y — `val_metric` для каждой стратегии
- [ ] разбор состава queried по nuisance-факторам: `layout_mode`, `bg_type`, `foreground_ratio`,
      `has_scale_bar`, `has_overlay_text` — что выбирала каждая стратегия
- [ ] краткий вывод:
  - кто выиграл на малом бюджете
  - кто начал насыщаться
  - где diversity дал меньше дублей
  - не подменяет ли uncertainty "информативность" отбором технически неудобных карточек

### 20. Финальный data-centric отчёт (10 баллов)
Status: `missing`

Нужно сделать:
- [ ] `REPORT_DATA_CENTRIC.md` (новый файл) или раздел в `REPORT.md`, в котором:
  - какая стратегия и какой бюджет дали лучший результат
  - насколько improved pipeline лучше baseline (численно)
  - какие типы проблем нашли в корпусе: конфликтующие метки, редкие классы, плохие фото, повторы,
    случаи неопределённости, артефакты формата карточки
  - **рекомендации куратору**:
    - какие словари стандартизовать (например, синонимы материала)
    - какие классы объединить
    - какие объекты приоритетно проверять вручную (top из review-таблицы)
    - какие новые фото наиболее полезно добавить
    - какие правила preprocessing принять по умолчанию (pad, не cross-object mosaic, etc.)
    - какие image-flags хранить в metadata (`layout_mode`, `has_scale_bar`, `foreground_ratio`)
- [ ] финальный inference CSV: `artifacts/preds/inference_data_centric.csv` с `image_file`, `auto_description`
      и доп. полями
- [ ] 5 примеров, где data-centric улучшение исправило ошибку baseline (с картинками)
- [ ] 5 случаев, которые остались сложными даже после улучшений

### 21. Финальный data-centric чек-лист (3 балла)
Status: `missing`

Нужно сделать:
- [ ] в конце `REPORT_DATA_CENTRIC.md` короткий итог:
  - какие поля выбраны основными
  - как устроены `train_seed`, `val_gold`, `test_gold`, `pool_candidate`
  - какой baseline использовался
  - какие uncertainty-сигналы считали
  - какие query-стратегии сравнили
  - какие бюджеты использовали
  - какая стратегия победила и на сколько улучшила baseline
  - 2-3 главных вывода о качестве корпуса
- [ ] приложить артефакты: `review_table.csv`, queried ids, learning curves, лучший AL-чекпоинт,
      итоговый inference CSV

## Обязательные артефакты перед сдачей

- [ ] `REPORT_DATA_CENTRIC.md` или соответствующий раздел в `REPORT.md`
- [ ] `configs/data_centric.yaml`
- [ ] `data/processed/data_centric/manifest_data_centric.csv` (manifest со всеми пулами)
- [ ] `data/processed/data_centric/{train_seed,val_gold,test_gold,pool_candidate}.csv`
- [ ] `data/processed/data_centric/pool_candidate_oracle.csv` (скрытые истинные метки)
- [ ] `src/similis_baseline/data_centric_split.py`
- [ ] `src/similis_baseline/data_centric_sanity_check.py`
- [ ] `src/similis_baseline/uncertainty_signals.py`
- [ ] `src/similis_baseline/extract_embeddings.py`
- [ ] `src/similis_baseline/al_strategies.py`
- [ ] `src/similis_baseline/al_loop.py`
- [ ] `artifacts/checkpoints/data_centric/baseline_seed_best.pt` (через Yandex Disk, как в baseline)
- [ ] `artifacts/reports/data_centric/baseline_seed_train_log_report/*`
- [ ] `artifacts/active_learning/pool_uncertainty.csv`
- [ ] `artifacts/embeddings/{pool}_embeddings.npy`
- [ ] `artifacts/review/review_table.csv` (≥30 объектов)
- [ ] `artifacts/active_learning/{random,uncertainty,diversity}/{budget}/{queried_ids.csv,metrics.json}`
- [ ] `artifacts/active_learning/comparison/results.csv`
- [ ] `artifacts/active_learning/comparison/learning_curves.png`
- [ ] `artifacts/preds/inference_data_centric.csv`
- [ ] `data_centric_report.ipynb` (по аналогии с `baseline_report.ipynb`)

## Минимальный критерий "можно сдавать data-centric"

Перед финальной сдачей обязательно закрыть:
- [ ] валидный новый сплит (#7) и sanity-check (#9)
- [ ] обученный baseline на `train_seed` (#11)
- [ ] минимум 2 сигнала неопределённости (#12)
- [ ] review-таблица минимум на 30 объектов (#14)
- [ ] random + uncertainty + diversity стратегии (#15-#17)
- [ ] AL-симуляция с одинаковым recipe и seed для всех стратегий (#18)
- [ ] сводная таблица + learning curves (#19)
- [ ] финальные рекомендации куратору (#20)

Желательно (для полного балла):
- [ ] эмбеддинги + nearest neighbors визуализация (#13)
- [ ] калибровка вероятностей / reliability diagram (бонус)
- [ ] coarse-to-fine словарь как третья гипотеза (бонус)
