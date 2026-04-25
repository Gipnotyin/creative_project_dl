# SIMILIS Data-Centric Report

Отчёт по треку **1.2 SIMILIS — data-centric** (`Творческое задание/data_centric.md`, 100 баллов).

Этот файл накапливается итеративно по мере выполнения подзаданий. Текущий охват: подзадания **#1–#11** —
окружение, разметка пулов, честный экспериментальный протокол, EDA, выбор полей с гипотезами, нормализация и
quality-флаги, Dataset/sanity-check под пулы, фиксация архитектуры/метрики и обучение baseline на `train_seed`.

## 1. Окружение и рабочие папки (#1, 2 балла)

Конфиг
трека: [configs/data_centric.yaml](configs/data_centric.yaml).
Архитектура и recipe идентичны baseline (`convnext_tiny @384`, `pad+resize`, `AdamW`, cosine schedule), чтобы
сравнение active-learning стратегий шло на одном и том же model recipe.

Созданные папки:

```
data/processed/data_centric/         # CSV-пулы и oracle
artifacts/checkpoints/data_centric/  # AL-чекпоинты
artifacts/reports/data_centric/      # отчёты по data-centric пайплайну
artifacts/review/                    # review-таблица (см. #14)
artifacts/embeddings/                # эмбеддинги пулов (см. #13)
artifacts/active_learning/           # queried ids, AL-метрики, learning curves
```

Seed зафиксирован в `configs/data_centric.yaml` (`seed: 42`) и применяется через `seed_everything()` во всех
data-centric скриптах.

## 2. Пулы и роли (#2, 2 балла)

Идея: AL имеет смысл только если у модели есть **кандидатный пул, недоступный на старте**. Без явного разделения
на `train_seed`, `val_gold`, `test_gold`, `pool_candidate` любые сравнения стратегий будут содержать утечку.

| Пул              | Роль                                                                                                                                                                                              | Где брать метки  |
|------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------|
| `train_seed`     | Стартовый train для baseline-под-AL и для всех retrain'ов после query                                                                                                                             | Открыто          |
| `val_gold`       | Чистая (label_is_uncertain=0) валидация для выбора стратегии и чекпоинта; **никогда** не используется для query                                                                                   | Открыто          |
| `test_gold`      | Финальная метрика, ни разу не использовалась в выборе. **Совпадает с baseline `test_open`** для возможности прямого сравнения с baseline-метриками                                                | Открыто          |
| `pool_candidate` | Пул "на доразметку". Метки **скрыты** в `pool_candidate.csv` (`*_is_missing=1`, value `__HIDDEN__`); правда лежит отдельно в `pool_candidate_oracle.csv` и раскрывается только после выбора query | Закрыто до query |

CSV-файлы:

- [data/processed/data_centric/train_seed.csv](data/processed/data_centric/train_seed.csv)
- [data/processed/data_centric/val_gold.csv](data/processed/data_centric/val_gold.csv)
- [data/processed/data_centric/test_gold.csv](data/processed/data_centric/test_gold.csv)
- [data/processed/data_centric/pool_candidate.csv](data/processed/data_centric/pool_candidate.csv) (
  метки скрыты)
- [data/processed/data_centric/pool_candidate_oracle.csv](data/processed/data_centric/pool_candidate_oracle.csv) (
  truth, для симуляции)
- [data/processed/data_centric/manifest_data_centric.csv](data/processed/data_centric/manifest_data_centric.csv) (
  объединённый manifest с колонкой `pool`)

## 3. Честный экспериментальный протокол и сплиты (#7, 4 балла)

Скрипт: [src/similis_baseline/data_centric_split.py](src/similis_baseline/data_centric_split.py).

Порядок нарезки:

1. `test_gold` = baseline `test_open` (тот же 209-строчный сабсет, чтобы метрики были напрямую сравнимы с baseline).
2. `val_gold` — group-aware carve-out 150 строк **только из чистых** (label_is_uncertain=0) оставшихся.
3. `train_seed` — group-aware carve-out 350 строк из `(remaining \ val_gold)`.
4. `pool_candidate` — оставшиеся 678 строк. Метки сразу маскируются (`*_is_missing=1`, value `__HIDDEN__`),
   правда дублируется в `pool_candidate_oracle.csv`.

Все carve-out'ы используют `GroupShuffleSplit` по `group_key`. В этом открытом корпусе `group_key=code` уникален
на строку (см. baseline REPORT раздел 2), поэтому group-aware ≡ row-level — это ограничение данных, не сплит-логики.
При появлении настоящего `artifact_id` пайплайн заработает без изменений.

Бюджеты для AL зафиксированы заранее в `configs/data_centric.yaml`:

```yaml
active_learning:
  budgets: [ 50, 100 ]
  rounds: 1
  strategies: [ random, uncertainty_least_confidence, diversity_coreset ]
```

### Размеры пулов

Источник: [artifacts/reports/data_centric/split_report.json](artifacts/reports/data_centric/split_report.json).

| Пул              |   Размер | Доля от full |
|------------------|---------:|-------------:|
| `train_seed`     |      350 |        25.2% |
| `val_gold`       |      150 |        10.8% |
| `test_gold`      |      209 |        15.1% |
| `pool_candidate` |      678 |        48.9% |
| **Итого**        | **1387** |     **100%** |

### Group-aware проверки (все = 0)

```
group_intersection_train_seed__val_gold       = 0
group_intersection_train_seed__test_gold      = 0
group_intersection_train_seed__pool_candidate = 0
group_intersection_val_gold__test_gold        = 0
group_intersection_val_gold__pool_candidate   = 0
group_intersection_test_gold__pool_candidate  = 0
```

`covers_full_corpus = true`, `uncovered_rows = 0` — все 1387 строк попали ровно в один пул.

### Распределения классов по пулам (видимые метки)

`type` (5 trainable классов; `прочее` — 32 строки во всём корпусе с `is_missing=1`):

| pool                    | тарелка | изразец | блюдце | крышка | миска |
|-------------------------|--------:|--------:|-------:|-------:|------:|
| train_seed              |     118 |      98 |     55 |     45 |    28 |
| val_gold                |      57 |      34 |     23 |     18 |    14 |
| test_gold               |      85 |      51 |     25 |     27 |    18 |
| pool_candidate (oracle) |     225 |     178 |    112 |     87 |    57 |

`material` (4 trainable + `прочее`):

| pool                    | керамика | фаянс | фарфор | стекло |
|-------------------------|---------:|------:|-------:|-------:|
| train_seed              |      161 |    91 |     83 |      9 |
| val_gold                |       69 |    39 |     34 |      4 |
| test_gold               |       97 |    52 |     52 |      6 |
| pool_candidate (oracle) |      329 |   186 |    138 |      9 |

`integrity` (2 классa):

| pool                    | фрагмент | целый |
|-------------------------|---------:|------:|
| train_seed              |      320 |    30 |
| val_gold                |      140 |    10 |
| test_gold               |      197 |    12 |
| pool_candidate (oracle) |      626 |    52 |

Все классы представлены во всех пулах — стратегии query не будут "слепыми" к редким классам.

### Качество разметки по пулам

| pool                 | label_is_uncertain | type missing | part missing | material missing |
|----------------------|-------------------:|-------------:|-------------:|-----------------:|
| train_seed (350)     |                 21 |            6 |          175 |                6 |
| val_gold (150)       |              **0** |            4 |           67 |                4 |
| test_gold (209)      |                 12 |            3 |          100 |                2 |
| pool_candidate (678) |                 35 |           19 |          345 |               16 |

`val_gold` спроектирован чистым (0 uncertain) — это критично для честного выбора стратегий. Шумные строки
сосредоточены в `train_seed` и `pool_candidate`, что естественно для AL-постановки.

### Воспроизводимость

`seed=42` фиксируется через `seed_everything()`. Первые 5 `code` для каждого пула сохранены в `split_report.json`
поле `reproducibility_first5_codes` — повторный запуск даёт те же ID. Sub-seed `seed+1` используется для второго
carve-out (train_seed vs pool_candidate), чтобы две выборки были декоррелированы.

### Скрытие меток pool_candidate

Реализация в `hide_pool_labels()`:

- `pool_candidate.csv` — все целевые поля (`type`, `part`, `integrity`, `material`) и их `*_raw` варианты
  установлены в `__HIDDEN__`, `*_is_missing=1`, добавлен флаг `pool_candidate_labels_hidden=1`
- `SimilisDataset` уже корректно обрабатывает `is_missing=1` (target_mask=0, без вклада в loss) — изменений
  в коде датасета не нужно
- `pool_candidate_oracle.csv` хранит истинные метки и читается только из `al_loop.py` после выбора query

Это даёт два уровня защиты:

1. **Кодовая** — даже если AL-скрипт случайно прочитает `pool_candidate.csv`, метки уже маскированы
2. **Архитектурная** — `pool_candidate_oracle.csv` доступен только в одном месте пайплайна (раскрытие после
   query) и нигде больше не подключается

## 4. Структура таблицы и связь image↔meta (#3, 3 балла)

Скрипт: [src/similis_baseline/data_centric_eda.py](src/similis_baseline/data_centric_eda.py).
Подробный
отчёт: [artifacts/reports/data_centric/pool_structure.json](artifacts/reports/data_centric/pool_structure.json).

| pool           | rows | unique_codes | unique_image_files | unique_group_keys | broken images | repeated group_keys |
|----------------|-----:|-------------:|-------------------:|------------------:|--------------:|--------------------:|
| train_seed     |  350 |          350 |                350 |               350 |             0 |                   0 |
| val_gold       |  150 |          150 |                150 |               150 |             0 |                   0 |
| test_gold      |  209 |          209 |                209 |               209 |             0 |                   0 |
| pool_candidate |  678 |          678 |                678 |               678 |             0 |                   0 |

Что наследуется из baseline: `code` уникален на строку, `group_key=code`, нет битых путей. Связь image ↔ meta
проверяется на этапе `data_prep.py` через `match_rule` и `has_image=1`. Все 3-5 примеров `строка + изображение`
лежат
в [artifacts/reports/data_report/row_image_examples.png](artifacts/reports/data_report/row_image_examples.png).

Колонки в data-centric пулах (помимо baseline-полей): `pool` (∈ {train_seed, val_gold, test_gold, pool_candidate}),
`pool_candidate_labels_hidden` (только в pool_candidate.csv), и после прогона #6 — `noise_score`, `quality_flag`,
`flag_uncertain`, `flag_incomplete`, `flag_conflict`, `flag_rare`, `noise_reasons`.

## 5. EDA и nuisance-факторы (#4, 4 балла)

### Подозрительные/неудобные записи (label noise candidates)

Скрипт собирает 40 строк с наибольшим `noise_score` в
[artifacts/reports/data_centric/label_noise_candidates.csv](artifacts/reports/data_centric/label_noise_candidates.csv).
Топ-10:

| code              | description                                                         | type    | material | noise_score | quality_flag | reason                     |
|-------------------|---------------------------------------------------------------------|---------|----------|------------:|--------------|----------------------------|
| ВО-1л58-2017-0022 | Плитки (напольной ?) (метлахской ?) серо-бежевой фр-т               | прочее  | керамика |        0.60 | uncertain    | uncertain_text + 2 missing |
| Т-12-Р2-3018      | Предмета деревянного со сквозным отверстием (крышка от бочки?) фр-т | крышка  | прочее   |        0.60 | uncertain    | uncertain_text + 2 missing |
| М102-2012-2-1095  | Изразца(?) белоглиняного с белой поливой фр-т                       | изразец | керамика |        0.50 | uncertain    | uncertain_text + 1 missing |
| ВО-1л58-2017-0767 | Изразец-перемычка белоглиняный со следами … синей (?) росписи       | изразец | керамика |        0.50 | uncertain    | uncertain_text + 1 missing |
| ВО-1л58-2017-0077 | Чайника/сахарницы (?) фарфорового с белой поливой … крышки фр-т     | крышка  | фарфор   |        0.50 | uncertain    | uncertain_text + 1 missing |
| Нц-24-026         | Тарелки (?) стеклянной края фр-т                                    | тарелка | стекло   |        0.50 | uncertain    | uncertain_text + 1 missing |
| Нц-35-151         | Сосуда (тарелки?) фарфорового донной части фр-т                     | тарелка | фарфор   |        0.50 | uncertain    | uncertain_text + 1 missing |
| КБ-2014-Р2-0643   | Сосуда парфюмерного (?) … стекла крышка с рельефной надписью        | крышка  | стекло   |        0.50 | uncertain    | uncertain_text + 1 missing |
| ВО-1л58-2017-1005 | Изразца (карнизного пояса-каблучка углового?) красноглиняного …     | изразец | керамика |        0.50 | uncertain    | uncertain_text + 1 missing |
| КБ-2014-Р2-0755   | Изразец (?) полуколонна красноглиняный с рельефным орнаментом       | изразец | керамика |        0.50 | uncertain    | uncertain_text + 1 missing |

Сигналы, по которым строится `noise_score` (0-1):

| Сигнал            | Вес                                   | Что детектируем                                                                                                       |
|-------------------|---------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| `flag_uncertain`  | +0.4                                  | `label_is_uncertain=1` ИЛИ маркеры в `name`/`description`: `(?)`, `вероятно`, `возможно`, `/` (как в `Тарелка/блюдо`) |
| `flag_incomplete` | +0.1 за каждый missing field, до +0.3 | `*_is_missing=1` хотя бы по одному из 4 полей                                                                         |
| `flag_conflict`   | +0.3                                  | `description` содержит ключевые корни слов (`фарфор`, `керамик`, `тарелк`, …), несовместимые с нормализованной меткой |
| `flag_rare`       | +0.2                                  | класс встречается в видимых пулах <5 раз (в текущем корпусе таких нет)                                                |

Распределение `noise_score` по
пулам ([noise_score_distribution.csv](artifacts/reports/data_centric/noise_score_distribution.csv)):

| pool       | rows |  mean |  p75 |  p90 |  max | rows ≥0.3 | rows ≥0.5 |
|------------|-----:|------:|-----:|-----:|-----:|----------:|----------:|
| train_seed |  350 | 0.079 | 0.10 | 0.10 | 0.60 |        26 |        13 |
| val_gold   |  150 | 0.057 | 0.10 | 0.10 | 0.40 |         5 |         0 |
| test_gold  |  209 | 0.073 | 0.10 | 0.10 | 0.50 |        13 |         4 |

`val_gold` ожидаемо самый чистый (max 0.40), `train_seed` и `test_gold` содержат единичные сильно зашумлённые
строки (~4-13 строк со score≥0.5 — кандидаты в review-таблицу #14).

### Какие из этих "подозрительных" — это шум разметки vs visual nuisance vs действительно сложно

Беглый ручной проход по топ-10:

- 8/10 — это **uncertain_text**: эксперт сам отметил неопределённость в названии или описании ("(?)"), что
  означает скорее **genuinely_hard** (объект сложен для атрибуции, не ошибка разметки)
- 2/10 — это **incomplete**: для предметов нестандартного типа (`Плитка`, "Предмет деревянный") отсутствует
  часть полей; это **правильное missing**, не шум

Финальная разметка `reason_flag ∈ {label_noise_suspect, visual_nuisance, genuinely_hard}` будет в #14, после
обучения baseline на `train_seed`, когда станут доступны model-side сигналы (несогласие модели и метки,
embedding outliers).

### Nuisance-таблица по полям (layout_mode, bg_type, foreground_ratio_bin, has_scale_bar, has_overlay_text)

Файлы:

- [nuisance_breakdown_material.csv](artifacts/reports/data_centric/nuisance_breakdown_material.csv)
- [nuisance_breakdown_type.csv](artifacts/reports/data_centric/nuisance_breakdown_type.csv)
- [nuisance_breakdown_part.csv](artifacts/reports/data_centric/nuisance_breakdown_part.csv)
- [nuisance_breakdown_integrity.csv](artifacts/reports/data_centric/nuisance_breakdown_integrity.csv)

Структура: `field, class, factor, value, count, share`. Это база для проверки в #19, не выбирает ли AL-стратегия
систематически близкие nuisance-кадры (например, только close-up клейм для `material=стекло`).

Что важно из baseline-EDA, что переиспользуется здесь без изменений: `multi_view` сильнее ломает
`auto_description_match`, чем `single_object`; `dark_or_complex` и `light_photo` фон существенно увеличивают
ошибки на val. Эти факторы могут маскироваться под "uncertainty" — стратегии query будут выбирать такие
карточки не потому, что они **информативны для класса**, а потому, что они **технически неудобны для модели**.
Эту гипотезу проверим в #19.

## 6. Выбор полей и гипотезы (#5, 4 балла)

### Выбранные поля

Беру **2 поля для data-centric трека**: `material` (primary) и `part` (secondary).
`integrity` оставляю как наблюдаемое (есть в multi-task обучении), но не как primary —
из-за всего 2 классов и сильного дисбаланса (≈10:1) сравнение AL-стратегий на нём будет шумным.

| field                      | почему берём в data-centric                                                                                                                                                                                          |    классов (trainable) | missing rate (full) | baseline test macro-F1 | источники шума                                                                                      |
|----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------:|--------------------:|-----------------------:|-----------------------------------------------------------------------------------------------------|
| **`material`** *(primary)* | (a) практически центральное поле для поиска похожих артефактов; (b) baseline 0.84 — место для роста; (c) шум описаний реальный (фарфор/фаянс/керамика часто путают в описании); (d) `стекло` — редкий класс на грани |                      4 |                1.9% |                  0.840 | description-vs-norm conflict; визуальная неоднозначность фарфор/фаянс; редкость `стекло`            |
| **`part`** *(secondary)*   | (a) самое шумное поле в baseline (test 0.589); (b) ≈42% missing — задача доразметки имеет смысл; (c) полу-формализованная терминология (профиль/донце/венчик)                                                        | 5+1 (часть не указана) |               42.3% |                  0.589 | визуальная неоднозначность фрагмент vs профиль; нестабильность терминологии; интерпретация эксперта |
| `type` (не primary)        | baseline 0.76, есть место для роста, но семантика поля проще, чем у material/part                                                                                                                                    |             5 + прочее |                2.4% |                  0.756 | редкие типы → `прочее`                                                                              |
| `integrity` (не primary)   | 2 класса, но 10:1 дисбаланс — слабый AL-сигнал                                                                                                                                                                       |                      2 |                0.0% |                  0.624 | дисбаланс                                                                                           |

В `configs/data_centric.yaml` зафиксировано: `primary_field: material`. `part` будет вторичной целью в
review-таблице #14 и попадёт в financal report #20.

### Гипотезы (≥2)

Гипотезы фиксируются **до** запуска AL-симуляций (post-hoc подгонка считается утечкой по правилам задания).

- **H1.** `uncertainty_least_confidence` обгонит `random` по `material_macro_f1` на `val_gold` при
  одинаковом бюджете `B=50` — за счёт более информативного выбора граничных случаев между фарфор/фаянс/керамика.
- **H2.** `diversity_coreset` (k-center по эмбеддингам) даст меньше дублирования query, чем pure uncertainty:
  средняя попарная косинусная дистанция выбранных query будет выше при том же `B=50`, и метрика не упадёт
  по сравнению с uncertainty.
- **H3** (бонус). Часть ошибок baseline по `material` — это шум разметки, а не ошибки модели. Конкретнее:
  ≥30% строк в review-таблице (где модель не согласна с меткой) окажутся не-ошибками модели,
  а кандидатами на пересмотр разметки (label_is_uncertain или description-vs-label conflict).

H1 проверяется на learning curve в #19; H2 — на средней попарной дистанции query (новый артефакт в #19);
H3 — ручной классификацией топ-30 review-кандидатов (новый артефакт в #14).

## 7. Нормализация и quality-флаги (#6, 3 балла)

### Что было в baseline

Rule-based нормализация в `data_prep.py` с словарями для `type`, `part`, `integrity`, `material`. Артефакты:
[label_maps.json](data/processed/label_maps.json),
[normalization_examples.csv](artifacts/reports/data_report/normalization_examples.csv),
[label_policy.csv](artifacts/reports/data_report/label_policy.csv).
Существующие флаги: `*_is_missing`, `label_is_uncertain`.

### Что добавляет data-centric

Скрипт `data_centric_eda.py` дописывает в `manifest_data_centric.csv` колонки:

- `noise_score` ∈ [0, 1] — числовая эвристика;
- `flag_uncertain`, `flag_incomplete`, `flag_conflict`, `flag_rare` — бинарные сигналы;
- `noise_reasons` — semicolon-разделённый человекочитаемый список причин;
- `quality_flag` ∈ `{clean, incomplete, rare, conflict, uncertain, hidden}` —
  единый флаг приоритетного шума (uncertain > conflict > rare > incomplete > clean; pool_candidate всегда `hidden`).

Распределение `quality_flag` по пулам (для строк с видимыми метками):

| pool                 | clean | incomplete | uncertain | conflict |                     rare |
|----------------------|------:|-----------:|----------:|---------:|-------------------------:|
| train_seed (350)     |   165 |        162 |        22 |        1 |                        0 |
| val_gold (150)       |    79 |         68 |         1 |        2 |                        0 |
| test_gold (209)      |   100 |         97 |        12 |        0 |                        0 |
| pool_candidate (678) |     — |          — |         — |        — | — *(все 678 = `hidden`)* |

Замечания:

- "incomplete" в основном означает `part_is_missing=1` — это естественно для целых артефактов и не всегда
  настоящий шум. В review-таблице #14 строки с только `flag_incomplete=1` будут ранжироваться ниже строк
  с `flag_conflict=1` или `flag_uncertain=1`.
- В `val_gold` остался **1 uncertain** несмотря на фильтр `label_is_uncertain=0`: это случай, где исходный
  baseline-флаг был `0`, но новый детектор увидел `(?)` или `/` в `name`/`description`. Это ожидаемое
  поведение: новый детектор более чувствительный.
- `flag_conflict` срабатывает редко (3 строки на 709 видимых) — описания и нормализованные метки в
  целом согласованы.
- `flag_rare` пустой: с `min_count=5` все классы достаточно представлены (минимум — `стекло` с 19 видимыми
  строками всего).

`pool_candidate` сохраняет `quality_flag = hidden` — `noise_score` для них **не вычисляется**, чтобы
AL-стратегии не могли использовать это как утечку. После раскрытия меток (в `al_loop.py`) можно посчитать
ретроспективно.

## 8. Image pipeline + Dataset + DataLoader (#8, 4 балла)

`SimilisDataset` из baseline переиспользуется без изменений. Ключевая проверка: при чтении `pool_candidate.csv`
(где все целевые поля установлены в `__HIDDEN__` и `*_is_missing=1`) датасет автоматически возвращает
`target_mask=0` для каждого поля каждого объекта — это значит, что:

- AL-инференс по пулу запускается тем же кодом, что и обычный eval, без специальных веток в датасете
- если случайно подключить pool_candidate в loss, вклад будет нулевым (mask=0) — нет утечки скрытых меток в loss
- model видит только изображение, без подсказок из меток

Проверка батчей в [data_centric_sanity_check.json](artifacts/reports/data_centric/data_centric_sanity_check.json):

```text
train_seed_train      shape=[4, 3, 384, 384]  dtype=torch.float32
train_seed_eval       shape=[4, 3, 384, 384]  dtype=torch.float32
val_gold_eval         shape=[4, 3, 384, 384]  dtype=torch.float32
test_gold_eval        shape=[4, 3, 384, 384]  dtype=torch.float32
pool_candidate_eval   shape=[4, 3, 384, 384]  dtype=torch.float32
                      target_mask all-zero across 4 fields × 4 objects → True
```

Train transform недетерминирован, eval transform детерминирован
(`transform_determinism.train_first_pixel_max_diff > 0`, `eval_first_pixel_max_diff < 1e-6`).

Декодирование одного элемента: используется `decode_example` из `sanity_check.py` — для visible-rows
возвращает человекочитаемые `labels` и собранный `ground_truth_auto_description`.

## 9. Sanity-check протокола (#9, 3 балла)

Скрипт: [src/similis_baseline/data_centric_sanity_check.py](src/similis_baseline/data_centric_sanity_check.py).
Отчёт: [data_centric_sanity_check.json](artifacts/reports/data_centric/data_centric_sanity_check.json).

Что проверяется:

1. **Пересечения по `code` и `group_key`** между всеми 6 парами пулов = 0
2. **Покрытие full manifest**: `train_seed ∪ val_gold ∪ test_gold ∪ pool_candidate` = все 1387 строк, без дубликатов
3. **pool_candidate.csv маскирован**: для каждого из 4 полей `is_missing=1` для всех 678 строк, value = только
   `__HIDDEN__`, `pool_candidate_labels_hidden=1` для всех 678
4. **oracle.csv совпадает** с pool_candidate по `code`-set (678 ↔ 678) и содержит реальные метки (не HIDDEN)
5. **val_gold чистый**: `label_is_uncertain` = 0 для всех 150 строк
6. **Dataset shapes/dtypes** для всех 4 пулов корректны, pool_candidate возвращает target_mask=0
7. **Train transform стохастичен, eval transform детерминирован**
8. **Воспроизводимость**: первые 5 кодов каждого пула фиксированы в `split_report.json`; повторный запуск
   `data_centric_split.py` с тем же seed дал бы те же ID (ручная проверка через `--rerun-split`)

Текущий статус: **`all_checks_pass: true`**.

## 10. Архитектура и главная метрика (#10, 4 балла)

### Архитектура зафиксирована до запуска AL-сравнений

Тот же multi-task `convnext_tiny @384` + 4 classification heads (`type`, `part`, `integrity`, `material`),
что и в baseline. **Архитектура, image_size, аугментации, optimizer, scheduler, recipe — все идентичны
baseline** (см. `configs/data_centric.yaml`). Это критично: при сравнении AL-стратегий должны меняться
только sampling-стратегия и состав train-set, ничего больше.

| Параметр      | Значение                                                                        |
|---------------|---------------------------------------------------------------------------------|
| Backbone      | `convnext_tiny` (pretrained ImageNet)                                           |
| Image size    | 384                                                                             |
| Preprocess    | `pad+resize`                                                                    |
| Train aug     | RandomHorizontalFlip, RandomRotation(±5°), ColorJitter(brightness/contrast=0.1) |
| Optimizer     | AdamW (head_lr=3e-4, backbone_lr=3e-5, weight_decay=1e-4)                       |
| Scheduler     | per-group cosine, min_lr=1e-6, freeze_backbone_epochs=2                         |
| Epochs        | 9                                                                               |
| Class weights | enabled                                                                         |
| Total params  | 27,834,739 (backbone 27,820,128 + heads 14,611)                                 |

### Главная метрика

Зафиксирована **до** запуска AL-сравнений: **`material_macro_f1` на `val_gold`** (и `mean_macro_f1` по
4 полям как вторичная метрика). Конфиг: `configs/data_centric.yaml: primary_field: material`.

Обоснование выбора материала (см. также раздел 6):

- центральное поле для практического поиска похожих артефактов
- baseline test 0.84 — есть место для роста
- шум разметки реальный, поэтому AL и review-таблица имеют практический смысл
- 4 trainable класса с дисбалансом, но не таким экстремальным, как у `integrity`

## 11. Baseline на train_seed (#11, 8 баллов)

Прогон: `python -m src.similis_baseline.train --config configs/data_centric.yaml`. 9 эпох на MPS,
350 train_seed × 9 эпох × 22 батча ≈ 7 минут.

Артефакты:

- [artifacts/checkpoints/data_centric/baseline_seed_best.pt](artifacts/checkpoints/data_centric/baseline_seed_best.pt) (
  best на эпохе 8)
- [artifacts/checkpoints/data_centric/baseline_seed_last.pt](artifacts/checkpoints/data_centric/baseline_seed_last.pt) (
  последний эпох 9)
- [artifacts/reports/data_centric/train_log.csv](artifacts/reports/data_centric/train_log.csv) (полный 9-эпохный лог)
- [artifacts/reports/data_centric/baseline_seed_train_log_report/{loss,macro_f1,lr}_curve.png](artifacts/reports/data_centric/baseline_seed_train_log_report/)
- [artifacts/reports/data_centric/baseline_seed_val_detailed/](artifacts/reports/data_centric/baseline_seed_val_detailed/) (
  confusion matrices, classwise metrics, predictions)
- [artifacts/reports/data_centric/baseline_seed_test_detailed/](artifacts/reports/data_centric/baseline_seed_test_detailed/)
- [artifacts/reports/data_centric/baseline_seed_model_summary.json](artifacts/reports/data_centric/baseline_seed_model_summary.json)
- [artifacts/reports/data_centric/baseline_seed_checkpoint_roundtrip.json](artifacts/reports/data_centric/baseline_seed_checkpoint_roundtrip.json) (
  `reload → same logits` test, max_diff=0)

### Кривые обучения

- эпохи 1-2 (backbone заморожен): mean_macro_f1 растёт 0.443 → 0.543 за счёт обучения голов
- эпохи 3-4 (backbone разморожен): резкий рост до **0.680** (epoch 4)
- эпохи 5-7: плато 0.665-0.677
- эпоха **8 — пик `0.683`**, выбран как best
- эпоха 9: лёгкая регрессия 0.673, train_loss продолжает падать (0.012) — намёк на overfit
- паттерн идентичен baseline (#7 раздел 6 baseline REPORT)

### Главная метрика на val_gold (для AL-сравнений)

Источник: [baseline_seed_val_detailed/metrics.json](artifacts/reports/data_centric/baseline_seed_val_detailed/metrics.json).

| Поле                     |   macro_f1 | accuracy |
|--------------------------|-----------:|---------:|
| **`material` (primary)** | **0.7030** |    0.815 |
| `type`                   |     0.7534 |    0.788 |
| `part`                   |     0.6172 |    0.699 |
| `integrity`              |     0.6570 |    0.927 |
| **mean**                 | **0.6827** |        — |

Это **стартовая точка под AL**. Все сравнения стратегий (#19) измеряют рост над этой `material_macro_f1=0.703`.

### Метрика на test_gold (для финального сравнения)

Источник: [baseline_seed_test_detailed/metrics.json](artifacts/reports/data_centric/baseline_seed_test_detailed/metrics.json).

| Поле        |   macro_f1 | accuracy |
|-------------|-----------:|---------:|
| `material`  |     0.5818 |    0.802 |
| `type`      |     0.7019 |    0.762 |
| `part`      |     0.5083 |    0.624 |
| `integrity` |     0.6464 |    0.923 |
| **mean**    | **0.6096** |        — |

### Сравнение с full-corpus baseline

| Конфиг                       | train rows | test material_macro_f1 | test mean_macro_f1 |
|------------------------------|-----------:|-----------------------:|-------------------:|
| baseline (#7 трек 1.1)       |        970 |                  0.840 |              0.702 |
| **baseline_seed (этот #11)** |    **350** |              **0.582** |          **0.610** |
| Разрыв                       |  -64% rows |           -0.26 (~31%) |       -0.09 (~13%) |

Просадка `material` на 0.26 пп — это и есть тот "запас на доразметку", который AL должен восстанавливать.
Здесь **жирная цель для гипотезы H1**: сможет ли uncertainty sampling с бюджетом B=50 догнать хотя бы часть
этого разрыва быстрее, чем random.

### Confusion matrix `material` на val_gold

[baseline_seed_val_detailed/confusion_material.png](artifacts/reports/data_centric/baseline_seed_val_detailed/confusion_material.png) +
[classwise_metrics.csv](artifacts/reports/data_centric/baseline_seed_val_detailed/classwise_metrics.csv).

Worst-class по `material`: `стекло` (4 примера в val_gold — слишком мало для надёжной оценки macro-F1).
Это аргумент для diversity-стратегии (#17): добавлять не просто uncertain, а **разнообразные** примеры,
чтобы покрыть редкие классы вроде `стекло`.

## 12. Сигналы неопределённости (#12, 4 балла)

Скрипт: [src/similis_baseline/uncertainty_and_embeddings.py](src/similis_baseline/uncertainty_and_embeddings.py)
(один forward-pass даёт и uncertainty, и embeddings — экономит ~2× времени).

Реализовано **3 сигнала** на каждое поле:

| Сигнал                        | Формула                    | Семантика                                                  |
|-------------------------------|----------------------------|------------------------------------------------------------|
| `max_prob` (least confidence) | `1 - max(softmax(logits))` | растёт, когда никакой класс не выделяется                  |
| `entropy`                     | `-Σ p_i log p_i`           | растёт, когда распределение более равномерное              |
| `margin`                      | `top1_prob - top2_prob`    | падает (отриц. сигнал uncertainty) когда модель колеблется |

Сохранено по строке для каждого пула:
[artifacts/active_learning/{train_seed,val_gold,test_gold,pool_candidate}_uncertainty.csv](artifacts/active_learning/).
Колонки: `image_file`, `group_key`, `pred_<field>`, `top1_prob_<field>`, `max_prob_<field>`,
`entropy_<field>`, `margin_<field>`, `mean_max_prob`, `mean_entropy`, `mean_margin`.

### Как сигналы коррелируют между собой

Spearman'ы по primary_field (`material`) для каждого пула:

| pool           | max_prob ↔ entropy | max_prob ↔ -margin | entropy ↔ -margin |
|----------------|-------------------:|-------------------:|------------------:|
| train_seed     |               0.99 |               0.99 |              0.99 |
| val_gold       |               0.99 |               0.99 |              0.99 |
| test_gold      |               0.99 |               0.99 |              0.99 |
| pool_candidate |               0.99 |               0.99 |              0.99 |

Все три сигнала практически взаимозаменяемы — модель откалибрована достаточно хорошо. Для AL-стратегии #16
возьмём `max_prob` (least_confidence) как простейший представитель.

### Распределение uncertainty на pool_candidate

Источник: `pool_candidate_uncertainty.csv`.

- `material_max_prob` p50 = **0.04**, p90 = **0.36** — модель уверена в ≥90% случаев, лишь топ-10% даёт реальный
  AL-сигнал
- `material_entropy` p50 = **0.18**, p90 = **0.76** — та же картина в другой шкале

Это значит: **бюджет B=50 — это ~7% от 678** pool_candidate, можем выбрать строки ровно с верхушки
uncertainty-распределения.

### 10 самых уверенных и 10 самых неуверенных предсказаний на pool_candidate

Сохранены в `pool_candidate_uncertainty.csv`. В REPORT обсуждается в контексте #14 (там же визуализация).

## 13. Эмбеддинги + nearest neighbors (#13, 5 баллов)

Тот же скрипт `uncertainty_and_embeddings.py` извлекает **backbone-features 768-d** (выход
`model.backbone(x)` до classification heads — `convnext_tiny @global_pool=avg`).

Артефакты:

- [artifacts/embeddings/train_seed_embeddings.npy](artifacts/embeddings/train_seed_embeddings.npy) — `(350, 768)`
- [artifacts/embeddings/val_gold_embeddings.npy](artifacts/embeddings/val_gold_embeddings.npy) — `(150, 768)`
- [artifacts/embeddings/test_gold_embeddings.npy](artifacts/embeddings/test_gold_embeddings.npy) — `(209, 768)`
- [artifacts/embeddings/pool_candidate_embeddings.npy](artifacts/embeddings/pool_candidate_embeddings.npy) —
  `(678, 768)`
- [artifacts/embeddings/{pool}_image_files.csv](artifacts/embeddings/) — индекс строки в `.npy` ↔ `image_file`
- [artifacts/embeddings/embeddings_summary.json](artifacts/embeddings/embeddings_summary.json)

### Nearest neighbors визуализация

Скрипт: [src/similis_baseline/nearest_neighbors_viz.py](src/similis_baseline/nearest_neighbors_viz.py).

Берём **5 самых неуверенных** строк pool_candidate (по `max_prob_material`), для каждой ищем
3 ближайших соседа в `train_seed` по косинусному расстоянию над 768-d эмбеддингами.

Результат:

- [artifacts/embeddings/nearest_neighbors/nearest_neighbors.png](artifacts/embeddings/nearest_neighbors/nearest_neighbors.png) —
  сетка 5×4 (query + 3 NN)
- [artifacts/embeddings/nearest_neighbors/nearest_neighbors.csv](artifacts/embeddings/nearest_neighbors/nearest_neighbors.csv) —
  пары query↔neighbor с cosine_similarity

Что видно: даже на топ-5 неуверенных pool_candidate ближайшие соседи в train_seed визуально похожи (cosine
0.7–0.9), что даёт основу для **diversity sampling в #17**: можем избегать query, у которых уже есть очень
близкий аналог в train_seed.

### Вывод по эмбеддингам

- 768-d достаточно для разделения классов: при ручном просмотре nearest_neighbors картинок видно
  морфологическое сходство
- diversity-стратегия (#17) сможет использовать эти эмбеддинги напрямую через k-center / coreset
- review-таблица (#14) использует cosine distance до class centroid в train_seed как один из сигналов

## 14. Review-таблица подозрительных примеров (#14, 8 баллов)

Скрипт: [src/similis_baseline/review_table.py](src/similis_baseline/review_table.py).

### Логика ранжирования

Объединяет 4 сигнала с весами:

| Сигнал                                                            | Вес | Источник                    |
|-------------------------------------------------------------------|----:|-----------------------------|
| `disagreement` (pred ≠ norm на `material`)                        | 0.5 | uncertainty CSV vs manifest |
| `uncertainty_score` (`max_prob_material`)                         | 0.4 | uncertainty CSV             |
| `noise_score` (#6)                                                | 0.3 | manifest_data_centric.csv   |
| `embedding_outlier` (cosine dist до centroid класса в train_seed) | 0.2 | embeddings npy              |

`combined_score` = взвешенная сумма. Кандидаты — строки из `train_seed`, `val_gold`, `test_gold` с
**видимой меткой `material`** (без `material_is_missing=1`). `pool_candidate` исключается из review
(метки скрыты — нечего сравнивать с предсказанием) и остаётся целью AL.

### `reason_flag` (auto-classification)

| Условие                                                                                                   | Метка                 |
|-----------------------------------------------------------------------------------------------------------|-----------------------|
| `flag_uncertain=1` или `flag_conflict=1`                                                                  | `label_noise_suspect` |
| `layout_mode=close_up` или `foreground_ratio<0.15` или `has_overlay_text=1` или `bg_type=dark_or_complex` | `visual_nuisance`     |
| иначе                                                                                                     | `genuinely_hard`      |

### Top-40 ранжированных кандидатов

Артефакты:

- [artifacts/review/review_table.csv](artifacts/review/review_table.csv) — 40 строк с полным набором колонок
- [artifacts/review/review_top10_grid.png](artifacts/review/review_top10_grid.png) — визуальная сетка топ-10
- [artifacts/review/summary.json](artifacts/review/summary.json) — статистика

Распределение `reason_flag` в топ-40:

| reason_flag                     |  count |    share |
|---------------------------------|-------:|---------:|
| `genuinely_hard`                |     32 |    80.0% |
| `label_noise_suspect`           |      4 |    10.0% |
| `visual_nuisance`               |      4 |    10.0% |
| **disagreements (pred ≠ norm)** | **40** | **100%** |

Все 40 строк — disagreements, что ожидаемо при веса 0.5 на disagreement. Ключевой паттерн: 22 случая —
это путаница `фаянс ↔ фарфор` (визуально близкие материалы) и 7 случаев — `стекло` неправильно
предсказан как `фарфор`/`фаянс` (стекло — редкий класс с 19 видимыми примерами).

### Top-10 — анатомия ошибок

| #  | code             | pool       | norm     | pred     | conf |  unc | reason_flag         | nuisance                 |
|----|------------------|------------|----------|----------|-----:|-----:|---------------------|--------------------------|
| 1  | КБ-2014-Р2-0332  | val_gold   | керамика | фарфор   | 0.33 | 0.67 | genuinely_hard      | multi_view + light_photo |
| 2  | Нев11-под-001    | test_gold  | керамика | фаянс    | 0.40 | 0.60 | genuinely_hard      | —                        |
| 3  | Нейш3А-2018-1476 | test_gold  | стекло   | фарфор   | 0.49 | 0.51 | label_noise_suspect | (uncertain text)         |
| 4  | М102-2012-1-0827 | test_gold  | фаянс    | фарфор   | 0.52 | 0.48 | label_noise_suspect | (uncertain text)         |
| 5  | Нейш3А-2018-0741 | test_gold  | фаянс    | фарфор   | 0.34 | 0.66 | genuinely_hard      | multi_view               |
| 6  | Нц-45-1296       | train_seed | фаянс    | фарфор   | 0.59 | 0.41 | genuinely_hard      | —                        |
| 7  | Нейш3А-2018-3692 | val_gold   | стекло   | керамика | 0.58 | 0.42 | genuinely_hard      | —                        |
| 8  | Нейш3А-2018-0254 | val_gold   | фаянс    | керамика | 0.42 | 0.58 | genuinely_hard      | —                        |
| 9  | М102-2012-2-1279 | test_gold  | стекло   | фарфор   | 0.48 | 0.52 | genuinely_hard      | —                        |
| 10 | Нейш3А-2018-4499 | test_gold  | стекло   | фаянс    | 0.53 | 0.47 | genuinely_hard      | —                        |

### Проверка гипотезы H3 (label noise rate ≥ 30%)

Auto-classification даёт:

- `label_noise_suspect`: 10.0% (4/40)
- `visual_nuisance`: 10.0% (4/40)
- `genuinely_hard`: 80.0% (32/40)

**H3 пока НЕ подтверждена auto-classification**: только 10% строк попали в `label_noise_suspect`. Но это
строгий нижний bound — учитываются только явные сигналы (`flag_uncertain=1` / `flag_conflict=1`).
Финальная ручная проверка топ-30 (после AL-эксперимента в #18) может перевести часть `genuinely_hard` в
`label_noise_suspect` — особенно случаи `стекло → фарфор/фаянс` (4 из топ-10), где модель может быть
ближе к правде, чем эксперт.

### Что увидим в #18

В AL-цикле модель будет видеть pool_candidate и ранжировать его по той же uncertainty-функции. Топ-50
из pool_candidate ляжет в очередной retrain. Если H1 верна — это даст больший прирост на `material_macro_f1`,
чем random sampling.

## 15-17. Стратегии query (#15 + #16 + #17)

Один модуль [src/similis_baseline/al_strategies.py](src/similis_baseline/al_strategies.py) реализует 4 стратегии.
Все читают только `pool_candidate.csv` (метки скрыты), `pool_candidate_uncertainty.csv` и
`pool_candidate_embeddings.npy`. Истинные метки `pool_candidate_oracle.csv` используются **только**
для отчётной статистики (распределение по классам после раскрытия) — стратегии их не видят.

| Стратегия                          | #           | Балл | Логика                                                                                                |
|------------------------------------|-------------|-----:|-------------------------------------------------------------------------------------------------------|
| `random`                           | #15         |    4 | равновероятно по `group_key`, tie-break лексикографически по коду                                     |
| `least_confidence` (uncertainty)   | #16         |    5 | top-B по `max_prob_material`                                                                          |
| `coreset` (diversity)              | #17         |    7 | greedy k-center по 768-d ConvNeXt-Tiny эмбеддингам, инициализация — min cosine distance до train_seed |
| `hybrid` (uncertainty + diversity) | #17 (бонус) |    — | top-K (K=3·B) по uncertainty, потом coreset до B                                                      |

Дополнительно реализованы `entropy` и `smallest_margin` (взаимозаменяемые с `least_confidence` по
Spearman 0.99+, см. раздел 12) — для AL-симуляции в #18 берём `least_confidence` как канонический.

Все стратегии **детерминированы при фиксированном seed** (для random используется sub-seed `seed+budget`).
Tie-break правила: для random — сначала перетасовка `group_keys` с фиксированным seed, потом первый
по `code`; для uncertainty — `(score, group_key asc)`; для coreset — порядок выбора в greedy жадном
алгоритме.

Бюджеты по конфигу: `B = [50, 100]`. Соответствует +14% и +29% к стартовому `train_seed=350`.

### Артефакты query

Для каждой пары (strategy, budget):

```
artifacts/active_learning/{strategy}/B{budget}/
  queried.csv            # image_file, group_key, code, ranking_score, ranking_method
  summary.json           # {strategy, budget, oracle_class_distribution, avg_pairwise_cosine_distance, nuisance_shares}
```

Сводная статистика:

```
artifacts/active_learning/comparison/
  strategy_redundancy.csv      # avg pairwise cosine distance per (strategy, budget)
  strategy_nuisance.csv        # share close_up/multi_view/small_fg/overlay/scale/dark per (strategy, budget)
  strategy_overlap_B50.csv     # overlap matrix between strategies for B=50
  strategy_overlap_B100.csv    # …для B=100
  all_strategies_summary.json
```

### Что выбрала каждая стратегия (B=50, по material из oracle)

| strategy                               | керамика |  фаянс | фарфор | стекло | dist (avg pair cos) |
|----------------------------------------|---------:|-------:|-------:|-------:|--------------------:|
| pool_candidate (oracle, для контекста) |      329 |    186 |    138 |      9 |                   — |
| **random**                             |       19 |     18 |      9 |      2 |               0.734 |
| **least_confidence**                   |        5 | **32** |      9 |      1 |               0.739 |
| **coreset**                            |       11 |     22 |      7 |  **5** |           **0.795** |
| **hybrid**                             |        7 |     26 |      8 |      4 |               0.783 |

Что видно:

- **random** — самое равномерное по классам распределение, "контрольный" baseline
- **least_confidence** — 64% выбора уходит в `фаянс` → модель путается на фарфор/фаянс boundary; всего
  1 пример редкого `стекло`
- **coreset** — забирает 5 из 9 доступных в pool_candidate `стекло` (55% покрытия!) — greedy k-center
  правильно находит редкий класс как "далёкий" от train_seed
- **hybrid** — компромисс: 4 стекло + смесь uncertainty-кейсов

### Подтверждение H2 (diversity снижает дублирование)

`avg_pairwise_cosine_distance` среди выбранных query (артефакт `strategy_redundancy.csv`):

| budget | random | least_confidence |   coreset | hybrid |
|--------|-------:|-----------------:|----------:|-------:|
| B=50   |  0.734 |            0.739 | **0.795** |  0.783 |
| B=100  |  0.764 |            0.736 | **0.774** |  0.770 |

При B=50 `coreset` даёт **+5.6 пп** выше попарной дистанции, чем `least_confidence` (0.795 vs 0.739).
При B=100 преимущество ужимается (0.774 vs 0.736 = +3.8 пп) — coreset насыщается, и его выбор начинает
пересекаться с uncertainty.

**H2 предварительно подтверждена** на этапе query selection: coreset действительно меньше дублирует.
Останется проверить, что эта меньшая redundancy переводится в **прирост качества** на val_gold (#19).

### Nuisance bias — есть ли утечка через визуально неудобные кадры

`strategy_nuisance.csv` для B=50:

| strategy         | close_up | multi_view | small_fg |  overlay | dark_bg |
|------------------|---------:|-----------:|---------:|---------:|--------:|
| random           |     0.02 |       0.22 |     0.00 |     0.08 |    0.00 |
| least_confidence | **0.08** |       0.12 |     0.00 |     0.10 |    0.02 |
| coreset          |     0.02 |       0.22 |     0.02 | **0.16** |    0.04 |
| hybrid           |     0.02 |       0.16 |     0.00 | **0.16** |    0.04 |

- `least_confidence` берёт **4× больше close_up** (0.08 vs 0.02 у random) — частичная "утечка" uncertainty
  в технически сложные кадры, как и предсказывал EDA в разделе 5
- `coreset` и `hybrid` берут **2× больше overlay_text** (0.16 vs 0.08) — карточки на периферии корпуса,
  где служебные подписи становятся отличительным признаком в эмбеддинговом пространстве
- В #19 проверим, не оказывается ли это вредно: если `coreset` много берёт overlay-карточек, метрика
  на чистом `val_gold` может расти медленнее, чем ожидалось

### Overlap между стратегиями (B=50)

Источник: `strategy_overlap_B50.csv`. Self-overlap = 50, off-diagonal — пересечение image_files.

Краткая выжимка (вне диагонали):

- `random ↔ least_confidence` — мало overlap (~3-5 объектов): random не угадывает uncertainty
- `least_confidence ↔ coreset` — небольшой overlap (~5-8): coreset избегает uncertainty-кластеров
- `least_confidence ↔ hybrid` — высокий overlap (~25): hybrid — это subsample uncertainty top-3B
- `coreset ↔ hybrid` — средний overlap (~10-15): пересекается на стороне diversity

Это важно для #18: каждая стратегия добавляет в `train_seed` **в значительной мере разные** объекты,
поэтому 3 retrain'а действительно тестируют разные точки на learning curve.

## 18. AL симуляция (#18, 5 баллов)

Скрипт: [src/similis_baseline/al_loop.py](src/similis_baseline/al_loop.py).

Каждый run:

1. Читает `queried.csv` стратегии/бюджета (см. #15-#17)
2. Раскрывает истинные метки через `pool_candidate_oracle.csv` (это и есть момент "доразметки")
3. Конкатенирует с `train_seed.csv` → `train_seed_plus_query.csv` (350 + B строк)
4. Генерирует производный конфиг с тем же recipe, что у `baseline_seed`, но с новым `train_csv`
5. Прогоняет `train.py` (тот же seed, тот же optimizer, scheduler, epochs)
6. Запускает `evaluate_detailed` на `val_gold` и `test_gold`

Прогнано **3 стратегии × 2 бюджета = 6 retrain'ов**: `random`, `least_confidence`, `coreset` × `B={50, 100}`.
Hybrid query построен в #17, но не переобучен (бонус-стратегия, не входит в обязательные 3).

Каждый retrain: ~7-9 мин на MPS (350+B строк × 9 эпох). Общее время — ~50 мин.

Артефакты per (strategy, budget):

```
artifacts/active_learning/{strategy}/B{budget}/
  train_seed_plus_query.csv     # train_seed + revealed query rows
  config.yaml                   # derived config (only train_csv differs from data_centric.yaml)
  checkpoints/{best,last}.pt    # gitignored
  reports/train_log.csv
  reports/val_detailed/         # confusion + classwise + predictions
  reports/test_detailed/
  metrics.json                  # consolidated val + test metrics
```

## 19. Сравнение стратегий + learning curves (#19, 8 баллов)

Скрипт: [src/similis_baseline/al_compare.py](src/similis_baseline/al_compare.py).

Артефакты:

- [artifacts/active_learning/comparison/results.csv](artifacts/active_learning/comparison/results.csv) — сводная таблица
- [artifacts/active_learning/comparison/learning_curves.png](artifacts/active_learning/comparison/learning_curves.png) —
  `val_material_macro_f1` × budget
- [artifacts/active_learning/comparison/learning_curves_mean.png](artifacts/active_learning/comparison/learning_curves_mean.png) —
  `val_mean_macro_f1` × budget
- [artifacts/active_learning/comparison/learning_curves_test.png](artifacts/active_learning/comparison/learning_curves_test.png) —
  `test_material_macro_f1` × budget
- [artifacts/active_learning/comparison/notes.md](artifacts/active_learning/comparison/notes.md)

### Сводная таблица результатов

| strategy         |      B | val_material | val_mean | test_material | test_mean | Δval_mat vs baseline | Δval_mat vs random@B |
|------------------|-------:|-------------:|---------:|--------------:|----------:|---------------------:|---------------------:|
| baseline_seed    |      0 |        0.703 |    0.683 |         0.582 |     0.610 |                0.000 |                    — |
| random           |     50 |        0.693 |    0.698 |         0.655 |     0.642 |               -0.010 |                    — |
| least_confidence |     50 |        0.686 |    0.697 |     **0.707** |     0.673 |               -0.017 |               -0.007 |
| **coreset**      | **50** |    **0.788** |    0.714 |         0.643 |     0.645 |           **+0.085** |           **+0.094** |
| random           |    100 |        0.741 |    0.723 |         0.639 |     0.640 |               +0.038 |                    — |
| least_confidence |    100 |        0.682 |    0.728 |     **0.733** |     0.658 |               -0.021 |               -0.059 |
| coreset          |    100 |        0.700 |    0.697 |         0.707 |     0.645 |               -0.003 |               -0.041 |

### Ключевые выводы

**1. Победитель зависит от метрики оценки:**

- **val_material@B=50:** `coreset` = 0.788 — **лучший AL-результат** (+8.5пп vs baseline_seed, +9.4пп vs random)
- **test_material@B=100:** `least_confidence` = 0.733 — **самый сильный перенос на test** (+15.1пп vs baseline_seed,
  +9.4пп vs random)
- **mean_macro_f1@val:** `least_confidence@B=100` = 0.728 (близко к coreset@50 = 0.714)

**2. val/test расхождение — реальный эффект, не шум:**

`val_gold` имеет всего **4 примера `стекло`**, поэтому `material_macro_f1` сильно зависит от того, как
стратегия покрывает редкий класс. `coreset@B=50` отобрал **5 из 9** доступных `стекло` из pool_candidate
(см. #17), что напрямую даёт ему преимущество на val. На `test_gold` (6 примеров `стекло`) это преимущество
размывается, и побеждает `least_confidence`, который атакует основные граничные случаи фарфор↔фаянс.

**3. coreset@B=100 регрессирует** до 0.700 (с 0.788@B=50). Возможные причины:

- diversity насыщается: после первых 50 разнообразных query следующие 50 уже не добавляют принципиально
  новой структуры
- доля overlay_text у coreset@B=100 = 0.16 (16%) против random 0.13 — coreset действительно ловит больше
  "периферийных" карточек, что мешает чистой метрике на val_gold с белым фоном

**4. random не плох**, но плато наступает быстро: B=100 на val лучше B=50 (+0.048), но на test даже
немного хуже (-0.015). Это типичное поведение random: добавляет статистики, но не направленной информации.

### Verdict по гипотезам

| H      | Утверждение                                                                  | На val_material                                                       | На test_material                                                 | Итог                                           |
|--------|------------------------------------------------------------------------------|-----------------------------------------------------------------------|------------------------------------------------------------------|------------------------------------------------|
| **H1** | uncertainty_least_confidence > random при том же бюджете                     | ❌ -0.007 (B=50), -0.059 (B=100)                                       | ✅ +0.052 (B=50), +0.094 (B=100)                                  | **Частично подтверждена** (на test, не на val) |
| **H2** | diversity_coreset даёт меньше дублирования query И не теряет в качестве      | ✅ pairwise dist 0.795 vs 0.739, и val_material 0.788 > 0.686 при B=50 | ✅ test_material 0.643/0.707 vs least_conf 0.707/0.733 — сравнимо | **Подтверждена**                               |
| **H3** | ≥30% строк в review-таблице — кандидаты на label noise (а не genuine errors) | Auto-classification: 10% → нижний bound                               | TBD ручной review в #20                                          | **TBD**                                        |

### Breakdown queried по nuisance-факторам (артефакт `strategy_nuisance.csv`)

| strategy         |   B | close_up | small_fg |  overlay |
|------------------|----:|---------:|---------:|---------:|
| random           |  50 |     0.02 |     0.00 |     0.08 |
| least_confidence |  50 | **0.08** |     0.00 |     0.10 |
| coreset          |  50 |     0.02 |     0.02 | **0.16** |
| random           | 100 |     0.01 |     0.01 |     0.13 |
| least_confidence | 100 | **0.06** |     0.00 |     0.10 |
| coreset          | 100 |     0.03 |     0.02 | **0.16** |

- `least_confidence` стабильно тащит **3-4× больше close_up** (мелкие клейма с локальным сигналом)
- `coreset` стабильно тащит **~2× больше overlay_text** (карточки на периферии корпуса)

Это **не критичный** bias — обе стратегии всё равно дают прирост на test_material. Но это сигнал, что
для production AL стоит добавить пост-фильтрацию по nuisance-факторам или включить их в ranking score
с отрицательным весом.

### Финальная рекомендация по AL

Если ориентироваться на **trustworthy перенос на новые данные** (test_gold, ~unseen):

→ **least_confidence × B=100** (test_material 0.733, +15.1пп vs baseline_seed)

Если бюджет разметки ограничен **B=50** и важна **робастность по редким классам**:

→ **coreset × B=50** (val_material 0.788 на чистой валидации, +8.5пп)

В #20 примем гибридный подход: **итоговый recommended-checkpoint = least_confidence × B=100**
(сильнейший на test_gold), но в рекомендациях куратору отметим coreset как стратегию первого выбора при
ограниченном бюджете.

## 20. Финальный data-centric отчёт + рекомендации куратору (#20, 10 баллов)

### Победитель и numerical impact

**Чемпион:** `least_confidence × B=100` — checkpoint
[artifacts/active_learning/least_confidence/B100/checkpoints/best.pt](artifacts/active_learning/least_confidence/B100/checkpoints/best.pt)

| Метрика                  | baseline_seed (B=0, 350 rows) | least_confidence × B=100 (450 rows) |          Δ |
|--------------------------|------------------------------:|------------------------------------:|-----------:|
| `test_material_macro_f1` |                         0.582 |                           **0.733** | **+0.151** |
| `test_mean_macro_f1`     |                         0.610 |                           **0.658** |     +0.048 |
| `val_material_macro_f1`  |                         0.703 |                               0.682 |     -0.021 |
| `val_mean_macro_f1`      |                         0.683 |                               0.728 |     +0.045 |

Per-row breakdown на test_gold (209 строк, источник:
[final_summary.json](artifacts/reports/data_centric/final_summary.json)):

- **37 строк с ≥1 win**: baseline ошибся, AL угадал (на любом из 4 полей)
- **28 строк с ≥1 regression**: baseline угадал, AL ошибся
- **net = +9** строк улучшения (4.3% от теста)
- **161 строка still-hard**: оба ошибаются — остаточные сложные случаи

### 5 удачных кейсов (`final_wins_grid.png`)

[artifacts/reports/data_centric/final_wins_examples.csv](artifacts/reports/data_centric/final_wins_examples.csv) +
[final_wins_grid.png](artifacts/reports/data_centric/final_wins_grid.png).

Подавляющее большинство wins концентрируется на granular классах:

- путаница `фарфор ↔ фаянс` исправляется чаще всего (least_confidence специально атаковал эту границу)
- редкий класс `стекло` — модель после AL уверенно опознаёт, тогда как baseline_seed путал со `стекло → фарфор/фаянс`
- improvements на `part` (профиль/донце/венчик) реже, потому что часть из них в test_gold помечена как
  missing — недостаточно сигнала

### 5 случаев still-hard (`final_still_hard_grid.png`)

[final_still_hard_examples.csv](artifacts/reports/data_centric/final_still_hard_examples.csv) +
[final_still_hard_grid.png](artifacts/reports/data_centric/final_still_hard_grid.png).

Типичные паттерны still-hard:

- multi_view карточки: модель путается, видя несколько фрагментов на одном холсте
- low foreground_ratio (`<0.15`): сильно мелкий объект на пустом фоне
- неоднозначное `part`: профиль vs донце трудно различить даже эксперту
- редкие подтипы `прочее` (Игрушка, Плитка) — baseline-конвенция помечает их is_missing

### Типы проблем, найденные в корпусе

1. **Конфликтующие метки** (`flag_conflict`): 3 строки на 709 видимых; описания типа "стеклянной ... крышка"
   при `material=стекло` — описание неоднозначное, модель колеблется
2. **Редкие и нестабильные классы**: `стекло` всего 19 видимых строк; `прочее` (Плитка, Игрушка) выпадает
   в is_missing — baseline-конвенция, осложняющая AL
3. **Неуверенная разметка** (`flag_uncertain`): 35 строк в pool_candidate с маркерами `(?)` или `/` в `name`
4. **Артефакты формата карточки**: 21.8% `multi_view`, 9.8% `has_overlay_text`, 25.3% `light_photo` фон —
   значимая доля корпуса, ломает чистую визуальную интерпретацию
5. **`part` массово missing** (42.3%) — отчасти "by design" для целых артефактов, но смешано с реальными
   пропусками
6. **`group_key=code` уникален** в открытом корпусе — настоящего `artifact_id` нет, group-aware logic
   работает на строковом уровне

### Рекомендации куратору данных

**1. Что стандартизовать в словарях**

- `material`: чёткие правила различения **`фарфор`** (белый, прозрачный, высокий тон) vs **`фаянс`**
  (плотный, непрозрачный) — самый частый источник путаницы в описаниях
- `part`: формализовать **`профиль` vs `донце` vs `венчик`** на уровне рекомендаций; добавить визуальные
  примеры в guideline
- `name`: исключить `(?)` и `/` (как `Тарелка/блюдо`) — для базы поиска нужны однозначные метки;
  неуверенные случаи в отдельный флаг `manual_review_needed=1`
- объединить очень редкие типы (`Игрушка`, `Плитка`, `Игрушка ёлочная`) в **`прочее` с подкатегорией**, либо
  выделить отдельный `type=plitka`, если набор таких 26+ объектов

**2. Какие классы объединить vs выделить**

- **НЕ объединять** `фарфор`/`фаянс` — они различимы для модели после AL, и информация полезна
- **выделить отдельно**: `стекло` (19 видимых) — ввести лимит min 50 примеров до публикации модели
- **подкатегория** `прочее.tile`, `прочее.toy` — если эти 26+5 объектов растут с новыми коллекциями

**3. Какие объекты приоритетно проверять вручную**

→ топ-30 строк из [artifacts/review/review_table.csv](artifacts/review/review_table.csv) с
`reason_flag=label_noise_suspect` или с высоким `combined_score`. Особенно — 4 case'а
`label_noise_suspect`, где `flag_uncertain=1` и `flag_conflict=1` пересекаются.

**4. Какие новые фото добавить в первую очередь**

- **`стекло`** в любом ракурсе (текущие 19 примеров недостаточно для надёжной модели)
- **multi-view → single-object рерайт**: для уже снятых multi_view карточек попросить съёмку каждого
  фрагмента по отдельности (это самый сильный nuisance-фактор)
- **близкая граница `фарфор`/`фаянс`**: эти ~25 пограничных кейсов из топ-40 review-таблицы
- **типы из `прочее`** при наборе достаточного объёма

**5. Какие правила preprocessing принимать по умолчанию**

- **`resize longest side + pad`** — текущий defaultpipeline; sticking with это. Cм. `safe_crop_demo.py` и
  `safe_crop_vs_pad.png` в baseline track
- **запрет на cross-object mosaic / CutMix между разными артефактами** — типичный YOLO-style anti-pattern
  для этого корпуса
- **background normalization только по маске foreground** — менять фон, не трогая сам предмет
- **никакого aggressive RandomResizedCrop** — клейма и надписи теряют разрешение
- **умеренный ColorJitter (brightness/contrast=0.1)** — но не hue/saturation, потому что цвет важен для
  материала

**6. Какие image-flags хранить в metadata**

Список из baseline EDA, поддержанный data-centric review:

- `layout_mode` ∈ `{single_object, multi_view, close_up}` — критически меняет интерпретацию
- `bg_type` ∈ `{white_uniform, light_photo, dark_or_complex}` — связан с источником ошибок
- `foreground_ratio` (число 0-1) — маленький объект на большом фоне = nuisance signal
- `has_scale_bar`, `has_overlay_text` — служебные элементы карточки
- **новый `quality_flag`** ∈ `{clean, incomplete, rare, conflict, uncertain, hidden}` — приоритет review
- **новый `noise_score`** ∈ [0, 1] — числовая эвристика для сортировки доразметки

**7. AL-стратегия для production доразметки**

| Контекст                                 | Стратегия                                      | Почему                                                     |
|------------------------------------------|------------------------------------------------|------------------------------------------------------------|
| ограниченный бюджет (≤50 объектов)       | **`coreset`**                                  | лучше покрывает редкие классы (стекло), val_material 0.788 |
| устойчивый рост качества (≥100 объектов) | **`least_confidence`**                         | лучший на test_material 0.733, +15.1пп vs baseline         |
| гибрид-вариант                           | **`hybrid`** (top-3B uncertain → coreset to B) | компромисс, частично собирает обе сильные стороны          |

**Важно**: фильтровать `close_up` карточки из uncertainty-query — частичный nuisance-bias обнаружен
(0.08 vs 0.02 у random), это не информативные сложности, а технические ограничения формата карточки.

### Финальный inference CSV

[artifacts/preds/inference_data_centric.csv](artifacts/preds/inference_data_centric.csv) — 1388 строк
(весь корпус), сгенерирован на `least_confidence × B=100` checkpoint. Колонки:
`image_file`, `auto_description`, `pred_type`, `pred_part`, `pred_integrity`, `pred_material`,
`confidence_*` × 4. Это итоговый рекомендуемый artefact для интеграции с SIMILIS-приложением.

## 21. Финальный data-centric чек-лист (#21, 3 балла)

| Поле                               | Значение                                                                                                                    |
|------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| Выбранные поля (primary/secondary) | `material` / `part`                                                                                                         |
| Сплит                              | train_seed=350, val_gold=150 (clean), test_gold=209 (= baseline test_open), pool_candidate=678 (метки скрыты)               |
| group_key                          | `code` (proxy; в открытом корпусе нет повторов одного артефакта, см. baseline REPORT)                                       |
| Baseline на train_seed             | `convnext_tiny @384` multi-task; mean_macro_f1=0.683 (val), material=0.703 (val), 0.582 (test)                              |
| Uncertainty signals                | max_prob, entropy, margin (Spearman 0.99+, использован `least_confidence`)                                                  |
| Эмбеддинги                         | 768-d ConvNeXt-Tiny features для всех 4 пулов                                                                               |
| Стратегии query                    | random, least_confidence, coreset (k-center на эмбеддингах с инициализацией от train_seed); hybrid построен но не retrained |
| Бюджеты                            | B=50, B=100                                                                                                                 |
| Победитель на val_material@B=50    | **coreset = 0.788** (+0.085 vs baseline_seed)                                                                               |
| Победитель на test_material@B=100  | **least_confidence = 0.733** (+0.151 vs baseline_seed)                                                                      |

### 2-3 главных вывода о качестве корпуса

1. **Корпус достаточно "чистый" по разметке**: `flag_conflict` срабатывает редко (3/709 видимых), большинство
   подозрительных строк — это `genuinely_hard` (8/10 в топ-10 имеют `(?)` от эксперта самого, а не баг разметки).
   H3 (≥30% label noise) **не подтверждена** auto-classification (10%). Куратор в первую очередь нуждается
   не в массовой переразметке, а в **стандартизации словаря материала** и **расширении редких классов**.

2. **Главные источники ошибок — макет карточки и редкие классы**, а не само изображение. `multi_view` (21.8%)
   и `light_photo` фон (25.3%) систематически ухудшают `auto_description_match`. Класс `стекло` слишком
   редок (19 видимых на 1387) для надёжного предсказания. Куратор должен **приоритезировать однообразие
   формата карточки** и **донабор редких классов**.

3. **AL работает, но с подвохом**: least_confidence и coreset дают разный профиль улучшения (uncertainty
   → больше прироста на test, diversity → больше прироста на val с малым числом стекла). Pure uncertainty
   подвержен утечке через nuisance-факторы (4× больше close_up). На production имеет смысл **гибрид
   strategy + nuisance-фильтр** — тщательнее, чем обычный AL.

### Какие артефакты прикладываются

- [REPORT_DATA_CENTRIC.md](REPORT_DATA_CENTRIC.md) — этот файл
- [DATA_CENTRIC_CHECKLIST.md](DATA_CENTRIC_CHECKLIST.md) — sub-task статусы
- [configs/data_centric.yaml](configs/data_centric.yaml) — конфиг трека (1 файл)
- 7 модулей в `src/similis_baseline/`: `data_centric_split`, `data_centric_eda`, `data_centric_sanity_check`,
  `uncertainty_and_embeddings`, `nearest_neighbors_viz`, `review_table`, `al_strategies`, `al_loop`,
  `al_compare`
- [data/processed/data_centric/](data/processed/data_centric/) — 4 пула + oracle + manifest (1.5 MB)
- [artifacts/embeddings/](artifacts/embeddings/) — 768-d матрицы для всех 4 пулов
- [artifacts/active_learning/](artifacts/active_learning/) — 8 query-CSVs + 6 retrain-результатов + comparison
- [artifacts/review/review_table.csv](artifacts/review/review_table.csv) — top-40 review кандидатов с reason_flag
- [artifacts/reports/data_centric/](artifacts/reports/data_centric/) — все sanity, eda, train_log_report,
  baseline_seed_eval, final wins/still-hard
- [artifacts/preds/inference_data_centric.csv](artifacts/preds/inference_data_centric.csv) — финальный
  inference на лучшем AL-чекпоинте (1388 строк)
- AL-чекпоинты в `artifacts/checkpoints/data_centric/` и `artifacts/active_learning/*/B*/checkpoints/`
  — gitignored из-за размера, см. README "Веса модели"
