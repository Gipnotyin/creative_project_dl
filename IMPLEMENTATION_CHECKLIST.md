# SIMILIS Baseline Implementation Checklist

Этот файл нужен как рабочий план доведения проекта до полного соответствия `task.txt`.

Статусы:
- `done` — реализовано и есть чем это показать
- `partial` — что-то есть, но пункт не закрыт полностью
- `missing` — реализации или обязательных артефактов нет
- `failed` — есть критичный дефект, из-за которого пункт нельзя считать выполненным

## Критичные блокеры

- Основной `best.pt` получен из чистого 9-эпохного retrain на `configs/baseline.yaml`, `train_log.csv` и
  `train_log_report/*.png` синхронизированы с этим checkpoint.
- Все 21 подзадание baseline закрыты (`done`).
- Сводный ноутбук: `baseline_report.ipynb`.
- Следующий этап — трек 1.2 (data-centric): **отдельная работа**, не входит в baseline.

## Рекомендуемый порядок реализации

1. Починить сквозной pipeline данных: `data_prep -> train -> predict`.
2. Привести preprocessing к safe-варианту: resize longest side + pad, единый train/val/predict.
3. Добавить sanity-check скрипты/ноутбук для задач 1-10.
4. Починить и прогнать inference, сохранить `artifacts/preds/inference.csv`.
5. Добавить расширенную валидацию, confusion matrix и error analysis.
6. Сделать `tiny-overfit` и `one-batch` debug.
7. Реализовать `test_open` evaluation.
8. Провести минимум 3 ablation experiments.
9. Собрать финальный notebook или markdown-отчёт с артефактами.
10. Сверить итог по пункту 21 и приложить все обязательные файлы.

## Чеклист по заданиям

### 1. Подготовить окружение и рабочие папки
Status: `done`

Уже есть:
- `requirements.txt`
- `configs/baseline.yaml`
- папки `data/raw`, `data/interim`, `data/processed`, `splits`, `artifacts/*`
- фиксация seed в `utils.py` и `data_prep.py`
- воспроизводимый environment report в `sanity_check.py`

Нужно доделать:
- [x] добавить воспроизводимый environment report: `python`, `torch`, `torchvision`, `cuda`
- [x] добавить в ноутбук/отчёт явное создание и проверку папок
- [x] показать детерминизм именно в отчёте, а не только в коде

### 2. Загрузить данные и показать, что raw-часть читается корректно
Status: `done`

Уже есть:
- pipeline чтения CSV и сканирования изображений
- `prep_stats.json`
- пути к CSV и изображениям зафиксированы в `README.md` и `baseline.yaml`

Нужно доделать:
- [x] явно показать путь к CSV и корню изображений
- [x] вывести число строк CSV и число файлов изображений на диске в финальном отчёте
- [x] зафиксировать способ получения данных в ноутбуке/README

### 3. Проверить структуру датасета и связать изображения с метаданными
Status: `done`

Уже есть:
- матчинг `code -> image_file`
- `group_key`
- `has_image`
- печать колонок и debug-примеров в `data_prep.py`

Нужно доделать:
- [x] посчитать и показать число битых путей
- [x] посчитать и показать дубли `image_file`
- [x] посчитать и показать повторы по `group_key`
- [x] показать 3-5 примеров `строка + изображение`
- [x] оформить выбор `group_key` и риск leakage в явном виде

### 4. Провести визуальный и табличный EDA корпуса
Status: `done`

Уже есть:
- базовые распределения по `name`, `material`, `fragm`, `type`, `part`, `integrity`
- histogram aspect ratio
- `image_size_stats.csv`

Нужно доделать:
- [x] сетка минимум из 12 изображений с подписями
- [x] распределение числа изображений на `group_key`
- [x] 5-10 проблемных изображений
- [x] краткий вывод по сложностям корпуса
- [x] выделить 2-3 визуальных режима: `close-up`, `multi-view`, `single object`
- [x] добавить эвристики или ручную разметку для `layout_mode`, `bg_type`, `has_scale_bar`, `has_overlay_text`, `foreground_ratio`
- [x] отдельно проверить, не читает ли модель карточку вместо артефакта

### 5. Исследовать структуру описаний и выбрать поля для baseline
Status: `done`

Уже есть:
- выбраны 4 поля: `type`, `part`, `integrity`, `material`
- краткое обоснование есть в `README.md`

Нужно доделать:
- [x] показать 15-20 реальных `description`
- [x] сделать таблицу кандидатов на поля: почему видно, тип цели, число классов, доля пропусков, риск неоднозначности
- [x] явно перечислить поля, которые исключаем из baseline, и почему

### 6. Нормализовать словари и собрать целевые колонки
Status: `done`

Уже есть:
- rule-based нормализация и target columns
- `label_maps.json`
- missing flags по полям
- распределения по нормализованным полям в `artifacts/figures/dist_type.png`, `dist_part.png`, `dist_integrity.png`, `dist_material.png`

Нужно доделать:
- [x] показать `raw -> normalized` минимум для двух полей
- [x] показать синонимы/сокращения/опечатки и правила сведения
- [x] показать распределения после нормализации
- [x] зафиксировать обработку `missing`, `unknown`, `rare`, `uncertain`
- [x] либо реально применить `rare_class_min_count`, либо убрать его из конфига
- [x] добавить/зафиксировать `label_is_uncertain`, если хотим ссылаться на неоднозначность

### 7. Спроектировать шаблон auto_description и правила уверенности
Status: `done`

Уже есть:
- порядок полей и пороги есть в `predict.py` и `baseline.yaml`
- предусмотрен fallback `не удалось уверенно собрать описание`

Нужно доделать:
- [x] явно разделить обязательные и опциональные поля
- [x] показать минимум 5 примеров `pred fields -> auto_description`
- [x] показать минимум 1 пример низкой уверенности
- [x] сохранить шаблон как часть воспроизводимого артефакта и сослаться на него в отчёте

### 8. Сделать grouped split без утечки данных
Status: `done`

Уже есть:
- `GroupShuffleSplit`
- проверка нулевого пересечения групп
- сплиты создаются воспроизводимо по seed
- `data_prep.py` теперь обновляет и `data/processed/*.csv`, и зеркальные файлы в `splits/*.csv`
- `train.py`, `evaluate.py` и `sanity_check.py` используют те же `data/processed/*.csv`
- честное обоснование выбора `group_key=code` в `README.md` и `REPORT.md`: в открытом корпусе нет повторов одного
  артефакта, поэтому group-aware split совпадает с row-level split (это ограничение данных, не pipeline)

Нужно доделать:
- [x] унифицировать место хранения сплитов
- [x] сделать `train_inner`, `val_inner`, `test_open` в едином формате
- [x] показать распределения классов в train/val/test минимум для 2-3 полей
- [x] показать повторный запуск с тем же seed и те же split’ы
- [x] зафиксировать честную аргументацию по group_key (в корпусе нет групп размера >1, поэтому "неразорванные группы"
      физически отсутствуют — это проверенное свойство данных, а не дефект pipeline)

### 9. Подготовить image preprocessing и аугментации
Status: `done`

Уже есть:
- единый `PadToSquare -> Resize -> ToTensor -> Normalize` для val/test/predict
- общий safe preprocessing без растягивания исходной картинки в квадрат
- train augmentation и eval transform разделены внутри одного `build_transforms`
- safe content-aware crop vs pad vs aggressive center-crop сравнение в
  `artifacts/reports/transform_report/safe_crop_vs_pad.png` (скрипт `src/similis_baseline/safe_crop_demo.py`)
- финальный `best.pt` получен на текущем `pad` preprocessing — `train_log.csv` синхронизирован с checkpoint

Нужно доделать:
- [x] реализовать safe preprocessing: `resize longest side + pad`
- [x] выровнять train/val/test/predict transforms
- [x] добавить нормализацию изображений единообразно
- [x] показать 3 одинаковых изображения после train-аугментаций
- [x] показать то же изображение на val/test transform
- [x] написать комментарий, какие аугментации опасны для `material`, покрытия, клейм и надписей
- [x] показать safe crop vs pad (`safe_crop_vs_pad.png`)
- [x] переобучить финальный checkpoint на текущем preprocessing

### 10. Реализовать Dataset + DataLoader и сделать финальный sanity-check
Status: `done`

Уже есть:
- `SimilisDataset`
- `image`, `targets`, `target_mask`, `metadata`
- `train_loader`, `val_loader`
- `test_loader` в `train.py`
- `sanity_check.py` с проверкой shapes, dtypes, target mask, GT auto_description и детерминизма val transform

Нужно доделать:
- [x] добавить `test_loader`
- [x] показать `shape` одного батча
- [x] показать `dtype` изображений и меток
- [x] показать декодирование одного элемента обратно в человекочитаемые поля
- [x] сделать sanity-check ячейку: путь/картинка, отсутствие train aug на val, корректность `target_mask`, сборка GT `auto_description`
- [x] добавить отдельный артефакт с визуализацией пути/картинки в notebook или отчёт

### 11. Собрать baseline-модель для предсказания нескольких полей
Status: `done`

Уже есть:
- multi-task модель на одном backbone
- отдельные heads по полям
- logits по каждому полю
- рабочий `forward` подтверждается train/evaluate/predict пайплайном
- `artifacts/reports/model_summary.json`

Нужно доделать:
- [x] посчитать и показать число параметров
- [x] показать размер выхода по каждому полю на одном батче
- [x] явно показать, что `forward` проходит без ошибок
- [x] добавить схему/краткое описание модели в отчёт

### 12. Проверить loss и batch-метрики на одном батче
Status: `done`

Уже есть:
- masked CE loss
- расчёт accuracy и macro-F1 в eval

Нужно доделать:
- [x] сделать отдельный one-batch debug step
- [x] показать `logits shape`, `targets shape`
- [x] вывести loss по каждому полю и общий loss
- [x] показать `pred vs gt` хотя бы для двух полей на 10 объектах
- [x] показать batch-accuracy и желательно batch-macro-F1

### 13. Сделать tiny-overfit test
Status: `done`

Нужно доделать:
- [x] добавить tiny-subset из 32-64 изображений
- [x] отключить тяжёлые augmentation для теста
- [x] обучить до явного падения train-loss
- [x] сохранить кривые и короткий вывод

### 14. Реализовать полный train/eval цикл
Status: `done`

Уже есть:
- `train_one_epoch`
- `evaluate`
- логирование по эпохам
- выбор best checkpoint по `mean_macro_f1`
- sanity-check подтверждает отсутствие train augmentation на val/test
- полный train_log за 9 эпох в `artifacts/reports/train_log.csv` синхронизирован с текущим `best.pt` (`epoch=7`)
- графики `loss_curve.png`, `macro_f1_curve.png`, `lr_curve.png` в `artifacts/reports/train_log_report/`

Нужно доделать:
- [x] явно показать, что метрики меняются на всём цикле (9 строк лога, видна разморозка backbone на эпохе 3 и пик
      `mean_macro_f1=0.819` на эпохе 7)
- [x] добавить контроль отсутствия train aug в eval
- [x] при необходимости вынести train/eval метрики в отдельный отчёт или notebook (`baseline_report.ipynb`)

### 15. Выбрать optimizer, scheduler и обработку дисбаланса классов
Status: `done`

Уже есть:
- `AdamW`
- per-group cosine scheduler в `train.py`
- `lr=3e-4`
- `weight_decay=1e-4`
- class weights
- решение по числу эпох (9) задокументировано в `REPORT.md` секция "Решение по числу эпох" — пик metric на эпохе 7,
  ранний overfit с эпохи 8 (val_loss растёт), увеличивать бюджет без доп. регуляризации смысла нет

Нужно доделать:
- [x] показать дисбаланс хотя бы по одному полю
- [x] показать и объяснить стратегию работы с дисбалансом
- [x] построить график learning rate по эпохам
- [x] решить, достаточно ли текущих `9` эпох для baseline (зафиксировано: достаточно — пик на эпохе 7, после неё
      ранний overfit без доп. регуляризации)

### 16. Добавить чекпоинты, конфиги и логирование
Status: `done`

Уже есть:
- `last.pt`
- `best.pt`
- train log CSV
- в checkpoint сохраняются model/optimizer/scheduler/config/vocabs
- `val_metrics.json`
- `test_metrics.json`
- best checkpoint выбирается по `mean_macro_f1` в `train.py`
- `artifacts/reports/checkpoint_roundtrip.json`

Нужно доделать:
- [x] сохранить и показать `best_metrics.json` или эквивалентный итоговый файл
- [x] зафиксировать шаблон `auto_description` как часть checkpoint/config story
- [x] показать reload checkpoint и совпадение предсказания на одном примере
- [x] задокументировать, по какой метрике выбирается best

### 17. Посчитать метрики на валидации и разобрать ошибки по полям
Status: `done`

Уже есть:
- val accuracy и macro-F1 по полям в `train_log.csv`
- воспроизводимый `artifacts/reports/val_metrics.json`

Нужно доделать:
- [x] итоговая таблица метрик по полям
- [x] confusion matrix минимум для двух полей
- [x] 5 лучших и 5 худших классов хотя бы по одному полю
- [x] error analysis: визуальная неоднозначность vs шум разметки
- [x] breakdown ошибок по `layout_mode`, `bg_type`, `foreground_ratio`, scale bar, overlay text

### 18. Реализовать инференс-скрипт и генерацию auto_description
Status: `done`

Уже есть:
- `predict.py` запускается
- есть smoke-run `artifacts/preds/smoke_inference.csv`
- есть полный `artifacts/preds/inference.csv`
- `auto_description` собирается по фиксированным порогам и сохраняется вместе с `pred_*` и `confidence_*`

Нужно доделать:
- [x] починить `predict.py`
- [x] обеспечить совместимость preprocessing с train/val
- [x] прогнать инференс на реальных изображениях
- [x] сохранить `artifacts/preds/inference.csv`
- [x] показать пример итогового CSV
- [x] показать 5-10 реальных предсказаний
- [x] показать хотя бы 1 случай, где низкая уверенность меняет текст

### 19. Провести ablation study
Status: `done`

Уже есть:
- 3 сравнимых CPU-эксперимента: `configs/ablation_cpu_base.yaml`, `configs/ablation_cpu_no_class_weights.yaml`, `configs/ablation_cpu_stretch.yaml`
- сводная таблица: `artifacts/ablations/report/ablation_results.csv`
- markdown summary: `artifacts/ablations/report/ablation_summary.md`
- выбор финального baseline зафиксирован в `README.md` и `REPORT.md`

Нужно доделать:
- [x] сделать минимум 3 сравнимых эксперимента
- [x] менять по одному фактору за раз
- [x] свести результаты в одну таблицу
- [x] выбрать финальный конфиг по результатам
- [x] хотя бы одно сравнение посвятить preprocessing

### 20. Финальная модель, test / hidden inference и анализ ошибок
Status: `done`

Нужно доделать:
- [x] выбрать финальный конфиг
- [x] один раз оценить на `test_open`
- [x] показать итоговые метрики на `test_open`
- [x] показать 5 удачных и 5 неудачных кейсов
- [x] разобрать ошибки: фрагментарность, качество фото, неоднозначность материала, шум разметки, ограничения шаблона
- [x] показать случаи осторожного поведения модели
- [x] проверить зависимость от фона, линейки, подписи карточки

### 21. Финальный чек-лист перед сдачей
Status: `done`

Уже есть:
- `README.md`
- `REPORT.md`
- `requirements.txt`
- `configs/baseline.yaml`
- код по основным модулям
- `best.pt`
- `train_log.csv`
- `val_metrics.json`
- `test_metrics.json`
- `artifacts/preds/inference.csv`
- таблица экспериментов в `artifacts/ablations/report/ablation_results.csv`

Нужно доделать:
- [x] собрать финальный summary по полям, `group_key`, backbone, image size, аугментациям, optimizer/scheduler
- [x] зафиксировать лучшую `val`-метрику
- [x] зафиксировать итоговую `test_open`-метрику или `hidden inference` CSV
- [x] записать 2-3 ключевых вывода по экспериментам
- [x] приложить notebook или эквивалентный воспроизводимый отчёт
- [x] приложить таблицу экспериментов
- [x] приложить примеры удачных и неудачных предсказаний

## Обязательные артефакты перед сдачей

- [x] `README.md`
- [x] `requirements.txt`
- [x] `configs/baseline.yaml`
- [x] `src/similis_baseline/*.py`
- [x] `artifacts/checkpoints/best.pt`
- [x] `artifacts/preds/inference.csv`
- [x] notebook `.ipynb` или эквивалентный полный воспроизводимый отчёт
- [x] финальная таблица экспериментов
- [x] confusion matrices и error analysis артефакты
- [x] `test_open` evaluation report

## Минимальный критерий "можно продолжать реализацию"

Перед следующими экспериментами обязательно закрыть:
- [x] единый источник truth для split-файлов
- [x] рабочий `predict.py`
- [x] единый preprocessing для train/val/predict
- [x] sanity-check батча
- [x] one-batch debug
- [x] генерацию `inference.csv`
