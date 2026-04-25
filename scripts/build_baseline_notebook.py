"""Build a thin baseline_report.ipynb that renders existing artifacts inline.

The notebook only reads CSV/JSON/PNG already produced by the pipeline; it does
not retrain or re-evaluate. Run this script after evaluate_detailed/predict
have refreshed val_detailed, test_detailed, inference.csv.
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "baseline_report.ipynb"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


CELLS: list[dict] = []

CELLS.append(md(
    "# SIMILIS Baseline — итоговый отчёт\n"
    "\n"
    "Воспроизводимый сводный ноутбук по треку **1.1 SIMILIS — baseline**. Здесь нет обучения и нет вычислений на GPU — все\n"
    "цифры, графики и confusion matrices читаются из уже сохранённых артефактов pipeline.\n"
    "\n"
    "Pipeline-скрипты: `src/similis_baseline/data_prep.py`, `train.py`, `evaluate_detailed.py`, `predict.py`,\n"
    "`data_report.py`, `error_factor_report.py`, `ablation_report.py`, `cautious_examples.py`, `safe_crop_demo.py`,\n"
    "`one_batch_debug.py`, `tiny_overfit.py`.\n"
    "\n"
    "Подробное обоснование решений — в `REPORT.md`. Подробный gap-чеклист — в `IMPLEMENTATION_CHECKLIST.md`.\n"
))

CELLS.append(md("## 1. Окружение, seed и рабочие папки"))
CELLS.append(code(
    "import json, sys, platform, subprocess\n"
    "from pathlib import Path\n"
    "\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import torch\n"
    "import torchvision\n"
    "from IPython.display import Image as IPyImage, Markdown, display\n"
    "\n"
    "ROOT = Path('.').resolve()\n"
    "print('cwd          :', ROOT)\n"
    "print('python       :', sys.version.split()[0])\n"
    "print('platform     :', platform.platform())\n"
    "print('torch        :', torch.__version__)\n"
    "print('torchvision  :', torchvision.__version__)\n"
    "print('cuda         :', torch.cuda.is_available())\n"
    "print('mps          :', getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available())\n"
))
CELLS.append(code(
    "for p in [\n"
    "    'data/raw', 'data/processed', 'splits',\n"
    "    'artifacts/checkpoints', 'artifacts/preds', 'artifacts/reports',\n"
    "    'artifacts/figures', 'artifacts/ablations',\n"
    "]:\n"
    "    exists = Path(p).exists()\n"
    "    print(f'  {p:35s}  exists={exists}')\n"
))

CELLS.append(md(
    "## 2. Загрузка данных и raw-проверка\n"
    "\n"
    "Открытый сабсет: `data/raw/selected_by_name_iimk_subset_public.csv` + папка `data/raw/images/`.\n"
    "Сводка по матчингу CSV ↔ изображениям лежит в `data/processed/prep_stats.json` и в `artifacts/reports/data_report/report.json`.\n"
))
CELLS.append(code(
    "manifest = pd.read_csv('data/processed/full_manifest.csv')\n"
    "print('rows in full_manifest :', len(manifest))\n"
    "print('rows with valid image :', int(manifest['has_image'].sum()))\n"
    "print('rows missing image    :', int((~manifest['has_image'].astype(bool)).sum()))\n"
    "print()\n"
    "print('columns:', manifest.columns.tolist())\n"
    "manifest.head(5)\n"
))
CELLS.append(code(
    "with open('data/processed/prep_stats.json', encoding='utf-8') as f:\n"
    "    prep_stats = json.load(f)\n"
    "prep_stats\n"
))

CELLS.append(md(
    "## 3. Структура датасета и group_key\n"
    "\n"
    "Сплит делается group-aware по `group_key`. В этом открытом корпусе **все 1387 строк имеют уникальный `code`**,\n"
    "а никакая комбинация других колонок (`name`, `description`, `cultlayer+execorg+survyear`) не даёт реальных\n"
    "артефакт-уровневых повторов: совпадения в `description` оказываются разными находками с одинаковой\n"
    "обобщённой формулировкой. Поэтому group_key=code эквивалентен row-level split — корректно для этих данных,\n"
    "но **показать 2-3 неразорванные группы невозможно физически**: в корпусе нет групп размера >1.\n"
    "Когда подключатся данные с настоящим artifact_id, group-aware логика заработает без изменений в коде.\n"
))
CELLS.append(code(
    "with open('artifacts/reports/data_report/report.json', encoding='utf-8') as f:\n"
    "    dr = json.load(f)\n"
    "for k in [\n"
    "    'csv_rows','image_files_on_disk','rows_with_valid_image','broken_path_count',\n"
    "    'duplicate_image_file_count','repeated_group_key_count','seed','split_reproducible_from_seed',\n"
    "    'train_inner_size','val_inner_size','test_open_size',\n"
    "    'group_intersection_train_val','group_intersection_train_test','group_intersection_val_test',\n"
    "]:\n"
    "    if k in dr:\n"
    "        print(f'  {k:40s} {dr[k]}')\n"
))
CELLS.append(code(
    "display(IPyImage(filename='artifacts/reports/data_report/row_image_examples.png'))\n"
))

CELLS.append(md(
    "## 4. EDA — визуальные режимы и проблемные изображения\n"
    "\n"
    "Heuristics в `image_analysis.py`: `foreground_ratio`, `bg_type`, `layout_mode`, `has_scale_bar`, `has_overlay_text`.\n"
))
CELLS.append(code(
    "for fname in ['sample_grid.png','layout_mode_examples.png','problematic_images.png']:\n"
    "    p = Path('artifacts/reports/data_report')/fname\n"
    "    if p.exists():\n"
    "        display(Markdown(f'### {fname}'))\n"
    "        display(IPyImage(filename=str(p)))\n"
))
CELLS.append(code(
    "heur = pd.read_csv('artifacts/reports/data_report/heuristic_summary.csv')\n"
    "heur\n"
))

CELLS.append(md(
    "### Safe crop vs pad\n"
    "\n"
    "Сравнение pad vs safe content-aware crop vs aggressive center-crop на изображении с большим белым фоном.\n"
    "Safe crop сохраняет foreground, aggressive center-crop отрезает часть артефакта.\n"
    "Скрипт: `src/similis_baseline/safe_crop_demo.py`.\n"
))
CELLS.append(code(
    "display(IPyImage(filename='artifacts/reports/transform_report/safe_crop_vs_pad.png'))\n"
))

CELLS.append(md(
    "## 5. Выбор полей baseline и нормализация словарей\n"
    "\n"
    "Baseline предсказывает 4 нормализованных поля: `type`, `part`, `integrity`, `material`.\n"
    "Невизуальные поля (`size`, `cultlayer`, `survyear`, `execorg`, функция, датировка) сознательно исключены.\n"
))
CELLS.append(code(
    "fc = pd.read_csv('artifacts/reports/data_report/field_candidates.csv')\n"
    "fc\n"
))
CELLS.append(code(
    "norm = pd.read_csv('artifacts/reports/data_report/normalization_examples.csv')\n"
    "norm.head(20)\n"
))
CELLS.append(code(
    "policy = pd.read_csv('artifacts/reports/data_report/label_policy.csv')\n"
    "policy\n"
))

CELLS.append(md(
    "## 6. Шаблон auto_description и правила confidence\n"
    "\n"
    "Все поля confidence-gated (порог из `configs/baseline.yaml`). Если ни одно поле не прошло порог, возвращается\n"
    "fallback `не удалось уверенно собрать описание`.\n"
    "\n"
    "Порядок полей в auto_description:\n"
    "1. `type`\n"
    "2. `material`\n"
    "3. `part`\n"
    "4. слово `фрагмент`, если `integrity=фрагмент` и confidence прошёл порог.\n"
))
CELLS.append(code(
    "import yaml\n"
    "with open('configs/baseline.yaml', encoding='utf-8') as f:\n"
    "    cfg = yaml.safe_load(f)\n"
    "print('confidence_thresholds:', cfg['confidence_thresholds'])\n"
    "print('fields              :', cfg['fields'])\n"
))
CELLS.append(code(
    "infer = pd.read_csv('artifacts/preds/inference.csv')\n"
    "cols = ['image_file','auto_description','pred_type','pred_material','pred_part','pred_integrity']\n"
    "conf_cols = [c for c in infer.columns if c.startswith('confidence_')]\n"
    "infer[cols + conf_cols].head(10)\n"
))
CELLS.append(code(
    "# 1 пример с низкой confidence -> поле опущено или fallback\n"
    "low = infer.copy()\n"
    "low['min_conf'] = low[conf_cols].min(axis=1)\n"
    "low.sort_values('min_conf').head(5)[cols + ['min_conf']]\n"
))

CELLS.append(md(
    "## 7. Модель — multi-task ConvNeXt-Tiny\n"
    "\n"
    "Backbone + по одной classification head на каждое поле; logits на выходе (без softmax). Описание — `src/similis_baseline/model.py`.\n"
))
CELLS.append(code(
    "with open('artifacts/reports/model_summary.json', encoding='utf-8') as f:\n"
    "    ms = json.load(f)\n"
    "ms\n"
))

CELLS.append(md(
    "## 8. One-batch debug и tiny-overfit (sanity-checks)\n"
    "\n"
    "- `one_batch_debug.py` — shapes/dtypes, loss/acc по каждому полю на одном батче.\n"
    "- `tiny_overfit.py` — на 32–64 объектах модель должна заметно переобучиться.\n"
    "Артефакты лежат в `artifacts/reports/tiny_overfit_type_clean_current/`.\n"
))
CELLS.append(code(
    "tof_dir = Path('artifacts/reports/tiny_overfit_type_clean_current')\n"
    "if tof_dir.exists():\n"
    "    for png in sorted(tof_dir.glob('*.png')):\n"
    "        display(Markdown(f'### {png.name}'))\n"
    "        display(IPyImage(filename=str(png)))\n"
    "else:\n"
    "    print('tiny_overfit dir missing')\n"
))

CELLS.append(md(
    "## 9. Полный train/eval цикл — кривые обучения\n"
    "\n"
    "Лог `artifacts/reports/train_log.csv` соответствует текущему `artifacts/checkpoints/best.pt`.\n"
))
CELLS.append(code(
    "import matplotlib.pyplot as plt\n"
    "log = pd.read_csv('artifacts/reports/train_log.csv')\n"
    "log\n"
))
CELLS.append(code(
    "fig, axes = plt.subplots(1, 3, figsize=(16, 4))\n"
    "axes[0].plot(log['epoch'], log['train_loss'], 'o-', label='train_loss')\n"
    "axes[0].plot(log['epoch'], log['val_loss'], 'o-', label='val_loss')\n"
    "axes[0].set_xlabel('epoch'); axes[0].set_ylabel('loss'); axes[0].legend(); axes[0].grid(alpha=.3)\n"
    "axes[0].set_title('Loss curves')\n"
    "for f in ['type','part','integrity','material']:\n"
    "    col = f'{f}_macro_f1'\n"
    "    if col in log.columns:\n"
    "        axes[1].plot(log['epoch'], log[col], 'o-', label=f)\n"
    "axes[1].plot(log['epoch'], log['mean_macro_f1'], 'k--', label='mean', linewidth=2)\n"
    "axes[1].set_xlabel('epoch'); axes[1].set_ylabel('macro-F1'); axes[1].legend(fontsize=8); axes[1].grid(alpha=.3)\n"
    "axes[1].set_title('Per-field macro-F1')\n"
    "axes[2].plot(log['epoch'], log['head_lr'], 'o-', label='head_lr')\n"
    "axes[2].plot(log['epoch'], log['backbone_lr'], 'o-', label='backbone_lr')\n"
    "axes[2].set_xlabel('epoch'); axes[2].set_ylabel('lr'); axes[2].set_yscale('log'); axes[2].legend(); axes[2].grid(alpha=.3)\n"
    "axes[2].set_title('Learning rate schedule')\n"
    "plt.tight_layout(); plt.show()\n"
))

CELLS.append(md(
    "## 10. Метрики на валидации + confusion matrices\n"
))
CELLS.append(code(
    "with open('artifacts/reports/val_detailed/metrics.json', encoding='utf-8') as f:\n"
    "    val_metrics = json.load(f)\n"
    "val_metrics\n"
))
CELLS.append(code(
    "cw = pd.read_csv('artifacts/reports/val_detailed/classwise_metrics.csv')\n"
    "cw\n"
))
CELLS.append(code(
    "for f in ['type','part','integrity','material']:\n"
    "    p = Path('artifacts/reports/val_detailed') / f'confusion_{f}.png'\n"
    "    if p.exists():\n"
    "        display(Markdown(f'### val: {f}'))\n"
    "        display(IPyImage(filename=str(p)))\n"
))
CELLS.append(code(
    "with open('artifacts/reports/val_detailed/best_worst_classes.json', encoding='utf-8') as f:\n"
    "    bw = json.load(f)\n"
    "bw\n"
))

CELLS.append(md(
    "## 11. Метрики на открытом тесте\n"
))
CELLS.append(code(
    "with open('artifacts/reports/test_detailed/metrics.json', encoding='utf-8') as f:\n"
    "    test_metrics = json.load(f)\n"
    "test_metrics\n"
))
CELLS.append(code(
    "for f in ['type','part','integrity','material']:\n"
    "    p = Path('artifacts/reports/test_detailed') / f'confusion_{f}.png'\n"
    "    if p.exists():\n"
    "        display(Markdown(f'### test: {f}'))\n"
    "        display(IPyImage(filename=str(p)))\n"
))

CELLS.append(md(
    "## 12. Error analysis по факторам\n"
))
CELLS.append(code(
    "for split in ['val','test']:\n"
    "    p = Path(f'artifacts/reports/{split}_error_factors/factor_breakdown.csv')\n"
    "    if p.exists():\n"
    "        display(Markdown(f'### {split} factor_breakdown'))\n"
    "        display(pd.read_csv(p))\n"
))
CELLS.append(code(
    "for split in ['val','test']:\n"
    "    p = Path(f'artifacts/reports/{split}_error_factors/error_reason_breakdown.csv')\n"
    "    if p.exists():\n"
    "        display(Markdown(f'### {split} error_reason_breakdown'))\n"
    "        display(pd.read_csv(p))\n"
))

CELLS.append(md(
    "## 13. Ablation study\n"
    "\n"
    "3 сравнимых CPU-эксперимента: `pad+class_weights`, `pad+no_class_weights`, `stretch+class_weights` при\n"
    "`image_size=224`, `epochs=4`, `freeze_backbone_epochs=1`.\n"
))
CELLS.append(code(
    "abl = pd.read_csv('artifacts/ablations/report/ablation_results.csv')\n"
    "abl\n"
))
CELLS.append(code(
    "summary_md = Path('artifacts/ablations/report/ablation_summary.md').read_text(encoding='utf-8')\n"
    "display(Markdown(summary_md))\n"
))

CELLS.append(md(
    "## 14. Финальная модель — удачные и неудачные кейсы\n"
))
CELLS.append(code(
    "succ = pd.read_csv('artifacts/reports/test_detailed/successful_examples.csv').head(5)\n"
    "fail = pd.read_csv('artifacts/reports/test_detailed/failed_examples.csv').head(5)\n"
    "display(Markdown('### test: 5 удачных'))\n"
    "display(succ)\n"
    "display(Markdown('### test: 5 неудачных'))\n"
    "display(fail)\n"
))

CELLS.append(md(
    "## 15. Cautious behaviour — где модель не настаивает на поле\n"
))
CELLS.append(code(
    "for split in ['val','test']:\n"
    "    p = Path(f'artifacts/reports/{split}_cautious_examples/cautious_examples.csv')\n"
    "    if p.exists():\n"
    "        display(Markdown(f'### {split}: cautious examples'))\n"
    "        display(pd.read_csv(p).head(5))\n"
))

CELLS.append(md(
    "## 16. Финальный чек-лист (задание 21)\n"
    "\n"
    "| Пункт | Значение |\n"
    "|---|---|\n"
    "| Поля | `type`, `part`, `integrity`, `material` |\n"
    "| group_key | `code` (proxy; в открытом корпусе нет повторов артефактов, см. раздел 3) |\n"
    "| Backbone | `convnext_tiny` (pretrained) |\n"
    "| Размер изображения | 384 (safe `pad+resize`) |\n"
    "| Аугментации | RandomHorizontalFlip, RandomRotation(±5°), ColorJitter(brightness/contrast=0.1) |\n"
    "| Optimizer | AdamW (head_lr=3e-4, backbone_lr=3e-5, weight_decay=1e-4) |\n"
    "| Scheduler | per-group cosine, min_lr=1e-6, freeze_backbone 2 epochs |\n"
    "| Epochs | 9 |\n"
    "| Лучшая val mean_macro_f1 | см. ячейку выше (`artifacts/reports/val_detailed/metrics.json`) |\n"
    "| test_open mean_macro_f1 | см. ячейку выше (`artifacts/reports/test_detailed/metrics.json`) |\n"
    "| Полный train_log синхронизирован с best.pt | да, см. `artifacts/reports/train_log.csv` |\n"
    "\n"
    "Ключевые выводы:\n"
    "1. Multi-task baseline с 4 полями работает и даёт контролируемый `auto_description` без свободной генерации.\n"
    "2. Самое слабое поле — `part` (визуальная неоднозначность + высокая доля missing).\n"
    "3. Preprocessing и class_weights критически влияют на перенос с val на test_open: на коротком CPU-study\n"
    "   `stretch` выиграл val, но `pad+class_weights` лучше переносился на open-test.\n"
    "4. Ошибки сильнее связаны с layout-режимом и фоном, чем с наличием scale bar (которого в открытом корпусе почти нет).\n"
))


def main() -> None:
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
