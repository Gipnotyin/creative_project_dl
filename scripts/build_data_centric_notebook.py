"""Build data_centric_report.ipynb that renders existing artifacts inline.

Mirrors scripts/build_baseline_notebook.py but for the data-centric track:
loads CSV/JSON/PNG/NPY artifacts and renders them in a single executed notebook.
No retraining, no GPU work — purely report assembly.
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "data_centric_report.ipynb"


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
    "# SIMILIS Data-Centric — итоговый отчёт\n"
    "\n"
    "Воспроизводимый сводный ноутбук по треку **1.2 SIMILIS — data-centric**. Здесь нет обучения и нет вычислений на GPU —\n"
    "все цифры, графики и confusion matrices читаются из уже сохранённых артефактов pipeline.\n"
    "\n"
    "Pipeline-скрипты:\n"
    "`src/similis_baseline/data_centric_split.py`,\n"
    "`data_centric_eda.py`,\n"
    "`data_centric_sanity_check.py`,\n"
    "`uncertainty_and_embeddings.py`,\n"
    "`nearest_neighbors_viz.py`,\n"
    "`review_table.py`,\n"
    "`al_strategies.py`,\n"
    "`al_loop.py`,\n"
    "`al_compare.py`,\n"
    "`scripts/dc_final_analysis.py`.\n"
    "\n"
    "Подробное обоснование решений — в `REPORT_DATA_CENTRIC.md`. Подробный gap-чеклист — в `DATA_CENTRIC_CHECKLIST.md`.\n"
))

CELLS.append(md("## 1. Окружение, seed и data-centric папки"))
CELLS.append(code(
    "import json, sys, platform\n"
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
    "    'data/processed/data_centric',\n"
    "    'artifacts/checkpoints/data_centric',\n"
    "    'artifacts/reports/data_centric',\n"
    "    'artifacts/review',\n"
    "    'artifacts/embeddings',\n"
    "    'artifacts/active_learning',\n"
    "]:\n"
    "    exists = Path(p).exists()\n"
    "    print(f'  {p:42s} exists={exists}')\n"
))

CELLS.append(md(
    "## 2-3. Пулы и роли\n"
    "\n"
    "4 пула с фиксированным seed:\n"
    "\n"
    "| pool | роль | метки |\n"
    "|---|---|---|\n"
    "| `train_seed` | стартовый train для baseline-под-AL и retrain'ов после query | открыто |\n"
    "| `val_gold` | чистая (label_is_uncertain=0) валидация для выбора стратегии | открыто |\n"
    "| `test_gold` | финальная метрика, = baseline test_open для прямого сравнения | открыто |\n"
    "| `pool_candidate` | пул на доразметку, метки скрыты (`__HIDDEN__`); правда в `pool_candidate_oracle.csv` | закрыто до query |\n"
))
CELLS.append(code(
    "with open('artifacts/reports/data_centric/split_report.json', encoding='utf-8') as f:\n"
    "    sr = json.load(f)\n"
    "for k in ['seed','sizes','all_intersections_zero','covers_full_corpus']:\n"
    "    if k in sr:\n"
    "        print(f'  {k:35s} {sr[k]}')\n"
))
CELLS.append(code(
    "manifest = pd.read_csv('data/processed/data_centric/manifest_data_centric.csv')\n"
    "print('manifest rows:', len(manifest))\n"
    "print('pool counts :', manifest['pool'].value_counts().to_dict())\n"
    "manifest[['code','image_file','pool','quality_flag','noise_score','label_is_uncertain']].head(10)\n"
))

CELLS.append(md("## 4. EDA по пулам — структура, label noise, nuisance"))
CELLS.append(code(
    "with open('artifacts/reports/data_centric/pool_structure.json', encoding='utf-8') as f:\n"
    "    ps = json.load(f)\n"
    "rows = []\n"
    "for pool, info in ps.items():\n"
    "    rows.append({'pool': pool, **info})\n"
    "pd.DataFrame(rows)\n"
))
CELLS.append(code(
    "noise_dist = pd.read_csv('artifacts/reports/data_centric/noise_score_distribution.csv')\n"
    "noise_dist\n"
))
CELLS.append(code(
    "noise_top = pd.read_csv('artifacts/reports/data_centric/label_noise_candidates.csv').head(10)\n"
    "noise_top[['code','description','type','material','noise_score','quality_flag','noise_reasons']]\n"
))
CELLS.append(code(
    "for f in ['material','part','type','integrity']:\n"
    "    p = Path(f'artifacts/reports/data_centric/nuisance_breakdown_{f}.csv')\n"
    "    if p.exists():\n"
    "        df = pd.read_csv(p)\n"
    "        display(Markdown(f'### nuisance breakdown — {f}'))\n"
    "        display(df.head(20))\n"
))

CELLS.append(md(
    "## 5. Выбор полей и гипотезы\n"
    "\n"
    "Primary: **`material`** (4 trainable классов, baseline test 0.84 → есть место для роста, реальный шум описаний).\n"
    "Secondary: **`part`** (5 классов, baseline test 0.59, ~42% missing — есть запас на доразметку).\n"
    "\n"
    "Гипотезы (зафиксированы ДО запуска AL):\n"
    "\n"
    "- **H1.** uncertainty_least_confidence > random по `material_macro_f1` на val_gold при B=50\n"
    "- **H2.** diversity_coreset даёт меньше дублирования query (выше попарная дистанция) и не теряет в качестве\n"
    "- **H3.** ≥30% строк в review-таблице окажутся label_noise_suspect (а не genuine errors)\n"
))

CELLS.append(md(
    "## 6. Quality flags и noise_score в manifest\n"
    "\n"
    "К baseline-нормализации добавлены:\n"
    "- `noise_score` ∈ [0, 1]\n"
    "- `flag_uncertain`, `flag_incomplete`, `flag_conflict`, `flag_rare`\n"
    "- `noise_reasons` (semicolon-separated)\n"
    "- `quality_flag` ∈ `{clean, incomplete, rare, conflict, uncertain, hidden}` (приоритет: uncertain > conflict > rare > incomplete > clean)\n"
    "- `pool_candidate` всегда `quality_flag=hidden` (анти-leakage, noise_score не вычисляется)\n"
))
CELLS.append(code(
    "rows = []\n"
    "for pool in ['train_seed','val_gold','test_gold','pool_candidate']:\n"
    "    sub = manifest[manifest['pool']==pool]\n"
    "    flags = sub['quality_flag'].value_counts().to_dict() if 'quality_flag' in sub.columns else {}\n"
    "    rows.append({'pool': pool, 'rows': int(len(sub)), **flags})\n"
    "pd.DataFrame(rows).fillna(0)\n"
))

CELLS.append(md(
    "## 7. Sanity-check протокола (`all_checks_pass`)\n"
))
CELLS.append(code(
    "with open('artifacts/reports/data_centric/data_centric_sanity_check.json', encoding='utf-8') as f:\n"
    "    sc = json.load(f)\n"
    "for k in [\n"
    "    'all_checks_pass','all_intersections_zero','val_gold_clean_by_baseline_flag',\n"
    "]:\n"
    "    if k in sc: print(f'  {k:42s} {sc[k]}')\n"
    "print('  pool_candidate masked all_hidden:', sc['pool_candidate_masked']['all_hidden'])\n"
    "print('  pool_candidate target_mask all_zero:', sc['dataset_batch_checks']['pool_candidate_eval'].get('pool_candidate_target_mask_all_zero_first_n'))\n"
    "print('  transform train stochastic    :', sc['transform_determinism']['train_is_stochastic'])\n"
    "print('  transform eval deterministic  :', sc['transform_determinism']['eval_is_deterministic'])\n"
))

CELLS.append(md(
    "## 8-10. Архитектура, главная метрика и baseline_seed\n"
    "\n"
    "Multi-task `convnext_tiny @384` (идентично baseline). Главная метрика —\n"
    "`material_macro_f1` на `val_gold`.\n"
))
CELLS.append(code(
    "with open('artifacts/reports/data_centric/baseline_seed_model_summary.json', encoding='utf-8') as f:\n"
    "    ms = json.load(f)\n"
    "ms\n"
))
CELLS.append(code(
    "log = pd.read_csv('artifacts/reports/data_centric/train_log.csv')\n"
    "log\n"
))
CELLS.append(code(
    "for fname in ['loss_curve.png','macro_f1_curve.png','lr_curve.png']:\n"
    "    p = Path(f'artifacts/reports/data_centric/baseline_seed_train_log_report/{fname}')\n"
    "    if p.exists():\n"
    "        display(Markdown(f'### {fname}'))\n"
    "        display(IPyImage(filename=str(p)))\n"
))
CELLS.append(code(
    "with open('artifacts/reports/data_centric/baseline_seed_val_detailed/metrics.json', encoding='utf-8') as f:\n"
    "    bv = json.load(f)\n"
    "with open('artifacts/reports/data_centric/baseline_seed_test_detailed/metrics.json', encoding='utf-8') as f:\n"
    "    bt = json.load(f)\n"
    "display(Markdown('### baseline_seed — val_gold'))\n"
    "display(bv)\n"
    "display(Markdown('### baseline_seed — test_gold'))\n"
    "display(bt)\n"
))
CELLS.append(code(
    "p = Path('artifacts/reports/data_centric/baseline_seed_val_detailed/confusion_material.png')\n"
    "if p.exists():\n"
    "    display(Markdown('### baseline_seed val_gold — confusion `material`'))\n"
    "    display(IPyImage(filename=str(p)))\n"
))

CELLS.append(md(
    "## 12. Сигналы неопределённости\n"
    "\n"
    "3 сигнала на каждое поле: `max_prob` (1 - max softmax), `entropy`, `margin`.\n"
    "Spearman 0.99+ между ними на всех пулах — взаимозаменяемые. Для AL #16 берём `least_confidence`.\n"
))
CELLS.append(code(
    "with open('artifacts/embeddings/embeddings_summary.json', encoding='utf-8') as f:\n"
    "    es = json.load(f)\n"
    "rows = []\n"
    "for pool, info in es['pools'].items():\n"
    "    rows.append({\n"
    "        'pool': pool,\n"
    "        'rows': info['rows'],\n"
    "        'emb_shape': info['embeddings_shape'],\n"
    "        'max_prob_p50': info.get('material_max_prob_p50'),\n"
    "        'max_prob_p90': info.get('material_max_prob_p90'),\n"
    "        'entropy_p50': info.get('material_entropy_p50'),\n"
    "        'entropy_p90': info.get('material_entropy_p90'),\n"
    "        **info['spearman_correlations_primary_field'],\n"
    "    })\n"
    "pd.DataFrame(rows)\n"
))

CELLS.append(md(
    "## 13. Эмбеддинги + nearest neighbors\n"
    "\n"
    "768-d backbone-features, для топ-5 неуверенных pool_candidate показаны 3 ближайших соседа в train_seed.\n"
))
CELLS.append(code(
    "p = Path('artifacts/embeddings/nearest_neighbors/nearest_neighbors.png')\n"
    "if p.exists():\n"
    "    display(IPyImage(filename=str(p)))\n"
))
CELLS.append(code(
    "nn = pd.read_csv('artifacts/embeddings/nearest_neighbors/nearest_neighbors.csv')\n"
    "nn\n"
))

CELLS.append(md(
    "## 14. Review-таблица (топ-40 подозрительных)\n"
    "\n"
    "Ranking = 0.5·disagreement + 0.4·uncertainty + 0.3·noise_score + 0.2·embedding_outlier.\n"
    "`reason_flag` ∈ `{label_noise_suspect, visual_nuisance, genuinely_hard}`.\n"
))
CELLS.append(code(
    "rev = pd.read_csv('artifacts/review/review_table.csv')\n"
    "print('review rows:', len(rev))\n"
    "print('reason_flag distribution:', rev['reason_flag'].value_counts().to_dict())\n"
    "print('disagreement count    :', int(rev['disagreement'].sum()))\n"
    "rev[['pool','code','norm_label','pred_label','confidence','uncertainty_score','noise_score','combined_score','reason_flag','quality_flag']].head(10)\n"
))
CELLS.append(code(
    "p = Path('artifacts/review/review_top10_grid.png')\n"
    "if p.exists():\n"
    "    display(IPyImage(filename=str(p)))\n"
))

CELLS.append(md(
    "## 15-17. Стратегии query — отбор query без retrain\n"
    "\n"
    "4 стратегии: random (control), least_confidence, coreset (k-center на эмбеддингах), hybrid (top-3B uncertain → coreset to B).\n"
    "8 query-CSVs: 4 стратегии × 2 бюджета.\n"
))
CELLS.append(code(
    "redund = pd.read_csv('artifacts/active_learning/comparison/strategy_redundancy.csv')\n"
    "display(Markdown('### Avg pairwise cosine distance among queried (H2)'))\n"
    "display(redund)\n"
))
CELLS.append(code(
    "nuis = pd.read_csv('artifacts/active_learning/comparison/strategy_nuisance.csv')\n"
    "display(Markdown('### Nuisance shares among queried'))\n"
    "display(nuis)\n"
))
CELLS.append(code(
    "for B in [50, 100]:\n"
    "    p = Path(f'artifacts/active_learning/comparison/strategy_overlap_B{B}.csv')\n"
    "    if p.exists():\n"
    "        display(Markdown(f'### strategy overlap @ B={B} (image_files in common)'))\n"
    "        display(pd.read_csv(p))\n"
))
CELLS.append(code(
    "with open('artifacts/active_learning/comparison/all_strategies_summary.json', encoding='utf-8') as f:\n"
    "    asum = json.load(f)\n"
    "rows = []\n"
    "for s in asum['summaries']:\n"
    "    cdist = s.get(f\"{s['primary_field']}_oracle_class_distribution\", {})\n"
    "    rows.append({\n"
    "        'strategy': s['strategy'],\n"
    "        'budget': s['budget'],\n"
    "        **cdist,\n"
    "    })\n"
    "display(Markdown('### Class distribution in queried (after oracle reveal) on `material`'))\n"
    "pd.DataFrame(rows).fillna(0).astype({'budget': int})\n"
))

CELLS.append(md(
    "## 18-19. AL симуляция и сравнение\n"
    "\n"
    "3 стратегии × 2 бюджета = 6 retrain'ов. Recipe идентичен baseline_seed.\n"
))
CELLS.append(code(
    "results = pd.read_csv('artifacts/active_learning/comparison/results.csv')\n"
    "results\n"
))
CELLS.append(code(
    "for fname, title in [\n"
    "    ('learning_curves.png', 'val_material_macro_f1 vs budget (PRIMARY)'),\n"
    "    ('learning_curves_mean.png', 'val_mean_macro_f1 vs budget (4 fields)'),\n"
    "    ('learning_curves_test.png', 'test_material_macro_f1 vs budget (final eval)'),\n"
    "]:\n"
    "    p = Path(f'artifacts/active_learning/comparison/{fname}')\n"
    "    if p.exists():\n"
    "        display(Markdown(f'### {title}'))\n"
    "        display(IPyImage(filename=str(p)))\n"
))
CELLS.append(code(
    "notes = Path('artifacts/active_learning/comparison/notes.md').read_text(encoding='utf-8')\n"
    "display(Markdown(notes))\n"
))

CELLS.append(md(
    "## 20. Финальная модель — wins, still-hard, recommendations\n"
    "\n"
    "Победитель: `least_confidence × B=100` — лучший на test_material (0.733, +0.151 vs baseline_seed=0.582).\n"
))
CELLS.append(code(
    "with open('artifacts/reports/data_centric/final_summary.json', encoding='utf-8') as f:\n"
    "    fs = json.load(f)\n"
    "fs\n"
))
CELLS.append(code(
    "p = Path('artifacts/reports/data_centric/final_wins_grid.png')\n"
    "if p.exists():\n"
    "    display(Markdown('### 5 wins: baseline ошибся → AL угадал'))\n"
    "    display(IPyImage(filename=str(p)))\n"
))
CELLS.append(code(
    "p = Path('artifacts/reports/data_centric/final_still_hard_grid.png')\n"
    "if p.exists():\n"
    "    display(Markdown('### 5 still-hard: оба ошибаются'))\n"
    "    display(IPyImage(filename=str(p)))\n"
))
CELLS.append(code(
    "wins = pd.read_csv('artifacts/reports/data_centric/final_wins_examples.csv')\n"
    "display(Markdown('### final_wins_examples (top 5)'))\n"
    "display(wins[['image_file','gt_material','baseline_pred_material','al_pred_material','wins_count','regressions_count']])\n"
    "still = pd.read_csv('artifacts/reports/data_centric/final_still_hard_examples.csv')\n"
    "display(Markdown('### final_still_hard_examples (top 5)'))\n"
    "display(still[['image_file','gt_material','baseline_pred_material','al_pred_material','both_wrong_count']])\n"
))

CELLS.append(md(
    "## 21. Финальный data-centric чек-лист\n"
    "\n"
    "| Поле | Значение |\n"
    "|---|---|\n"
    "| Поля (primary/secondary) | `material` / `part` |\n"
    "| Сплит | train_seed=350, val_gold=150, test_gold=209, pool_candidate=678 |\n"
    "| Backbone | `convnext_tiny` (pretrained), 27.8M params |\n"
    "| Image size | 384, `pad+resize` |\n"
    "| Optimizer | AdamW (head_lr=3e-4, backbone_lr=3e-5, weight_decay=1e-4) |\n"
    "| Scheduler | per-group cosine, min_lr=1e-6, freeze_backbone 2 epochs, 9 epochs total |\n"
    "| Uncertainty signals | max_prob, entropy, margin (Spearman 0.99+) |\n"
    "| Стратегии | random, least_confidence, coreset (+ hybrid query без retrain) |\n"
    "| Бюджеты | B=50, B=100 |\n"
    "| Лучший val_material@B=50 | **coreset = 0.788** (+0.085 vs baseline_seed) |\n"
    "| Лучший test_material@B=100 | **least_confidence = 0.733** (+0.151 vs baseline_seed) |\n"
    "| Финальный inference CSV | `artifacts/preds/inference_data_centric.csv` (1388 строк) |\n"
    "\n"
    "**3 главных вывода о корпусе** (см. REPORT_DATA_CENTRIC.md раздел 21):\n"
    "1. Корпус достаточно чистый по разметке — H3 не подтверждена auto-classification (10% < 30%); куратору нужна стандартизация словаря материала, а не массовая переразметка.\n"
    "2. Главные источники ошибок — макет карточки (`multi_view`, `light_photo`) и редкие классы (`стекло` всего 19 видимых строк).\n"
    "3. Pure uncertainty подвержен утечке через nuisance (4× больше close_up vs random). Production AL = uncertainty + nuisance-фильтр.\n"
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
