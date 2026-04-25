"""Generate final data-centric inference + success/failure case comparison.

For #20:
  1. Run predict.py with the best AL checkpoint (least_confidence@B=100) over
     all images in data/raw/images → artifacts/preds/inference_data_centric.csv
  2. Compare per-row predictions on test_gold:
     baseline_seed_test_detailed/predictions.csv  (B=0)
     vs least_confidence/B100/reports/test_detailed/predictions.csv (B=100)
     - 5 wins:  baseline wrong → AL right (any of 4 fields)
     - 5 still-hard: both wrong on the same field

Outputs:
  artifacts/preds/inference_data_centric.csv
  artifacts/reports/data_centric/final_wins_examples.csv
  artifacts/reports/data_centric/final_still_hard_examples.csv
  artifacts/reports/data_centric/final_wins_grid.png
  artifacts/reports/data_centric/final_still_hard_grid.png
  artifacts/reports/data_centric/final_summary.json
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.similis_baseline.utils import ensure_dir, save_json  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

BEST_AL_CHECKPOINT = ROOT / "artifacts/active_learning/least_confidence/B100/checkpoints/best.pt"
BASELINE_PRED_CSV = ROOT / "artifacts/reports/data_centric/baseline_seed_test_detailed/predictions.csv"
AL_PRED_CSV = ROOT / "artifacts/active_learning/least_confidence/B100/reports/test_detailed/predictions.csv"
INFERENCE_OUT = ROOT / "artifacts/preds/inference_data_centric.csv"
RPT_DIR = ROOT / "artifacts/reports/data_centric"
FIELDS = ["type", "part", "integrity", "material"]


def run_inference() -> None:
    print(f"--- predict.py with {BEST_AL_CHECKPOINT.relative_to(ROOT)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.similis_baseline.predict",
            "--checkpoint",
            str(BEST_AL_CHECKPOINT),
            "--input-dir",
            "data/raw/images",
            "--output",
            str(INFERENCE_OUT),
        ],
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"predict.py failed rc={proc.returncode}")
    print(f"  saved → {INFERENCE_OUT.relative_to(ROOT)}")


def field_correctness(row: pd.Series, field: str) -> int | None:
    """Return 1 if correct, 0 if wrong, None if missing in GT."""
    gt = row.get(f"gt_{field}", "__MISSING__")
    if gt in (None, "", "__MISSING__") or pd.isna(gt):
        return None
    pred = row.get(f"pred_{field}", "")
    return int(str(gt) == str(pred))


def grid_from_rows(
    rows: pd.DataFrame,
    title: str,
    out_png: Path,
) -> None:
    n = len(rows)
    cols = 5
    rows_count = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_count, cols, figsize=(cols * 4.0, rows_count * 4.5))
    axes = np.array(axes).reshape(rows_count, cols)
    for i in range(rows_count * cols):
        ax = axes[i // cols, i % cols]
        if i >= n:
            ax.axis("off")
            continue
        r = rows.iloc[i]
        try:
            img = Image.open(r["image_file"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (256, 256), (200, 200, 200))
        ax.imshow(np.asarray(img))
        gt_b = "·".join(str(r.get(f"gt_{f}", "?"))[:8] for f in FIELDS)
        bs = "·".join(str(r.get(f"baseline_pred_{f}", "?"))[:8] for f in FIELDS)
        al = "·".join(str(r.get(f"al_pred_{f}", "?"))[:8] for f in FIELDS)
        ax.set_title(
            f"{r['image_file'].rsplit('/', 1)[-1]}\nGT: {gt_b}\nbase: {bs}\nAL:  {al}",
            fontsize=7.5,
        )
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dir(INFERENCE_OUT.parent)
    ensure_dir(RPT_DIR)

    if not BEST_AL_CHECKPOINT.exists():
        raise FileNotFoundError(f"missing {BEST_AL_CHECKPOINT}")

    if not INFERENCE_OUT.exists():
        run_inference()
    else:
        print(f"--- skipping inference (already exists): {INFERENCE_OUT.relative_to(ROOT)}")

    base = pd.read_csv(BASELINE_PRED_CSV)
    al = pd.read_csv(AL_PRED_CSV)

    keep = ["image_file", "group_key"] + [f"gt_{f}" for f in FIELDS] + [f"pred_{f}" for f in FIELDS] + [f"correct_{f}" for f in FIELDS] + ["pred_auto_description", "gt_auto_description", "auto_description_match"]
    keep = [c for c in keep if c in base.columns]

    merged = base[keep].merge(
        al[keep],
        on=["image_file", "group_key"],
        suffixes=("_base", "_al"),
    )

    for f in FIELDS:
        merged[f"baseline_pred_{f}"] = merged[f"pred_{f}_base"]
        merged[f"al_pred_{f}"] = merged[f"pred_{f}_al"]
        merged[f"gt_{f}"] = merged[f"gt_{f}_base"]
        merged[f"baseline_correct_{f}"] = merged[f"correct_{f}_base"]
        merged[f"al_correct_{f}"] = merged[f"correct_{f}_al"]

    fields_with_gt = [f for f in FIELDS if f"gt_{f}_base" in merged.columns]
    merged["wins_count"] = sum(
        ((merged[f"baseline_correct_{f}"] == 0) & (merged[f"al_correct_{f}"] == 1)).astype(int)
        for f in fields_with_gt
    )
    merged["regressions_count"] = sum(
        ((merged[f"baseline_correct_{f}"] == 1) & (merged[f"al_correct_{f}"] == 0)).astype(int)
        for f in fields_with_gt
    )
    merged["both_wrong_count"] = sum(
        ((merged[f"baseline_correct_{f}"] == 0) & (merged[f"al_correct_{f}"] == 0)).astype(int)
        for f in fields_with_gt
    )

    out_cols = [
        "image_file",
        "group_key",
        *[f"gt_{f}" for f in FIELDS],
        *[f"baseline_pred_{f}" for f in FIELDS],
        *[f"al_pred_{f}" for f in FIELDS],
        *[f"baseline_correct_{f}" for f in FIELDS],
        *[f"al_correct_{f}" for f in FIELDS],
        "wins_count",
        "regressions_count",
        "both_wrong_count",
    ]
    out_cols = [c for c in out_cols if c in merged.columns]
    summary_df = merged[out_cols].copy()

    wins = (
        summary_df[summary_df["wins_count"] > 0]
        .sort_values(["wins_count", "regressions_count"], ascending=[False, True])
        .head(5)
    )
    still_hard = (
        summary_df[summary_df["both_wrong_count"] > 0]
        .sort_values("both_wrong_count", ascending=False)
        .head(5)
    )

    wins.to_csv(RPT_DIR / "final_wins_examples.csv", index=False)
    still_hard.to_csv(RPT_DIR / "final_still_hard_examples.csv", index=False)
    summary_df.to_csv(RPT_DIR / "final_baseline_vs_al_per_row.csv", index=False)

    grid_from_rows(
        wins,
        "5 wins: baseline_seed (B=0) ошибается → least_confidence × B=100 угадывает",
        RPT_DIR / "final_wins_grid.png",
    )
    grid_from_rows(
        still_hard,
        "5 still-hard cases: оба ошибаются (baseline_seed и least_confidence × B=100)",
        RPT_DIR / "final_still_hard_grid.png",
    )

    summary = {
        "best_al_checkpoint": str(BEST_AL_CHECKPOINT.relative_to(ROOT)),
        "test_gold_rows": int(len(merged)),
        "rows_with_at_least_one_win": int((summary_df["wins_count"] > 0).sum()),
        "rows_with_at_least_one_regression": int((summary_df["regressions_count"] > 0).sum()),
        "rows_with_at_least_one_both_wrong": int((summary_df["both_wrong_count"] > 0).sum()),
        "fields_with_gt": fields_with_gt,
        "inference_csv": str(INFERENCE_OUT.relative_to(ROOT)),
        "inference_csv_rows": int(len(pd.read_csv(INFERENCE_OUT))),
    }
    save_json(summary, RPT_DIR / "final_summary.json")
    print("--- summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
