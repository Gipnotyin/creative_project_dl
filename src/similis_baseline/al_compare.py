"""AL strategy comparison + learning curves (#19).

Reads metrics from baseline_seed_best (B=0 reference) and from each
artifacts/active_learning/{strategy}/B{budget}/reports/{val,test}_detailed/metrics.json,
plus the redundancy/nuisance/overlap artifacts produced by al_strategies.py
and al_loop.py.

Outputs:
  artifacts/active_learning/comparison/
    results.csv                  — strategy, budget, val/test material/mean macro_f1,
                                   delta_vs_baseline, delta_vs_random,
                                   redundancy, nuisance shares
    learning_curves.png          — material_macro_f1 vs budget, one line per strategy
    learning_curves_mean.png     — mean_macro_f1 vs budget
    notes.md                     — short auto-generated summary
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import ensure_dir, load_config


def load_metrics(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_centric.yaml")
    parser.add_argument(
        "--output-dir",
        default="artifacts/active_learning/comparison",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    primary_field = cfg.get("primary_field", "material")
    budgets = list(cfg.get("active_learning", {}).get("budgets", [50, 100]))
    al_root = Path(cfg["active_learning_dir"])
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    # baseline_seed (B=0)
    baseline_val = load_metrics(
        Path("artifacts/reports/data_centric/baseline_seed_val_detailed/metrics.json")
    )
    baseline_test = load_metrics(
        Path("artifacts/reports/data_centric/baseline_seed_test_detailed/metrics.json")
    )
    baseline_val_primary = float(baseline_val.get(f"{primary_field}_macro_f1", 0.0))
    baseline_val_mean = float(baseline_val.get("mean_macro_f1", 0.0))
    baseline_test_primary = float(baseline_test.get(f"{primary_field}_macro_f1", 0.0))
    baseline_test_mean = float(baseline_test.get("mean_macro_f1", 0.0))

    # auto-discover strategies that actually have AL runs
    strategies = sorted(
        d.name
        for d in al_root.iterdir()
        if d.is_dir()
        and d.name not in ("comparison",)
        and any((d / f"B{b}").exists() for b in budgets)
    )
    print(f"discovered strategies: {strategies}")

    # redundancy / nuisance from al_strategies.py
    redundancy_csv = out_dir / "strategy_redundancy.csv"
    nuisance_csv = out_dir / "strategy_nuisance.csv"
    redundancy_df = pd.read_csv(redundancy_csv) if redundancy_csv.exists() else pd.DataFrame()
    nuisance_df = pd.read_csv(nuisance_csv) if nuisance_csv.exists() else pd.DataFrame()

    rows: List[Dict] = []
    rows.append(
        {
            "strategy": "baseline_seed",
            "budget": 0,
            "val_material_macro_f1": baseline_val_primary,
            "val_mean_macro_f1": baseline_val_mean,
            "test_material_macro_f1": baseline_test_primary,
            "test_mean_macro_f1": baseline_test_mean,
            "delta_val_material_vs_baseline": 0.0,
            "delta_val_mean_vs_baseline": 0.0,
            "delta_val_material_vs_random_same_budget": np.nan,
            "avg_pairwise_cosine_distance": np.nan,
            "close_up_share": np.nan,
            "small_foreground_share": np.nan,
            "has_overlay_text_share": np.nan,
            "notes": "B=0 reference (train_seed only)",
        }
    )

    # compute random baselines per budget for delta_vs_random
    random_val_primary_by_budget: Dict[int, float] = {}
    random_val_mean_by_budget: Dict[int, float] = {}

    # collect per-strategy results per budget
    for strategy in strategies:
        for budget in budgets:
            run_dir = al_root / strategy / f"B{budget}"
            val = load_metrics(run_dir / "reports" / "val_detailed" / "metrics.json")
            test = load_metrics(run_dir / "reports" / "test_detailed" / "metrics.json")
            v_pri = float(val.get(f"{primary_field}_macro_f1", 0.0)) if val else np.nan
            v_mean = float(val.get("mean_macro_f1", 0.0)) if val else np.nan
            t_pri = float(test.get(f"{primary_field}_macro_f1", 0.0)) if test else np.nan
            t_mean = float(test.get("mean_macro_f1", 0.0)) if test else np.nan

            redundancy = float("nan")
            close_up = small_fg = overlay = float("nan")
            if not redundancy_df.empty:
                m = (redundancy_df["strategy"] == strategy) & (redundancy_df["budget"] == budget)
                if m.any():
                    redundancy = float(redundancy_df.loc[m, "avg_pairwise_cosine_distance"].iloc[0])
            if not nuisance_df.empty:
                m = (nuisance_df["strategy"] == strategy) & (nuisance_df["budget"] == budget)
                if m.any():
                    nrow = nuisance_df.loc[m].iloc[0]
                    close_up = float(nrow.get("close_up_share", np.nan))
                    small_fg = float(nrow.get("small_foreground_share", np.nan))
                    overlay = float(nrow.get("has_overlay_text_share", np.nan))

            rows.append(
                {
                    "strategy": strategy,
                    "budget": int(budget),
                    "val_material_macro_f1": v_pri,
                    "val_mean_macro_f1": v_mean,
                    "test_material_macro_f1": t_pri,
                    "test_mean_macro_f1": t_mean,
                    "delta_val_material_vs_baseline": v_pri - baseline_val_primary
                    if not np.isnan(v_pri)
                    else np.nan,
                    "delta_val_mean_vs_baseline": v_mean - baseline_val_mean
                    if not np.isnan(v_mean)
                    else np.nan,
                    "delta_val_material_vs_random_same_budget": np.nan,  # filled later
                    "avg_pairwise_cosine_distance": redundancy,
                    "close_up_share": close_up,
                    "small_foreground_share": small_fg,
                    "has_overlay_text_share": overlay,
                    "notes": "",
                }
            )

            if strategy == "random":
                random_val_primary_by_budget[int(budget)] = v_pri
                random_val_mean_by_budget[int(budget)] = v_mean

    # fill delta_vs_random
    for r in rows:
        b = r["budget"]
        if b in random_val_primary_by_budget and r["strategy"] not in ("baseline_seed", "random"):
            base = random_val_primary_by_budget[b]
            if not np.isnan(r["val_material_macro_f1"]) and not np.isnan(base):
                r["delta_val_material_vs_random_same_budget"] = r["val_material_macro_f1"] - base

    df = pd.DataFrame(rows)
    df = df.sort_values(["budget", "strategy"]).reset_index(drop=True)
    df.to_csv(out_dir / "results.csv", index=False)
    print(f"saved → {out_dir / 'results.csv'}")

    # ---- plot learning curves
    def plot_curve(metric_col: str, title: str, out_png: Path) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        for strategy in strategies + ["baseline_seed"]:
            sub = df[df["strategy"] == strategy].sort_values("budget")
            if strategy == "baseline_seed":
                bx = [0]
                by = [sub[metric_col].iloc[0]]
                ax.scatter(bx, by, label="baseline_seed (B=0)", marker="*", s=140, color="black", zorder=5)
            else:
                # prepend baseline as B=0 anchor
                xs = [0] + sub["budget"].tolist()
                ys = [df[df["strategy"] == "baseline_seed"][metric_col].iloc[0]] + sub[metric_col].tolist()
                ax.plot(xs, ys, marker="o", linewidth=1.8, label=strategy)
        ax.set_xlabel("budget B (queried rows added to train_seed)")
        ax.set_ylabel(metric_col)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_png, dpi=120)
        plt.close(fig)

    plot_curve(
        "val_material_macro_f1",
        "Learning curves: material_macro_f1 on val_gold (PRIMARY)",
        out_dir / "learning_curves.png",
    )
    plot_curve(
        "val_mean_macro_f1",
        "Learning curves: mean_macro_f1 on val_gold (4 fields)",
        out_dir / "learning_curves_mean.png",
    )
    plot_curve(
        "test_material_macro_f1",
        "Learning curves: material_macro_f1 on test_gold (final eval)",
        out_dir / "learning_curves_test.png",
    )

    # ---- short notes
    notes_lines: List[str] = []
    notes_lines.append("# AL strategy comparison — auto notes\n")
    notes_lines.append(f"Primary metric: `{primary_field}_macro_f1` on val_gold.\n")
    notes_lines.append(
        f"Reference (B=0, train_seed only): val={baseline_val_primary:.4f}, test={baseline_test_primary:.4f}\n"
    )
    for budget in budgets:
        notes_lines.append(f"\n## Budget B={budget}\n")
        rows_b = df[df["budget"] == budget].sort_values("val_material_macro_f1", ascending=False)
        for _, r in rows_b.iterrows():
            notes_lines.append(
                f"- **{r['strategy']}** val_material={r['val_material_macro_f1']:.4f} "
                f"(Δvs_seed={r['delta_val_material_vs_baseline']:+.4f}, "
                f"Δvs_random={r['delta_val_material_vs_random_same_budget']:+.4f}) "
                f"test_material={r['test_material_macro_f1']:.4f} "
                f"redund={r['avg_pairwise_cosine_distance']:.3f}\n"
            )
    (out_dir / "notes.md").write_text("".join(notes_lines), encoding="utf-8")
    print(f"saved → {out_dir / 'notes.md'}")


if __name__ == "__main__":
    main()
