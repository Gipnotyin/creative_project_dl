"""Active learning simulation: retrain for each strategy × budget (#18).

For each (strategy, budget):

  1. Read queried.csv from artifacts/active_learning/{strategy}/B{budget}/
  2. Merge with pool_candidate_oracle.csv to reveal true labels
     (this is the "доразметка" step — done only after query selection)
  3. Concatenate with train_seed.csv → train_seed_plus_query.csv
  4. Generate a derived config that overrides train_csv + checkpoint_dir +
     reports_dir; everything else identical to configs/data_centric.yaml
     (same architecture, optimizer, scheduler, epochs, seed)
  5. Invoke `python -m src.similis_baseline.train --config <derived>`
  6. Run evaluate_detailed on val_gold and test_gold against the new best.pt

This produces the artifacts needed for #19 (comparison + learning curves):

  artifacts/active_learning/{strategy}/B{budget}/
    train_seed_plus_query.csv
    config.yaml
    metrics.json                      — final val_gold + test_gold mean_macro_f1
    checkpoints/{best,last}.pt        — local checkpoints (heavy, gitignored)
    reports/train_log.csv
    reports/val_detailed/             — full eval on val_gold
    reports/test_detailed/            — full eval on test_gold
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

from .utils import ensure_dir, load_config, save_json


def reveal_query_labels(
    queried_csv: Path,
    oracle_csv: Path,
) -> pd.DataFrame:
    queried = pd.read_csv(queried_csv)
    oracle = pd.read_csv(oracle_csv)
    keep_image_files = set(queried["image_file"].astype(str))
    revealed = oracle[oracle["image_file"].astype(str).isin(keep_image_files)].copy()
    if len(revealed) != len(queried):
        raise RuntimeError(
            f"oracle merge mismatch: {len(revealed)} found vs {len(queried)} queried"
        )
    return revealed


def build_train_csv(
    train_seed_csv: Path,
    revealed: pd.DataFrame,
    out_csv: Path,
) -> int:
    seed_df = pd.read_csv(train_seed_csv)
    common_cols = [c for c in seed_df.columns if c in revealed.columns]
    revealed = revealed[common_cols].copy()
    combined = pd.concat([seed_df[common_cols], revealed], axis=0).reset_index(drop=True)
    combined.to_csv(out_csv, index=False)
    return int(len(combined))


def write_run_config(
    base_cfg: Dict,
    train_csv: Path,
    run_dir: Path,
    out_path: Path,
) -> None:
    cfg = dict(base_cfg)
    cfg["train_csv"] = str(train_csv)
    cfg["checkpoint_dir"] = str(run_dir / "checkpoints")
    cfg["reports_dir"] = str(run_dir / "reports")
    cfg["preds_dir"] = str(run_dir / "preds")
    cfg["figures_dir"] = str(run_dir / "figures")
    out_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")


def run_train(config_path: Path, log_path: Path) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent.parent)
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.run(
            [sys.executable, "-m", "src.similis_baseline.train", "--config", str(config_path)],
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
        )
    return int(proc.returncode)


def run_evaluate(checkpoint: Path, split: str, output_dir: Path, log_path: Path) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent.parent)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n--- evaluate_detailed --split {split}\n")
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.similis_baseline.evaluate_detailed",
                "--checkpoint",
                str(checkpoint),
                "--split",
                split,
                "--output-dir",
                str(output_dir),
            ],
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
        )
    return int(proc.returncode)


def collect_metrics(run_dir: Path) -> Dict:
    out: Dict = {}
    for split in ["val", "test"]:
        mpath = run_dir / "reports" / f"{split}_detailed" / "metrics.json"
        if mpath.exists():
            out[f"{split}_metrics"] = json.loads(mpath.read_text(encoding="utf-8"))
    bpath = run_dir / "reports" / "best_metrics.json"
    if bpath.exists():
        out["best_metrics"] = json.loads(bpath.read_text(encoding="utf-8"))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_centric.yaml")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["random", "least_confidence", "coreset"],
        help="default trio: random, least_confidence, coreset (skips hybrid for #18)",
    )
    parser.add_argument("--budgets", nargs="+", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true",
                        help="skip combos where best.pt already exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = load_config(args.config)
    budgets = args.budgets or list(base_cfg.get("active_learning", {}).get("budgets", [50, 100]))

    al_root = Path(base_cfg["active_learning_dir"])
    train_seed_csv = Path(base_cfg["train_seed_csv"])
    oracle_csv = Path(base_cfg["pool_candidate_oracle_csv"])

    summary: List[Dict] = []
    for strategy in args.strategies:
        for budget in budgets:
            print(f"\n========== {strategy} × B={budget} ==========")
            run_dir = al_root / strategy / f"B{budget}"
            ensure_dir(run_dir)

            best_pt = run_dir / "checkpoints" / "best.pt"
            if args.skip_existing and best_pt.exists():
                print(f"  skipping (best.pt exists at {best_pt})")
                summary.append(
                    {
                        "strategy": strategy,
                        "budget": int(budget),
                        "status": "skipped",
                        **collect_metrics(run_dir),
                    }
                )
                continue

            queried_csv = run_dir / "queried.csv"
            if not queried_csv.exists():
                print(f"  ERROR: missing {queried_csv}")
                continue

            revealed = reveal_query_labels(queried_csv, oracle_csv)
            print(f"  revealed {len(revealed)} rows from oracle")

            combined_csv = run_dir / "train_seed_plus_query.csv"
            n_train = build_train_csv(train_seed_csv, revealed, combined_csv)
            print(f"  train CSV: {n_train} rows ({len(pd.read_csv(train_seed_csv))} seed + {len(revealed)} queried)")

            run_config = run_dir / "config.yaml"
            write_run_config(
                base_cfg=base_cfg,
                train_csv=combined_csv,
                run_dir=run_dir,
                out_path=run_config,
            )

            log_path = run_dir / "train_stdout.log"
            print(f"  training... → {log_path}")
            rc = run_train(run_config, log_path)
            if rc != 0:
                print(f"  train returncode={rc}; skipping eval")
                summary.append(
                    {
                        "strategy": strategy,
                        "budget": int(budget),
                        "status": f"train_failed_rc{rc}",
                    }
                )
                continue

            best_pt = run_dir / "checkpoints" / "best.pt"
            print(f"  evaluate val_gold...")
            run_evaluate(
                best_pt,
                "val",
                run_dir / "reports" / "val_detailed",
                log_path,
            )
            print(f"  evaluate test_gold...")
            run_evaluate(
                best_pt,
                "test",
                run_dir / "reports" / "test_detailed",
                log_path,
            )

            metrics = collect_metrics(run_dir)
            entry = {
                "strategy": strategy,
                "budget": int(budget),
                "train_rows": int(n_train),
                "status": "ok",
                **metrics,
            }
            save_json(entry, run_dir / "metrics.json")
            summary.append(entry)
            print(
                f"  done. val_mean_macro_f1={metrics.get('val_metrics', {}).get('mean_macro_f1', 'NA')} "
                f"val_material_macro_f1={metrics.get('val_metrics', {}).get('material_macro_f1', 'NA')}"
            )

    cmp_dir = al_root / "comparison"
    ensure_dir(cmp_dir)
    save_json(
        {"runs": summary, "config": args.config},
        cmp_dir / "al_loop_summary.json",
    )
    print(f"\nSummary saved → {cmp_dir / 'al_loop_summary.json'}")


if __name__ == "__main__":
    main()
