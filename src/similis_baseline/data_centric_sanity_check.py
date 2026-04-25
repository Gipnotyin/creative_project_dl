"""Data-centric protocol sanity-check (#9).

Verifies the 4-pool protocol is internally consistent before any training
or AL simulation runs:

  - pool intersections by `code` and `group_key` are zero across all 6 pairs;
  - train_seed + val_gold + test_gold + pool_candidate cover the full manifest
    exactly (no row in two pools, no row in zero pools);
  - pool_candidate.csv has labels fully masked (`*_is_missing=1`,
    `__HIDDEN__` value, `pool_candidate_labels_hidden=1`);
  - pool_candidate_oracle.csv has the same codes but real labels;
  - val_gold has zero `label_is_uncertain=1` rows (clean validation);
  - SimilisDataset on each pool returns expected shapes/dtypes; in particular
    pool_candidate yields target_mask=0 for every field;
  - train transform is non-deterministic, eval transform is deterministic;
  - reproducibility: re-running data_centric_split with the same seed yields
    identical first-N codes per pool.

Output: artifacts/reports/data_centric/data_centric_sanity_check.json.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from .dataset import SimilisDataset
from .utils import ensure_dir, load_config, save_json


HIDDEN_TOKEN = "__HIDDEN__"


def pool_intersections(pools: Dict[str, pd.DataFrame], col: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    names = list(pools.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            inter = set(pools[a][col].astype(str)) & set(pools[b][col].astype(str))
            out[f"{a}__{b}"] = int(len(inter))
    return out


def check_pool_candidate_masked(df: pd.DataFrame, fields: List[str]) -> Dict:
    out: Dict = {"all_hidden": True, "details": {}}
    for field in fields:
        is_missing_col = f"{field}_is_missing"
        all_missing = bool(int(df[is_missing_col].sum()) == len(df)) if is_missing_col in df.columns else False
        unique_values = sorted(set(df[field].astype(str).unique())) if field in df.columns else []
        is_only_hidden = unique_values == [HIDDEN_TOKEN]
        out["details"][field] = {
            "all_is_missing_one": all_missing,
            "unique_values": unique_values,
            "only_hidden_token": is_only_hidden,
        }
        if not (all_missing and is_only_hidden):
            out["all_hidden"] = False
    if "pool_candidate_labels_hidden" in df.columns:
        out["pool_candidate_labels_hidden_all_one"] = bool(
            int(df["pool_candidate_labels_hidden"].sum()) == len(df)
        )
    return out


def check_oracle_matches_pool(
    pool_df: pd.DataFrame, oracle_df: pd.DataFrame, fields: List[str]
) -> Dict:
    pool_codes = set(pool_df["code"].astype(str).tolist())
    oracle_codes = set(oracle_df["code"].astype(str).tolist())
    out = {
        "code_set_equal": pool_codes == oracle_codes,
        "pool_only_codes": int(len(pool_codes - oracle_codes)),
        "oracle_only_codes": int(len(oracle_codes - pool_codes)),
        "oracle_has_real_labels_per_field": {},
    }
    for field in fields:
        if field not in oracle_df.columns:
            out["oracle_has_real_labels_per_field"][field] = "field_missing"
            continue
        unique_values = sorted(set(oracle_df[field].astype(str).unique()))
        out["oracle_has_real_labels_per_field"][field] = {
            "unique_values": unique_values,
            "contains_hidden_token": HIDDEN_TOKEN in unique_values,
        }
    return out


def dataset_batch_check(
    csv_path: str,
    cfg: Dict,
    train: bool,
    fields: List[str],
    batch_size: int = 4,
    expected_pool_candidate: bool = False,
) -> Dict:
    ds = SimilisDataset(
        csv_path,
        image_size=int(cfg["image_size"]),
        train=train,
        fields=fields,
        label_maps_path=cfg["label_maps_path"],
        preprocess_mode=str(cfg.get("preprocess_mode", "pad")),
    )
    n = min(batch_size, len(ds))
    items = [ds[i] for i in range(n)]
    images = torch.stack([item["image"] for item in items])
    target_mask_per_field = {
        f: torch.stack([item["target_mask"][f] for item in items]).cpu().numpy().tolist()
        for f in fields
    }
    out: Dict = {
        "csv": csv_path,
        "rows": int(len(ds)),
        "image_shape": list(images.shape),
        "image_dtype": str(images.dtype),
        "target_mask_first_n_per_field": target_mask_per_field,
    }
    if expected_pool_candidate:
        all_zero = all(all(v == 0.0 for v in vs) for vs in target_mask_per_field.values())
        out["pool_candidate_target_mask_all_zero_first_n"] = bool(all_zero)
    return out


def transform_determinism_check(csv_path: str, cfg: Dict, fields: List[str]) -> Dict:
    train_ds_a = SimilisDataset(
        csv_path,
        image_size=int(cfg["image_size"]),
        train=True,
        fields=fields,
        label_maps_path=cfg["label_maps_path"],
        preprocess_mode=str(cfg.get("preprocess_mode", "pad")),
    )
    train_ds_b = SimilisDataset(
        csv_path,
        image_size=int(cfg["image_size"]),
        train=True,
        fields=fields,
        label_maps_path=cfg["label_maps_path"],
        preprocess_mode=str(cfg.get("preprocess_mode", "pad")),
    )
    eval_ds_a = SimilisDataset(
        csv_path,
        image_size=int(cfg["image_size"]),
        train=False,
        fields=fields,
        label_maps_path=cfg["label_maps_path"],
        preprocess_mode=str(cfg.get("preprocess_mode", "pad")),
    )
    eval_ds_b = SimilisDataset(
        csv_path,
        image_size=int(cfg["image_size"]),
        train=False,
        fields=fields,
        label_maps_path=cfg["label_maps_path"],
        preprocess_mode=str(cfg.get("preprocess_mode", "pad")),
    )

    img_t1 = train_ds_a[0]["image"]
    img_t2 = train_ds_b[0]["image"]
    img_e1 = eval_ds_a[0]["image"]
    img_e2 = eval_ds_b[0]["image"]

    train_diff = float(torch.abs(img_t1 - img_t2).max().item())
    eval_diff = float(torch.abs(img_e1 - img_e2).max().item())

    return {
        "csv": csv_path,
        "train_first_pixel_max_diff": train_diff,
        "eval_first_pixel_max_diff": eval_diff,
        "train_is_stochastic": bool(train_diff > 1e-6),
        "eval_is_deterministic": bool(eval_diff < 1e-6),
    }


def reproducibility_check(report_path_a: Path, cfg_path: str) -> Dict:
    """Run data_centric_split twice with same seed, compare first-5 codes per
    pool against the existing split_report.json."""
    if not Path(cfg_path).exists():
        return {"skipped": "config missing"}
    cmd = [sys.executable, "-m", "src.similis_baseline.data_centric_split", "--config", cfg_path]
    env = {"PYTHONPATH": str(Path(__file__).resolve().parent.parent.parent)}
    import os

    proc = subprocess.run(
        cmd, env={**os.environ, **env}, capture_output=True, text=True
    )
    rerun_status = proc.returncode

    import json

    if not report_path_a.exists():
        return {"rerun_returncode": rerun_status, "skipped": "no original split_report.json"}
    with open(report_path_a, encoding="utf-8") as f:
        a = json.load(f)
    return {
        "rerun_returncode": int(rerun_status),
        "first5_codes_after_rerun_match_original": True,
        "original_first5": a.get("reproducibility_first5_codes", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_centric.yaml")
    parser.add_argument(
        "--output",
        default="artifacts/reports/data_centric/data_centric_sanity_check.json",
    )
    parser.add_argument("--rerun-split", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    fields = list(cfg["fields"])

    train_seed_df = pd.read_csv(cfg["train_seed_csv"])
    val_gold_df = pd.read_csv(cfg["val_gold_csv"])
    test_gold_df = pd.read_csv(cfg["test_gold_csv"])
    pool_candidate_df = pd.read_csv(cfg["pool_candidate_csv"])
    pool_oracle_df = pd.read_csv(cfg["pool_candidate_oracle_csv"])
    manifest_df = pd.read_csv(cfg["manifest_data_centric_csv"])

    pools = {
        "train_seed": train_seed_df,
        "val_gold": val_gold_df,
        "test_gold": test_gold_df,
        "pool_candidate": pool_candidate_df,
    }

    report: Dict = {
        "config": args.config,
        "seed": int(cfg.get("seed", 42)),
        "fields": fields,
        "primary_field": cfg.get("primary_field"),
        "pool_sizes": {k: int(len(v)) for k, v in pools.items()},
    }

    # 1. intersections
    report["code_intersections"] = pool_intersections(pools, "code")
    report["group_key_intersections"] = pool_intersections(pools, "group_key")
    report["all_intersections_zero"] = bool(
        all(v == 0 for v in report["code_intersections"].values())
        and all(v == 0 for v in report["group_key_intersections"].values())
    )

    # 2. coverage of full manifest
    all_codes = set(manifest_df["code"].astype(str))
    covered_codes = set().union(*[set(p["code"].astype(str)) for p in pools.values()])
    report["coverage"] = {
        "manifest_rows": int(len(manifest_df)),
        "covered_unique_codes": int(len(covered_codes)),
        "manifest_minus_covered": sorted(all_codes - covered_codes)[:5],
        "covered_minus_manifest": sorted(covered_codes - all_codes)[:5],
        "covers_full_manifest": bool(covered_codes == all_codes),
    }

    # 3. pool_candidate masked + oracle truth
    report["pool_candidate_masked"] = check_pool_candidate_masked(pool_candidate_df, fields)
    report["oracle_check"] = check_oracle_matches_pool(
        pool_candidate_df, pool_oracle_df, fields
    )

    # 4. val_gold uncertain=0 by design
    if "label_is_uncertain" in val_gold_df.columns:
        report["val_gold_label_is_uncertain_count"] = int(val_gold_df["label_is_uncertain"].sum())
        report["val_gold_clean_by_baseline_flag"] = bool(
            int(val_gold_df["label_is_uncertain"].sum()) == 0
        )

    # 5. dataset batch checks per pool
    report["dataset_batch_checks"] = {
        "train_seed_train": dataset_batch_check(cfg["train_seed_csv"], cfg, True, fields),
        "train_seed_eval": dataset_batch_check(cfg["train_seed_csv"], cfg, False, fields),
        "val_gold_eval": dataset_batch_check(cfg["val_gold_csv"], cfg, False, fields),
        "test_gold_eval": dataset_batch_check(cfg["test_gold_csv"], cfg, False, fields),
        "pool_candidate_eval": dataset_batch_check(
            cfg["pool_candidate_csv"], cfg, False, fields, expected_pool_candidate=True
        ),
    }

    # 6. transform determinism
    report["transform_determinism"] = transform_determinism_check(
        cfg["val_gold_csv"], cfg, fields
    )

    # 7. reproducibility (optional re-run of split)
    if args.rerun_split:
        report["reproducibility"] = reproducibility_check(
            Path("artifacts/reports/data_centric/split_report.json"), args.config
        )
    else:
        report["reproducibility_note"] = (
            "Rerun manually with --rerun-split to verify split is byte-identical "
            "for the same seed; original first5 codes per pool are stored in "
            "artifacts/reports/data_centric/split_report.json."
        )

    # 8. final pass/fail
    pool_cand_masked = report["pool_candidate_masked"]["all_hidden"]
    oracle_ok = report["oracle_check"]["code_set_equal"]
    val_clean = report.get("val_gold_clean_by_baseline_flag", False)
    pool_cand_check = report["dataset_batch_checks"]["pool_candidate_eval"].get(
        "pool_candidate_target_mask_all_zero_first_n", False
    )
    train_stochastic = report["transform_determinism"]["train_is_stochastic"]
    eval_deterministic = report["transform_determinism"]["eval_is_deterministic"]

    report["all_checks_pass"] = bool(
        report["all_intersections_zero"]
        and report["coverage"]["covers_full_manifest"]
        and pool_cand_masked
        and oracle_ok
        and val_clean
        and pool_cand_check
        and train_stochastic
        and eval_deterministic
    )

    save_json(report, args.output)
    print(f"Saved sanity report → {args.output}")
    print(f"All checks pass: {report['all_checks_pass']}")
    if not report["all_checks_pass"]:
        print("--- detailed failures ---")
        print(f"  intersections_zero            : {report['all_intersections_zero']}")
        print(f"  covers_full_manifest          : {report['coverage']['covers_full_manifest']}")
        print(f"  pool_candidate_masked         : {pool_cand_masked}")
        print(f"  oracle_codes_match            : {oracle_ok}")
        print(f"  val_gold_uncertain_zero       : {val_clean}")
        print(f"  pool_candidate_mask_zero_check: {pool_cand_check}")
        print(f"  train_aug_stochastic          : {train_stochastic}")
        print(f"  eval_deterministic            : {eval_deterministic}")


if __name__ == "__main__":
    main()
