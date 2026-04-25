"""Build data-centric pools: train_seed / val_gold / test_gold / pool_candidate.

Reads the full manifest produced by data_prep.py and creates 4 disjoint pools
with hidden labels for pool_candidate (oracle CSV stored separately so the model
cannot peek during query-strategy selection).

Pools:
  - test_gold       — reused from baseline test_open (final-eval surface stays
                      the same as baseline, allowing direct comparison).
  - val_gold        — clean rows (label_is_uncertain=0), group-aware carve-out.
                      Used for choosing strategy/checkpoint, NOT for AL queries.
  - train_seed      — small initial labelled set for AL baseline.
  - pool_candidate  — everything else; labels masked in pool_candidate.csv,
                      truth kept in pool_candidate_oracle.csv.

Reproducibility: deterministic split by seed (default cfg["seed"]); group-aware
via GroupShuffleSplit on `group_key`. In the current open corpus group_key is
unique per row, so this is effectively a row split — but the pipeline will
honour real artifact_id when it appears.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .utils import ensure_dir, load_config, save_json, seed_everything


HIDDEN_TOKEN = "__HIDDEN__"


def split_off(
    df: pd.DataFrame,
    n_to_split_off: int,
    seed: int,
    group_col: str = "group_key",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Group-aware carve-out. Returns (rest, split_off) where split_off has
    approximately `n_to_split_off` rows.
    """
    n_to_split_off = max(1, min(int(n_to_split_off), len(df) - 1))
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=n_to_split_off,
        random_state=seed,
    )
    rest_idx, split_idx = next(splitter.split(df, groups=df[group_col]))
    return df.iloc[rest_idx].copy(), df.iloc[split_idx].copy()


def hide_pool_labels(df: pd.DataFrame, fields: List[str]) -> pd.DataFrame:
    """Mask labels for pool_candidate: set *_is_missing=1, blank values, set raw to HIDDEN."""
    df = df.copy()
    for field in fields:
        df[f"{field}_is_missing"] = 1
        df[field] = HIDDEN_TOKEN
        raw_col = f"{field}_raw"
        if raw_col in df.columns:
            df[raw_col] = HIDDEN_TOKEN
    df["pool_candidate_labels_hidden"] = 1
    return df


def class_distribution(df: pd.DataFrame, fields: List[str]) -> Dict[str, Dict[str, int]]:
    """Per-field class counts ignoring missing rows."""
    out: Dict[str, Dict[str, int]] = {}
    for field in fields:
        missing_col = f"{field}_is_missing"
        if missing_col in df.columns:
            sub = df[df[missing_col] == 0]
        else:
            sub = df
        counts = sub[field].value_counts(dropna=False).to_dict()
        out[field] = {str(k): int(v) for k, v in counts.items()}
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_centric.yaml")
    parser.add_argument(
        "--output-report",
        default="artifacts/reports/data_centric/split_report.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    fields = list(cfg["fields"])
    full_df = pd.read_csv(cfg["full_manifest_path"])
    test_open_df = pd.read_csv(cfg["test_open_csv"])

    n_full = len(full_df)
    n_test_open = len(test_open_df)
    print(f"Source: {n_full} rows in full manifest, {n_test_open} in baseline test_open")

    if "code" not in full_df.columns or "group_key" not in full_df.columns:
        raise ValueError("full_manifest must have `code` and `group_key` columns")

    # Drop rows without an image (baseline data_prep already filters, but defensive)
    if "has_image" in full_df.columns:
        full_df = full_df[full_df["has_image"] == 1].copy()
        if len(full_df) != n_full:
            print(f"  dropped {n_full - len(full_df)} rows without image")

    full_df = full_df.reset_index(drop=True)

    # 1. test_gold = baseline test_open
    test_gold_codes = set(test_open_df["code"].astype(str).tolist())
    test_gold_df = full_df[full_df["code"].astype(str).isin(test_gold_codes)].copy()
    if len(test_gold_df) != len(test_open_df):
        print(
            f"  warning: only {len(test_gold_df)}/{len(test_open_df)} test_open rows found in full_manifest"
        )

    # 2. Remaining = full - test_gold
    remaining = full_df[~full_df["code"].astype(str).isin(test_gold_codes)].copy()

    # 3. val_gold: split off from clean-only rows (label_is_uncertain=0)
    if "label_is_uncertain" in remaining.columns:
        clean_remaining = remaining[remaining["label_is_uncertain"] == 0].copy()
    else:
        clean_remaining = remaining.copy()

    target_val = int(cfg.get("val_gold_size", 150))
    target_val = min(target_val, len(clean_remaining) - 1)
    _, val_gold_df = split_off(clean_remaining, n_to_split_off=target_val, seed=seed)

    val_gold_codes = set(val_gold_df["code"].astype(str).tolist())

    # 4. After val_gold removal, split rest into train_seed + pool_candidate
    rest_after_val = remaining[~remaining["code"].astype(str).isin(val_gold_codes)].copy()

    target_seed = int(cfg.get("train_seed_size", 350))
    pool_candidate_df_unmasked, train_seed_df = split_off(
        rest_after_val,
        n_to_split_off=target_seed,
        seed=seed + 1,  # different sub-seed to decorrelate from val_gold draw
    )

    # Save oracle (true labels for pool_candidate) before masking
    pool_oracle_df = pool_candidate_df_unmasked.copy()
    pool_candidate_df = hide_pool_labels(pool_candidate_df_unmasked, fields)

    # Tag pool membership in a unified manifest for convenience
    manifest_df = full_df.copy()
    manifest_df["pool"] = "unassigned"
    manifest_df.loc[manifest_df["code"].astype(str).isin(test_gold_codes), "pool"] = "test_gold"
    manifest_df.loc[
        manifest_df["code"].astype(str).isin(val_gold_codes), "pool"
    ] = "val_gold"
    train_seed_codes = set(train_seed_df["code"].astype(str).tolist())
    pool_candidate_codes = set(pool_oracle_df["code"].astype(str).tolist())
    manifest_df.loc[
        manifest_df["code"].astype(str).isin(train_seed_codes), "pool"
    ] = "train_seed"
    manifest_df.loc[
        manifest_df["code"].astype(str).isin(pool_candidate_codes), "pool"
    ] = "pool_candidate"

    # Save all CSVs
    out_dir = Path(cfg["data_centric_dir"])
    ensure_dir(out_dir)

    train_seed_df.to_csv(cfg["train_seed_csv"], index=False)
    val_gold_df.to_csv(cfg["val_gold_csv"], index=False)
    test_gold_df.to_csv(cfg["test_gold_csv"], index=False)
    pool_candidate_df.to_csv(cfg["pool_candidate_csv"], index=False)
    pool_oracle_df.to_csv(cfg["pool_candidate_oracle_csv"], index=False)
    manifest_df.to_csv(cfg["manifest_data_centric_csv"], index=False)

    # ----- Sanity checks (echoed in report)
    pool_codes = {
        "train_seed": train_seed_codes,
        "val_gold": val_gold_codes,
        "test_gold": test_gold_codes,
        "pool_candidate": pool_candidate_codes,
    }
    pool_groups = {
        "train_seed": set(train_seed_df["group_key"].astype(str)),
        "val_gold": set(val_gold_df["group_key"].astype(str)),
        "test_gold": set(test_gold_df["group_key"].astype(str)),
        "pool_candidate": set(pool_oracle_df["group_key"].astype(str)),
    }
    intersections: Dict[str, int] = {}
    pairs = [
        ("train_seed", "val_gold"),
        ("train_seed", "test_gold"),
        ("train_seed", "pool_candidate"),
        ("val_gold", "test_gold"),
        ("val_gold", "pool_candidate"),
        ("test_gold", "pool_candidate"),
    ]
    for a, b in pairs:
        intersections[f"group_intersection_{a}__{b}"] = int(
            len(pool_groups[a] & pool_groups[b])
        )

    sizes = {
        "full_manifest": int(len(full_df)),
        "test_gold": int(len(test_gold_df)),
        "val_gold": int(len(val_gold_df)),
        "train_seed": int(len(train_seed_df)),
        "pool_candidate": int(len(pool_oracle_df)),
        "covered_total": int(
            len(test_gold_df) + len(val_gold_df) + len(train_seed_df) + len(pool_oracle_df)
        ),
    }
    sizes["uncovered_rows"] = sizes["full_manifest"] - sizes["covered_total"]

    # Class distributions per pool, only for fields that are visible
    class_dists = {
        "train_seed": class_distribution(train_seed_df, fields),
        "val_gold": class_distribution(val_gold_df, fields),
        "test_gold": class_distribution(test_gold_df, fields),
        "pool_candidate_oracle": class_distribution(pool_oracle_df, fields),
    }

    # Quality flags counts
    flags_summary: Dict[str, Dict[str, int]] = {}
    for pool_name, df in [
        ("train_seed", train_seed_df),
        ("val_gold", val_gold_df),
        ("test_gold", test_gold_df),
        ("pool_candidate_oracle", pool_oracle_df),
    ]:
        if "label_is_uncertain" in df.columns:
            uncertain = int(df["label_is_uncertain"].sum())
        else:
            uncertain = 0
        per_field_missing = {
            field: int(df[f"{field}_is_missing"].sum())
            if f"{field}_is_missing" in df.columns
            else 0
            for field in fields
        }
        flags_summary[pool_name] = {
            "label_is_uncertain": uncertain,
            "per_field_missing": per_field_missing,
        }

    # Reproducibility check: re-run split with same seed -> same first 5 codes per pool
    repro = {}
    for pool_name, df in [
        ("train_seed", train_seed_df),
        ("val_gold", val_gold_df),
        ("pool_candidate_oracle", pool_oracle_df),
    ]:
        repro[pool_name + "_first5_codes"] = list(df["code"].astype(str).head(5))

    report = {
        "seed": seed,
        "config": str(args.config),
        "fields": fields,
        "primary_field": cfg.get("primary_field"),
        "sizes": sizes,
        "group_intersections": intersections,
        "all_intersections_zero": bool(all(v == 0 for v in intersections.values())),
        "covers_full_corpus": bool(sizes["uncovered_rows"] == 0),
        "class_distributions": class_dists,
        "flags_summary": flags_summary,
        "reproducibility_first5_codes": repro,
        "outputs": {
            "train_seed_csv": cfg["train_seed_csv"],
            "val_gold_csv": cfg["val_gold_csv"],
            "test_gold_csv": cfg["test_gold_csv"],
            "pool_candidate_csv": cfg["pool_candidate_csv"],
            "pool_candidate_oracle_csv": cfg["pool_candidate_oracle_csv"],
            "manifest_data_centric_csv": cfg["manifest_data_centric_csv"],
        },
    }

    save_json(report, args.output_report)
    print()
    print("=" * 60)
    print("Pool sizes:", sizes)
    print("All group intersections zero:", report["all_intersections_zero"])
    print("Covers full corpus:", report["covers_full_corpus"])
    print(f"Saved split report -> {args.output_report}")


if __name__ == "__main__":
    main()
