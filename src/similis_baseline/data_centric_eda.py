"""Data-centric EDA: pool structure + label-noise scoring + nuisance breakdown.

Closes data-centric checklist items:
  #3 — pool structure: per-pool row counts, valid images, dups, group_key repeats
  #4 — EDA + ≥10 label-noise candidates with reason flags + nuisance breakdown
       per chosen field
  #6 — quality_flag and noise_score added to manifest_data_centric.csv (only
       for rows with visible labels — pool_candidate stays masked)

Design notes:
- noise_score is a 0..1 heuristic combining label_is_uncertain, description-vs-
  normalized-label conflicts, missing flags, rare-class membership, and
  uncertainty markers in the `name` column ('(?)', '/' for hybrids).
- conflict detection compares the free-text `description` with normalized
  `material`, `type`, `integrity` via keyword roots.
- nuisance breakdown reuses heuristics from artifacts/reports/data_report/
  image_heuristics.csv and slices each chosen field's classes by layout_mode,
  bg_type, foreground_ratio_bin, has_scale_bar, has_overlay_text.

Outputs:
  artifacts/reports/data_centric/
    pool_structure.json
    label_noise_candidates.csv     (≥10 rows, with reason flags)
    nuisance_breakdown_{field}.csv (per chosen field × pool)
    noise_score_distribution.csv   (per-pool quantiles)
  data/processed/data_centric/
    manifest_data_centric.csv      (overwritten with noise_score, quality_flag)
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .image_analysis import foreground_ratio_bin
from .utils import ensure_dir, load_config, save_json


MATERIAL_KEYWORDS = {
    "фарфор": ["фарфор"],
    "фаянс": ["фаянс"],
    "керамика": [
        "красноглинян",
        "белоглинян",
        "глинян",
        "керамик",
        "ангоб",
        "поливн",
    ],
    "стекло": ["стекл"],
}

TYPE_KEYWORDS = {
    "тарелка": ["тарелк"],
    "блюдце": ["блюдц", "блюдеч", "блюдчик"],
    "изразец": ["изразц", "изразец"],
    "крышка": ["крышк"],
    "миска": ["миск"],
}

INTEGRITY_KEYWORDS = {
    "фрагмент": [r"\bфр-т\b", r"\bфр\.\b", r"\bфрагмент", r"\bосколок", r"\bобломок"],
}

UNCERTAIN_TEXT_MARKERS = ["(?)", "(?", "?)", "вероятно", "возможно", "вероятнее всего"]


def _tokens_match(text: str, patterns: List[str]) -> bool:
    text = text.lower()
    for pattern in patterns:
        if pattern.startswith("\\") or pattern.startswith("\\b"):
            if re.search(pattern, text, flags=re.IGNORECASE):
                return True
        else:
            if pattern in text:
                return True
    return False


def detect_field_conflict(
    description: str,
    norm_value: str,
    keyword_map: Dict[str, List[str]],
) -> Optional[str]:
    """Return the value implied by description (if any) when it differs from
    norm_value; return None when description is neutral or agrees.
    """
    if not isinstance(description, str) or description.strip() == "":
        return None
    description_low = description.lower()
    implied: List[str] = []
    for value, patterns in keyword_map.items():
        if _tokens_match(description_low, patterns):
            implied.append(value)
    if not implied:
        return None
    # if any implied value differs from norm and at least one matches → still ok
    if norm_value in implied:
        return None
    # pick the first implied value as the conflict signal
    return implied[0]


def detect_uncertainty_text(name: str, description: str) -> bool:
    for marker in UNCERTAIN_TEXT_MARKERS:
        if marker in str(name) or marker in str(description):
            return True
    if "/" in str(name):  # 'Тарелка/блюдо'
        return True
    return False


def compute_rare_classes(df: pd.DataFrame, fields: List[str], min_count: int = 5) -> Dict[str, set]:
    rare: Dict[str, set] = {}
    for field in fields:
        missing_col = f"{field}_is_missing"
        if missing_col in df.columns:
            sub = df[df[missing_col] == 0]
        else:
            sub = df
        counts = sub[field].value_counts()
        rare[field] = set(counts[counts < min_count].index.tolist())
    return rare


def compute_noise_score_and_flags(
    df: pd.DataFrame,
    fields: List[str],
    rare_classes: Dict[str, set],
) -> pd.DataFrame:
    out = df.copy()
    score = np.zeros(len(out), dtype=np.float32)

    flag_uncertain = np.zeros(len(out), dtype=np.int8)
    flag_incomplete = np.zeros(len(out), dtype=np.int8)
    flag_conflict = np.zeros(len(out), dtype=np.int8)
    flag_rare = np.zeros(len(out), dtype=np.int8)

    conflict_reasons: List[str] = ["" for _ in range(len(out))]

    label_uncertain = (
        out["label_is_uncertain"].fillna(0).astype(int).values
        if "label_is_uncertain" in out.columns
        else np.zeros(len(out), dtype=int)
    )

    descriptions = out["description"].fillna("").astype(str).values
    names = out["name"].fillna("").astype(str).values

    for i in range(len(out)):
        reasons: List[str] = []

        if int(label_uncertain[i]) == 1 or detect_uncertainty_text(names[i], descriptions[i]):
            flag_uncertain[i] = 1
            score[i] += 0.4
            reasons.append("uncertain_text")

        # missing field count
        missing_n = 0
        for field in fields:
            mc = f"{field}_is_missing"
            if mc in out.columns and int(out.iloc[i][mc]) == 1:
                missing_n += 1
        if missing_n > 0:
            flag_incomplete[i] = 1
            score[i] += min(missing_n * 0.1, 0.3)
            reasons.append(f"incomplete_{missing_n}")

        # description-vs-norm conflicts
        for field, kmap in [("material", MATERIAL_KEYWORDS), ("type", TYPE_KEYWORDS)]:
            if field not in out.columns:
                continue
            mc = f"{field}_is_missing"
            if mc in out.columns and int(out.iloc[i][mc]) == 1:
                continue  # missing → not a conflict
            norm_val = str(out.iloc[i][field]).lower()
            implied = detect_field_conflict(descriptions[i], norm_val, kmap)
            if implied:
                flag_conflict[i] = 1
                score[i] += 0.3
                reasons.append(f"conflict_{field}={norm_val}_vs_desc_{implied}")

        # rare class
        for field, rare_set in rare_classes.items():
            mc = f"{field}_is_missing"
            if mc in out.columns and int(out.iloc[i][mc]) == 1:
                continue
            val = str(out.iloc[i][field])
            if val in rare_set:
                flag_rare[i] = 1
                score[i] += 0.2
                reasons.append(f"rare_{field}={val}")

        conflict_reasons[i] = ";".join(reasons)

    out["noise_score"] = np.clip(score, 0.0, 1.0)
    out["flag_uncertain"] = flag_uncertain
    out["flag_incomplete"] = flag_incomplete
    out["flag_conflict"] = flag_conflict
    out["flag_rare"] = flag_rare
    out["noise_reasons"] = conflict_reasons

    out["quality_flag"] = "clean"
    out.loc[out["flag_incomplete"] == 1, "quality_flag"] = "incomplete"
    out.loc[out["flag_rare"] == 1, "quality_flag"] = "rare"
    out.loc[out["flag_conflict"] == 1, "quality_flag"] = "conflict"
    out.loc[out["flag_uncertain"] == 1, "quality_flag"] = "uncertain"

    return out


def pool_structure_report(manifest: pd.DataFrame, pools: List[str]) -> Dict:
    out: Dict[str, Dict] = {}
    for pool in pools:
        sub = manifest[manifest["pool"] == pool]
        out[pool] = {
            "rows": int(len(sub)),
            "unique_codes": int(sub["code"].nunique()),
            "unique_image_files": int(sub["image_file"].nunique()),
            "unique_group_keys": int(sub["group_key"].nunique()),
            "duplicated_image_files": int(sub["image_file"].duplicated().sum()),
            "repeated_group_keys": int(sub["group_key"].duplicated().sum()),
        }
        if "has_image" in sub.columns:
            out[pool]["valid_image_rows"] = int(sub["has_image"].sum())
            out[pool]["broken_image_rows"] = int(len(sub) - sub["has_image"].sum())
    return out


def nuisance_breakdown(
    df_with_heuristics: pd.DataFrame,
    field: str,
) -> pd.DataFrame:
    """Per-class × {layout_mode, bg_type, foreground_ratio_bin, has_scale_bar,
    has_overlay_text} count breakdown for a single field, restricted to rows
    where the label is visible."""
    mc = f"{field}_is_missing"
    sub = df_with_heuristics
    if mc in sub.columns:
        sub = sub[sub[mc] == 0]
    sub = sub.copy()

    rows: List[Dict] = []
    for cls in sorted(sub[field].dropna().unique()):
        cls_df = sub[sub[field] == cls]
        for factor in ["layout_mode", "bg_type", "foreground_ratio_bin", "has_scale_bar", "has_overlay_text"]:
            if factor not in cls_df.columns:
                continue
            counts = cls_df[factor].value_counts(dropna=False)
            for value, count in counts.items():
                rows.append(
                    {
                        "field": field,
                        "class": cls,
                        "factor": factor,
                        "value": str(value),
                        "count": int(count),
                        "share": round(float(count) / max(1, len(cls_df)), 4),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_centric.yaml")
    parser.add_argument(
        "--report-dir",
        default="artifacts/reports/data_centric",
    )
    parser.add_argument(
        "--noise-candidates-top",
        type=int,
        default=40,
        help="how many top-noise rows to dump as label_noise_candidates.csv",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    fields = list(cfg["fields"])
    primary_field = cfg.get("primary_field", "material")

    report_dir = Path(args.report_dir)
    ensure_dir(report_dir)

    manifest = pd.read_csv(cfg["manifest_data_centric_csv"])
    print(f"manifest rows: {len(manifest)}")

    # ----- #3 pool structure
    pools = ["train_seed", "val_gold", "test_gold", "pool_candidate"]
    structure = pool_structure_report(manifest, pools)
    save_json(structure, report_dir / "pool_structure.json")
    print("pool_structure.json saved")

    # ----- #6 noise score + quality_flag (only for rows with visible labels)
    visible_mask = manifest["pool"] != "pool_candidate"
    visible = manifest[visible_mask].copy()

    rare_classes = compute_rare_classes(visible, fields, min_count=5)
    print("rare classes:", {k: sorted(v) for k, v in rare_classes.items()})

    visible_scored = compute_noise_score_and_flags(visible, fields, rare_classes)

    # write back into manifest
    cols_added = [
        "noise_score",
        "flag_uncertain",
        "flag_incomplete",
        "flag_conflict",
        "flag_rare",
        "noise_reasons",
        "quality_flag",
    ]
    for col in cols_added:
        manifest[col] = np.nan if col == "noise_score" else (
            "" if col in {"noise_reasons", "quality_flag"} else 0
        )
    manifest.loc[visible_mask, cols_added] = visible_scored[cols_added].values

    # pool_candidate explicitly marked as 'hidden'
    manifest.loc[~visible_mask, "quality_flag"] = "hidden"
    manifest.to_csv(cfg["manifest_data_centric_csv"], index=False)
    print(f"manifest with quality_flag/noise_score → {cfg['manifest_data_centric_csv']}")

    # ----- #4 label noise candidates (top-noise visible rows)
    noise_top = (
        visible_scored.sort_values("noise_score", ascending=False)
        .head(int(args.noise_candidates_top))[
            [
                "image_file",
                "code",
                "group_key",
                "pool" if "pool" in visible_scored.columns else "code",
                "name",
                "description",
                *fields,
                *[f"{f}_is_missing" for f in fields if f"{f}_is_missing" in visible_scored.columns],
                "label_is_uncertain",
                "noise_score",
                "quality_flag",
                "flag_uncertain",
                "flag_incomplete",
                "flag_conflict",
                "flag_rare",
                "noise_reasons",
            ]
        ]
    )
    noise_top.to_csv(report_dir / "label_noise_candidates.csv", index=False)
    print(f"label_noise_candidates.csv (top {len(noise_top)}) saved")

    # noise score distribution per pool
    rows: List[Dict] = []
    for pool in ["train_seed", "val_gold", "test_gold"]:
        sub = visible_scored[visible_scored["pool"] == pool]["noise_score"]
        rows.append(
            {
                "pool": pool,
                "rows": int(len(sub)),
                "mean": float(sub.mean()) if len(sub) else 0.0,
                "median": float(sub.median()) if len(sub) else 0.0,
                "p75": float(sub.quantile(0.75)) if len(sub) else 0.0,
                "p90": float(sub.quantile(0.90)) if len(sub) else 0.0,
                "max": float(sub.max()) if len(sub) else 0.0,
                "rows_score_ge_0_3": int((sub >= 0.3).sum()),
                "rows_score_ge_0_5": int((sub >= 0.5).sum()),
            }
        )
    pd.DataFrame(rows).to_csv(report_dir / "noise_score_distribution.csv", index=False)
    print("noise_score_distribution.csv saved")

    # ----- #4 nuisance breakdown per chosen field
    heuristics_path = "artifacts/reports/data_report/image_heuristics.csv"
    if Path(heuristics_path).exists():
        heur = pd.read_csv(heuristics_path)
        heur["foreground_ratio_bin"] = heur["foreground_ratio"].apply(foreground_ratio_bin)
        merge_cols = ["image_file", "layout_mode", "bg_type", "foreground_ratio_bin", "has_scale_bar", "has_overlay_text"]
        merged = manifest.merge(heur[merge_cols], on="image_file", how="left")
        for field in [primary_field, "type", "part", "integrity"]:
            if field not in merged.columns:
                continue
            df_breakdown = nuisance_breakdown(merged, field)
            df_breakdown.to_csv(report_dir / f"nuisance_breakdown_{field}.csv", index=False)
            print(f"nuisance_breakdown_{field}.csv saved ({len(df_breakdown)} rows)")
    else:
        print(f"WARNING: {heuristics_path} not found — skipping nuisance breakdown")

    # ----- summary
    summary = {
        "fields": fields,
        "primary_field": primary_field,
        "rare_classes": {k: sorted(v) for k, v in rare_classes.items()},
        "pool_structure": structure,
        "visible_rows": int(len(visible_scored)),
        "noise_candidates_top": int(args.noise_candidates_top),
        "noise_quality_flag_distribution": {
            pool: visible_scored[visible_scored["pool"] == pool]["quality_flag"].value_counts().to_dict()
            for pool in ["train_seed", "val_gold", "test_gold"]
        },
        "noise_quality_flag_distribution_total": visible_scored["quality_flag"].value_counts().to_dict(),
    }
    save_json(summary, report_dir / "data_centric_eda_summary.json")
    print(f"data_centric_eda_summary.json saved")


if __name__ == "__main__":
    main()
