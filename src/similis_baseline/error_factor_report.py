from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .image_analysis import foreground_ratio_bin
from .utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-csv", type=str, required=True)
    parser.add_argument("--heuristics-csv", type=str, default="artifacts/reports/data_report/image_heuristics.csv")
    parser.add_argument("--manifest-csv", type=str, default="data/processed/full_manifest.csv")
    parser.add_argument("--split-name", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def labeled_acc(group: pd.DataFrame, field: str) -> float | None:
    mask_col = f"{field}_mask"
    correct_col = f"correct_{field}"
    if mask_col not in group.columns or correct_col not in group.columns:
        return None
    labeled = group[group[mask_col] == 1.0]
    if labeled.empty:
        return None
    return float(labeled[correct_col].mean())


def infer_error_reason(row: pd.Series) -> str:
    if int(row.get("any_error", 0)) == 0:
        return "correct"
    if int(row.get("label_is_uncertain", 0)) == 1:
        return "uncertain_or_label_noise"
    if float(row.get("mean_confidence", 1.0)) < 0.75:
        return "visual_ambiguity_low_confidence"
    if str(row.get("layout_mode", "")) != "single_object":
        return "layout_complexity"
    if int(row.get("has_scale_bar", 0)) == 1 or int(row.get("has_overlay_text", 0)) == 1:
        return "shortcut_risk_layout"
    if str(row.get("bg_type", "")) == "dark_or_complex" or float(row.get("foreground_ratio", 1.0)) < 0.08:
        return "hard_visual_conditions"
    return "other_model_error"


def build_breakdown(df: pd.DataFrame, factors: List[str], output_path: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    for factor in factors:
        for value, group in df.groupby(factor, dropna=False):
            row = {
                "factor": factor,
                "value": str(value),
                "rows": int(len(group)),
                "any_error_rate": float(group["any_error"].mean()),
                "auto_description_match_rate": float(group["auto_description_match"].mean()),
                "mean_num_field_errors": float(group["num_field_errors"].mean()),
                "mean_confidence": float(group["mean_confidence"].mean()),
            }
            for field in ["type", "part", "integrity", "material"]:
                row[f"{field}_acc_labeled"] = labeled_acc(group, field)
            rows.append(row)

    breakdown = pd.DataFrame(rows).sort_values(["factor", "rows", "any_error_rate"], ascending=[True, False, False])
    breakdown.to_csv(output_path, index=False)
    return breakdown


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    predictions = pd.read_csv(args.pred_csv)
    heuristics = pd.read_csv(args.heuristics_csv)
    manifest = pd.read_csv(args.manifest_csv)
    manifest = manifest[
        [
            "image_file",
            "code",
            "name",
            "description",
            "type",
            "part",
            "integrity",
            "material",
            "label_is_uncertain",
        ]
    ].copy()

    merged = predictions.merge(heuristics, on="image_file", how="left").merge(manifest, on="image_file", how="left", suffixes=("", "_manifest"))
    merged["foreground_ratio_bin"] = merged["foreground_ratio"].map(foreground_ratio_bin)
    merged["error_reason"] = merged.apply(infer_error_reason, axis=1)

    merged.to_csv(output_dir / "predictions_with_factors.csv", index=False)

    factor_breakdown = build_breakdown(
        merged,
        factors=["layout_mode", "bg_type", "has_scale_bar", "has_overlay_text", "foreground_ratio_bin"],
        output_path=output_dir / "factor_breakdown.csv",
    )

    reason_breakdown = (
        merged.groupby("error_reason", dropna=False)
        .agg(
            rows=("image_file", "count"),
            any_error_rate=("any_error", "mean"),
            auto_description_match_rate=("auto_description_match", "mean"),
            mean_num_field_errors=("num_field_errors", "mean"),
            mean_confidence=("mean_confidence", "mean"),
        )
        .reset_index()
        .sort_values(["rows", "any_error_rate"], ascending=[False, False])
    )
    reason_breakdown.to_csv(output_dir / "error_reason_breakdown.csv", index=False)

    error_cases = merged[merged["any_error"] == 1].sort_values(
        ["num_field_errors", "mean_confidence"],
        ascending=[False, True],
    )
    error_cases.to_csv(output_dir / "error_cases.csv", index=False)

    summary = {
        "split_name": args.split_name,
        "rows": int(len(merged)),
        "any_error_rate": float(merged["any_error"].mean()),
        "auto_description_match_rate": float(merged["auto_description_match"].mean()),
        "layout_modes": merged["layout_mode"].value_counts(normalize=True).to_dict(),
        "bg_types": merged["bg_type"].value_counts(normalize=True).to_dict(),
    }
    save_json(summary, output_dir / "summary.json")

    print(summary)
    print(f"Saved error factor report to {output_dir}")


if __name__ == "__main__":
    main()
