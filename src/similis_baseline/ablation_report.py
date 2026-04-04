from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .utils import ensure_dir, load_config, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--output-dir", type=str, default="artifacts/ablations/report")
    return parser.parse_args()


def maybe_load_json(path: Path) -> Dict:
    if path.exists():
        return load_json(path)
    return {}


def maybe_load_train_log(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def build_row(config_path: str) -> Dict:
    cfg = load_config(config_path)
    reports_dir = Path(cfg["reports_dir"])
    best_metrics = maybe_load_json(reports_dir / "best_metrics.json")
    val_metrics = maybe_load_json(reports_dir / "val_metrics.json")
    test_metrics = maybe_load_json(reports_dir / "test_metrics.json")
    train_log = maybe_load_train_log(reports_dir / "train_log.csv")

    best_epoch = None
    if not train_log.empty and "mean_macro_f1" in train_log.columns:
        best_row = train_log.sort_values("mean_macro_f1", ascending=False).iloc[0]
        best_epoch = int(best_row["epoch"])

    return {
        "experiment": Path(config_path).stem,
        "config_path": config_path,
        "backbone": cfg.get("backbone"),
        "image_size": cfg.get("image_size"),
        "epochs": cfg.get("epochs"),
        "freeze_backbone_epochs": cfg.get("freeze_backbone_epochs", 0),
        "preprocess_mode": cfg.get("preprocess_mode", "pad"),
        "use_class_weights": bool(cfg.get("use_class_weights", True)),
        "best_epoch": best_epoch,
        "logged_best_val_mean_macro_f1": best_metrics.get("best_val_mean_macro_f1"),
        "val_mean_macro_f1": val_metrics.get("mean_macro_f1"),
        "val_type_macro_f1": val_metrics.get("type_macro_f1"),
        "val_part_macro_f1": val_metrics.get("part_macro_f1"),
        "val_integrity_macro_f1": val_metrics.get("integrity_macro_f1"),
        "val_material_macro_f1": val_metrics.get("material_macro_f1"),
        "test_mean_macro_f1": test_metrics.get("mean_macro_f1"),
    }


def write_markdown(df: pd.DataFrame, output_path: Path) -> None:
    sortable = df.copy()
    sort_column = "val_mean_macro_f1" if "val_mean_macro_f1" in sortable.columns else "logged_best_val_mean_macro_f1"
    sortable = sortable.sort_values(sort_column, ascending=False, na_position="last")
    best_name = sortable.iloc[0]["experiment"] if not sortable.empty else "n/a"

    lines: List[str] = []
    lines.append("# Ablation Study")
    lines.append("")
    lines.append(f"Best experiment by validation macro-F1: `{best_name}`")
    lines.append("")
    lines.append("| experiment | preprocess | class_weights | image_size | epochs | best_epoch | val_mean_macro_f1 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for _, row in sortable.iterrows():
        lines.append(
            "| {experiment} | {preprocess_mode} | {use_class_weights} | {image_size} | {epochs} | {best_epoch} | {val_mean_macro_f1} |".format(
                experiment=row["experiment"],
                preprocess_mode=row["preprocess_mode"],
                use_class_weights=row["use_class_weights"],
                image_size=row["image_size"],
                epochs=row["epochs"],
                best_epoch=row["best_epoch"] if pd.notna(row["best_epoch"]) else "n/a",
                val_mean_macro_f1=f"{row['val_mean_macro_f1']:.4f}" if pd.notna(row["val_mean_macro_f1"]) else "n/a",
            )
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = [build_row(config_path) for config_path in args.configs]
    df = pd.DataFrame(rows).sort_values("val_mean_macro_f1", ascending=False, na_position="last")

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    df.to_csv(output_dir / "ablation_results.csv", index=False)
    write_markdown(df, output_dir / "ablation_summary.md")

    print(df)
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
