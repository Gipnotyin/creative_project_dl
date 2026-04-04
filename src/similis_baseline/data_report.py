from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageFile

from .data_prep import IMAGE_EXTS, grouped_split
from .image_analysis import analyze_image, foreground_ratio_bin
from .utils import ensure_dir, load_config, save_json

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--output-dir", type=str, default="artifacts/reports/data_report")
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--num-row-image-examples", type=int, default=5)
    parser.add_argument("--problematic-count", type=int, default=10)
    parser.add_argument("--layout-examples-per-mode", type=int, default=3)
    return parser.parse_args()


def shorten(text: str, width: int = 70) -> str:
    value = "" if text is None else str(text)
    return textwrap.shorten(value.replace("\n", " "), width=width, placeholder="...")


def render_grid(rows: pd.DataFrame, output_path: Path, title_columns: List[str], cols: int = 4) -> None:
    if rows.empty:
        return

    cols = max(1, cols)
    rows_count = (len(rows) + cols - 1) // cols
    plt.figure(figsize=(cols * 4.5, rows_count * 4.5))

    for plot_idx, (_, row) in enumerate(rows.iterrows(), start=1):
        plt.subplot(rows_count, cols, plot_idx)
        try:
            with Image.open(row["image_file"]) as image:
                image = image.convert("RGB")
                image.thumbnail((600, 600))
                plt.imshow(image)
        except Exception:
            plt.text(0.5, 0.5, "failed to load", ha="center", va="center")

        title_lines = []
        for column in title_columns:
            value = row.get(column, "")
            if pd.isna(value) or value == "":
                continue
            title_lines.append(f"{column}: {shorten(value, width=48)}")
        plt.title("\n".join(title_lines), fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_group_hist(group_counts: pd.Series, output_path: Path) -> None:
    if group_counts.empty:
        return

    plt.figure(figsize=(7, 4))
    plt.hist(group_counts.values, bins=min(20, max(5, int(group_counts.max()))))
    plt.xlabel("images per group_key")
    plt.ylabel("count")
    plt.title("Distribution of group sizes")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def collect_split_distribution(split_name: str, split_df: pd.DataFrame, fields: List[str]) -> List[Dict]:
    rows = []
    for field in fields:
        counts = split_df[field].value_counts(dropna=False)
        total = int(counts.sum())
        for label, count in counts.items():
            rows.append(
                {
                    "split": split_name,
                    "field": field,
                    "label": str(label),
                    "count": int(count),
                    "ratio": float(count / max(total, 1)),
                }
            )
    return rows


def pick_diverse_examples(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if df.empty:
        return df.head(0).copy()

    picked_parts = []
    if "type" in df.columns:
        for _, part in df.groupby("type", sort=True):
            picked_parts.append(part.sample(n=1, random_state=seed))
            if sum(len(chunk) for chunk in picked_parts) >= n:
                break

    picked = pd.concat(picked_parts, ignore_index=True) if picked_parts else df.head(0).copy()
    used = set(picked.get("image_file", pd.Series(dtype=str)).astype(str).tolist())
    remaining = df[~df["image_file"].astype(str).isin(used)]
    if len(picked) < n and not remaining.empty:
        picked = pd.concat(
            [
                picked,
                remaining.sample(n=min(n - len(picked), len(remaining)), random_state=seed),
            ],
            ignore_index=True,
        )
    return picked.head(n).copy()


def build_field_candidate_table(df: pd.DataFrame) -> pd.DataFrame:
    selected_rows = []
    for field, visibility, risk in [
        ("type", "yes", "medium"),
        ("part", "partial", "high"),
        ("integrity", "yes", "low"),
        ("material", "partial", "medium"),
    ]:
        missing_col = f"{field}_is_missing"
        selected_rows.append(
            {
                "candidate_field": field,
                "visible_from_image": visibility,
                "target_type": "categorical",
                "num_classes": int(df[field].nunique(dropna=False)),
                "missing_ratio": float(df[missing_col].fillna(0).mean()) if missing_col in df.columns else 0.0,
                "ambiguity_risk": risk,
                "baseline_decision": "selected",
                "comment": {
                    "type": "core object category",
                    "part": "useful, but noisy and often missing",
                    "integrity": "strong visual cue",
                    "material": "visible, but sometimes probabilistic",
                }[field],
            }
        )

    excluded_rows = [
        {
            "candidate_field": "size",
            "visible_from_image": "no",
            "target_type": "free_text",
            "num_classes": int(df["size"].fillna("").astype(str).nunique()),
            "missing_ratio": float(df["size"].isna().mean()),
            "ambiguity_risk": "high",
            "baseline_decision": "excluded",
            "comment": "requires external scale or card metadata",
        },
        {
            "candidate_field": "cultlayer",
            "visible_from_image": "no",
            "target_type": "categorical",
            "num_classes": int(df["cultlayer"].fillna("").astype(str).nunique()),
            "missing_ratio": float(df["cultlayer"].isna().mean()),
            "ambiguity_risk": "high",
            "baseline_decision": "excluded",
            "comment": "context field, not an image attribute",
        },
        {
            "candidate_field": "survyear",
            "visible_from_image": "no",
            "target_type": "categorical",
            "num_classes": int(df["survyear"].fillna("").astype(str).nunique()),
            "missing_ratio": float(df["survyear"].isna().mean()),
            "ambiguity_risk": "high",
            "baseline_decision": "excluded",
            "comment": "excavation metadata, not visual content",
        },
    ]
    return pd.DataFrame(selected_rows + excluded_rows)


def build_normalization_examples(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []

    material_df = df[
        ["code", "material_raw", "material", "description"]
    ].dropna(subset=["material_raw"]).drop_duplicates(subset=["material_raw", "material"])
    for _, row in material_df.sort_values(["material", "material_raw"]).head(20).iterrows():
        rows.append(
            {
                "field": "material",
                "raw_value": str(row["material_raw"]),
                "normalized_value": str(row["material"]),
                "example_code": str(row["code"]),
                "example_text": shorten(str(row.get("description", "")), width=100),
            }
        )

    type_df = df[df["type_is_missing"] == 0][["code", "name", "description", "type"]].copy()
    type_df["raw_value"] = type_df["name"].astype(str) + " | " + type_df["description"].astype(str)
    type_df = type_df.drop_duplicates(subset=["type", "raw_value"])
    for _, row in type_df.sort_values(["type", "code"]).head(20).iterrows():
        rows.append(
            {
                "field": "type",
                "raw_value": shorten(str(row["raw_value"]), width=110),
                "normalized_value": str(row["type"]),
                "example_code": str(row["code"]),
                "example_text": shorten(str(row.get("description", "")), width=100),
            }
        )

    return pd.DataFrame(rows)


def build_policy_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case": "missing",
                "handling": "keep dedicated *_is_missing flag and mask target in loss/metrics",
            },
            {
                "case": "unknown",
                "handling": "map unresolved values to coarse fallback class like прочее or часть не указана",
            },
            {
                "case": "rare",
                "handling": "do not collapse automatically; rely on coarse normalization and report classwise support",
            },
            {
                "case": "uncertain",
                "handling": "mark row with label_is_uncertain based on textual markers like ? / вероятно / возможно",
            },
        ]
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    manifest_path = Path(cfg.get("full_manifest_path", "data/processed/full_manifest.csv"))
    train_path = Path(cfg["train_csv"])
    val_path = Path(cfg["val_csv"])
    test_path = Path(cfg["test_csv"])
    images_dir = Path(cfg["images_dir"])
    fields = list(cfg["fields"])
    seed = int(cfg["seed"])

    manifest = pd.read_csv(manifest_path)
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    image_paths = sorted(
        p
        for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    broken_paths_df = manifest[manifest["image_file"].notna()].copy()
    broken_paths_df["path_exists"] = broken_paths_df["image_file"].map(lambda x: Path(x).exists())
    broken_paths_df = broken_paths_df[~broken_paths_df["path_exists"]].copy()

    duplicate_image_df = (
        manifest[manifest["image_file"].notna()]["image_file"]
        .value_counts()
        .rename_axis("image_file")
        .reset_index(name="count")
    )
    duplicate_image_df = duplicate_image_df[duplicate_image_df["count"] > 1].copy()

    group_counts = (
        manifest["group_key"]
        .astype(str)
        .value_counts()
        .rename_axis("group_key")
        .reset_index(name="count")
    )
    repeated_groups_df = group_counts[group_counts["count"] > 1].copy()

    heuristics_rows = [analyze_image(path) for path in manifest["image_file"].dropna().astype(str).unique().tolist()]
    heuristics_df = pd.DataFrame(heuristics_rows)
    heuristics_df["foreground_ratio_bin"] = heuristics_df["foreground_ratio"].map(foreground_ratio_bin)
    heuristics_df.to_csv(output_dir / "image_heuristics.csv", index=False)

    manifest_with_heuristics = manifest.merge(heuristics_df, on="image_file", how="left")
    current_split_df = pd.concat(
        [
            train_df.assign(split="train"),
            val_df.assign(split="val"),
            test_df.assign(split="test"),
        ],
        ignore_index=True,
    )
    split_with_heuristics = current_split_df.merge(heuristics_df, on="image_file", how="left")

    current_group_split = (
        current_split_df[["group_key", "split"]]
        .drop_duplicates()
        .sort_values(["group_key", "split"])
        .reset_index(drop=True)
    )
    regenerated_split_df = grouped_split(
        manifest[manifest["has_image"] == 1].copy().reset_index(drop=True),
        group_col="group_key",
        seed=seed,
        val_size=float(cfg["split"]["val_size"]),
        test_size=float(cfg["split"]["test_size"]),
    )
    regenerated_group_split = (
        regenerated_split_df[["group_key", "split"]]
        .drop_duplicates()
        .sort_values(["group_key", "split"])
        .reset_index(drop=True)
    )

    split_distribution_rows = []
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_distribution_rows.extend(collect_split_distribution(split_name, split_df, fields))
    pd.DataFrame(split_distribution_rows).to_csv(output_dir / "split_class_distribution.csv", index=False)

    heuristic_summary_rows = []
    for column in ["layout_mode", "bg_type", "has_scale_bar", "has_overlay_text", "foreground_ratio_bin"]:
        counts = heuristics_df[column].value_counts(dropna=False)
        for value, count in counts.items():
            heuristic_summary_rows.append(
                {
                    "feature": column,
                    "value": str(value),
                    "count": int(count),
                    "ratio": float(count / max(len(heuristics_df), 1)),
                }
            )
    pd.DataFrame(heuristic_summary_rows).to_csv(output_dir / "heuristic_summary.csv", index=False)

    field_candidates = build_field_candidate_table(manifest)
    field_candidates.to_csv(output_dir / "field_candidates.csv", index=False)

    normalization_examples = build_normalization_examples(manifest)
    normalization_examples.to_csv(output_dir / "normalization_examples.csv", index=False)

    policy_table = build_policy_table()
    policy_table.to_csv(output_dir / "label_policy.csv", index=False)

    description_examples = pick_diverse_examples(manifest_with_heuristics, n=20, seed=seed)[
        [
            "code",
            "name",
            "description",
            "material_raw" if "material_raw" in manifest_with_heuristics.columns else "material",
            "material",
            "type",
            "part",
            "integrity",
            "label_is_uncertain" if "label_is_uncertain" in manifest_with_heuristics.columns else "type_is_missing",
        ]
    ].copy()
    description_examples.to_csv(output_dir / "description_examples.csv", index=False)

    row_image_examples = pick_diverse_examples(
        manifest_with_heuristics[manifest_with_heuristics["image_file"].notna()].copy(),
        n=args.num_row_image_examples,
        seed=seed,
    )
    row_image_examples.to_csv(output_dir / "row_image_examples.csv", index=False)
    render_grid(
        row_image_examples,
        output_dir / "row_image_examples.png",
        title_columns=["code", "type", "material", "description"],
        cols=min(3, max(1, len(row_image_examples))),
    )

    random_sample = pick_diverse_examples(
        manifest_with_heuristics[manifest_with_heuristics["image_file"].notna()].copy(),
        n=args.num_samples,
        seed=seed,
    )
    render_grid(
        random_sample,
        output_dir / "sample_grid.png",
        title_columns=["code", "type", "material"],
        cols=4,
    )
    plot_group_hist(group_counts.set_index("group_key")["count"], output_dir / "group_size_hist.png")

    problematic_images = (
        manifest_with_heuristics[manifest_with_heuristics["image_file"].notna()]
        .sort_values(["problem_score", "aspect_ratio"], ascending=[False, False])
        .head(args.problematic_count)
        .copy()
    )
    problematic_images.to_csv(output_dir / "problematic_images.csv", index=False)
    render_grid(
        problematic_images,
        output_dir / "problematic_images.png",
        title_columns=["code", "layout_mode", "bg_type", "foreground_ratio", "problem_score"],
        cols=4,
    )

    layout_example_parts = []
    for layout_mode in ["single_object", "multi_view", "close_up"]:
        part = manifest_with_heuristics[manifest_with_heuristics["layout_mode"] == layout_mode].copy()
        if part.empty:
            continue
        layout_example_parts.append(part.head(args.layout_examples_per_mode))
    layout_examples = pd.concat(layout_example_parts, ignore_index=True) if layout_example_parts else manifest.head(0).copy()
    layout_examples.to_csv(output_dir / "layout_mode_examples.csv", index=False)
    render_grid(
        layout_examples,
        output_dir / "layout_mode_examples.png",
        title_columns=["code", "layout_mode", "foreground_ratio", "component_count"],
        cols=3,
    )

    shortcut_examples = manifest_with_heuristics[
        (manifest_with_heuristics["has_scale_bar"] == 1) | (manifest_with_heuristics["has_overlay_text"] == 1)
    ].copy()
    shortcut_examples = shortcut_examples.sort_values(
        ["has_scale_bar", "has_overlay_text", "problem_score"],
        ascending=[False, False, False],
    ).head(20)
    shortcut_examples.to_csv(output_dir / "shortcut_risk_examples.csv", index=False)

    intact_group_examples = []
    repeated_current_groups = (
        current_split_df["group_key"].astype(str).value_counts().rename_axis("group_key").reset_index(name="count")
    )
    repeated_current_groups = repeated_current_groups[repeated_current_groups["count"] > 1].head(3)
    for _, row in repeated_current_groups.iterrows():
        group_key = row["group_key"]
        group_rows = current_split_df[current_split_df["group_key"].astype(str) == str(group_key)].copy()
        intact_group_examples.append(
            {
                "group_key": str(group_key),
                "count": int(row["count"]),
                "split": str(group_rows["split"].iloc[0]),
                "example_images": group_rows["image_file"].head(3).tolist(),
            }
        )
    pd.DataFrame(intact_group_examples).to_csv(output_dir / "intact_group_examples.csv", index=False)

    train_groups = set(train_df["group_key"].astype(str))
    val_groups = set(val_df["group_key"].astype(str))
    test_groups = set(test_df["group_key"].astype(str))

    report = {
        "manifest_path": str(manifest_path),
        "images_dir": str(images_dir),
        "full_rows": int(len(manifest)),
        "rows_with_images": int(manifest["has_image"].sum()) if "has_image" in manifest.columns else None,
        "images_found_on_disk": int(len(image_paths)),
        "broken_path_count": int(len(broken_paths_df)),
        "duplicate_image_file_count": int(len(duplicate_image_df)),
        "repeated_group_key_count": int(len(repeated_groups_df)),
        "uncertain_row_count": int(manifest.get("label_is_uncertain", pd.Series(dtype=int)).fillna(0).sum()),
        "split_rows": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "group_overlap": {
            "train_val": int(len(train_groups & val_groups)),
            "train_test": int(len(train_groups & test_groups)),
            "val_test": int(len(val_groups & test_groups)),
        },
        "split_reproducible_from_seed": bool(current_group_split.equals(regenerated_group_split)),
        "layout_mode_distribution": heuristics_df["layout_mode"].value_counts(normalize=True).to_dict(),
        "bg_type_distribution": heuristics_df["bg_type"].value_counts(normalize=True).to_dict(),
        "foreground_ratio": {
            "median": float(heuristics_df["foreground_ratio"].median()),
            "p10": float(heuristics_df["foreground_ratio"].quantile(0.10)),
            "p90": float(heuristics_df["foreground_ratio"].quantile(0.90)),
        },
        "scale_bar_ratio": float(heuristics_df["has_scale_bar"].mean()),
        "overlay_text_ratio": float(heuristics_df["has_overlay_text"].mean()),
        "intact_group_examples": intact_group_examples,
        "corpus_difficulty_summary": [
            f"multi_view ratio: {heuristics_df['layout_mode'].eq('multi_view').mean():.3f}",
            f"close_up ratio: {heuristics_df['layout_mode'].eq('close_up').mean():.3f}",
            f"scale_bar ratio: {heuristics_df['has_scale_bar'].mean():.3f}",
            f"overlay_text ratio: {heuristics_df['has_overlay_text'].mean():.3f}",
        ],
    }

    manifest.head(5).to_csv(output_dir / "sample_rows.csv", index=False)
    duplicate_image_df.to_csv(output_dir / "duplicate_image_files.csv", index=False)
    repeated_groups_df.to_csv(output_dir / "repeated_group_keys.csv", index=False)
    broken_paths_df.to_csv(output_dir / "broken_paths.csv", index=False)
    split_with_heuristics.to_csv(output_dir / "split_with_heuristics.csv", index=False)
    save_json(report, output_dir / "report.json")

    print(report)
    print(f"Saved data report to {output_dir}")


if __name__ == "__main__":
    main()
