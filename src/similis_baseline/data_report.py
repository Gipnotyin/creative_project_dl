from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageFile

from .data_prep import IMAGE_EXTS, grouped_split
from .utils import ensure_dir, load_config, save_json

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--output-dir", type=str, default="artifacts/reports/data_report")
    parser.add_argument("--num-samples", type=int, default=12)
    return parser.parse_args()


def render_sample_grid(df: pd.DataFrame, output_path: Path, num_samples: int, seed: int) -> None:
    sample_df = df[df["image_file"].notna()].sample(
        n=min(num_samples, int(df["image_file"].notna().sum())),
        random_state=seed,
    )
    if sample_df.empty:
        return

    cols = 4
    rows = (len(sample_df) + cols - 1) // cols
    plt.figure(figsize=(cols * 4, rows * 4))

    for plot_idx, (_, row) in enumerate(sample_df.iterrows(), start=1):
        plt.subplot(rows, cols, plot_idx)
        try:
            with Image.open(row["image_file"]) as image:
                image = image.convert("RGB")
                image.thumbnail((512, 512))
                plt.imshow(image)
            title = str(row.get("code", row.get("group_key", "")))
            plt.title(title, fontsize=9)
        except Exception:
            plt.text(0.5, 0.5, "failed to load", ha="center", va="center")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_group_hist(group_counts: pd.Series, output_path: Path) -> None:
    if group_counts.empty:
        return

    plt.figure(figsize=(7, 4))
    plt.hist(group_counts.values, bins=min(20, max(5, group_counts.max())))
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

    current_split_df = pd.concat(
        [
            train_df.assign(split="train"),
            val_df.assign(split="val"),
            test_df.assign(split="test"),
        ],
        ignore_index=True,
    )
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

    render_sample_grid(manifest, output_dir / "sample_grid.png", num_samples=args.num_samples, seed=seed)
    plot_group_hist(group_counts.set_index("group_key")["count"], output_dir / "group_size_hist.png")

    manifest.head(5).to_csv(output_dir / "sample_rows.csv", index=False)
    duplicate_image_df.to_csv(output_dir / "duplicate_image_files.csv", index=False)
    repeated_groups_df.to_csv(output_dir / "repeated_group_keys.csv", index=False)
    broken_paths_df.to_csv(output_dir / "broken_paths.csv", index=False)

    intact_group_examples = []
    repeated_current_groups = (
        current_split_df["group_key"].astype(str).value_counts().rename_axis("group_key").reset_index(name="count")
    )
    repeated_current_groups = repeated_current_groups[repeated_current_groups["count"] > 1].head(3)
    for _, row in repeated_current_groups.iterrows():
        group_key = row["group_key"]
        group_rows = current_split_df[current_split_df["group_key"].astype(str) == str(group_key)]
        intact_group_examples.append(
            {
                "group_key": str(group_key),
                "count": int(row["count"]),
                "split": group_rows["split"].iloc[0],
                "example_images": group_rows["image_file"].head(3).tolist(),
            }
        )

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
        "intact_group_examples": intact_group_examples,
    }
    save_json(report, output_dir / "report.json")

    print(report)
    print(f"Saved data report to {output_dir}")


if __name__ == "__main__":
    main()
