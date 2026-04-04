from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-log", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def maybe_plot(df: pd.DataFrame, columns: list[str], title: str, ylabel: str, output_path: Path) -> None:
    available = [column for column in columns if column in df.columns]
    if not available:
        return

    plt.figure(figsize=(8, 4))
    for column in available:
        plt.plot(df["epoch"], df[column], marker="o", label=column)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    df = pd.read_csv(args.train_log)
    maybe_plot(df, ["train_loss", "val_loss"], "Loss Curves", "loss", output_dir / "loss_curve.png")
    maybe_plot(
        df,
        ["mean_macro_f1", "type_macro_f1", "part_macro_f1", "integrity_macro_f1", "material_macro_f1"],
        "Macro-F1 Curves",
        "macro_f1",
        output_dir / "macro_f1_curve.png",
    )
    maybe_plot(df, ["head_lr", "backbone_lr"], "Learning Rate Curves", "lr", output_path=output_dir / "lr_curve.png")

    best_row = df.sort_values("mean_macro_f1", ascending=False).iloc[0].to_dict() if "mean_macro_f1" in df.columns else {}
    summary = {
        "rows": int(len(df)),
        "columns": df.columns.tolist(),
        "best_row": best_row,
    }
    save_json(summary, output_dir / "summary.json")

    print(summary)
    print(f"Saved train log report to {output_dir}")


if __name__ == "__main__":
    main()
