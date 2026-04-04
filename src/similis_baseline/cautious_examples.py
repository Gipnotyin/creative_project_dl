from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch

from .predict import build_auto_description
from .utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="artifacts/checkpoints/best.pt")
    parser.add_argument("--pred-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--limit", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    thresholds = ckpt["config"].get("confidence_thresholds", {})
    fields: List[str] = list(ckpt["config"]["fields"])

    df = pd.read_csv(args.pred_csv)
    rows: List[Dict] = []

    for _, row in df.iterrows():
        pred_labels = {field: row.get(f"pred_{field}") for field in fields}
        conf = {field: float(row.get(f"confidence_{field}", 0.0)) for field in fields}
        full_description = build_auto_description(pred_labels, conf, {field: 0.0 for field in fields})
        cautious_description = build_auto_description(pred_labels, conf, thresholds)
        dropped_fields = [
            field
            for field in fields
            if conf[field] < float(thresholds.get(field, 0.0))
        ]
        if full_description == cautious_description or not dropped_fields:
            continue

        item = {
            "image_file": row.get("image_file"),
            "full_description_no_thresholds": full_description,
            "cautious_description": cautious_description,
            "dropped_fields": ",".join(dropped_fields),
        }
        for field in fields:
            item[f"pred_{field}"] = pred_labels[field]
            item[f"confidence_{field}"] = conf[field]
            item[f"threshold_{field}"] = float(thresholds.get(field, 0.0))
        rows.append(item)

    cautious_df = pd.DataFrame(rows).sort_values(
        [f"confidence_{fields[0]}"] if rows else ["image_file"],
        ascending=True,
    ).head(args.limit)

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    cautious_df.to_csv(output_dir / "cautious_examples.csv", index=False)
    save_json(
        {
            "rows": int(len(cautious_df)),
            "fields": fields,
            "thresholds": thresholds,
        },
        output_dir / "summary.json",
    )

    print({"rows": int(len(cautious_df)), "output_dir": str(output_dir)})
    print(f"Saved cautious examples to {output_dir}")


if __name__ == "__main__":
    main()
