from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import SimilisDataset
from .model import SimilisMultiTaskModel
from .train import build_class_weights, evaluate
from .utils import ensure_dir, load_json, torch_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="artifacts/checkpoints/best.pt")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    fields = cfg["fields"]
    label_maps_path = cfg.get("label_maps_path", "data/processed/label_maps.json")
    label_maps = load_json(label_maps_path)

    split_to_csv = {
        "train": cfg["train_csv"],
        "val": cfg["val_csv"],
        "test": cfg["test_csv"],
    }
    split_csv = split_to_csv[args.split]

    dataset = SimilisDataset(
        split_csv,
        image_size=cfg["image_size"],
        train=False,
        fields=fields,
        label_maps_path=label_maps_path,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = SimilisMultiTaskModel(
        cfg["backbone"],
        {field: len(ckpt["vocabs"][field]) for field in fields},
        pretrained=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    device = torch_device()
    model.to(device)

    class_weights = build_class_weights(
        train_csv=cfg["train_csv"],
        fields=fields,
        label_maps=label_maps,
        max_weight=5.0,
    )
    metrics = evaluate(model, loader, device, fields, cfg["loss_weights"], class_weights)
    metrics["split"] = args.split
    metrics["rows"] = len(dataset)

    output_path = Path(args.output) if args.output else Path(cfg["reports_dir"]) / f"{args.split}_metrics.json"
    ensure_dir(output_path.parent)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(metrics)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
