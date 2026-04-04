from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from .dataset import SimilisDataset
from .model import SimilisMultiTaskModel
from .train import build_class_weights, masked_ce_loss
from .utils import ensure_dir, load_config, load_json, save_json, seed_everything, torch_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-preview", type=int, default=10)
    parser.add_argument("--output-json", type=str, default="artifacts/reports/one_batch_debug.json")
    parser.add_argument(
        "--output-csv",
        type=str,
        default="artifacts/reports/one_batch_debug_preview.csv",
    )
    parser.add_argument(
        "--use-pretrained",
        action="store_true",
        help="Load pretrained timm weights when no checkpoint is provided.",
    )
    return parser.parse_args()


def build_preview_rows(
    batch: Dict,
    outputs: Dict[str, torch.Tensor],
    fields: List[str],
    idx_to_label: Dict[str, Dict[int, str]],
    num_preview: int,
) -> List[Dict]:
    rows = []
    preview_size = min(num_preview, int(batch["image"].shape[0]))

    for index in range(preview_size):
        row = {
            "row_index": index,
            "image_file": batch["metadata"]["image_file"][index],
            "group_key": batch["metadata"]["group_key"][index],
        }

        for field in fields:
            target_mask = float(batch["target_mask"][field][index].item())
            target_idx = int(batch["targets"][field][index].item())
            pred_idx = int(outputs[field][index].argmax().item())
            probs = torch.softmax(outputs[field][index], dim=-1)
            row[f"{field}_mask"] = target_mask
            row[f"gt_{field}"] = idx_to_label[field].get(target_idx, "__MISSING__") if target_mask else "__MISSING__"
            row[f"pred_{field}"] = idx_to_label[field].get(pred_idx, "__UNKNOWN__")
            row[f"conf_{field}"] = round(float(probs[pred_idx].item()), 6)

        rows.append(row)

    return rows


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])

    label_maps_path = cfg.get("label_maps_path", "data/processed/label_maps.json")
    label_maps = load_json(label_maps_path)
    fields = list(cfg["fields"])
    num_classes = {field: len(label_maps[field]["field_to_idx"]) for field in fields}
    idx_to_label = {
        field: {int(idx): label for idx, label in label_maps[field]["idx_to_field"].items()}
        for field in fields
    }

    dataset = SimilisDataset(
        cfg["train_csv"],
        image_size=cfg["image_size"],
        train=True,
        fields=fields,
        label_maps_path=label_maps_path,
        preprocess_mode=str(cfg.get("preprocess_mode", "pad")),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or cfg["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    batch = next(iter(loader))

    model = SimilisMultiTaskModel(
        cfg["backbone"],
        num_classes=num_classes,
        pretrained=bool(args.use_pretrained and not args.checkpoint),
    )
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])

    device = torch_device()
    model.to(device)
    model.eval()

    images = batch["image"].to(device)
    outputs = model(images)

    class_weights = None
    if bool(cfg.get("use_class_weights", True)):
        class_weights = build_class_weights(
            train_csv=cfg["train_csv"],
            fields=fields,
            label_maps=label_maps,
            max_weight=5.0,
        )
    loss_weights = cfg["loss_weights"]

    report = {
        "device": device,
        "batch_size": int(images.shape[0]),
        "image_shape": list(batch["image"].shape),
        "image_dtype": str(batch["image"].dtype),
        "target_dtypes": {field: str(batch["targets"][field].dtype) for field in fields},
        "fields": {},
    }

    total_loss = 0.0
    for field in fields:
        logits = outputs[field]
        targets = batch["targets"][field].to(device)
        mask = batch["target_mask"][field].to(device)
        field_loss = masked_ce_loss(
            logits,
            targets,
            mask,
            class_weights=class_weights[field].to(device) if class_weights is not None else None,
        )
        weighted_loss = loss_weights[field] * field_loss
        total_loss += weighted_loss

        preds = logits.argmax(dim=1)
        keep = mask.detach().cpu().numpy().astype(bool)
        targets_cpu = targets.detach().cpu().numpy()
        preds_cpu = preds.detach().cpu().numpy()

        if keep.any():
            acc = float(accuracy_score(targets_cpu[keep], preds_cpu[keep]))
            macro_f1 = float(
                f1_score(targets_cpu[keep], preds_cpu[keep], average="macro", zero_division=0)
            )
        else:
            acc = 0.0
            macro_f1 = 0.0

        report["fields"][field] = {
            "logits_shape": list(logits.shape),
            "targets_shape": list(targets.shape),
            "mask_sum": float(mask.sum().item()),
            "loss": float(field_loss.item()),
            "weighted_loss": float(weighted_loss.item()),
            "batch_accuracy": acc,
            "batch_macro_f1": macro_f1,
        }

    report["total_loss"] = float(total_loss.item())

    preview_rows = build_preview_rows(
        batch=batch,
        outputs={field: outputs[field].detach().cpu() for field in fields},
        fields=fields,
        idx_to_label=idx_to_label,
        num_preview=args.num_preview,
    )

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    ensure_dir(output_json.parent)
    ensure_dir(output_csv.parent)
    save_json(report, output_json)
    pd.DataFrame(preview_rows).to_csv(output_csv, index=False)

    print(report)
    print(f"Saved report to {output_json}")
    print(f"Saved preview to {output_csv}")


if __name__ == "__main__":
    main()
