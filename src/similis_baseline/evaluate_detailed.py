from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from .dataset import SimilisDataset
from .model import SimilisMultiTaskModel
from .predict import build_auto_description
from .train import build_class_weights, masked_ce_loss
from .utils import ensure_dir, load_json, save_json, torch_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="artifacts/checkpoints/best.pt")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def plot_confusion(cm: np.ndarray, labels: List[str], title: str, path: Path) -> None:
    fig_w = max(6, 0.8 * len(labels))
    fig_h = max(5, 0.6 * len(labels))
    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)

    threshold = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(int(cm[i, j])),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def build_gt_description(row: Dict[str, str], fields: List[str], thresholds: Dict[str, float]) -> str:
    labels = {}
    conf = {}
    for field in fields:
        value = row.get(f"gt_{field}", "__MISSING__")
        if value == "__MISSING__":
            labels[field] = "неизвестно"
            conf[field] = 0.0
        else:
            labels[field] = value
            conf[field] = 1.0
    return build_auto_description(labels, conf, thresholds)


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    fields = list(cfg["fields"])
    label_maps_path = cfg.get("label_maps_path", "data/processed/label_maps.json")
    label_maps = load_json(label_maps_path)
    thresholds = cfg.get("confidence_thresholds", {})

    split_to_csv = {
        "train": cfg["train_csv"],
        "val": cfg["val_csv"],
        "test": cfg["test_csv"],
    }
    dataset = SimilisDataset(
        split_to_csv[args.split],
        image_size=cfg["image_size"],
        train=False,
        fields=fields,
        label_maps_path=label_maps_path,
        preprocess_mode=str(cfg.get("preprocess_mode", "pad")),
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
    model.eval()

    class_weights = None
    if bool(cfg.get("use_class_weights", True)):
        class_weights = build_class_weights(
            train_csv=cfg["train_csv"],
            fields=fields,
            label_maps=label_maps,
            max_weight=5.0,
        )

    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg["reports_dir"]) / f"{args.split}_detailed"
    ensure_dir(output_dir)

    idx_to_label = {
        field: {int(idx): label for idx, label in label_maps[field]["idx_to_field"].items()}
        for field in fields
    }
    label_order = {
        field: [idx_to_label[field][idx] for idx in sorted(idx_to_label[field].keys())]
        for field in fields
    }
    label_ids = {
        field: sorted(idx_to_label[field].keys())
        for field in fields
    }

    losses = []
    all_gt = {field: [] for field in fields}
    all_pred = {field: [] for field in fields}
    sample_rows: List[Dict] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            outputs = model(images)
            total_loss = 0.0

            pred_indices = {}
            pred_confidences = {}

            for field in fields:
                targets = batch["targets"][field].to(device)
                mask = batch["target_mask"][field].to(device)
                loss = masked_ce_loss(
                    outputs[field],
                    targets,
                    mask,
                    class_weights=class_weights[field].to(device) if class_weights is not None else None,
                )
                total_loss += cfg["loss_weights"][field] * loss

                probs = torch.softmax(outputs[field], dim=1)
                conf, pred = probs.max(dim=1)
                pred_indices[field] = pred.detach().cpu()
                pred_confidences[field] = conf.detach().cpu()

                keep = mask.detach().cpu().numpy().astype(bool)
                all_gt[field].extend(targets.detach().cpu().numpy()[keep].tolist())
                all_pred[field].extend(pred.detach().cpu().numpy()[keep].tolist())

            losses.append(float(total_loss.item()))

            batch_size = int(images.shape[0])
            for row_index in range(batch_size):
                row = {
                    "image_file": batch["metadata"]["image_file"][row_index],
                    "group_key": batch["metadata"]["group_key"][row_index],
                }
                pred_labels = {}
                pred_conf = {}
                num_field_errors = 0

                for field in fields:
                    mask_value = float(batch["target_mask"][field][row_index].item())
                    gt_idx = int(batch["targets"][field][row_index].item())
                    pred_idx = int(pred_indices[field][row_index].item())
                    pred_label = idx_to_label[field][pred_idx]
                    gt_label = idx_to_label[field][gt_idx] if mask_value else "__MISSING__"
                    confidence = float(pred_confidences[field][row_index].item())

                    row[f"gt_{field}"] = gt_label
                    row[f"pred_{field}"] = pred_label
                    row[f"confidence_{field}"] = round(confidence, 6)
                    row[f"{field}_mask"] = mask_value
                    row[f"correct_{field}"] = int(mask_value == 1.0 and gt_idx == pred_idx)

                    if mask_value == 1.0 and gt_idx != pred_idx:
                        num_field_errors += 1

                    pred_labels[field] = pred_label
                    pred_conf[field] = confidence

                row["pred_auto_description"] = build_auto_description(pred_labels, pred_conf, thresholds)
                row["gt_auto_description"] = build_gt_description(row, fields, thresholds)
                row["auto_description_match"] = int(
                    row["pred_auto_description"] == row["gt_auto_description"]
                )
                row["num_field_errors"] = num_field_errors
                row["mean_confidence"] = round(
                    float(np.mean([row[f"confidence_{field}"] for field in fields])),
                    6,
                )
                row["any_error"] = int(num_field_errors > 0)
                sample_rows.append(row)

    metrics = {
        "split": args.split,
        "rows": len(dataset),
        "val_loss": float(np.mean(losses)) if losses else 0.0,
    }
    macro_f1_values = []

    classwise_rows = []
    best_worst_classes = {}
    for field in fields:
        if all_gt[field]:
            metrics[f"{field}_acc"] = float(accuracy_score(all_gt[field], all_pred[field]))
            metrics[f"{field}_macro_f1"] = float(
                f1_score(all_gt[field], all_pred[field], average="macro", zero_division=0)
            )
            macro_f1_values.append(metrics[f"{field}_macro_f1"])
        else:
            metrics[f"{field}_acc"] = 0.0
            metrics[f"{field}_macro_f1"] = 0.0

        precision, recall, f1, support = precision_recall_fscore_support(
            all_gt[field],
            all_pred[field],
            labels=label_ids[field],
            zero_division=0,
        )
        for pos, (idx, label) in enumerate(zip(label_ids[field], label_order[field])):
            classwise_rows.append(
                {
                    "field": field,
                    "label_idx": idx,
                    "label": label,
                    "precision": float(precision[pos]),
                    "recall": float(recall[pos]),
                    "f1": float(f1[pos]),
                    "support": int(support[pos]),
                }
            )

        cm = confusion_matrix(all_gt[field], all_pred[field], labels=label_ids[field])
        pd.DataFrame(cm, index=label_order[field], columns=label_order[field]).to_csv(
            output_dir / f"confusion_{field}.csv"
        )
        plot_confusion(
            cm=cm,
            labels=label_order[field],
            title=f"{args.split} confusion: {field}",
            path=output_dir / f"confusion_{field}.png",
        )

    metrics["mean_macro_f1"] = float(np.mean(macro_f1_values)) if macro_f1_values else 0.0

    predictions = pd.DataFrame(sample_rows)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    predictions[predictions["any_error"] == 1].sort_values(
        ["num_field_errors", "mean_confidence"],
        ascending=[False, False],
    ).head(50).to_csv(output_dir / "failed_examples.csv", index=False)
    predictions[predictions["any_error"] == 0].sort_values(
        "mean_confidence",
        ascending=False,
    ).head(50).to_csv(output_dir / "successful_examples.csv", index=False)

    classwise_df = pd.DataFrame(classwise_rows)
    classwise_df.to_csv(output_dir / "classwise_metrics.csv", index=False)

    for field in fields:
        field_df = classwise_df[(classwise_df["field"] == field) & (classwise_df["support"] > 0)].copy()
        best_worst_classes[field] = {
            "best": field_df.sort_values(["f1", "support"], ascending=[False, False]).head(5).to_dict("records"),
            "worst": field_df.sort_values(["f1", "support"], ascending=[True, False]).head(5).to_dict("records"),
        }
    save_json(best_worst_classes, output_dir / "best_worst_classes.json")
    save_json(metrics, output_dir / "metrics.json")

    print(metrics)
    print(f"Saved detailed evaluation to {output_dir}")


if __name__ == "__main__":
    main()
