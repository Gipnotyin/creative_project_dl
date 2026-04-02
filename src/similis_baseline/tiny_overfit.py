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
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .dataset import SimilisDataset
from .model import SimilisMultiTaskModel
from .train import build_class_weights, masked_ce_loss
from .utils import ensure_dir, load_config, load_json, save_json, seed_everything, torch_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--fields",
        type=str,
        default=None,
        help="Comma-separated subset of fields to train, e.g. type,material",
    )
    parser.add_argument("--subset-size", type=int, default=32)
    parser.add_argument(
        "--balance-by",
        type=str,
        default=None,
        help="Sample a more balanced tiny subset by this field.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default="artifacts/reports/tiny_overfit")
    parser.add_argument("--use-pretrained", action="store_true")
    parser.add_argument(
        "--class-weights-source",
        choices=["none", "train", "subset"],
        default="none",
    )
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument(
        "--reset-heads",
        action="store_true",
        help="Reinitialize task heads after loading checkpoint.",
    )
    parser.add_argument(
        "--scheduler",
        choices=["none", "cosine"],
        default="none",
    )
    parser.add_argument(
        "--train-aug",
        action="store_true",
        help="Enable train-time augmentations for the tiny subset. Disabled by default.",
    )
    parser.add_argument(
        "--exclude-uncertain",
        action="store_true",
        help="Drop rows whose raw text contains uncertainty markers like '?', 'вероятно', 'возможно'.",
    )
    parser.add_argument(
        "--require-complete-labels",
        action="store_true",
        help="Keep only rows with non-missing labels for all active fields.",
    )
    parser.add_argument(
        "--exclude-part-labels",
        type=str,
        default=None,
        help="Comma-separated part labels to exclude from the tiny subset candidate pool.",
    )
    return parser.parse_args()


def collect_epoch_metrics(model, loader, device, fields, loss_weights, class_weights):
    model.eval()
    losses = []
    field_losses = {field: [] for field in fields}
    all_gt = {field: [] for field in fields}
    all_pred = {field: [] for field in fields}

    with torch.no_grad():
        for batch in tqdm(loader, desc="tiny-metrics", leave=False):
            images = batch["image"].to(device)
            outputs = model(images)
            total_loss = 0.0

            for field in fields:
                targets = batch["targets"][field].to(device)
                mask = batch["target_mask"][field].to(device)
                cw = class_weights[field].to(device) if class_weights is not None else None
                loss = masked_ce_loss(outputs[field], targets, mask, class_weights=cw)
                total_loss += loss_weights[field] * loss
                field_losses[field].append(float(loss.item()))

                preds = outputs[field].argmax(dim=1)
                keep = mask.detach().cpu().numpy().astype(bool)
                all_gt[field].extend(targets.detach().cpu().numpy()[keep].tolist())
                all_pred[field].extend(preds.detach().cpu().numpy()[keep].tolist())

            losses.append(float(total_loss.item()))

        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.detach_()
                parameter.grad.zero_()

    metrics = {"subset_loss": float(np.mean(losses)) if losses else 0.0}
    macro_f1_values = []
    for field in fields:
        if not all_gt[field]:
            metrics[f"{field}_acc"] = 0.0
            metrics[f"{field}_macro_f1"] = 0.0
            metrics[f"{field}_subset_loss"] = float(np.mean(field_losses[field])) if field_losses[field] else 0.0
            continue
        metrics[f"{field}_subset_loss"] = float(np.mean(field_losses[field])) if field_losses[field] else 0.0
        metrics[f"{field}_acc"] = float(accuracy_score(all_gt[field], all_pred[field]))
        metrics[f"{field}_macro_f1"] = float(
            f1_score(all_gt[field], all_pred[field], average="macro", zero_division=0)
        )
        macro_f1_values.append(metrics[f"{field}_macro_f1"])
    metrics["mean_macro_f1"] = float(np.mean(macro_f1_values)) if macro_f1_values else 0.0
    return metrics


def train_one_epoch(model, loader, optimizer, device, fields, loss_weights, class_weights):
    model.train()
    losses = []

    for batch in tqdm(loader, desc="tiny-train", leave=False):
        images = batch["image"].to(device)
        outputs = model(images)
        total_loss = 0.0

        for field in fields:
            targets = batch["targets"][field].to(device)
            mask = batch["target_mask"][field].to(device)
            cw = class_weights[field].to(device) if class_weights is not None else None
            loss = masked_ce_loss(outputs[field], targets, mask, class_weights=cw)
            total_loss += loss_weights[field] * loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        losses.append(float(total_loss.item()))

    return float(np.mean(losses)) if losses else 0.0


def summarize_subset(df: pd.DataFrame, fields: List[str]) -> Dict:
    field_counts = {}
    for field in fields:
        field_counts[field] = df[field].value_counts(dropna=False).to_dict()
    missing_counts = {
        field: int(df.get(f"{field}_is_missing", pd.Series(dtype=int)).fillna(0).sum())
        for field in fields
    }
    return {
        "rows": int(len(df)),
        "field_value_counts": field_counts,
        "missing_label_counts": missing_counts,
    }


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def reset_model_heads(model: torch.nn.Module) -> None:
    for module in model.heads.values():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()


def filter_candidate_pool(
    df: pd.DataFrame,
    fields: List[str],
    exclude_uncertain: bool,
    require_complete_labels: bool,
    exclude_part_labels: List[str] | None = None,
) -> tuple[pd.DataFrame, Dict]:
    work_df = df.copy()
    work_df["_dataset_index"] = work_df.index
    summary = {
        "rows_before_filters": int(len(work_df)),
        "excluded_uncertain_rows": 0,
        "excluded_missing_label_rows": 0,
        "excluded_part_label_rows": 0,
    }

    if exclude_uncertain:
        text = (
            work_df.get("name", pd.Series("", index=work_df.index)).fillna("").astype(str)
            + " "
            + work_df.get("description", pd.Series("", index=work_df.index)).fillna("").astype(str)
            + " "
            + work_df.get("fragm", pd.Series("", index=work_df.index)).fillna("").astype(str)
        ).str.lower()
        uncertain_mask = text.str.contains(r"\?|вероят|возможно|неяс|\(\?\)|сомнит", regex=True)
        summary["excluded_uncertain_rows"] = int(uncertain_mask.sum())
        work_df = work_df[~uncertain_mask].copy()

    if require_complete_labels:
        complete_mask = pd.Series(True, index=work_df.index)
        for field in fields:
            missing_col = f"{field}_is_missing"
            if missing_col in work_df.columns:
                complete_mask &= work_df[missing_col].fillna(0).astype(int).eq(0)
        summary["excluded_missing_label_rows"] = int((~complete_mask).sum())
        work_df = work_df[complete_mask].copy()

    if exclude_part_labels:
        if "part" in fields and "part" in work_df.columns:
            exclude_values = {value for value in exclude_part_labels if value}
            exclude_mask = work_df["part"].astype(str).isin(exclude_values)
            summary["excluded_part_label_rows"] = int(exclude_mask.sum())
            work_df = work_df[~exclude_mask].copy()

    summary["rows_after_filters"] = int(len(work_df))
    return work_df.reset_index(drop=True), summary


def choose_subset_indices(
    df: pd.DataFrame,
    subset_size: int,
    seed: int,
    balance_by: str | None = None,
) -> List[int]:
    if not balance_by:
        rng = np.random.default_rng(seed)
        return rng.permutation(len(df))[:subset_size].tolist()

    work_df = df.reset_index(drop=True).copy()
    if balance_by not in work_df.columns:
        raise ValueError(f"Cannot balance by missing column: {balance_by}")

    missing_col = f"{balance_by}_is_missing"
    if missing_col in work_df.columns:
        work_df = work_df[work_df[missing_col].fillna(0).astype(int) == 0].copy()

    rng = np.random.default_rng(seed)
    label_to_indices = {}
    for label, part in work_df.groupby(balance_by):
        indices = part.index.to_list()
        rng.shuffle(indices)
        label_to_indices[str(label)] = indices

    ordered_labels = sorted(label_to_indices.keys())
    chosen = []
    exhausted = False
    while len(chosen) < subset_size and not exhausted:
        exhausted = True
        for label in ordered_labels:
            if label_to_indices[label]:
                chosen.append(label_to_indices[label].pop())
                exhausted = False
                if len(chosen) >= subset_size:
                    break

    if not chosen:
        raise ValueError(f"No samples available after balancing by {balance_by}")
    return chosen[:subset_size]


def build_class_weights_from_df(
    df: pd.DataFrame,
    fields: List[str],
    label_maps: Dict,
    max_weight: float = 5.0,
) -> Dict[str, torch.Tensor]:
    class_weights = {}

    for field in fields:
        field_to_idx = {
            str(k): int(v)
            for k, v in label_maps[field]["field_to_idx"].items()
        }
        num_classes = len(field_to_idx)
        counts = np.zeros(num_classes, dtype=np.float32)

        missing_col = f"{field}_is_missing"
        has_missing_col = missing_col in df.columns

        for _, row in df.iterrows():
            if has_missing_col and int(row[missing_col]) == 1:
                continue
            value = str(row[field])
            if value not in field_to_idx:
                continue
            counts[field_to_idx[value]] += 1.0

        weights = np.ones(num_classes, dtype=np.float32)
        nonzero = counts > 0
        if nonzero.any():
            weights[nonzero] = counts[nonzero].sum() / (counts[nonzero] * nonzero.sum())
            weights = np.clip(weights, 1.0, max_weight)
            weights[nonzero] = weights[nonzero] / weights[nonzero].mean()

        class_weights[field] = torch.tensor(weights, dtype=torch.float32)

    return class_weights


def plot_history(history: pd.DataFrame, fields: List[str], output_dir: Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(history["epoch"], history["train_loss"], marker="o")
    plt.xlabel("epoch")
    plt.ylabel("train_loss")
    plt.title("Tiny Overfit Train Loss")
    plt.tight_layout()
    plt.savefig(output_dir / "train_loss.png")
    plt.close()

    if "subset_loss" in history.columns:
        plt.figure(figsize=(8, 4))
        plt.plot(history["epoch"], history["subset_loss"], marker="o")
        plt.xlabel("epoch")
        plt.ylabel("subset_loss")
        plt.title("Tiny Overfit Subset Loss")
        plt.tight_layout()
        plt.savefig(output_dir / "subset_loss.png")
        plt.close()

    plt.figure(figsize=(10, 5))
    for field in fields:
        plt.plot(history["epoch"], history[f"{field}_acc"], marker="o", label=f"{field}_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Tiny Overfit Field Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "train_accuracy.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    for field in fields:
        plt.plot(history["epoch"], history[f"{field}_macro_f1"], marker="o", label=f"{field}_macro_f1")
    plt.plot(history["epoch"], history["mean_macro_f1"], marker="o", linewidth=2, label="mean_macro_f1")
    plt.xlabel("epoch")
    plt.ylabel("macro_f1")
    plt.title("Tiny Overfit Macro-F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "train_macro_f1.png")
    plt.close()


def collect_prediction_rows(
    model,
    loader,
    device,
    fields: List[str],
    idx_to_label: Dict[str, Dict[int, str]],
) -> pd.DataFrame:
    rows = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            outputs = model(images)
            batch_size = int(images.shape[0])

            probs_by_field = {
                field: torch.softmax(outputs[field], dim=1).detach().cpu()
                for field in fields
            }
            pred_by_field = {
                field: probs_by_field[field].argmax(dim=1)
                for field in fields
            }

            for row_index in range(batch_size):
                row = {
                    "image_file": batch["metadata"]["image_file"][row_index],
                    "group_key": batch["metadata"]["group_key"][row_index],
                }
                num_field_errors = 0

                for field in fields:
                    mask_value = float(batch["target_mask"][field][row_index].item())
                    gt_idx = int(batch["targets"][field][row_index].item())
                    pred_idx = int(pred_by_field[field][row_index].item())
                    conf = float(probs_by_field[field][row_index, pred_idx].item())

                    gt_label = idx_to_label[field].get(gt_idx, "__UNKNOWN__") if mask_value else "__MISSING__"
                    pred_label = idx_to_label[field].get(pred_idx, "__UNKNOWN__")
                    correct = int(mask_value == 1.0 and gt_idx == pred_idx)

                    row[f"{field}_mask"] = mask_value
                    row[f"gt_{field}"] = gt_label
                    row[f"pred_{field}"] = pred_label
                    row[f"confidence_{field}"] = round(conf, 6)
                    row[f"correct_{field}"] = correct

                    if mask_value == 1.0 and not correct:
                        num_field_errors += 1

                row["num_field_errors"] = num_field_errors
                row["all_fields_correct"] = int(num_field_errors == 0)
                rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])

    label_maps_path = cfg.get("label_maps_path", "data/processed/label_maps.json")
    label_maps = load_json(label_maps_path)
    fields = (
        [field.strip() for field in args.fields.split(",") if field.strip()]
        if args.fields
        else list(cfg["fields"])
    )
    exclude_part_labels = (
        [value.strip() for value in args.exclude_part_labels.split(",") if value.strip()]
        if args.exclude_part_labels
        else None
    )
    num_classes = {field: len(label_maps[field]["field_to_idx"]) for field in fields}
    idx_to_label = {
        field: {int(idx): label for idx, label in label_maps[field]["idx_to_field"].items()}
        for field in fields
    }
    image_size = args.image_size if args.image_size is not None else cfg["image_size"]

    dataset = SimilisDataset(
        cfg["train_csv"],
        image_size=image_size,
        train=bool(args.train_aug),
        fields=fields,
        label_maps_path=label_maps_path,
    )
    candidate_df, candidate_summary = filter_candidate_pool(
        df=dataset.df,
        fields=fields,
        exclude_uncertain=bool(args.exclude_uncertain),
        require_complete_labels=bool(args.require_complete_labels),
        exclude_part_labels=exclude_part_labels,
    )
    if candidate_df.empty:
        raise ValueError("No candidate rows left after tiny-overfit filters")

    subset_size = min(args.subset_size, len(candidate_df))
    candidate_positions = choose_subset_indices(
        df=candidate_df,
        subset_size=subset_size,
        seed=int(cfg["seed"]),
        balance_by=args.balance_by,
    )
    subset_indices = candidate_df.iloc[candidate_positions]["_dataset_index"].astype(int).tolist()
    subset = Subset(dataset, subset_indices)

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    subset_df = dataset.df.iloc[subset_indices].copy()
    subset_df.to_csv(output_dir / "tiny_subset.csv", index=False)
    save_json(
        {
            **summarize_subset(subset_df, fields),
            "candidate_pool": candidate_summary,
            "exclude_uncertain": bool(args.exclude_uncertain),
            "require_complete_labels": bool(args.require_complete_labels),
            "exclude_part_labels": exclude_part_labels or [],
        },
        output_dir / "tiny_subset_summary.json",
    )

    train_loader = DataLoader(
        subset,
        batch_size=min(args.batch_size, subset_size),
        shuffle=True,
        num_workers=0,
    )
    eval_loader = DataLoader(
        subset,
        batch_size=min(args.batch_size, subset_size),
        shuffle=False,
        num_workers=0,
    )

    device = torch_device()
    model = SimilisMultiTaskModel(
        cfg["backbone"],
        num_classes=num_classes,
        pretrained=bool(args.use_pretrained and not args.checkpoint),
    ).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if args.reset_heads:
        reset_model_heads(model)

    if args.freeze_backbone:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False

    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr if args.lr is not None else cfg["lr"],
        weight_decay=args.weight_decay if args.weight_decay is not None else cfg["weight_decay"],
    )
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    loss_weights = {field: float(cfg["loss_weights"][field]) for field in fields}
    loss_weight_sum = sum(loss_weights.values())
    if loss_weight_sum > 0:
        loss_weights = {field: value / loss_weight_sum for field, value in loss_weights.items()}

    class_weights = None
    if args.class_weights_source == "train":
        class_weights = build_class_weights(
            train_csv=cfg["train_csv"],
            fields=fields,
            label_maps=label_maps,
            max_weight=5.0,
        )
    elif args.class_weights_source == "subset":
        class_weights = build_class_weights_from_df(
            df=subset_df,
            fields=fields,
            label_maps=label_maps,
            max_weight=5.0,
        )

    history_rows = []
    best_mean_macro_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            fields=fields,
            loss_weights=loss_weights,
            class_weights=class_weights,
        )
        metrics = collect_epoch_metrics(
            model=model,
            loader=eval_loader,
            device=device,
            fields=fields,
            loss_weights=loss_weights,
            class_weights=class_weights,
        )
        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history_rows.append(row)
        best_mean_macro_f1 = max(best_mean_macro_f1, row["mean_macro_f1"])
        print(row)

    history = pd.DataFrame(history_rows)
    history.to_csv(output_dir / "history.csv", index=False)
    plot_history(history, fields, output_dir)

    predictions = collect_prediction_rows(
        model=model,
        loader=eval_loader,
        device=device,
        fields=fields,
        idx_to_label=idx_to_label,
    )
    subset_export = subset_df.reset_index(drop=True).copy()
    predictions_export = pd.concat(
        [subset_export, predictions.drop(columns=["image_file", "group_key"], errors="ignore")],
        axis=1,
    )
    predictions_export.to_csv(output_dir / "tiny_subset_predictions.csv", index=False)
    predictions_export[predictions_export["num_field_errors"] > 0].to_csv(
        output_dir / "tiny_subset_errors.csv",
        index=False,
    )

    summary = {
        "subset_size": subset_size,
        "epochs": args.epochs,
        "batch_size": min(args.batch_size, subset_size),
        "steps_per_epoch": len(train_loader),
        "total_optimizer_steps": int(len(train_loader) * args.epochs),
        "image_size": image_size,
        "fields": fields,
        "balance_by": args.balance_by,
        "train_aug_enabled": bool(args.train_aug),
        "class_weights_source": args.class_weights_source,
        "exclude_uncertain": bool(args.exclude_uncertain),
        "require_complete_labels": bool(args.require_complete_labels),
        "exclude_part_labels": exclude_part_labels or [],
        "candidate_pool": candidate_summary,
        "checkpoint": args.checkpoint,
        "scheduler": args.scheduler,
        "freeze_backbone": bool(args.freeze_backbone),
        "reset_heads": bool(args.reset_heads),
        "trainable_parameters": count_trainable_parameters(model),
        "start_train_loss": float(history.iloc[0]["train_loss"]),
        "end_train_loss": float(history.iloc[-1]["train_loss"]),
        "start_subset_loss": float(history.iloc[0]["subset_loss"]),
        "end_subset_loss": float(history.iloc[-1]["subset_loss"]),
        "best_mean_macro_f1": float(best_mean_macro_f1),
        "final_metrics": history.iloc[-1].to_dict(),
        "prediction_rows": int(len(predictions_export)),
        "overfit_success": bool(
            history.iloc[-1]["subset_loss"] <= history.iloc[0]["subset_loss"] * 0.6
            or history.iloc[-1]["mean_macro_f1"] >= 0.9
        ),
    }
    save_json(summary, output_dir / "summary.json")

    print(summary)
    print(f"Saved tiny-overfit artifacts to {output_dir}")


if __name__ == "__main__":
    main()
