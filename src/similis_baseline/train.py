from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import SimilisDataset
from .model import SimilisMultiTaskModel
from .utils import ensure_dir, load_config, load_json, save_json, seed_everything, torch_device


def build_class_weights(
    train_csv: str,
    fields,
    label_maps: Dict,
    max_weight: float = 5.0,
) -> Dict[str, torch.Tensor]:
    df = pd.read_csv(train_csv)
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

            idx = field_to_idx[value]
            counts[idx] += 1.0

        # inverse frequency
        weights = np.zeros(num_classes, dtype=np.float32)
        nonzero = counts > 0
        if nonzero.any():
            weights[nonzero] = counts[nonzero].sum() / (counts[nonzero] * nonzero.sum())

            # мягкая отсечка, чтобы редкие классы не взрывали loss
            weights = np.clip(weights, 1.0, max_weight)

            # нормируем так, чтобы средний вес по непустым классам был ~1
            weights[nonzero] = weights[nonzero] / weights[nonzero].mean()

        # для пустых классов просто 1.0
        weights[~nonzero] = 1.0

        class_weights[field] = torch.tensor(weights, dtype=torch.float32)

    return class_weights


def masked_ce_loss(logits, targets, mask, class_weights=None):
    loss = torch.nn.functional.cross_entropy(
        logits,
        targets,
        reduction="none",
        weight=class_weights,
    )
    loss = loss * mask
    denom = mask.sum().clamp(min=1.0)
    return loss.sum() / denom


def set_backbone_trainable(model, trainable: bool) -> None:
    for parameter in model.backbone.parameters():
        parameter.requires_grad = trainable


def build_optimizer(model, cfg) -> AdamW:
    head_lr = float(cfg.get("head_lr", cfg["lr"]))
    backbone_lr = float(cfg.get("backbone_lr", head_lr * float(cfg.get("backbone_lr_scale", 0.1))))
    weight_decay = float(cfg["weight_decay"])

    return AdamW(
        [
            {
                "params": list(model.backbone.parameters()),
                "lr": backbone_lr,
                "weight_decay": weight_decay,
                "name": "backbone",
            },
            {
                "params": list(model.heads.parameters()),
                "lr": head_lr,
                "weight_decay": weight_decay,
                "name": "heads",
            },
        ]
    )


class SafeFineTuneScheduler:
    def __init__(self, optimizer, cfg):
        self.optimizer = optimizer
        self.total_epochs = int(cfg["epochs"])
        self.freeze_backbone_epochs = int(cfg.get("freeze_backbone_epochs", 0))
        self.min_lr = float(cfg.get("min_lr", 0.0))
        self.schedule = str(cfg.get("lr_schedule", "cosine"))
        self.base_lrs = {
            group["name"]: float(group["lr"])
            for group in optimizer.param_groups
        }
        self.last_epoch = 0

    def _cosine_lr(self, base_lr: float, epoch_index: int, num_epochs: int) -> float:
        if num_epochs <= 1:
            return base_lr
        progress = epoch_index / max(num_epochs - 1, 1)
        return self.min_lr + 0.5 * (base_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))

    def _lr_for_group(self, group_name: str, epoch: int) -> float:
        base_lr = self.base_lrs[group_name]
        if self.schedule != "cosine":
            return base_lr

        if group_name == "backbone":
            if epoch <= self.freeze_backbone_epochs:
                return 0.0
            active_epochs = max(1, self.total_epochs - self.freeze_backbone_epochs)
            active_epoch_index = epoch - self.freeze_backbone_epochs - 1
            return self._cosine_lr(base_lr, active_epoch_index, active_epochs)

        epoch_index = epoch - 1
        return self._cosine_lr(base_lr, epoch_index, self.total_epochs)

    def step(self, epoch: int) -> None:
        self.last_epoch = int(epoch)
        for group in self.optimizer.param_groups:
            group["lr"] = self._lr_for_group(group["name"], epoch)

    def state_dict(self) -> Dict:
        return {
            "last_epoch": self.last_epoch,
            "total_epochs": self.total_epochs,
            "freeze_backbone_epochs": self.freeze_backbone_epochs,
            "min_lr": self.min_lr,
            "schedule": self.schedule,
            "base_lrs": self.base_lrs,
        }


def evaluate(model, loader, device, fields, loss_weights, class_weights):
    model.eval()
    losses = []
    all_gt = {f: [] for f in fields}
    all_pred = {f: [] for f in fields}

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            outputs = model(images)
            total_loss = 0.0

            for field in fields:
                y = batch["targets"][field].to(device)
                m = batch["target_mask"][field].to(device)
                cw = class_weights[field].to(device) if class_weights is not None else None

                loss = masked_ce_loss(outputs[field], y, m, class_weights=cw)
                total_loss += loss_weights[field] * loss

                preds = outputs[field].argmax(dim=1)
                keep = m.cpu().numpy().astype(bool)
                all_gt[field].extend(y.cpu().numpy()[keep].tolist())
                all_pred[field].extend(preds.cpu().numpy()[keep].tolist())

            losses.append(float(total_loss.item()))

    metrics = {"val_loss": float(np.mean(losses))}
    macro_f1_values = []

    for field in fields:
        if len(all_gt[field]) == 0:
            metrics[f"{field}_acc"] = 0.0
            metrics[f"{field}_macro_f1"] = 0.0
            continue

        metrics[f"{field}_acc"] = float(accuracy_score(all_gt[field], all_pred[field]))
        metrics[f"{field}_macro_f1"] = float(
            f1_score(all_gt[field], all_pred[field], average="macro", zero_division=0)
        )
        macro_f1_values.append(metrics[f"{field}_macro_f1"])

    metrics["mean_macro_f1"] = float(np.mean(macro_f1_values)) if macro_f1_values else 0.0
    return metrics


def train_one_epoch(model, loader, optimizer, device, fields, loss_weights, class_weights, max_grad_norm=None):
    model.train()
    running = []

    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device)
        outputs = model(images)
        total_loss = 0.0

        for field in fields:
            y = batch["targets"][field].to(device)
            m = batch["target_mask"][field].to(device)
            cw = class_weights[field].to(device) if class_weights is not None else None

            loss = masked_ce_loss(outputs[field], y, m, class_weights=cw)
            total_loss += loss_weights[field] * loss

        optimizer.zero_grad()
        total_loss.backward()
        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        running.append(float(total_loss.item()))

    return float(np.mean(running))


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_metric, config, vocabs):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "epoch": epoch,
            "best_metric": best_metric,
            "config": config,
            "vocabs": vocabs,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    device = torch_device()

    ensure_dir(cfg["checkpoint_dir"])
    ensure_dir(cfg["reports_dir"])

    label_maps_path = cfg.get("label_maps_path", "data/processed/label_maps.json")
    label_maps = load_json(label_maps_path)
    fields = cfg["fields"]
    preprocess_mode = str(cfg.get("preprocess_mode", "pad"))
    use_class_weights = bool(cfg.get("use_class_weights", True))

    vocabs = {
        f: label_maps[f]["field_to_idx"]
        for f in fields
    }

    num_classes = {
        f: len(label_maps[f]["field_to_idx"])
        for f in fields
    }

    train_ds = SimilisDataset(
        cfg["train_csv"],
        image_size=cfg["image_size"],
        train=True,
        fields=fields,
        label_maps_path=label_maps_path,
        preprocess_mode=preprocess_mode,
    )
    val_ds = SimilisDataset(
        cfg["val_csv"],
        image_size=cfg["image_size"],
        train=False,
        fields=fields,
        label_maps_path=label_maps_path,
        preprocess_mode=preprocess_mode,
    )
    test_ds = SimilisDataset(
        cfg["test_csv"],
        image_size=cfg["image_size"],
        train=False,
        fields=fields,
        label_maps_path=label_maps_path,
        preprocess_mode=preprocess_mode,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
    )

    print(
        {
            "train_rows": len(train_ds),
            "val_rows": len(val_ds),
            "test_rows": len(test_ds),
            "test_batches": len(test_loader),
        }
    )

    class_weights = None
    if use_class_weights:
        class_weights = build_class_weights(
            train_csv=cfg["train_csv"],
            fields=fields,
            label_maps=label_maps,
            max_weight=5.0,
        )

        print("Class weights:")
        for field in fields:
            idx_to_field = {
                int(k): v for k, v in label_maps[field]["idx_to_field"].items()
            }
            weights_np = class_weights[field].numpy()
            printable = {
                idx_to_field[i]: round(float(weights_np[i]), 4)
                for i in range(len(weights_np))
            }
            print(field, printable)
    else:
        print("Class weights: disabled")

    model = SimilisMultiTaskModel(
        cfg["backbone"],
        num_classes=num_classes,
        pretrained=bool(cfg.get("pretrained", True)),
    ).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = SafeFineTuneScheduler(optimizer, cfg)
    loss_weights = cfg["loss_weights"]
    max_grad_norm = float(cfg.get("max_grad_norm", 0.0))

    print(
        {
            "head_lr": float(cfg.get("head_lr", cfg["lr"])),
            "backbone_lr": float(cfg.get("backbone_lr", float(cfg["lr"]) * float(cfg.get("backbone_lr_scale", 0.1)))),
            "freeze_backbone_epochs": int(cfg.get("freeze_backbone_epochs", 0)),
            "max_grad_norm": max_grad_norm,
            "lr_schedule": str(cfg.get("lr_schedule", "cosine")),
            "min_lr": float(cfg.get("min_lr", 0.0)),
            "preprocess_mode": preprocess_mode,
            "use_class_weights": use_class_weights,
        }
    )

    best_metric = -1.0
    log_path = Path(cfg["reports_dir"]) / "train_log.csv"

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = None

        for epoch in range(1, cfg["epochs"] + 1):
            backbone_trainable = epoch > int(cfg.get("freeze_backbone_epochs", 0))
            set_backbone_trainable(model, backbone_trainable)
            scheduler.step(epoch)

            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                fields,
                loss_weights,
                class_weights,
                max_grad_norm=max_grad_norm,
            )
            metrics = evaluate(
                model,
                val_loader,
                device,
                fields,
                loss_weights,
                class_weights,
            )

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                **metrics,
                "backbone_trainable": int(backbone_trainable),
                "backbone_lr": optimizer.param_groups[0]["lr"],
                "head_lr": optimizer.param_groups[1]["lr"],
            }

            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()

            writer.writerow(row)
            f.flush()
            print(row)

            save_checkpoint(
                Path(cfg["checkpoint_dir"]) / "last.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_metric,
                cfg,
                vocabs,
            )

            if metrics["mean_macro_f1"] > best_metric:
                best_metric = metrics["mean_macro_f1"]
                save_checkpoint(
                    Path(cfg["checkpoint_dir"]) / "best.pt",
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_metric,
                    cfg,
                    vocabs,
                )

    save_json(
        {"best_val_mean_macro_f1": best_metric},
        Path(cfg["reports_dir"]) / "best_metrics.json",
    )


if __name__ == "__main__":
    main()
