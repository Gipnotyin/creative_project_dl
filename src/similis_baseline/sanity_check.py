from __future__ import annotations

import argparse
import platform

import torch
import torchvision
from torch.utils.data import DataLoader

from .dataset import SimilisDataset
from .predict import build_auto_description
from .utils import load_config, load_json, mps_available, seed_everything, torch_device


def decode_example(batch, index, fields, idx_to_label, thresholds):
    labels = {}
    confidences = {}

    for field in fields:
        mask = float(batch["target_mask"][field][index].item())
        confidences[field] = mask
        if mask == 0.0:
            labels[field] = "неизвестно"
            continue

        label_idx = int(batch["targets"][field][index].item())
        labels[field] = idx_to_label[field][label_idx]

    auto_description = build_auto_description(labels, confidences, thresholds)
    return labels, auto_description


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])

    label_maps = load_json(cfg["label_maps_path"])
    fields = cfg["fields"]
    idx_to_label = {
        field: {int(idx): label for idx, label in label_maps[field]["idx_to_field"].items()}
        for field in fields
    }

    train_ds = SimilisDataset(
        cfg["train_csv"],
        image_size=cfg["image_size"],
        train=True,
        fields=fields,
        label_maps_path=cfg["label_maps_path"],
    )
    val_ds = SimilisDataset(
        cfg["val_csv"],
        image_size=cfg["image_size"],
        train=False,
        fields=fields,
        label_maps_path=cfg["label_maps_path"],
    )
    test_ds = SimilisDataset(
        cfg["test_csv"],
        image_size=cfg["image_size"],
        train=False,
        fields=fields,
        label_maps_path=cfg["label_maps_path"],
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    train_labels, train_auto_description = decode_example(
        train_batch,
        index=0,
        fields=fields,
        idx_to_label=idx_to_label,
        thresholds=cfg.get("confidence_thresholds", {}),
    )

    train_repeat_equal = torch.allclose(train_ds[0]["image"], train_ds[0]["image"])
    val_repeat_equal = torch.allclose(val_ds[0]["image"], val_ds[0]["image"])

    print("Environment")
    print(
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "device": torch_device(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "mps_available": mps_available(),
            "mps_built": getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_built(),
        }
    )

    print("Dataset sizes")
    print(
        {
            "train_rows": len(train_ds),
            "val_rows": len(val_ds),
            "test_rows": len(test_ds),
        }
    )

    print("Batch shapes and dtypes")
    print(
        {
            "train_image_shape": tuple(train_batch["image"].shape),
            "train_image_dtype": str(train_batch["image"].dtype),
            "val_image_shape": tuple(val_batch["image"].shape),
            "val_image_dtype": str(val_batch["image"].dtype),
            "target_dtypes": {field: str(train_batch["targets"][field].dtype) for field in fields},
        }
    )

    print("Augmentation sanity")
    print(
        {
            "train_repeat_allclose": bool(train_repeat_equal),
            "val_repeat_allclose": bool(val_repeat_equal),
        }
    )

    print("Decoded example")
    print(
        {
            "metadata": {
                "image_file": train_batch["metadata"]["image_file"][0],
                "group_key": train_batch["metadata"]["group_key"][0],
            },
            "labels": train_labels,
            "target_mask": {field: float(train_batch["target_mask"][field][0].item()) for field in fields},
            "ground_truth_auto_description": train_auto_description,
        }
    )

    print("Loader sizes")
    print(
        {
            "train_batches": len(train_loader),
            "val_batches": len(val_loader),
            "test_batches": len(test_loader),
        }
    )


if __name__ == "__main__":
    main()
