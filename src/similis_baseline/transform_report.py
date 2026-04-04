from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile

from .dataset import IMAGENET_MEAN, IMAGENET_STD, build_transforms
from .utils import ensure_dir, load_config, save_json, seed_everything

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--image-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="artifacts/reports/transform_report")
    parser.add_argument("--train-variants", type=int, default=3)
    return parser.parse_args()


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    image = tensor.detach().cpu() * std + mean
    image = image.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return image


def select_image_path(cfg: dict, image_path: str | None) -> str:
    if image_path:
        return image_path
    train_df = pd.read_csv(cfg["train_csv"])
    return str(train_df.iloc[0]["image_file"])


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg["seed"]))

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    selected_path = select_image_path(cfg, args.image_path)
    with Image.open(selected_path) as image:
        original = image.convert("RGB")
        original_np = np.asarray(original)

    current_mode = str(cfg.get("preprocess_mode", "pad"))
    compare_mode = "stretch" if current_mode != "stretch" else "pad"

    train_transform = build_transforms(
        image_size=int(cfg["image_size"]),
        train=True,
        preprocess_mode=current_mode,
    )
    eval_transform = build_transforms(
        image_size=int(cfg["image_size"]),
        train=False,
        preprocess_mode=current_mode,
    )
    compare_transform = build_transforms(
        image_size=int(cfg["image_size"]),
        train=False,
        preprocess_mode=compare_mode,
    )

    images = [("original", original_np)]
    for index in range(args.train_variants):
        images.append((f"train_aug_{index + 1}", denormalize_image(train_transform(original))))
    images.append((f"eval_{current_mode}", denormalize_image(eval_transform(original))))
    images.append((f"eval_{compare_mode}", denormalize_image(compare_transform(original))))

    cols = len(images)
    plt.figure(figsize=(cols * 3.5, 4.0))
    for index, (title, array) in enumerate(images, start=1):
        plt.subplot(1, cols, index)
        plt.imshow(array)
        plt.title(title, fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "transform_examples.png")
    plt.close()

    summary = {
        "image_path": selected_path,
        "image_size": int(cfg["image_size"]),
        "current_preprocess_mode": current_mode,
        "compare_preprocess_mode": compare_mode,
        "train_variants": int(args.train_variants),
    }
    save_json(summary, output_dir / "summary.json")

    print(summary)
    print(f"Saved transform report to {output_dir}")


if __name__ == "__main__":
    main()
