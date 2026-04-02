from __future__ import annotations

from typing import Dict, List

import pandas as pd
import torch
from PIL import Image, ImageFile, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .utils import load_json

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class PadToSquare:
    def __init__(self, fill: int | tuple[int, int, int] = (255, 255, 255)):
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if width == height:
            return image

        max_side = max(width, height)
        pad_w = max_side - width
        pad_h = max_side - height
        left = pad_w // 2
        top = pad_h // 2
        right = pad_w - left
        bottom = pad_h - top
        return ImageOps.expand(image, border=(left, top, right, bottom), fill=self.fill)


def build_transforms(image_size: int, train: bool):
    resize = transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR)
    to_tensor = [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    if train:
        return transforms.Compose(
            [
                PadToSquare(),
                resize,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(
                    degrees=5,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=(255, 255, 255),
                ),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                *to_tensor,
            ]
        )
    return transforms.Compose(
        [
            PadToSquare(),
            resize,
            *to_tensor,
        ]
    )


class SimilisDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        image_size: int,
        train: bool,
        fields: List[str],
        label_maps_path: str = "data/processed/label_maps.json",
    ):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.fields = fields
        self.transform = build_transforms(image_size=image_size, train=train)

        label_maps = load_json(label_maps_path)
        self.field_to_idx: Dict[str, Dict[str, int]] = {
            field: {str(k): int(v) for k, v in label_maps[field]["field_to_idx"].items()}
            for field in fields
        }

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, image_path: str) -> Image.Image:
        img = Image.open(image_path).convert("RGB")
        return img

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        image_path = row["image_file"]
        image = self._load_image(image_path)
        image = self.transform(image)

        targets = {}
        target_mask = {}

        for field in self.fields:
            value = str(row[field])

            missing_col = f"{field}_is_missing"
            is_missing = int(row[missing_col]) if missing_col in row else 0

            if is_missing == 1 or value not in self.field_to_idx[field]:
                targets[field] = torch.tensor(0, dtype=torch.long)
                target_mask[field] = torch.tensor(0.0, dtype=torch.float32)
            else:
                targets[field] = torch.tensor(self.field_to_idx[field][value], dtype=torch.long)
                target_mask[field] = torch.tensor(1.0, dtype=torch.float32)

        metadata = {
            "image_file": str(row["image_file"]),
            "group_key": str(row["group_key"]) if "group_key" in row else "",
        }

        return {
            "image": image,
            "targets": targets,
            "target_mask": target_mask,
            "metadata": metadata,
        }
