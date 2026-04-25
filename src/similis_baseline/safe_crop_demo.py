from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFile

from .image_analysis import connected_components, estimate_foreground_mask
from .utils import ensure_dir

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


DEFAULT_IMAGE = "data/raw/images/Нц-45-0012_orig.jpg"


def pad_to_square(image: Image.Image, fill=(255, 255, 255)) -> Image.Image:
    width, height = image.size
    if width == height:
        return image
    side = max(width, height)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas


def safe_content_aware_crop(image: Image.Image, margin_ratio: float = 0.08) -> tuple[Image.Image, tuple[int, int, int, int]]:
    arr = np.asarray(image)
    mask, _, _, _ = estimate_foreground_mask(arr)
    components = connected_components(mask, min_area=max(6, int(mask.size * 0.0015)))

    if not components:
        width, height = image.size
        return image, (0, 0, width, height)

    x0 = min(c["x0"] for c in components)
    y0 = min(c["y0"] for c in components)
    x1 = max(c["x1"] for c in components)
    y1 = max(c["y1"] for c in components)

    height, width = arr.shape[:2]
    bbox_w = x1 - x0 + 1
    bbox_h = y1 - y0 + 1
    margin_x = int(bbox_w * margin_ratio)
    margin_y = int(bbox_h * margin_ratio)

    cx0 = max(0, x0 - margin_x)
    cy0 = max(0, y0 - margin_y)
    cx1 = min(width - 1, x1 + margin_x)
    cy1 = min(height - 1, y1 + margin_y)

    cropped = image.crop((cx0, cy0, cx1 + 1, cy1 + 1))
    return cropped, (cx0, cy0, cx1, cy1)


def aggressive_center_crop(image: Image.Image, target_side: int) -> Image.Image:
    width, height = image.size
    short_side = min(width, height)
    left = (width - short_side) // 2
    top = (height - short_side) // 2
    cropped = image.crop((left, top, left + short_side, top + short_side))
    return cropped.resize((target_side, target_side), Image.BILINEAR)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=str, default=DEFAULT_IMAGE)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument(
        "--output-path",
        type=str,
        default="artifacts/reports/transform_report/safe_crop_vs_pad.png",
    )
    args = parser.parse_args()

    src = Image.open(args.image_path).convert("RGB")
    side = int(args.image_size)

    padded = pad_to_square(src).resize((side, side), Image.BILINEAR)
    cropped_safe, bbox = safe_content_aware_crop(src, margin_ratio=0.08)
    cropped_safe_resized = pad_to_square(cropped_safe).resize((side, side), Image.BILINEAR)
    aggressive = aggressive_center_crop(src, side)

    panels = [
        ("original", np.asarray(src), f"{src.size[0]}×{src.size[1]}"),
        ("pad + resize (current baseline)", np.asarray(padded), f"{side}×{side}"),
        ("safe content-aware crop + pad + resize", np.asarray(cropped_safe_resized), f"{side}×{side}"),
        ("aggressive center crop (anti-pattern)", np.asarray(aggressive), f"{side}×{side}"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, (title, arr, sub) in zip(axes, panels):
        ax.imshow(arr)
        ax.set_title(f"{title}\n{sub}", fontsize=10)
        ax.axis("off")

    fig.suptitle(
        "Safe crop vs pad: content-aware crop сохраняет foreground без потери информации, "
        "тогда как aggressive center-crop отрезает часть артефакта",
        fontsize=11,
    )
    fig.tight_layout()

    output_path = Path(args.output_path)
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")
    print(f"  source: {args.image_path}")
    print(f"  bbox(safe-crop): {bbox}  src: {src.size}")


if __name__ == "__main__":
    main()
