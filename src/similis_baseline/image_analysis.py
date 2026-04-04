from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def open_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def resize_longest_side(image: Image.Image, max_side: int) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(new_size, Image.BILINEAR)


def border_pixels(arr: np.ndarray, border: int = 8) -> np.ndarray:
    h, w = arr.shape[:2]
    border = max(1, min(border, h // 4, w // 4))
    strips = [
        arr[:border, :, :].reshape(-1, 3),
        arr[-border:, :, :].reshape(-1, 3),
        arr[:, :border, :].reshape(-1, 3),
        arr[:, -border:, :].reshape(-1, 3),
    ]
    return np.concatenate(strips, axis=0)


def estimate_foreground_mask(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    border = border_pixels(arr)
    bg_color = np.median(border, axis=0)
    border_brightness = border.mean(axis=1)
    border_mean = float(border_brightness.mean())
    border_std = float(border_brightness.std())

    diff = np.linalg.norm(arr.astype(np.float32) - bg_color.astype(np.float32), axis=2)
    if border_mean >= 225:
        threshold = 22.0
    elif border_mean >= 180:
        threshold = 28.0
    else:
        threshold = 35.0

    mask = diff > threshold
    return mask, bg_color, border_mean, border_std


def connected_components(mask: np.ndarray, min_area: int = 1) -> List[Dict]:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    components: List[Dict] = []

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue

            queue = deque([(y, x)])
            visited[y, x] = True
            area = 0
            x0 = x1 = x
            y0 = y1 = y

            while queue:
                cy, cx = queue.pop()
                area += 1
                if cx < x0:
                    x0 = cx
                if cx > x1:
                    x1 = cx
                if cy < y0:
                    y0 = cy
                if cy > y1:
                    y1 = cy

                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue
                    if visited[ny, nx] or not mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    queue.append((ny, nx))

            if area < min_area:
                continue

            width = x1 - x0 + 1
            height = y1 - y0 + 1
            components.append(
                {
                    "area": int(area),
                    "x0": int(x0),
                    "y0": int(y0),
                    "x1": int(x1),
                    "y1": int(y1),
                    "width": int(width),
                    "height": int(height),
                    "touches_border": bool(x0 == 0 or y0 == 0 or x1 == w - 1 or y1 == h - 1),
                }
            )

    components.sort(key=lambda item: item["area"], reverse=True)
    return components


def classify_bg_type(border_mean: float, border_std: float) -> str:
    if border_mean >= 235 and border_std <= 18:
        return "white_uniform"
    if border_mean >= 185:
        return "light_photo"
    return "dark_or_complex"


def classify_layout_mode(foreground_ratio: float, component_areas: List[float]) -> str:
    largest = component_areas[0] if component_areas else 0.0
    second = component_areas[1] if len(component_areas) > 1 else 0.0
    if second >= 0.16 or (second >= 0.10 and foreground_ratio >= 0.55 and largest >= 0.18):
        return "multi_view"
    if foreground_ratio <= 0.18 and largest <= 0.18:
        return "close_up"
    return "single_object"


def analyze_image(path: str | Path, analysis_max_side: int = 160) -> Dict:
    with open_rgb(path) as image:
        original_width, original_height = image.size
        resized = resize_longest_side(image, analysis_max_side)
        arr = np.asarray(resized)

    gray = arr.mean(axis=2)
    mask, bg_color, border_mean, border_std = estimate_foreground_mask(arr)
    area = int(mask.size)
    min_component_area = max(6, int(area * 0.0015))
    foreground_components = connected_components(mask, min_area=min_component_area)

    foreground_ratio = float(mask.mean())
    component_area_ratios = [component["area"] / max(area, 1) for component in foreground_components]
    largest_component_ratio = float(component_area_ratios[0]) if component_area_ratios else 0.0
    second_component_ratio = float(component_area_ratios[1]) if len(component_area_ratios) > 1 else 0.0

    dark_threshold = 110 if border_mean >= 185 else 70
    dark_mask = gray < dark_threshold
    dark_components = connected_components(dark_mask, min_area=4)
    has_scale_bar = any(
        component["touches_border"]
        and component["area"] / max(area, 1) >= 0.002
        and max(component["width"], component["height"]) / max(1, min(component["width"], component["height"])) >= 5.0
        for component in dark_components
    )
    small_margin_components = sum(
        4 <= component["area"] <= 180
        and (
            component["y0"] <= int(arr.shape[0] * 0.18)
            or component["y1"] >= int(arr.shape[0] * 0.82)
        )
        for component in dark_components
    )
    has_overlay_text = small_margin_components >= 6

    bg_type = classify_bg_type(border_mean=border_mean, border_std=border_std)
    layout_mode = classify_layout_mode(
        foreground_ratio=foreground_ratio,
        component_areas=component_area_ratios,
    )

    problem_score = 0.0
    problem_score += abs(np.log(max(original_width / max(original_height, 1), 1e-6)))
    problem_score += max(0.0, 0.12 - foreground_ratio) * 8.0
    problem_score += 0.6 if layout_mode == "multi_view" else 0.0
    problem_score += 0.4 if bg_type == "dark_or_complex" else 0.0
    problem_score += 0.3 if has_scale_bar else 0.0
    problem_score += 0.3 if has_overlay_text else 0.0

    return {
        "image_file": str(path),
        "image_width": int(original_width),
        "image_height": int(original_height),
        "aspect_ratio": float(original_width / max(original_height, 1)),
        "analysis_width": int(arr.shape[1]),
        "analysis_height": int(arr.shape[0]),
        "foreground_ratio": foreground_ratio,
        "bg_type": bg_type,
        "layout_mode": layout_mode,
        "component_count": int(len(foreground_components)),
        "largest_component_ratio": largest_component_ratio,
        "second_component_ratio": second_component_ratio,
        "has_scale_bar": int(has_scale_bar),
        "has_overlay_text": int(has_overlay_text),
        "border_brightness_mean": border_mean,
        "border_brightness_std": border_std,
        "problem_score": float(problem_score),
        "bg_color_r": float(bg_color[0]),
        "bg_color_g": float(bg_color[1]),
        "bg_color_b": float(bg_color[2]),
    }


def foreground_ratio_bin(value: float) -> str:
    if value < 0.08:
        return "lt_0.08"
    if value < 0.15:
        return "0.08_0.15"
    if value < 0.30:
        return "0.15_0.30"
    return "ge_0.30"
