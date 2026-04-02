from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def mps_available() -> bool:
    return (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )


def load_config(path: str | os.PathLike) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | os.PathLike) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], path: str | os.PathLike) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str | os.PathLike) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text)
    if text.lower() == "nan":
        return ""
    text = text.lower().strip()
    text = text.replace("ё", "е")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def slugify_code(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"[^0-9a-zа-я]+", "", text)
    return text


def torch_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if mps_available():
        return "mps"
    return "cpu"
