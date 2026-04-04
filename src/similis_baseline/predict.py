from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from PIL import Image, ImageFile
from tqdm import tqdm

from .dataset import build_transforms
from .model import SimilisMultiTaskModel
from .utils import ensure_dir, torch_device

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True



def softmax_conf(logits: torch.Tensor):
    probs = torch.softmax(logits, dim=-1)
    conf, idx = probs.max(dim=-1)
    return conf.item(), idx.item()



def build_auto_description(pred: Dict[str, str], conf: Dict[str, float], thr: Dict[str, float]) -> str:
    tokens: List[str] = []
    if conf.get("type", 0.0) >= thr.get("type", 0.0) and pred.get("type") not in [None, "неизвестно", "__OTHER__"]:
        tokens.append(pred["type"])
    if conf.get("material", 0.0) >= thr.get("material", 0.0) and pred.get("material") not in [None, "неизвестно", "__OTHER__"]:
        tokens.append(pred["material"])
    if conf.get("part", 0.0) >= thr.get("part", 0.0) and pred.get("part") not in [None, "неизвестно", "часть не указана", "__OTHER__"]:
        tokens.append(pred["part"])
    if conf.get("integrity", 0.0) >= thr.get("integrity", 0.0) and pred.get("integrity") == "фрагмент":
        tokens.append("фрагмент")
    return " ".join(tokens).strip() or "не удалось уверенно собрать описание"



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="artifacts/checkpoints/best.pt")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="artifacts/preds/inference.csv")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    vocabs = ckpt["vocabs"]
    idx_to_label = {field: {idx: label for label, idx in vocab.items()} for field, vocab in vocabs.items()}
    fields = cfg["fields"]

    model = SimilisMultiTaskModel(
        cfg["backbone"],
        {f: len(vocabs[f]) for f in fields},
        pretrained=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    device = torch_device()
    model.to(device)

    transform = build_transforms(
        image_size=cfg["image_size"],
        train=False,
        preprocess_mode=str(cfg.get("preprocess_mode", "pad")),
    )
    input_dir = Path(args.input_dir)
    paths = sorted(
        p
        for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    )
    if args.limit is not None:
        paths = paths[: args.limit]

    rows = []
    for path in tqdm(paths, desc="predict"):
        img = Image.open(path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(x)
        pred_labels = {}
        confidences = {}
        for field in fields:
            conf, idx = softmax_conf(outputs[field][0])
            pred_labels[field] = idx_to_label[field][idx]
            confidences[field] = conf
        auto_desc = build_auto_description(pred_labels, confidences, cfg.get("confidence_thresholds", {}))
        row = {
            "image_file": str(path.relative_to(input_dir)),
            "auto_description": auto_desc,
        }
        for field in fields:
            row[f"pred_{field}"] = pred_labels[field]
            row[f"confidence_{field}"] = round(confidences[field], 6)
        rows.append(row)

    ensure_dir(Path(args.output).parent)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
