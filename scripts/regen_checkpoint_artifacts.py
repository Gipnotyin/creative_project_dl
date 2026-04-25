"""Regenerate model_summary.json and checkpoint_roundtrip.json for current best.pt.

These files are referenced from README/REPORT but are not produced by any
pipeline script. This is a one-shot sanity helper: load the checkpoint, count
parameters, and verify reload->same prediction on one fixed image.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

from src.similis_baseline.dataset import build_transforms
from src.similis_baseline.model import SimilisMultiTaskModel
from src.similis_baseline.utils import torch_device

ROOT = Path(__file__).resolve().parent.parent


def load_model(checkpoint_path: Path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt["config"]
    vocabs = ckpt["vocabs"]
    fields = list(cfg["fields"])
    model = SimilisMultiTaskModel(
        cfg["backbone"],
        {f: len(vocabs[f]) for f in fields},
        pretrained=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt, cfg


def count_params(model) -> dict:
    backbone = sum(p.numel() for p in model.backbone.parameters())
    heads = sum(p.numel() for p in model.heads.parameters())
    return {
        "total_params": int(backbone + heads),
        "backbone_params": int(backbone),
        "heads_params": int(heads),
        "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }


def first_image_tensor(cfg) -> torch.Tensor:
    import pandas as pd

    df = pd.read_csv(cfg["val_csv"])
    image_path = str(df.iloc[0]["image_file"])
    transform = build_transforms(
        image_size=int(cfg["image_size"]),
        train=False,
        preprocess_mode=str(cfg.get("preprocess_mode", "pad")),
    )
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0), image_path


def main() -> None:
    checkpoint_path = ROOT / "artifacts/checkpoints/best.pt"
    model_a, ckpt, cfg = load_model(checkpoint_path)

    summary = {
        "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
        "backbone": cfg["backbone"],
        "image_size": int(cfg["image_size"]),
        "fields": list(cfg["fields"]),
        "epoch": int(ckpt.get("epoch", -1)),
        "best_metric": float(ckpt.get("best_metric", 0.0)),
        **count_params(model_a),
    }
    summary_path = ROOT / "artifacts/reports/model_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")
    print(summary)

    device = torch_device()
    image_tensor, image_path = first_image_tensor(cfg)
    image_tensor = image_tensor.to(device)
    model_a.to(device)
    with torch.no_grad():
        out_a = {f: v.detach().cpu().numpy().tolist() for f, v in model_a(image_tensor).items()}

    model_b, _, _ = load_model(checkpoint_path)
    model_b.to(device)
    with torch.no_grad():
        out_b = {f: v.detach().cpu().numpy().tolist() for f, v in model_b(image_tensor).items()}

    max_diff = {
        f: float(max(abs(a - b) for a, b in zip(out_a[f][0], out_b[f][0])))
        for f in cfg["fields"]
    }
    roundtrip = {
        "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
        "test_image": image_path,
        "device": device,
        "max_logit_diff_per_field": max_diff,
        "all_fields_match": all(d < 1e-6 for d in max_diff.values()),
    }
    rt_path = ROOT / "artifacts/reports/checkpoint_roundtrip.json"
    rt_path.write_text(json.dumps(roundtrip, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {rt_path}")
    print(roundtrip)


if __name__ == "__main__":
    main()
