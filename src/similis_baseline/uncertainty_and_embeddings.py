"""Extract uncertainty signals (#12) and embeddings (#13) for all 4 pools.

For each row in train_seed / val_gold / test_gold / pool_candidate, runs a single
forward pass of `baseline_seed_best.pt` and captures:

  - logits per field → uncertainty: max_prob (1-max softmax), entropy, margin
  - backbone features (penultimate layer, before heads) → 768-d embedding
    (ConvNeXt-Tiny default)

Outputs:
  artifacts/active_learning/{pool}_uncertainty.csv
    image_file, group_key, pred_<field>, max_prob_<field>,
    entropy_<field>, margin_<field>, mean_max_prob, mean_entropy, mean_margin
  artifacts/embeddings/{pool}_embeddings.npy        — (N, D) float32
  artifacts/embeddings/{pool}_image_files.csv       — index = row order in .npy
  artifacts/embeddings/embeddings_summary.json      — shapes + checksum

The same forward pass produces both signals, saving ~2x inference time vs.
two separate scripts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import SimilisDataset
from .model import SimilisMultiTaskModel
from .utils import ensure_dir, load_config, save_json, torch_device


def compute_uncertainty(logits: torch.Tensor) -> Dict[str, torch.Tensor]:
    """logits: (B, C). Returns max_prob (1 - max softmax), entropy, margin (top1-top2)."""
    probs = torch.softmax(logits, dim=-1)
    top2 = probs.topk(min(2, probs.shape[-1]), dim=-1)
    top1 = top2.values[:, 0]
    second = top2.values[:, 1] if probs.shape[-1] >= 2 else torch.zeros_like(top1)
    least_conf = 1.0 - top1
    eps = 1e-12
    entropy = -(probs * (probs + eps).log()).sum(dim=-1)
    margin = top1 - second
    return {
        "max_prob": least_conf,
        "entropy": entropy,
        "margin": margin,
        "pred_idx": probs.argmax(dim=-1),
        "top1_prob": top1,
    }


def run_inference(
    csv_path: str,
    cfg: Dict,
    model: SimilisMultiTaskModel,
    fields: List[str],
    idx_to_label: Dict[str, Dict[int, str]],
    device: str,
    batch_size: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Returns (uncertainty_df, embeddings_matrix) in order of csv_path rows."""
    ds = SimilisDataset(
        csv_path,
        image_size=int(cfg["image_size"]),
        train=False,
        fields=fields,
        label_maps_path=cfg["label_maps_path"],
        preprocess_mode=str(cfg.get("preprocess_mode", "pad")),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_rows: List[Dict] = []
    all_embs: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            features = model.backbone(images)  # (B, D)
            all_embs.append(features.detach().cpu().numpy().astype(np.float32))

            per_field_results = {}
            for field in fields:
                logits = model.heads[model.field_to_head_name[field]](features)
                u = compute_uncertainty(logits)
                per_field_results[field] = {k: v.detach().cpu().numpy() for k, v in u.items()}

            B = images.shape[0]
            for i in range(B):
                row: Dict = {
                    "image_file": batch["metadata"]["image_file"][i],
                    "group_key": batch["metadata"]["group_key"][i],
                }
                max_probs = []
                entropies = []
                margins = []
                for field in fields:
                    pi = int(per_field_results[field]["pred_idx"][i])
                    row[f"pred_{field}"] = idx_to_label[field].get(pi, f"idx_{pi}")
                    row[f"top1_prob_{field}"] = float(per_field_results[field]["top1_prob"][i])
                    row[f"max_prob_{field}"] = float(per_field_results[field]["max_prob"][i])
                    row[f"entropy_{field}"] = float(per_field_results[field]["entropy"][i])
                    row[f"margin_{field}"] = float(per_field_results[field]["margin"][i])
                    max_probs.append(row[f"max_prob_{field}"])
                    entropies.append(row[f"entropy_{field}"])
                    margins.append(row[f"margin_{field}"])
                row["mean_max_prob"] = float(np.mean(max_probs))
                row["mean_entropy"] = float(np.mean(entropies))
                row["mean_margin"] = float(np.mean(margins))
                all_rows.append(row)

    df = pd.DataFrame(all_rows)
    embs = np.concatenate(all_embs, axis=0)
    return df, embs


def signal_correlations(df: pd.DataFrame, primary_field: str) -> Dict[str, float]:
    """Spearman corr between max_prob, entropy, -margin on primary_field."""
    from scipy.stats import spearmanr

    a = df[f"max_prob_{primary_field}"].values
    b = df[f"entropy_{primary_field}"].values
    c = -df[f"margin_{primary_field}"].values
    out = {}
    out["max_prob_vs_entropy"] = float(spearmanr(a, b).correlation)
    out["max_prob_vs_neg_margin"] = float(spearmanr(a, c).correlation)
    out["entropy_vs_neg_margin"] = float(spearmanr(b, c).correlation)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_centric.yaml")
    parser.add_argument(
        "--checkpoint",
        default="artifacts/checkpoints/data_centric/baseline_seed_best.pt",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    fields = list(cfg["fields"])
    primary_field = cfg.get("primary_field", "material")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    vocabs = ckpt["vocabs"]
    idx_to_label = {
        f: {int(idx): label for label, idx in vocabs[f].items()}
        for f in fields
    }

    device = torch_device()
    model = SimilisMultiTaskModel(
        cfg["backbone"],
        {f: len(vocabs[f]) for f in fields},
        pretrained=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    out_unc = Path(cfg["active_learning_dir"])
    ensure_dir(out_unc)
    out_emb = Path(cfg["embeddings_dir"])
    ensure_dir(out_emb)

    pool_csvs = {
        "train_seed": cfg["train_seed_csv"],
        "val_gold": cfg["val_gold_csv"],
        "test_gold": cfg["test_gold_csv"],
        "pool_candidate": cfg["pool_candidate_csv"],
    }

    summary: Dict = {
        "checkpoint": args.checkpoint,
        "primary_field": primary_field,
        "fields": fields,
        "embedding_dim": None,
        "device": device,
        "pools": {},
    }

    for pool_name, csv_path in pool_csvs.items():
        print(f"--- {pool_name} ({csv_path})")
        df, embs = run_inference(
            csv_path=csv_path,
            cfg=cfg,
            model=model,
            fields=fields,
            idx_to_label=idx_to_label,
            device=device,
            batch_size=args.batch_size,
        )

        df.to_csv(out_unc / f"{pool_name}_uncertainty.csv", index=False)
        np.save(out_emb / f"{pool_name}_embeddings.npy", embs)
        df[["image_file", "group_key"]].to_csv(
            out_emb / f"{pool_name}_image_files.csv", index=False
        )

        corrs = signal_correlations(df, primary_field)
        summary["pools"][pool_name] = {
            "rows": int(len(df)),
            "embeddings_shape": list(embs.shape),
            "spearman_correlations_primary_field": corrs,
            f"{primary_field}_max_prob_p50": float(np.median(df[f"max_prob_{primary_field}"])),
            f"{primary_field}_max_prob_p90": float(np.percentile(df[f"max_prob_{primary_field}"], 90)),
            f"{primary_field}_entropy_p50": float(np.median(df[f"entropy_{primary_field}"])),
            f"{primary_field}_entropy_p90": float(np.percentile(df[f"entropy_{primary_field}"], 90)),
        }
        if summary["embedding_dim"] is None and embs.size:
            summary["embedding_dim"] = int(embs.shape[1])

        print(f"  uncertainty CSV: {out_unc / (pool_name + '_uncertainty.csv')}")
        print(f"  embeddings    : shape={embs.shape}  → {out_emb / (pool_name + '_embeddings.npy')}")

    save_json(summary, out_emb / "embeddings_summary.json")
    print("\nDone. Summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
