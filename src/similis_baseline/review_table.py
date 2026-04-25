"""Build review table of ≥30 suspicious examples (#14).

Combines:
  - model-side signals: disagreement (pred != norm_label), uncertainty (max_prob),
    embedding outlier (distance from class centroid in train_seed)
  - label-side signals: noise_score, flag_uncertain/conflict/incomplete from #6
  - image-side signals: layout_mode, bg_type, foreground_ratio, scale_bar, overlay

Scope: rows in {train_seed, val_gold, test_gold} where the primary field label
is visible (so disagreement can be measured). pool_candidate stays masked and
is the target of AL queries instead.

Output:
  artifacts/review/review_table.csv     — ≥30 ranked rows with reason_flag
  artifacts/review/review_top10_grid.png — visual sample of 10 cases
  artifacts/review/summary.json
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFile

from .image_analysis import foreground_ratio_bin
from .utils import ensure_dir, load_config, save_json

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def class_centroids(
    embeddings: np.ndarray,
    labels: List[str],
) -> Dict[str, np.ndarray]:
    df = pd.DataFrame({"label": labels})
    df["pos"] = np.arange(len(df))
    centroids: Dict[str, np.ndarray] = {}
    for cls, sub in df.groupby("label"):
        positions = sub["pos"].values.astype(int)
        centroids[cls] = embeddings[positions].mean(axis=0)
    return centroids


def cosine_distance_to(vec: np.ndarray, centroid: np.ndarray) -> float:
    a = vec / (np.linalg.norm(vec) + 1e-12)
    b = centroid / (np.linalg.norm(centroid) + 1e-12)
    return 1.0 - float(a @ b)


def auto_reason_flag(row: pd.Series) -> str:
    if int(row.get("flag_uncertain", 0)) == 1 or int(row.get("flag_conflict", 0)) == 1:
        return "label_noise_suspect"
    layout = str(row.get("layout_mode", ""))
    fg = float(row.get("foreground_ratio", 1.0))
    overlay = int(row.get("has_overlay_text", 0))
    bg = str(row.get("bg_type", ""))
    if layout == "close_up" or fg < 0.15 or overlay == 1 or bg == "dark_or_complex":
        return "visual_nuisance"
    return "genuinely_hard"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_centric.yaml")
    parser.add_argument("--top-n", type=int, default=40)
    parser.add_argument("--output-dir", default="artifacts/review")
    parser.add_argument(
        "--weights",
        type=str,
        default="disagreement=0.5,uncertainty=0.4,noise=0.3,embedding=0.2",
    )
    return parser.parse_args()


def parse_weights(spec: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for chunk in spec.split(","):
        k, v = chunk.split("=")
        out[k.strip()] = float(v)
    return out


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    primary_field = cfg.get("primary_field", "material")
    weights = parse_weights(args.weights)

    embs_dir = Path(cfg["embeddings_dir"])
    al_dir = Path(cfg["active_learning_dir"])
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    manifest = pd.read_csv(cfg["manifest_data_centric_csv"])
    heur_path = Path("artifacts/reports/data_report/image_heuristics.csv")
    if heur_path.exists():
        heur = pd.read_csv(heur_path)
        heur["foreground_ratio_bin"] = heur["foreground_ratio"].apply(foreground_ratio_bin)
    else:
        heur = pd.DataFrame()

    seed_emb = np.load(embs_dir / "train_seed_embeddings.npy")
    seed_idx = pd.read_csv(embs_dir / "train_seed_image_files.csv")
    seed_unc = pd.read_csv(al_dir / "train_seed_uncertainty.csv")

    seed_meta = manifest[manifest["pool"] == "train_seed"][
        ["image_file", primary_field, f"{primary_field}_is_missing"]
    ].rename(columns={primary_field: f"norm_{primary_field}"})
    seed_emb_with_label = seed_idx.merge(seed_meta, on="image_file", how="left")
    visible_seed_mask = (
        (seed_emb_with_label[f"{primary_field}_is_missing"] == 0)
        & (seed_emb_with_label[f"norm_{primary_field}"].notna())
    )
    visible_emb = seed_emb[visible_seed_mask.values]
    visible_labels = seed_emb_with_label.loc[visible_seed_mask, f"norm_{primary_field}"].astype(str).tolist()

    centroids = class_centroids(visible_emb, visible_labels)
    print(f"  centroids built for classes: {sorted(centroids.keys())}")

    rows: List[Dict] = []

    for pool_name in ["train_seed", "val_gold", "test_gold"]:
        unc = pd.read_csv(al_dir / f"{pool_name}_uncertainty.csv")
        idx_df = pd.read_csv(embs_dir / f"{pool_name}_image_files.csv")
        embs = np.load(embs_dir / f"{pool_name}_embeddings.npy")
        idx_df = idx_df.copy()
        idx_df["pos"] = np.arange(len(idx_df))

        sub_manifest = manifest[manifest["pool"] == pool_name].copy()
        merged = sub_manifest.merge(unc, on=["image_file", "group_key"], how="left")
        merged = merged.merge(
            idx_df[["image_file", "pos"]], on="image_file", how="left"
        )

        if not heur.empty:
            heur_cols = [
                "image_file",
                "layout_mode",
                "bg_type",
                "foreground_ratio",
                "foreground_ratio_bin",
                "has_scale_bar",
                "has_overlay_text",
            ]
            merged = merged.merge(heur[heur_cols], on="image_file", how="left")

        for _, r in merged.iterrows():
            mc = f"{primary_field}_is_missing"
            if mc in r and int(r[mc]) == 1:
                continue  # only consider rows with visible primary label

            norm = str(r.get(primary_field, ""))
            pred = str(r.get(f"pred_{primary_field}", ""))
            disagreement = int(norm != pred and norm not in ("", "nan"))
            top1_conf = float(r.get(f"top1_prob_{primary_field}", 1.0))
            unc_score = float(r.get(f"max_prob_{primary_field}", 0.0))
            noise_score = float(r.get("noise_score", 0.0)) if not pd.isna(r.get("noise_score", np.nan)) else 0.0

            pos = int(r["pos"])
            emb = embs[pos]
            if norm in centroids:
                emb_outlier = cosine_distance_to(emb, centroids[norm])
            else:
                emb_outlier = 0.0

            score = (
                weights.get("disagreement", 0.0) * disagreement
                + weights.get("uncertainty", 0.0) * unc_score
                + weights.get("noise", 0.0) * noise_score
                + weights.get("embedding", 0.0) * emb_outlier
            )

            row: Dict = {
                "pool": pool_name,
                "image_file": r["image_file"],
                "group_key": r.get("group_key", ""),
                "code": r.get("code", ""),
                "name": r.get("name", ""),
                "description": r.get("description", ""),
                "raw_label": r.get(f"{primary_field}_raw", r.get(primary_field, "")),
                "norm_label": norm,
                "pred_label": pred,
                "confidence": round(top1_conf, 4),
                "uncertainty_score": round(unc_score, 4),
                "noise_score": round(noise_score, 4),
                "embedding_outlier": round(emb_outlier, 4),
                "disagreement": disagreement,
                "combined_score": round(score, 4),
                "quality_flag": r.get("quality_flag", ""),
                "flag_uncertain": int(r.get("flag_uncertain", 0)),
                "flag_incomplete": int(r.get("flag_incomplete", 0)),
                "flag_conflict": int(r.get("flag_conflict", 0)),
                "flag_rare": int(r.get("flag_rare", 0)),
                "noise_reasons": r.get("noise_reasons", ""),
                "layout_mode": r.get("layout_mode", ""),
                "bg_type": r.get("bg_type", ""),
                "foreground_ratio": round(float(r.get("foreground_ratio", 0.0)), 4)
                if not pd.isna(r.get("foreground_ratio", np.nan))
                else None,
                "has_scale_bar": int(r.get("has_scale_bar", 0)) if not pd.isna(r.get("has_scale_bar", np.nan)) else 0,
                "has_overlay_text": int(r.get("has_overlay_text", 0)) if not pd.isna(r.get("has_overlay_text", np.nan)) else 0,
            }
            row["reason_flag"] = auto_reason_flag(row)
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("combined_score", ascending=False).reset_index(drop=True)
    review_top = df.head(args.top_n).copy()
    review_top.to_csv(out_dir / "review_table.csv", index=False)

    # --- visual sample of top 10
    top10 = review_top.head(10)
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    axes = axes.flatten()
    for i, (_, r) in enumerate(top10.iterrows()):
        try:
            img = Image.open(r["image_file"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (256, 256), (200, 200, 200))
        axes[i].imshow(np.asarray(img))
        title = (
            f"#{i + 1} {r['pool']} | {r['reason_flag']}\n"
            f"norm={r['norm_label']}  pred={r['pred_label']}\n"
            f"conf={r['confidence']:.2f}  unc={r['uncertainty_score']:.2f}\n"
            f"score={r['combined_score']:.3f}"
        )
        axes[i].set_title(title, fontsize=8)
        axes[i].axis("off")
    fig.suptitle(
        "Review top-10: ranked by combined disagreement + uncertainty + noise + embedding-outlier "
        f"on '{primary_field}'",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "review_top10_grid.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "primary_field": primary_field,
        "weights": weights,
        "candidate_pool_rows": int(len(df)),
        "top_n_saved": int(len(review_top)),
        "reason_flag_distribution_topN": review_top["reason_flag"].value_counts().to_dict(),
        "pool_distribution_topN": review_top["pool"].value_counts().to_dict(),
        "disagreement_count_topN": int(review_top["disagreement"].sum()),
        "label_noise_suspect_share_topN": float(
            (review_top["reason_flag"] == "label_noise_suspect").mean()
        ),
    }
    save_json(summary, out_dir / "summary.json")
    print(f"Saved → {out_dir / 'review_table.csv'} ({len(review_top)} rows)")
    print(f"Saved → {out_dir / 'review_top10_grid.png'}")
    print(f"Reason flag distribution (top {args.top_n}):", summary["reason_flag_distribution_topN"])
    print(f"Disagreements in top-{args.top_n}:", summary["disagreement_count_topN"])


if __name__ == "__main__":
    main()
