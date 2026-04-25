"""Nearest neighbors visualization (#13).

For 5 query images from pool_candidate, finds 3 nearest neighbors in train_seed
by cosine distance over backbone embeddings (768-d, ConvNeXt-Tiny). Saves a
grid PNG and a CSV of pairs for inspection. The query is intentionally diverse:
the 5 most-uncertain pool_candidate rows by max_prob_material, so the picture
shows what the model "thinks" they look like in train_seed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFile

from .utils import ensure_dir, load_config, save_json

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_centric.yaml")
    parser.add_argument(
        "--output-dir",
        default="artifacts/embeddings/nearest_neighbors",
    )
    parser.add_argument("--num-queries", type=int, default=5)
    parser.add_argument("--num-neighbors", type=int, default=3)
    args = parser.parse_args()

    cfg = load_config(args.config)
    primary_field = cfg.get("primary_field", "material")

    embs_dir = Path(cfg["embeddings_dir"])
    al_dir = Path(cfg["active_learning_dir"])
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    pool_emb = np.load(embs_dir / "pool_candidate_embeddings.npy")
    pool_idx = pd.read_csv(embs_dir / "pool_candidate_image_files.csv")
    pool_unc = pd.read_csv(al_dir / "pool_candidate_uncertainty.csv")
    seed_emb = np.load(embs_dir / "train_seed_embeddings.npy")
    seed_idx = pd.read_csv(embs_dir / "train_seed_image_files.csv")
    seed_unc = pd.read_csv(al_dir / "train_seed_uncertainty.csv")

    pool_unc = pool_unc.merge(pool_idx, on=["image_file", "group_key"], how="left")
    seed_unc = seed_unc.merge(seed_idx, on=["image_file", "group_key"], how="left")

    pool_with_pos = pool_idx.copy()
    pool_with_pos["pos"] = np.arange(len(pool_idx))
    pool_unc = pool_unc.merge(pool_with_pos, on=["image_file", "group_key"], how="left")

    # Pick the top-N most uncertain by max_prob on primary_field
    query_df = pool_unc.sort_values(
        f"max_prob_{primary_field}", ascending=False
    ).head(args.num_queries)
    query_positions = query_df["pos"].values.astype(int)

    sim = cosine_similarity(pool_emb[query_positions], seed_emb)  # (Q, N_seed)

    pairs_rows = []
    fig, axes = plt.subplots(
        args.num_queries,
        args.num_neighbors + 1,
        figsize=((args.num_neighbors + 1) * 3.0, args.num_queries * 3.0),
    )
    if args.num_queries == 1:
        axes = np.expand_dims(axes, 0)

    for qi, q_pos in enumerate(query_positions):
        q_row = pool_unc.iloc[int(np.where(pool_unc["pos"] == q_pos)[0][0])]
        q_path = q_row["image_file"]
        q_pred = q_row.get(f"pred_{primary_field}", "?")
        q_conf = q_row.get(f"top1_prob_{primary_field}", 0.0)

        try:
            img = Image.open(q_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (256, 256), (255, 255, 255))
        axes[qi, 0].imshow(np.asarray(img))
        axes[qi, 0].set_title(
            f"QUERY (pool)\n{Path(q_path).name}\npred={q_pred}, conf={q_conf:.2f}",
            fontsize=8,
        )
        axes[qi, 0].axis("off")

        nn_idx = np.argsort(-sim[qi])[: args.num_neighbors]
        for ni, j in enumerate(nn_idx):
            n_row = seed_unc.iloc[int(j)]
            n_path = n_row["image_file"]
            n_pred = n_row.get(f"pred_{primary_field}", "?")
            n_sim = float(sim[qi, j])
            try:
                ni_img = Image.open(n_path).convert("RGB")
            except Exception:
                ni_img = Image.new("RGB", (256, 256), (255, 255, 255))
            axes[qi, ni + 1].imshow(np.asarray(ni_img))
            axes[qi, ni + 1].set_title(
                f"NN{ni + 1} (seed)\n{Path(n_path).name}\npred={n_pred}, cos={n_sim:.3f}",
                fontsize=8,
            )
            axes[qi, ni + 1].axis("off")

            pairs_rows.append(
                {
                    "query_image": q_path,
                    "query_pred_field": primary_field,
                    "query_pred": q_pred,
                    "query_conf": float(q_conf),
                    "neighbor_rank": ni + 1,
                    "neighbor_image": n_path,
                    "neighbor_pred": n_pred,
                    "cosine_similarity": n_sim,
                }
            )

    fig.suptitle(
        "Nearest neighbors: 5 most uncertain pool_candidate queries → 3 NN in train_seed (cosine on 768-d ConvNeXt-Tiny features)",
        fontsize=10,
    )
    fig.tight_layout()
    out_png = out_dir / "nearest_neighbors.png"
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(pairs_rows).to_csv(out_dir / "nearest_neighbors.csv", index=False)
    save_json(
        {
            "num_queries": int(args.num_queries),
            "num_neighbors": int(args.num_neighbors),
            "embedding_dim": int(pool_emb.shape[1]),
            "primary_field": primary_field,
            "ranking_signal": f"max_prob_{primary_field}",
            "output_png": str(out_png),
        },
        out_dir / "summary.json",
    )
    print(f"Saved → {out_png}")
    print(f"Saved → {out_dir / 'nearest_neighbors.csv'}")


if __name__ == "__main__":
    main()
