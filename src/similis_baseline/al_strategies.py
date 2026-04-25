"""Active learning query strategies (#15, #16, #17).

Each strategy takes pool_candidate (with hidden labels) and returns B `image_file`
entries to be revealed via `pool_candidate_oracle.csv` in #18. Selection is
deterministic for a given seed.

Strategies:
  - random            — uniform random by group_key (#15, control)
  - least_confidence  — top-B by max_prob on primary_field (#16)
  - entropy           — top-B by entropy on primary_field (#16)
  - smallest_margin   — top-B by -margin on primary_field (#16)
  - coreset           — greedy k-center on embeddings, initialized from train_seed (#17)
  - hybrid            — top-K (K=3·B) by max_prob, then coreset to B (#17)

Outputs per strategy/budget:
  artifacts/active_learning/{strategy}/B{budget}/queried.csv  — selection + scores
  artifacts/active_learning/{strategy}/B{budget}/summary.json — quality metrics

Comparison artifacts (across strategies, per budget):
  artifacts/active_learning/comparison/strategy_overlap_B{budget}.csv
  artifacts/active_learning/comparison/strategy_redundancy.csv     — avg pairwise distance
  artifacts/active_learning/comparison/strategy_nuisance.csv       — share of nuisance kadrów
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .image_analysis import foreground_ratio_bin
from .utils import ensure_dir, load_config, save_json, seed_everything


# ----------------------------- Strategies


def random_query(
    pool_df: pd.DataFrame,
    budget: int,
    seed: int,
) -> pd.DataFrame:
    """Group-aware uniform random selection. Tie-break by group_key alphabetically."""
    rng = np.random.RandomState(seed)
    group_keys = pool_df["group_key"].astype(str).tolist()
    unique_groups = sorted(set(group_keys))
    rng.shuffle(unique_groups)
    chosen_groups = unique_groups[: int(budget)]
    selected = pool_df[pool_df["group_key"].astype(str).isin(set(chosen_groups))].copy()
    # if multiple rows per group (not in this corpus), keep first by lex code
    selected = selected.sort_values("code").drop_duplicates("group_key").head(int(budget))
    selected["ranking_score"] = float("nan")
    selected["ranking_method"] = "random"
    return selected.reset_index(drop=True)


def uncertainty_query(
    pool_df: pd.DataFrame,
    budget: int,
    method: str,
    primary_field: str,
) -> pd.DataFrame:
    """Top-B by uncertainty signal on primary_field; tie-break lexicographic by group_key."""
    if method == "least_confidence":
        score_col = f"max_prob_{primary_field}"
        ascending = False
    elif method == "entropy":
        score_col = f"entropy_{primary_field}"
        ascending = False
    elif method == "smallest_margin":
        score_col = f"margin_{primary_field}"
        ascending = True  # smallest margin → most uncertain
    else:
        raise ValueError(f"unknown uncertainty method: {method}")

    if score_col not in pool_df.columns:
        raise KeyError(f"missing column {score_col}; run uncertainty_and_embeddings.py first")

    df = pool_df.copy()
    df["_score_for_sort"] = df[score_col].astype(float)
    sorted_df = df.sort_values(
        ["_score_for_sort", "group_key"],
        ascending=[ascending, True],
    )
    selected = sorted_df.drop_duplicates("group_key").head(int(budget))
    selected = selected.copy()
    selected["ranking_score"] = selected[score_col]
    selected["ranking_method"] = method
    return selected.drop(columns=["_score_for_sort"]).reset_index(drop=True)


def coreset_kcenter_greedy(
    pool_emb: np.ndarray,
    seed_emb: np.ndarray,
    budget: int,
) -> List[int]:
    """Greedy k-center over cosine distance. Init: min cosine distance of each
    pool point to all seed points (so we pick points farthest from train_seed)."""
    pool_norm = pool_emb / (np.linalg.norm(pool_emb, axis=1, keepdims=True) + 1e-12)
    seed_norm = seed_emb / (np.linalg.norm(seed_emb, axis=1, keepdims=True) + 1e-12)
    sims_to_seed = pool_norm @ seed_norm.T  # (P, S)
    min_dist = 1.0 - sims_to_seed.max(axis=1)  # (P,)

    selected: List[int] = []
    for _ in range(int(budget)):
        idx = int(np.argmax(min_dist))
        selected.append(idx)
        # mark as selected so it cannot be picked again
        min_dist[idx] = -np.inf
        # update with distance to the new selected point
        sim_to_new = pool_norm @ pool_norm[idx]
        new_dist = 1.0 - sim_to_new
        # keep min_dist of -inf for already-selected
        for s in selected:
            new_dist[s] = -np.inf
        min_dist = np.minimum(min_dist, new_dist)
    return selected


def coreset_query(
    pool_df: pd.DataFrame,
    pool_emb: np.ndarray,
    seed_emb: np.ndarray,
    budget: int,
) -> pd.DataFrame:
    indices = coreset_kcenter_greedy(pool_emb, seed_emb, budget)
    selected = pool_df.iloc[indices].copy()
    selected["ranking_score"] = np.arange(len(indices), dtype=np.float32)
    selected["ranking_method"] = "coreset"
    return selected.reset_index(drop=True)


def hybrid_query(
    pool_df: pd.DataFrame,
    pool_emb: np.ndarray,
    seed_emb: np.ndarray,
    budget: int,
    primary_field: str,
    top_k_multiplier: int = 3,
) -> pd.DataFrame:
    """Top-K most uncertain (K = top_k_multiplier · budget), then coreset to B."""
    k = min(int(top_k_multiplier) * int(budget), len(pool_df))
    top_unc = uncertainty_query(pool_df, k, "least_confidence", primary_field)
    top_unc_idx_in_pool = pool_df["image_file"].reset_index().set_index("image_file")["index"]
    top_indices = [int(top_unc_idx_in_pool[f]) for f in top_unc["image_file"]]
    top_emb = pool_emb[top_indices]

    sub_indices = coreset_kcenter_greedy(top_emb, seed_emb, budget)
    final_indices = [top_indices[i] for i in sub_indices]
    selected = pool_df.iloc[final_indices].copy()
    selected["ranking_score"] = np.arange(len(final_indices), dtype=np.float32)
    selected["ranking_method"] = f"hybrid_top{top_k_multiplier}x_coreset"
    return selected.reset_index(drop=True)


# ----------------------------- Quality metrics


def avg_pairwise_cosine_distance(emb: np.ndarray) -> float:
    if len(emb) < 2:
        return 0.0
    norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    sims = norm @ norm.T
    iu = np.triu_indices(len(emb), k=1)
    return float(1.0 - sims[iu].mean())


def selected_oracle_class_distribution(
    selected_df: pd.DataFrame,
    oracle_df: pd.DataFrame,
    primary_field: str,
) -> Dict[str, int]:
    merged = selected_df[["image_file"]].merge(
        oracle_df[["image_file", primary_field, f"{primary_field}_is_missing"]],
        on="image_file",
        how="left",
    )
    visible = merged[merged[f"{primary_field}_is_missing"] == 0]
    return {
        str(k): int(v)
        for k, v in visible[primary_field].value_counts().to_dict().items()
    }


def selected_nuisance_share(
    selected_df: pd.DataFrame,
    heur: pd.DataFrame,
) -> Dict[str, float]:
    if heur.empty:
        return {}
    merged = selected_df[["image_file"]].merge(heur, on="image_file", how="left")
    n = max(1, len(merged))
    return {
        "close_up_share": float((merged["layout_mode"] == "close_up").sum() / n),
        "multi_view_share": float((merged["layout_mode"] == "multi_view").sum() / n),
        "small_foreground_share": float((merged["foreground_ratio"] < 0.15).sum() / n),
        "has_overlay_text_share": float(merged["has_overlay_text"].fillna(0).sum() / n),
        "has_scale_bar_share": float(merged["has_scale_bar"].fillna(0).sum() / n),
        "dark_or_complex_bg_share": float((merged["bg_type"] == "dark_or_complex").sum() / n),
    }


# ----------------------------- Orchestration


def run_strategy(
    name: str,
    pool_df: pd.DataFrame,
    pool_emb: np.ndarray,
    seed_emb: np.ndarray,
    oracle_df: pd.DataFrame,
    heur: pd.DataFrame,
    budget: int,
    seed: int,
    primary_field: str,
    out_root: Path,
) -> Tuple[pd.DataFrame, Dict]:
    if name == "random":
        selected = random_query(pool_df, budget, seed)
    elif name == "least_confidence":
        selected = uncertainty_query(pool_df, budget, "least_confidence", primary_field)
    elif name == "entropy":
        selected = uncertainty_query(pool_df, budget, "entropy", primary_field)
    elif name == "smallest_margin":
        selected = uncertainty_query(pool_df, budget, "smallest_margin", primary_field)
    elif name == "coreset":
        selected = coreset_query(pool_df, pool_emb, seed_emb, budget)
    elif name == "hybrid":
        selected = hybrid_query(pool_df, pool_emb, seed_emb, budget, primary_field)
    else:
        raise ValueError(f"unknown strategy: {name}")

    out_dir = out_root / name / f"B{budget}"
    ensure_dir(out_dir)
    keep_cols = [
        "image_file",
        "group_key",
        "code",
        "ranking_score",
        "ranking_method",
    ]
    keep_cols = [c for c in keep_cols if c in selected.columns]
    selected[keep_cols].to_csv(out_dir / "queried.csv", index=False)

    pool_idx = pool_df["image_file"].reset_index().set_index("image_file")["index"]
    sel_indices = [int(pool_idx[f]) for f in selected["image_file"]]
    sel_emb = pool_emb[sel_indices]

    oracle_dist = selected_oracle_class_distribution(selected, oracle_df, primary_field)
    nuisance = selected_nuisance_share(selected, heur)
    redundancy = avg_pairwise_cosine_distance(sel_emb)

    summary = {
        "strategy": name,
        "budget": int(budget),
        "selected_count": int(len(selected)),
        "primary_field": primary_field,
        f"{primary_field}_oracle_class_distribution": oracle_dist,
        "avg_pairwise_cosine_distance": redundancy,
        "nuisance_shares": nuisance,
    }
    save_json(summary, out_dir / "summary.json")
    return selected, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_centric.yaml")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["random", "least_confidence", "coreset", "hybrid"],
    )
    parser.add_argument("--budgets", nargs="+", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    primary_field = cfg.get("primary_field", "material")
    budgets = args.budgets or list(cfg.get("active_learning", {}).get("budgets", [50, 100]))

    al_dir = Path(cfg["active_learning_dir"])
    embs_dir = Path(cfg["embeddings_dir"])
    out_root = al_dir
    ensure_dir(out_root)

    pool_unc = pd.read_csv(al_dir / "pool_candidate_uncertainty.csv")
    pool_emb = np.load(embs_dir / "pool_candidate_embeddings.npy")
    pool_idx = pd.read_csv(embs_dir / "pool_candidate_image_files.csv")
    seed_emb = np.load(embs_dir / "train_seed_embeddings.npy")

    pool_df = pool_unc.merge(
        pool_idx,
        on=["image_file", "group_key"],
        how="left",
    )
    if "code" not in pool_df.columns:
        oracle_codes = pd.read_csv(cfg["pool_candidate_oracle_csv"])[
            ["image_file", "code"]
        ]
        pool_df = pool_df.merge(oracle_codes, on="image_file", how="left")

    oracle_df = pd.read_csv(cfg["pool_candidate_oracle_csv"])

    heur_path = Path("artifacts/reports/data_report/image_heuristics.csv")
    if heur_path.exists():
        heur = pd.read_csv(heur_path)
        heur["foreground_ratio_bin"] = heur["foreground_ratio"].apply(foreground_ratio_bin)
    else:
        heur = pd.DataFrame()

    print(f"pool_candidate rows: {len(pool_df)}, primary_field: {primary_field}")
    print(f"strategies: {args.strategies}, budgets: {budgets}")

    all_summaries: List[Dict] = []
    for budget in budgets:
        for strategy in args.strategies:
            print(f"\n--- {strategy} @ B={budget}")
            _, summary = run_strategy(
                name=strategy,
                pool_df=pool_df,
                pool_emb=pool_emb,
                seed_emb=seed_emb,
                oracle_df=oracle_df,
                heur=heur,
                budget=int(budget),
                seed=seed + int(budget),
                primary_field=primary_field,
                out_root=out_root,
            )
            print(json.dumps(summary, ensure_ascii=False))
            all_summaries.append(summary)

    # comparison artifacts
    cmp_dir = out_root / "comparison"
    ensure_dir(cmp_dir)
    pd.DataFrame(
        [
            {
                "strategy": s["strategy"],
                "budget": s["budget"],
                "avg_pairwise_cosine_distance": s["avg_pairwise_cosine_distance"],
            }
            for s in all_summaries
        ]
    ).to_csv(cmp_dir / "strategy_redundancy.csv", index=False)

    nuis_rows = []
    for s in all_summaries:
        row = {"strategy": s["strategy"], "budget": s["budget"], **s["nuisance_shares"]}
        nuis_rows.append(row)
    pd.DataFrame(nuis_rows).to_csv(cmp_dir / "strategy_nuisance.csv", index=False)

    # overlap matrices per budget
    selected_per_strategy: Dict[Tuple[str, int], set] = {}
    for budget in budgets:
        for strategy in args.strategies:
            qcsv = out_root / strategy / f"B{budget}" / "queried.csv"
            sel = pd.read_csv(qcsv)
            selected_per_strategy[(strategy, budget)] = set(sel["image_file"].tolist())

    for budget in budgets:
        rows = []
        for s_a in args.strategies:
            row = {"strategy": s_a}
            for s_b in args.strategies:
                a = selected_per_strategy[(s_a, budget)]
                b = selected_per_strategy[(s_b, budget)]
                row[s_b] = int(len(a & b))
            rows.append(row)
        pd.DataFrame(rows).to_csv(cmp_dir / f"strategy_overlap_B{budget}.csv", index=False)

    save_json(
        {
            "primary_field": primary_field,
            "budgets": [int(b) for b in budgets],
            "strategies": list(args.strategies),
            "summaries": all_summaries,
        },
        cmp_dir / "all_strategies_summary.json",
    )
    print(f"\nDone. Comparison artifacts → {cmp_dir}")


if __name__ == "__main__":
    main()
