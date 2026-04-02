import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_text(s: object) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    s = s.replace("ё", "е")
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_key(s: object) -> str:
    s = normalize_text(s)
    s = s.replace("м", "m")
    s = re.sub(r"\.[a-z0-9]+$", "", s)
    s = re.sub(r"[\s\-_(){}\[\],.;:]+", "", s)
    return s


def code_variants(code: object) -> List[str]:
    raw = "" if pd.isna(code) else str(code).strip()
    base = normalize_text(raw)

    variants = {
        normalize_key(raw),
        normalize_key(base),
        normalize_key(base.replace("м", "m")),
        normalize_key(base.replace("m", "м")),
        normalize_key(re.sub(r"[\s\-_()]+", "", base)),
        normalize_key(re.sub(r"[\s\-_()]+", "", base.replace("м", "m"))),
    }

    parts = re.split(r"[-_\s]+", base)
    if len(parts) >= 2:
        variants.add(normalize_key("".join(parts)))
        variants.add(normalize_key("_".join(parts)))
        variants.add(normalize_key("-".join(parts)))

    variants = [v for v in variants if v]
    variants = sorted(set(variants), key=lambda x: (len(x), x), reverse=True)
    return variants


def normalize_filename_tokens(path: Path) -> Dict[str, str]:
    stem = path.stem
    stem_norm = normalize_key(stem)

    return {
        "stem_raw": stem,
        "stem_norm": stem_norm,
        "name_norm": normalize_key(path.name),
    }


def build_image_index(images_dir: Path) -> Tuple[List[Path], Dict[str, List[Path]], List[Tuple[str, Path]]]:
    image_paths = sorted(
        [
            p
            for p in images_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
    )

    exact_index: Dict[str, List[Path]] = {}
    normalized_items: List[Tuple[str, Path]] = []

    for p in image_paths:
        toks = normalize_filename_tokens(p)
        stem_norm = toks["stem_norm"]
        exact_index.setdefault(stem_norm, []).append(p)
        normalized_items.append((stem_norm, p))

    return image_paths, exact_index, normalized_items


def choose_best_candidate(candidates: List[Path]) -> Path:
    def score(p: Path) -> Tuple[int, int, str]:
        stem = p.stem.lower()

        penalties = 0
        if re.search(r"[\(\[]\d+[\)\]]$", stem):
            penalties += 2
        if re.search(r"(copy|копия|thumb|small|preview)", stem):
            penalties += 3

        suffix_bonus = 0
        if re.search(r"(_1|-1|\s1)$", stem):
            suffix_bonus += 1

        return (penalties, -suffix_bonus, str(p))

    return sorted(candidates, key=score)[0]


def resolve_image_path_with_reason(
    code: object,
    exact_index: Dict[str, List[Path]],
    normalized_items: List[Tuple[str, Path]],
) -> Tuple[Optional[str], str]:
    variants = code_variants(code)

    for variant in variants:
        candidates = exact_index.get(variant, [])
        if len(candidates) == 1:
            return str(candidates[0]), "exact"
        if len(candidates) > 1:
            return str(choose_best_candidate(candidates)), "exact_multi"

    for variant in variants:
        candidates = [p for stem, p in normalized_items if stem.startswith(variant)]
        if len(candidates) == 1:
            return str(candidates[0]), "startswith"
        if len(candidates) > 1:
            return str(choose_best_candidate(candidates)), "startswith_multi"

    for variant in variants:
        candidates = [p for stem, p in normalized_items if variant in stem]
        if len(candidates) == 1:
            return str(candidates[0]), "contains"
        if len(candidates) > 1:
            return str(choose_best_candidate(candidates)), "contains_multi"

    return None, "missing"


def normalize_material(value: object) -> Tuple[str, int]:
    s = normalize_text(value)
    if not s:
        return "прочее", 1

    if re.search(r"фарфор", s):
        return "фарфор", 0

    if re.search(r"фаянс", s):
        return "фаянс", 0

    if re.search(r"стекл", s):
        return "стекло", 0

    if re.search(r"красноглин|белоглин|керамик|глин", s):
        return "керамика", 0

    return "прочее", 1


def normalize_integrity(row: pd.Series) -> Tuple[str, int]:
    text = " ".join(
        [
            normalize_text(row.get("fragm", "")),
            normalize_text(row.get("name", "")),
            normalize_text(row.get("description", "")),
        ]
    )

    if not text.strip():
        return "неизвестно", 1

    fragment_patterns = [
        r"фраг",
        r"облом",
        r"оскол",
        r"часть",
        r"венчик",
        r"донце",
        r"стенк",
        r"профил",
        r"горлышк",
        r"край",
        r"ручк",
    ]
    whole_patterns = [
        r"\bцел(ый|ая|ое|ые)\b",
    ]

    if any(re.search(p, text) for p in fragment_patterns):
        return "фрагмент", 0
    if any(re.search(p, text) for p in whole_patterns):
        return "целый", 0

    return "неизвестно", 1


def normalize_part(row: pd.Series) -> Tuple[str, int]:
    text = " ".join(
        [
            normalize_text(row.get("fragm", "")),
            normalize_text(row.get("name", "")),
            normalize_text(row.get("description", "")),
        ]
    )

    if not text.strip():
        return "часть не указана", 1

    if re.search(r"венчик", text):
        return "венчик", 0

    if re.search(r"донц|дно", text):
        return "донце", 0

    if re.search(r"профил", text):
        return "профиль", 0

    if re.search(r"ручк", text):
        return "ручка", 0

    if re.search(r"стенк|край|бортик|горлышк|горло", text):
        return "прочее", 0

    return "часть не указана", 1


def normalize_type(row: pd.Series) -> Tuple[str, int]:
    text = " ".join(
        [
            normalize_text(row.get("name", "")),
            normalize_text(row.get("description", "")),
        ]
    )

    if not text.strip():
        return "прочее", 1

    if re.search(r"израз", text):
        return "изразец", 0

    if re.search(r"тарел", text):
        return "тарелка", 0

    if re.search(r"блюдц", text):
        return "блюдце", 0

    if re.search(r"крышк", text):
        return "крышка", 0

    if re.search(r"миск", text):
        return "миска", 0

    return "прочее", 1


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    type_vals = df.apply(normalize_type, axis=1)
    part_vals = df.apply(normalize_part, axis=1)
    integrity_vals = df.apply(normalize_integrity, axis=1)
    material_vals = df["material"].apply(normalize_material)

    df["type"] = type_vals.apply(lambda x: x[0])
    df["type_is_missing"] = type_vals.apply(lambda x: x[1]).astype(int)

    df["part"] = part_vals.apply(lambda x: x[0])
    df["part_is_missing"] = part_vals.apply(lambda x: x[1]).astype(int)

    df["integrity"] = integrity_vals.apply(lambda x: x[0])
    df["integrity_is_missing"] = integrity_vals.apply(lambda x: x[1]).astype(int)

    df["material"] = material_vals.apply(lambda x: x[0])
    df["material_is_missing"] = material_vals.apply(lambda x: x[1]).astype(int)

    return df


def build_group_key(row: pd.Series) -> str:
    code = row.get("code", "")
    if pd.notna(code) and str(code).strip():
        return str(code).strip()
    fallback = f"{row.get('name', '')}__{row.get('description', '')}"
    return normalize_text(fallback)[:200]


def grouped_split(
    df: pd.DataFrame,
    group_col: str,
    seed: int,
    val_size: float,
    test_size: float,
) -> pd.DataFrame:
    work_df = df.copy()

    groups = work_df[group_col].astype(str).values
    idx = np.arange(len(work_df))

    gss_outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(gss_outer.split(idx, groups=groups))

    train_val_df = work_df.iloc[train_val_idx].copy()
    test_df = work_df.iloc[test_idx].copy()

    train_val_groups = train_val_df[group_col].astype(str).values
    rel_val_size = val_size / max(1e-8, (1.0 - test_size))

    gss_inner = GroupShuffleSplit(n_splits=1, test_size=rel_val_size, random_state=seed)
    inner_idx = np.arange(len(train_val_df))
    train_idx, val_idx = next(gss_inner.split(inner_idx, groups=train_val_groups))

    train_df = train_val_df.iloc[train_idx].copy()
    val_df = train_val_df.iloc[val_idx].copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    out = pd.concat([train_df, val_df, test_df], axis=0).sort_index()
    return out


def save_split_files(
    df: pd.DataFrame,
    split_paths: Dict[str, Path],
    mirror_dir: Optional[Path] = None,
) -> None:
    for path in split_paths.values():
        ensure_dir(path.parent)

    for split_name, output_path in split_paths.items():
        part = df[df["split"] == split_name].copy()
        part.to_csv(output_path, index=False)

    if mirror_dir is None:
        return

    ensure_dir(mirror_dir)
    mirror_paths = {
        "train": mirror_dir / "train.csv",
        "val": mirror_dir / "val.csv",
        "test": mirror_dir / "test_open.csv",
    }
    for split_name, output_path in mirror_paths.items():
        part = df[df["split"] == split_name].copy()
        part.to_csv(output_path, index=False)


def save_label_maps(df: pd.DataFrame, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    label_maps = {}
    for col in ["type", "part", "integrity", "material"]:
        values = sorted(df[col].dropna().astype(str).unique().tolist())
        field_to_idx = {v: i for i, v in enumerate(values)}
        idx_to_field = {i: v for v, i in field_to_idx.items()}
        label_maps[col] = {
            "field_to_idx": field_to_idx,
            "idx_to_field": idx_to_field,
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(label_maps, f, ensure_ascii=False, indent=2)


def print_debug_examples(df: pd.DataFrame) -> None:
    print(df.head())
    print(df.columns.tolist())
    cols = ["type", "part", "integrity", "material"]
    print(df[cols].head(10))


def print_match_debug(df: pd.DataFrame) -> None:
    print("\nMatch rule distribution:")
    print(df["match_rule"].value_counts(dropna=False).to_string())

    missing = df[df["has_image"] == 0][["code", "name"]].head(30)
    if len(missing) > 0:
        print("\nExamples with missing images:")
        print(missing.to_string(index=False))


def print_group_split_checks(df: pd.DataFrame) -> None:
    train_groups = set(df[df["split"] == "train"]["group_key"].astype(str))
    val_groups = set(df[df["split"] == "val"]["group_key"].astype(str))
    test_groups = set(df[df["split"] == "test"]["group_key"].astype(str))

    print("\nGroup overlap checks:")
    print("train ∩ val :", len(train_groups & val_groups))
    print("train ∩ test:", len(train_groups & test_groups))
    print("val ∩ test  :", len(val_groups & test_groups))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--images-dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    csv_path = Path(args.csv)
    images_dir = Path(args.images_dir)

    out_processed_dir = Path(cfg.get("paths", {}).get("processed_dir", "data/processed"))
    out_splits_dir = Path(cfg.get("paths", {}).get("splits_dir", "splits"))
    full_manifest_path = Path(cfg.get("full_manifest_path", out_processed_dir / "full_manifest.csv"))
    matched_manifest_path = Path(cfg.get("matched_manifest_path", out_processed_dir / "matched_manifest.csv"))
    label_maps_path = Path(cfg.get("label_maps_path", out_processed_dir / "label_maps.json"))
    prep_stats_path = Path(cfg.get("prep_stats_path", out_processed_dir / "prep_stats.json"))
    split_paths = {
        "train": Path(cfg.get("train_csv", out_processed_dir / "train.csv")),
        "val": Path(cfg.get("val_csv", out_processed_dir / "val.csv")),
        "test": Path(cfg.get("test_csv", out_processed_dir / "test_open.csv")),
    }

    ensure_dir(out_processed_dir)
    ensure_dir(out_splits_dir)
    ensure_dir(full_manifest_path.parent)
    ensure_dir(matched_manifest_path.parent)
    ensure_dir(label_maps_path.parent)
    ensure_dir(prep_stats_path.parent)

    df = pd.read_csv(csv_path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    image_paths, exact_index, normalized_items = build_image_index(images_dir)

    resolved = df["code"].apply(
        lambda x: resolve_image_path_with_reason(x, exact_index, normalized_items)
    )
    df["image_file"] = resolved.apply(lambda x: x[0])
    df["match_rule"] = resolved.apply(lambda x: x[1])
    df["has_image"] = df["image_file"].notna().astype(int)

    df["group_key"] = df.apply(build_group_key, axis=1)

    df = add_targets(df)

    df.to_csv(full_manifest_path, index=False)

    matched_df = df[df["has_image"] == 1].copy().reset_index(drop=True)

    val_size = float(cfg.get("split", {}).get("val_size", 0.15))
    test_size = float(cfg.get("split", {}).get("test_size", 0.15))

    split_df = grouped_split(
        matched_df,
        group_col="group_key",
        seed=seed,
        val_size=val_size,
        test_size=test_size,
    )

    split_df.to_csv(matched_manifest_path, index=False)

    save_split_files(split_df, split_paths=split_paths, mirror_dir=out_splits_dir)
    save_label_maps(split_df, label_maps_path)

    stats = {
        "full_rows": int(len(df)),
        "rows_with_images": int(len(matched_df)),
        "train_rows": int((split_df["split"] == "train").sum()),
        "val_rows": int((split_df["split"] == "val").sum()),
        "test_rows": int((split_df["split"] == "test").sum()),
        "image_match_rate": float(df["has_image"].mean()),
        "images_found_on_disk": int(len(image_paths)),
    }

    with open(prep_stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(stats)
    print_debug_examples(df)
    print_match_debug(df)
    print_group_split_checks(split_df)


if __name__ == "__main__":
    main()
