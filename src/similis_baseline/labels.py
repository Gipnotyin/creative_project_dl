from __future__ import annotations

import re
from typing import Dict

import pandas as pd

from .utils import normalize_text


TYPE_MAP = {
    "изразец": "изразец",
    "тарелка": "тарелка",
    "тарелка/блюдо": "тарелка",
    "блюдце": "блюдце",
    "миска": "миска",
    "миска (?)": "миска",
    "крышка": "крышка",
    "плитка": "плитка",
    "игрушка": "игрушка",
    "игрушка елочная": "игрушка",
}

MATERIAL_RULES = [
    (r"фарфор|фарфоров", "фарфор"),
    (r"фаянс|фаянсов", "фаянс"),
    (r"белоглинян", "белоглиняная керамика"),
    (r"красноглинян", "красноглиняная керамика"),
    (r"светлоглинян", "светлоглиняная керамика"),
    (r"каменная масса", "каменная масса"),
    (r"стекл", "стекло"),
    (r"глина", "глина"),
    (r"дерев", "дерево"),
    (r"сланец", "сланец"),
    (r"песчаник", "песчаник"),
    (r"камень", "камень"),
    (r"керамик", "керамика"),
]

PART_RULES = [
    (r"венчик|венчика|венчика фр", "венчик"),
    (r"донце|донца|придонн", "донце"),
    (r"стенка|стенки", "стенка"),
    (r"профиль|профиля", "профиль"),
    (r"горлышк|горло", "горлышко"),
    (r"ручк", "ручка"),
    (r"край|края", "край"),
]

INTEGRITY_RULES = [
    (r"фрагмент|фр\.?-?т|обломок", "фрагмент"),
    (r"целый", "целый"),
]



def normalize_type(value: str) -> str:
    value = normalize_text(value)
    return TYPE_MAP.get(value, value if value else "неизвестно")



def normalize_material(value: str, fallback_text: str = "") -> str:
    value = normalize_text(value)
    text = f"{value} {normalize_text(fallback_text)}".strip()
    for pattern, label in MATERIAL_RULES:
        if re.search(pattern, text):
            return label
    return "неизвестно"



def extract_part(text: str) -> str:
    text = normalize_text(text)
    for pattern, label in PART_RULES:
        if re.search(pattern, text):
            return label
    return "часть не указана"



def extract_integrity(fragm: str, description: str = "") -> str:
    text = f"{normalize_text(fragm)} {normalize_text(description)}".strip()
    for pattern, label in INTEGRITY_RULES:
        if re.search(pattern, text):
            return label
    return "неизвестно"



def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["type"] = out["name"].map(normalize_type)
    out["part"] = (out["name"].fillna("") + " " + out["description"].fillna("") + " " + out["fragm"].fillna("")).map(extract_part)
    out["integrity"] = [extract_integrity(f, d) for f, d in zip(out["fragm"].fillna(""), out["description"].fillna(""))]
    out["material"] = [normalize_material(m, d) for m, d in zip(out["material"].fillna(""), out["description"].fillna(""))]

    for col in ["type", "part", "integrity", "material"]:
        out[f"{col}_is_missing"] = out[col].isin(["", "неизвестно", "часть не указана"]).astype(int)

    return out



def build_vocab(df: pd.DataFrame, field: str, rare_class_min_count: int = 1) -> Dict[str, int]:
    counts = df[field].fillna("неизвестно").value_counts()
    labels = []
    for label, cnt in counts.items():
        if cnt >= rare_class_min_count:
            labels.append(label)
    if "__OTHER__" not in labels:
        labels.append("__OTHER__")
    if "неизвестно" not in labels:
        labels.append("неизвестно")
    labels = sorted(set(labels))
    return {label: idx for idx, label in enumerate(labels)}



def apply_vocab(value: str, vocab: Dict[str, int]) -> int:
    value = value if value in vocab else "__OTHER__"
    return vocab[value]
