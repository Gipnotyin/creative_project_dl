from __future__ import annotations

import argparse
import base64
import os
from io import BytesIO
from pathlib import Path
from typing import Dict, List

import pandas as pd
from PIL import Image


FIELD_LABELS = {
    "pred_type": "Тип",
    "pred_material": "Материал",
    "pred_part": "Часть / зона",
    "pred_integrity": "Целостность",
}

CONF_LABELS = {
    "pred_type": "confidence_type",
    "pred_material": "confidence_material",
    "pred_part": "confidence_part",
    "pred_integrity": "confidence_integrity",
}


def parse_scalar(value: str):
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null", "nan"}:
        return None
    try:
        if "." in lowered:
            return float(lowered)
        return int(lowered)
    except Exception:
        return value


def apply_filter(df: pd.DataFrame, filter_col: str, filter_value: str, filter_mode: str) -> pd.DataFrame:
    if filter_col not in df.columns:
        raise ValueError(f"Unknown --filter-col: {filter_col}")

    series = df[filter_col]
    parsed_value = parse_scalar(filter_value)

    if filter_mode == "eq":
        if parsed_value is None:
            return df[series.isna()].copy()
        return df[series.astype(str) == str(parsed_value)].copy()
    if filter_mode == "contains":
        return df[series.astype(str).str.contains(str(filter_value), case=False, na=False)].copy()

    numeric = pd.to_numeric(series, errors="coerce")
    compare_value = float(parsed_value)
    if filter_mode == "gt":
        return df[numeric > compare_value].copy()
    if filter_mode == "ge":
        return df[numeric >= compare_value].copy()
    if filter_mode == "lt":
        return df[numeric < compare_value].copy()
    if filter_mode == "le":
        return df[numeric <= compare_value].copy()

    raise ValueError(f"Unsupported filter mode: {filter_mode}")


def resolve_image_path(image_path: str, images_root: str | None = None) -> Path:
    path = Path(image_path)
    if path.exists():
        return path

    if images_root is not None:
        rooted = Path(images_root) / path
        if rooted.exists():
            return rooted

    raise FileNotFoundError(f"Cannot resolve image path: {image_path}")


def image_to_base64(image_path: str, max_size: int = 900, images_root: str | None = None) -> str:
    resolved_path = resolve_image_path(image_path, images_root=images_root)
    img = Image.open(resolved_path).convert("RGB")
    img.thumbnail((max_size, max_size))

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def image_to_link(
    image_path: str,
    out_html_path: str,
    images_root: str | None = None,
) -> str:
    resolved_path = resolve_image_path(image_path, images_root=images_root).resolve()
    out_dir = Path(out_html_path).resolve().parent
    try:
        return os.path.relpath(resolved_path, out_dir)
    except Exception:
        return str(resolved_path)


def conf_badge(value: float | None) -> str:
    if value is None:
        return ""
    if value >= 0.75:
        return "✅"
    if value >= 0.45:
        return "🟡"
    return "⚪"


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def build_items(row: pd.Series) -> List[Dict]:
    items = []
    for pred_col, label in FIELD_LABELS.items():
        if pred_col not in row:
            continue

        value = row[pred_col]
        if pd.isna(value) or str(value).strip() == "":
            continue

        conf_col = CONF_LABELS.get(pred_col)
        conf_val = safe_float(row[conf_col]) if conf_col in row else None

        items.append(
            {
                "label": label,
                "value": str(value),
                "badge": conf_badge(conf_val),
                "confidence": conf_val,
            }
        )
    return items


def render_html(
    df: pd.DataFrame,
    title: str = "SIMILIS preview",
    images_root: str | None = None,
    out_html_path: str | None = None,
    image_mode: str = "auto",
) -> str:
    cards = []
    resolved_image_mode = image_mode
    if resolved_image_mode == "auto":
        resolved_image_mode = "embed" if len(df) <= 50 else "link"

    for i, row in df.iterrows():
        if resolved_image_mode == "embed":
            img_src = image_to_base64(row["image_file"], images_root=images_root)
        elif resolved_image_mode == "link":
            if out_html_path is None:
                raise ValueError("out_html_path is required when image_mode='link'")
            img_src = image_to_link(
                row["image_file"],
                out_html_path=out_html_path,
                images_root=images_root,
            )
        else:
            raise ValueError(f"Unsupported image_mode: {resolved_image_mode}")
        auto_description = row.get("auto_description", "")
        items = build_items(row)

        li_html = []
        for item in items:
            conf_text = ""
            if item["confidence"] is not None:
                conf_text = f' <span class="conf">({item["confidence"]:.2f})</span>'

            li_html.append(
                f"<li><b>{item['label']}:</b> {item['value']} {item['badge']}{conf_text}</li>"
            )

        li_block = "\n".join(li_html)

        cards.append(
            f"""
            <section class="card">
              <h2>Пример {i + 1}</h2>
              <div class="img-wrap">
                <img src="{img_src}" alt="artifact_{i+1}" loading="lazy" decoding="async">
              </div>
              <p class="desc"><b>Описание:</b> {auto_description}</p>
              <ul>
                {li_block}
              </ul>
            </section>
            """
        )

    cards_html = "\n".join(cards)

    html = f"""
    <!doctype html>
    <html lang="ru">
    <head>
      <meta charset="utf-8">
      <title>{title}</title>
      <style>
        body {{
          font-family: Arial, Helvetica, sans-serif;
          margin: 24px;
          background: #f5f5f5;
          color: #222;
        }}
        h1 {{
          font-size: 42px;
          margin-bottom: 32px;
        }}
        .card {{
          background: white;
          border-radius: 14px;
          padding: 24px;
          margin-bottom: 28px;
          box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }}
        .card h2 {{
          font-size: 30px;
          margin-top: 0;
          margin-bottom: 20px;
        }}
        .img-wrap {{
          margin: 12px 0 20px 0;
        }}
        .img-wrap img {{
          max-width: 760px;
          max-height: 620px;
          width: auto;
          height: auto;
          display: block;
          background: #fafafa;
        }}
        .desc {{
          font-size: 22px;
          line-height: 1.45;
          margin: 18px 0;
        }}
        ul {{
          font-size: 22px;
          line-height: 1.6;
          padding-left: 28px;
        }}
        .conf {{
          color: #666;
          font-size: 18px;
        }}
      </style>
    </head>
    <body>
      <h1>{title}</h1>
      {cards_html}
    </body>
    </html>
    """
    return html


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-csv", required=True, type=str)
    parser.add_argument("--out-html", required=True, type=str)
    parser.add_argument("--images-root", type=str, default=None)
    parser.add_argument("--title", type=str, default="SIMILIS — примеры предсказаний")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sort-by", type=str, default=None)
    parser.add_argument("--sort-desc", action="store_true")
    parser.add_argument("--filter-col", type=str, default=None)
    parser.add_argument("--filter-value", type=str, default=None)
    parser.add_argument(
        "--filter-mode",
        choices=["eq", "contains", "gt", "ge", "lt", "le"],
        default="eq",
    )
    parser.add_argument(
        "--image-mode",
        choices=["auto", "embed", "link"],
        default="auto",
        help="embed = inline base64, link = use file paths, auto = embed for small reports and link for large ones",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.pred_csv)
    if (args.filter_col is None) != (args.filter_value is None):
        raise ValueError("--filter-col and --filter-value must be passed together")
    if args.filter_col is not None and args.filter_value is not None:
        df = apply_filter(df, args.filter_col, args.filter_value, args.filter_mode)
    if args.sort_by is not None:
        if args.sort_by not in df.columns:
            raise ValueError(f"Unknown --sort-by column: {args.sort_by}")
        df = df.sort_values(args.sort_by, ascending=not args.sort_desc, kind="stable")
    if args.offset < 0:
        raise ValueError("--offset must be >= 0")
    df = df.iloc[args.offset : args.offset + args.limit].copy()

    html = render_html(
        df,
        title=args.title,
        images_root=args.images_root,
        out_html_path=args.out_html,
        image_mode=args.image_mode,
    )

    out_path = Path(args.out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
