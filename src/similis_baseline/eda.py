from __future__ import annotations

import argparse
from pathlib import Path

import warnings
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageFile

from .utils import ensure_dir

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

def image_size_stats(paths):
    rows = []
    for p in paths[:200]:
        try:
            with Image.open(p) as img:
                img.draft("RGB", (4096, 4096))
                w, h = img.size
            rows.append({"path": str(p), "width": w, "height": h, "aspect": w / max(h, 1)})
        except Exception:
            continue
    return pd.DataFrame(rows)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default="data/processed/full_manifest.csv")
    parser.add_argument("--outdir", type=str, default="artifacts/figures")
    args = parser.parse_args()

    ensure_dir(args.outdir)
    df = pd.read_csv(args.manifest)
    print(df.head())
    print(df.columns.tolist())
    print(df[["type", "part", "integrity", "material"]].head(10))

    for col in ["name", "material", "fragm", "type", "part", "integrity"]:
        vc = df[col].value_counts(dropna=False).head(20)
        plt.figure(figsize=(10, 4))
        vc.plot(kind="bar")
        plt.title(col)
        plt.tight_layout()
        plt.savefig(Path(args.outdir) / f"dist_{col}.png")
        plt.close()

    image_paths = [Path(p) for p in df["image_file"].dropna().tolist()]
    stats = image_size_stats(image_paths)
    if not stats.empty:
        stats.to_csv(Path(args.outdir) / "image_size_stats.csv", index=False)
        plt.figure(figsize=(6, 4))
        plt.hist(stats["aspect"], bins=30)
        plt.title("Aspect ratio")
        plt.tight_layout()
        plt.savefig(Path(args.outdir) / "aspect_ratio_hist.png")
        plt.close()


if __name__ == "__main__":
    main()
