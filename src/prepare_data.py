"""Build the Phase 1 dataset from both sources and write a stratified
train/val/test split to data/processed/<split>/<class>/.

* Zenodo 4639543: filename prefix -> class; HEIC converted; EXIF rotation fixed.
* Kaggle Alhamdan: one fruit on a plain grey background, ~15% of a 5184x3456
  frame, so each image is auto-cropped to a square around the fruit.

Stratification is on class+source, so every split holds both photo styles of a
variety that has both.
"""
import concurrent.futures as cf
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from sklearn.model_selection import train_test_split

from config import (ALHAMDAN_DIR, ALHAMDAN_MAP, CLASSES, PROCESSED_DIR, RAW_DIR, SEED, SPLIT, SPLITS_CSV,
                    STORE_MAX_SIDE, ZENODO_MAP)

register_heif_opener()
EXTS = {".jpg", ".jpeg", ".png", ".heic"}


def zenodo_class(filename):
    return ZENODO_MAP.get(re.sub(r"[\s_\-\(]*\d.*$", "", filename))


def collect():
    rows = [{"path": str(p), "label": zenodo_class(p.name), "source": "zenodo"}
            for p in sorted(RAW_DIR.iterdir()) if p.suffix.lower() in EXTS and zenodo_class(p.name)]
    for folder, label in ALHAMDAN_MAP.items():
        rows += [{"path": str(p), "label": label, "source": "alhamdan"}
                 for p in sorted((ALHAMDAN_DIR / folder).iterdir()) if p.suffix.lower() in EXTS]
    df = pd.DataFrame(rows)
    return df[df.label.isin(CLASSES)].reset_index(drop=True)


def crop_to_fruit(img, margin=0.3):
    """Square crop around the fruit on a plain background."""
    small = np.asarray(img.resize((img.width // 4, img.height // 4)))
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
    border = np.concatenate([lab[:8].reshape(-1, 3), lab[-8:].reshape(-1, 3),
                             lab[:, :8].reshape(-1, 3), lab[:, -8:].reshape(-1, 3)])
    dist = np.linalg.norm(lab - np.median(border, 0), axis=2)
    dist = cv2.GaussianBlur(np.clip(dist * 4, 0, 255).astype(np.uint8), (5, 5), 0)
    _, mask = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    H, W = mask.shape
    # Ignore blobs touching the frame edge (vignetting / table shadow), keep the biggest other one.
    inner = [i for i in range(1, n) if stats[i, 0] > 0 and stats[i, 1] > 0
             and stats[i, 0] + stats[i, 2] < W and stats[i, 1] + stats[i, 3] < H]
    if not inner:
        return img
    x, y, w, h, _ = stats[max(inner, key=lambda i: stats[i, cv2.CC_STAT_AREA])] * 4
    side = min(int(max(w, h) * (1 + 2 * margin)), img.width, img.height)
    cx, cy = x + w // 2, y + h // 2
    left = int(np.clip(cx - side // 2, 0, max(img.width - side, 0)))
    top = int(np.clip(cy - side // 2, 0, max(img.height - side, 0)))
    return img.crop((left, top, left + side, top + side))


def process(row):
    src, out, source = row
    img = Image.open(src)
    if source == "alhamdan":
        img.draft("RGB", (img.width // 2, img.height // 2))
    img = ImageOps.exif_transpose(img).convert("RGB")
    if source == "alhamdan":
        img = crop_to_fruit(img)
    img.thumbnail((STORE_MAX_SIDE, STORE_MAX_SIDE), Image.LANCZOS)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, quality=95)


def main():
    df = collect()
    too_few = df.label.value_counts().loc[lambda s: s < 20]
    if len(too_few):
        raise SystemExit(f"Too few images for {dict(too_few)}: is the Zenodo download in {RAW_DIR} complete?")
    strata = df.label + "|" + df.source
    # A class+source group too small to spread over three splits is stratified by class only.
    counts = strata.map(strata.value_counts())
    strata = strata.where(counts >= 10, df.label)
    train, rest = train_test_split(df, train_size=SPLIT["train"], stratify=strata, random_state=SEED)
    val_frac = SPLIT["val"] / (SPLIT["val"] + SPLIT["test"])
    val, test = train_test_split(rest, train_size=val_frac, stratify=strata[rest.index], random_state=SEED)
    df = pd.concat([train.assign(split="train"), val.assign(split="val"), test.assign(split="test")])
    df["out"] = [
        str(PROCESSED_DIR / r.split / r.label / f"{r.source}__{Path(r.path).stem}.jpg")
        for r in df.itertuples()
    ]

    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)
    jobs = [(r.path, Path(r.out), r.source) for r in df.itertuples()]
    with cf.ProcessPoolExecutor() as pool:
        for i, _ in enumerate(pool.map(process, jobs, chunksize=8), 1):
            if i % 200 == 0:
                print(f"{i}/{len(jobs)}", flush=True)
    df.to_csv(SPLITS_CSV, index=False)
    print(pd.crosstab([df.label, df.source], df.split, margins=True).to_string())


if __name__ == "__main__":
    main()
