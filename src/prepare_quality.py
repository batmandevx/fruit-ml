"""Build the Phase 2 datasets under data/quality/.

grade/<split>/Grade-k/     Kaggle variety/size/grade set: 3,004 single-fruit crops of 4
                           Pakistani varieties (Aseel, Fasli Toto, Gajar, Kupro). Split
                           70/15/15, stratified on variety+size+grade.
maturity/<split>/<stage>/  Immature / Khalal / Rutab / Tamar.
                           * Zenodo 10143465 (Moroccan orchards): bunch crops from
                             scripts/download_maturity.py. Each stage is one photo session per
                             variety, so the whole MATURITY_TEST_VARIETY goes to test and a
                             random split is used only for train/val.
                           * Alhamdan studio photos: the "Rutab" folder, and an equal number of
                             Tamar fruits sampled from the other varieties. Single fruits on a
                             plain background, the way a dealer photographs them.
grading_index.csv          one row per grading-set fruit: variety, size, grade, split and the
                           fruit's length/width/area in pixels (for the size model in quality.py).

Every image is padded (not cropped) to a square, so no part of the fruit is cut
off. Filenames start with "<source-or-variety>__", which evaluate.py reports
accuracy by.

Usage: python src/prepare_quality.py [--only grade|maturity]
"""
import argparse
import concurrent.futures as cf
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

from config import (ALHAMDAN_DIR, GRADING_DIR, MATURITY_SRC_DIR, MATURITY_TEST_VARIETY, QUALITY_DIR, SEED, SPLIT,
                    STORE_MAX_SIDE)
from features import segment_fruit
from prepare_data import crop_to_fruit

TAMAR_FOLDERS = ["Ajwa", "Medjool", "Sokari", "Sugaey", "Nabtat Ali", "Meneifi", "Shaishe", "Galaxy"]


def pad_square(img):
    """Pad to a square with the median border colour."""
    a = np.asarray(img)
    border = np.concatenate([a[:4].reshape(-1, 3), a[-4:].reshape(-1, 3), a[:, :4].reshape(-1, 3), a[:, -4:].reshape(-1, 3)])
    fill = tuple(int(c) for c in np.median(border, 0))
    side = max(img.size)
    out = Image.new("RGB", (side, side), fill)
    out.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    return out


def measure(img):
    """Fruit length / width / area in pixels of the original crop."""
    rgb = cv2.copyMakeBorder(np.asarray(img), 40, 40, 40, 40, cv2.BORDER_REPLICATE)
    mask = segment_fruit(rgb)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    (_, _), (a, b), _ = cv2.minAreaRect(c)
    return {"length_px": max(a, b), "width_px": min(a, b), "area_px": cv2.contourArea(c)}


def process(job):
    src, out, kind = job
    img = ImageOps.exif_transpose(Image.open(src))
    if kind == "alhamdan":
        img.draft("RGB", (img.width // 2, img.height // 2))
        img = crop_to_fruit(img.convert("RGB"))
    img = img.convert("RGB")
    geom = measure(img) if kind == "grading" else {}
    img = pad_square(img)
    img.thumbnail((STORE_MAX_SIDE, STORE_MAX_SIDE), Image.LANCZOS)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, quality=95)
    return geom


def split3(df, strata):
    train, rest = train_test_split(df, train_size=SPLIT["train"], stratify=strata, random_state=SEED)
    val_frac = SPLIT["val"] / (SPLIT["val"] + SPLIT["test"])
    val, test = train_test_split(rest, train_size=val_frac, stratify=strata[rest.index], random_state=SEED)
    return pd.concat([train.assign(split="train"), val.assign(split="val"), test.assign(split="test")])


def grading_rows():
    rows = [{"path": str(p), "variety": p.parts[-4], "size": p.parts[-3], "grade": p.parts[-2]}
            for p in sorted(GRADING_DIR.glob("*/*/*/*")) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    df = pd.DataFrame(rows)
    df = split3(df, df.variety + "|" + df["size"] + "|" + df.grade)
    df["out"] = [str(QUALITY_DIR / "grade" / r.split / r.grade /
                     f"{r.variety.replace(' ', '-')}__{r.size}__{Path(r.path).stem}.jpg") for r in df.itertuples()]
    df["kind"] = "grading"
    return df


def maturity_rows():
    crops = MATURITY_SRC_DIR / "crops"
    mor = pd.DataFrame([{"path": str(p), "stage": p.parent.name, "variety": p.name.split("__")[0]}
                        for p in sorted(crops.glob("*/*.jpg"))])
    if mor.empty:
        raise SystemExit(f"No crops in {crops}: run scripts/download_maturity.py first")
    test = mor[mor.variety == MATURITY_TEST_VARIETY].assign(split="test")
    rest = mor[mor.variety != MATURITY_TEST_VARIETY]
    val_frac = SPLIT["val"] / (SPLIT["train"] + SPLIT["val"])
    tr, va = train_test_split(rest, test_size=val_frac, stratify=rest.variety + rest.stage, random_state=SEED)
    mor = pd.concat([tr.assign(split="train"), va.assign(split="val"), test])
    mor["source"] = "morocco-" + mor.variety
    mor["kind"] = "morocco"

    rutab = sorted((ALHAMDAN_DIR / "Rutab").iterdir())
    rng = np.random.default_rng(SEED)
    per = int(np.ceil(len(rutab) / len(TAMAR_FOLDERS)))
    tamar = [p for f in TAMAR_FOLDERS for p in rng.choice(sorted((ALHAMDAN_DIR / f).iterdir()), per, replace=False)]
    alh = pd.DataFrame([{"path": str(p), "stage": "Rutab"} for p in rutab] +
                       [{"path": str(p), "stage": "Tamar"} for p in tamar])
    alh = split3(alh, alh.stage).assign(source="alhamdan", kind="alhamdan")

    df = pd.concat([mor, alh], ignore_index=True)
    df["out"] = [str(QUALITY_DIR / "maturity" / r.split / r.stage / f"{r.source}__{Path(r.path).stem}.jpg")
                 for r in df.itertuples()]
    return df


def run(df):
    jobs = [(r.path, Path(r.out), r.kind) for r in df.itertuples()]
    with cf.ProcessPoolExecutor() as pool:
        return list(pool.map(process, jobs, chunksize=16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["grade", "maturity"])
    args = ap.parse_args()
    for sub in ["grade", "maturity"]:
        if args.only in (None, sub) and (QUALITY_DIR / sub).exists():
            shutil.rmtree(QUALITY_DIR / sub)

    if args.only in (None, "grade"):
        g = grading_rows()
        g = pd.concat([g.reset_index(drop=True), pd.DataFrame(run(g))], axis=1)
        g.drop(columns=["kind"]).to_csv(QUALITY_DIR / "grading_index.csv", index=False)
        print(pd.crosstab([g.variety, g.grade], g.split, margins=True).to_string(), "\n")

    if args.only in (None, "maturity"):
        m = maturity_rows()
        run(m)
        m.drop(columns=["kind"]).to_csv(QUALITY_DIR / "maturity_index.csv", index=False)
        print(pd.crosstab([m.source, m.stage], m.split, margins=True).to_string())


if __name__ == "__main__":
    main()
