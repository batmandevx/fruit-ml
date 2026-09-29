"""Fetch a balanced sample of the Moroccan date maturity dataset (Zenodo 10143465)
and save the labelled boxes as crops to data/maturity/crops/<stage>/.

The record is ~40 GB of zips of orchard photos. Only the sampled images are
read, with HTTP range requests (remotezip), and only the crops are kept.
Each maturity stage was photographed in its own session, so images are sampled
evenly over a session (consecutive burst shots are near-duplicates).

Crop filenames start with the variety so prepare_quality.py can hold out a
whole variety for testing.

Usage: python scripts/download_maturity.py [--per-group 60]
"""
import argparse
import concurrent.futures as cf
import io
import json
import pathlib
import urllib.request
import zipfile
from collections import defaultdict

from PIL import Image, ImageOps
from remotezip import RemoteZip

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "maturity"
RECORD = "https://zenodo.org/api/records/10143465/files/{}/content"
ZIPS = ["bouisthami.zip", "kholt.zip", "Boumajhoul.zip", "Boumajhoul2.zip",
        "Boufagous.zip", "Boufagous2.zip", "Boufagous3.zip"]
STAGES = ["Immature", "Khalal", "Rutab", "Tamar"]  # YOLO class ids 0-3
MIN_BOX_AREA = 0.004  # fraction of the frame; smaller boxes are too blurry to crop
CROP_MAX_SIDE = 512


def annotations():
    ann = OUT / "ann" / "Annotations"
    if not ann.exists():
        buf = io.BytesIO(urllib.request.urlopen(RECORD.format("Annotations.zip")).read())
        zipfile.ZipFile(buf).extractall(OUT / "ann")
    return ann / "object detection" / "date fruits maturity"


def zip_index():
    cache = OUT / "zip_index.json"
    if cache.exists():
        return json.loads(cache.read_text())
    idx = {}
    for z in ZIPS:
        with RemoteZip(RECORD.format(z)) as rz:
            idx[z] = [i.filename for i in rz.infolist() if not i.is_dir()]
    cache.write_text(json.dumps(idx))
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-group", type=int, default=60, help="images per (variety, session)")
    args = ap.parse_args()

    labels = annotations()
    groups = defaultdict(list)  # (variety, session) -> [(zip, member, label_file)]
    for z, members in zip_index().items():
        for m in members:
            parts = m.split("/")
            lab = labels / (pathlib.Path(m).stem + ".txt")
            if len(parts) == 3 and lab.exists() and lab.stat().st_size:
                variety = parts[0].rstrip("2345").lower()
                groups[(variety, parts[1].upper())].append((z, m, lab))

    todo = defaultdict(list)  # zip -> sampled members
    for key, items in sorted(groups.items()):
        items.sort(key=lambda t: t[1])
        step = max(1, len(items) / args.per_group)
        picked = [items[int(i * step)] for i in range(min(args.per_group, len(items)))]
        print(f"{key}: {len(items)} labelled, sampling {len(picked)}")
        for z, m, lab in picked:
            todo[z].append((m, lab))

    # One stream per zip in parallel: a single Zenodo stream tops out around 1.5 MB/s.
    with cf.ThreadPoolExecutor(len(todo)) as pool:
        list(pool.map(lambda kv: fetch_zip(*kv), todo.items()))
    for s in STAGES:
        print(s, len(list((OUT / "crops" / s).glob("*.jpg"))))


def fetch_zip(z, items):
    for attempt in range(5):
        try:
            _fetch(z, items)
            return
        except Exception as e:  # dropped connection: reopen and skip what is done
            print(f"{z}: {e!r}, retry {attempt + 1}", flush=True)


def _fetch(z, items):
    n_crops = 0
    with RemoteZip(RECORD.format(z)) as rz:
        for i, (m, lab) in enumerate(items):
            variety = m.split("/")[0].rstrip("2345").lower()
            stem = pathlib.Path(m).stem
            if any((OUT / "crops").glob(f"*/{variety}__{stem}__*.jpg")):
                continue
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(rz.read(m)))).convert("RGB")
            W, H = img.size
            for j, line in enumerate(lab.read_text().split("\n")):
                f = line.split()
                if len(f) != 5:
                    continue
                c, cx, cy, w, h = int(f[0]), *map(float, f[1:])
                if w * h < MIN_BOX_AREA:
                    continue
                # 10% context margin around the box
                box = (max(0, (cx - w * 0.6) * W), max(0, (cy - h * 0.6) * H),
                       min(W, (cx + w * 0.6) * W), min(H, (cy + h * 0.6) * H))
                crop = img.crop(tuple(map(int, box)))
                crop.thumbnail((CROP_MAX_SIDE, CROP_MAX_SIDE), Image.LANCZOS)
                dest = OUT / "crops" / STAGES[c] / f"{variety}__{stem}__{j}.jpg"
                dest.parent.mkdir(parents=True, exist_ok=True)
                crop.save(dest, quality=92)
                n_crops += 1
            if (i + 1) % 20 == 0:
                print(f"{z}: {i + 1}/{len(items)} images, {n_crops} crops", flush=True)


if __name__ == "__main__":
    main()
