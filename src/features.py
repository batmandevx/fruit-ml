"""Hand-crafted colour / shape / texture descriptors of the fruit in an image.

The fruit is segmented with GrabCut (initialised from a centred rectangle), so
the numbers describe fruit pixels only, not the background. Shape values are in
pixels of a 512-px-long image: without a scale reference in the photo they are
relative, not physical, sizes.

Usage:
  python src/features.py <image>            # print features of one image
  python src/features.py --class-profiles   # per-class profile over the train split
"""
import argparse
import json
import sys

import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.cluster import KMeans

WORK_SIDE = 512


def load_rgb(path):
    from PIL import Image, ImageOps
    from pillow_heif import register_heif_opener

    register_heif_opener()
    img = np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))
    scale = WORK_SIDE / max(img.shape[:2])
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _plain_background_mask(rgb):
    """Fast path for studio photos: pixels far (in Lab) from the border colour.
    Returns None when the border is not a plain background."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    border = np.concatenate([lab[:6].reshape(-1, 3), lab[-6:].reshape(-1, 3),
                             lab[:, :6].reshape(-1, 3), lab[:, -6:].reshape(-1, 3)])
    if border.std(0).max() > 12:
        return None
    dist = np.linalg.norm(lab - np.median(border, 0), axis=2)
    dist = cv2.GaussianBlur(np.clip(dist * 4, 0, 255).astype(np.uint8), (5, 5), 0)
    _, mask = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def segment_fruit(rgb):
    """Binary mask (uint8 0/255) of fruit pixels."""
    fast = _plain_background_mask(rgb)
    if fast is not None:
        return _clean(fast)
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (int(w * 0.04), int(h * 0.04), int(w * 0.92), int(h * 0.92))
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    fruit = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return _clean(fruit)


def _clean(fruit):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fruit = cv2.morphologyEx(cv2.morphologyEx(fruit, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    # Keep blobs that are at least 10% of the largest one (drops specks).
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fruit)
    if n <= 1:
        return fruit
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + np.flatnonzero(areas >= 0.1 * areas.max())
    fruit = np.isin(labels, keep).astype(np.uint8) * 255
    # Fill holes (specular highlights can look like background).
    contours, _ = cv2.findContours(fruit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.drawContours(fruit, contours, -1, 255, cv2.FILLED)


def remove_background(rgb):
    """Everything outside the (slightly dilated) fruit mask set to the median background colour."""
    fruit = cv2.dilate(segment_fruit(rgb), np.ones((9, 9), np.uint8)) > 0
    if not fruit.any() or fruit.all():
        return rgb
    out = rgb.copy()
    out[~fruit] = np.median(rgb[~fruit], 0)
    return out


def color_name(h_deg, s, v):
    if v < 0.18 or (v < 0.33 and s < 0.35):
        return "black / very dark brown"
    if s < 0.18:
        return "greyish / dull"
    if h_deg < 12 or h_deg >= 340:
        return "dark red-brown" if v < 0.45 else "reddish brown"
    if h_deg < 25:
        return "dark brown" if v < 0.4 else "amber brown"
    if h_deg < 45:
        return "golden brown" if v < 0.6 else "amber / golden yellow"
    if h_deg < 70:
        return "yellow"
    return "greenish yellow"


def color_features(rgb, mask):
    px = rgb[mask > 0]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV_FULL)[mask > 0].astype(float)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[mask > 0].astype(float)
    lab[:, 0] *= 100 / 255
    lab[:, 1:] -= 128
    # circular mean for hue
    ang = hsv[:, 0] / 255 * 2 * np.pi
    hue = (np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) + 360) % 360
    sat, val = hsv[:, 1].mean() / 255, hsv[:, 2].mean() / 255

    km = KMeans(n_clusters=3, n_init=4, random_state=0).fit(px[:: max(1, len(px) // 5000)].astype(float))
    share = np.bincount(km.labels_, minlength=3) / len(km.labels_)
    order = share.argsort()[::-1]
    dominant = [
        {"hex": "#%02x%02x%02x" % tuple(int(c) for c in km.cluster_centers_[i]), "share": round(float(share[i]), 3)}
        for i in order
    ]
    # Colour uniformity: spread of Lab colour around its mean (mean ΔE76), mapped to 0-1.
    delta_e = np.linalg.norm(lab - lab.mean(0), axis=1).mean()
    return {
        "color_name": color_name(hue, sat, val),
        "mean_rgb": [round(float(c), 1) for c in px.mean(0)],
        "mean_hue_deg": round(float(hue), 1),
        "mean_saturation": round(float(sat), 3),
        "mean_brightness": round(float(val), 3),
        "mean_lab": [round(float(c), 1) for c in lab.mean(0)],
        "dominant_colors": dominant,
        "color_spread_deltaE": round(float(delta_e), 2),
        "color_uniformity": round(float(np.exp(-delta_e / 15)), 3),
    }


def shape_features(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    big = [c for c in contours if cv2.contourArea(c) > 0]
    c = max(big, key=cv2.contourArea)
    area, perim = cv2.contourArea(c), cv2.arcLength(c, True)
    hull_area = cv2.contourArea(cv2.convexHull(c))
    x, y, w, h = cv2.boundingRect(c)
    if len(c) >= 5:
        (_, _), (a1, a2), _ = cv2.fitEllipse(c)
        major, minor = max(a1, a2), min(a1, a2)
    else:
        major, minor = max(w, h), min(w, h)
    ar = major / max(minor, 1e-6)
    shape = "round" if ar < 1.25 else "oval" if ar < 1.6 else "oblong / elongated" if ar < 2.2 else "very elongated"
    return {
        "shape": shape,
        "n_fruit_regions": len([b for b in big if cv2.contourArea(b) >= 0.1 * area]),
        "area_px": int(area),
        "length_px": round(float(major), 1),
        "width_px": round(float(minor), 1),
        "aspect_ratio": round(float(ar), 3),
        "circularity": round(float(4 * np.pi * area / perim**2), 3),
        "solidity": round(float(area / hull_area), 3),
        "extent": round(float(area / (w * h)), 3),
        "eccentricity": round(float(np.sqrt(1 - (minor / major) ** 2)), 3),
    }


def texture_features(rgb, mask):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    x, y, w, h = cv2.boundingRect(mask)
    g, m = gray[y : y + h, x : x + w], mask[y : y + h, x : x + w] > 0
    # GLCM on fruit pixels only: background set to level 0 and level 0 excluded.
    q = (g // 8 + 1).astype(np.uint8)
    q[~m] = 0
    glcm = graycomatrix(q, [1, 3], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=33)[1:, 1:]
    glcm = glcm / np.maximum(glcm.sum(axis=(0, 1), keepdims=True), 1)
    props = {p: round(float(graycoprops(glcm, p).mean()), 4)
             for p in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]}

    lbp = local_binary_pattern(g, 8, 1, "uniform")[m]
    hist = np.bincount(lbp.astype(int), minlength=10) / max(lbp.size, 1)
    lbp_entropy = float(-(hist[hist > 0] * np.log2(hist[hist > 0])).sum())

    inner = cv2.erode(mask, np.ones((9, 9), np.uint8)) > 0
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 60, 150) > 0
    wrinkle = float(edges[inner].mean()) if inner.any() else 0.0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    specular = (hsv[..., 2] > 225) & (hsv[..., 1] < 60)
    gloss = float(specular[mask > 0].mean())
    level = lambda v, lo, hi: "low" if v < lo else "medium" if v < hi else "high"
    return {
        **{f"glcm_{k}": v for k, v in props.items()},
        "lbp_entropy": round(lbp_entropy, 3),
        "wrinkle_index": round(wrinkle, 4),
        "wrinkling": level(wrinkle, 0.04, 0.10),
        "gloss_index": round(gloss, 4),
        "gloss": level(gloss, 0.005, 0.02),
        "surface": "smooth" if props["homogeneity"] > 0.45 and wrinkle < 0.05 else "textured / wrinkled",
    }


def extract(path_or_rgb):
    rgb = load_rgb(path_or_rgb) if not isinstance(path_or_rgb, np.ndarray) else path_or_rgb
    mask = segment_fruit(rgb)
    if mask.sum() < 0.01 * 255 * mask.size:
        return {"error": "fruit not found"}, mask
    feats = {
        "fruit_fraction_of_image": round(float((mask > 0).mean()), 3),
        "color": color_features(rgb, mask),
        "shape": shape_features(mask),
        "texture": texture_features(rgb, mask),
    }
    return feats, mask


def flatten(feats):
    out = {}
    for group, d in feats.items():
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, (int, float)):
                    out[f"{group}.{k}"] = v
        elif isinstance(d, (int, float)):
            out[group] = d
    return out


def _profile_row(p):
    feats, _ = extract(p)
    if "error" in feats:
        return None
    return {"class": p.parent.name, "source": p.name.split("__")[0], "file": p.name, **flatten(feats),
            "color.color_name": feats["color"]["color_name"], "shape.shape": feats["shape"]["shape"]}


def class_profiles():
    from config import OUTPUT_DIR, PROCESSED_DIR

    import concurrent.futures as cf

    paths = sorted((PROCESSED_DIR / "train").glob("*/*.jpg"))
    with cf.ProcessPoolExecutor() as pool:
        rows = [r for r in pool.map(_profile_row, paths, chunksize=8) if r]
    df = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_DIR / "train_image_features.csv", index=False)
    num = df.select_dtypes("number").columns
    prof = df.groupby("class")[num].median().round(4)
    for col in ["color.color_name", "shape.shape"]:
        prof[col] = df.groupby("class")[col].agg(lambda s: s.mode().iat[0])
    prof.to_csv(OUTPUT_DIR / "class_feature_profiles.csv")
    (OUTPUT_DIR / "class_feature_profiles.json").write_text(prof.to_json(orient="index", indent=2))
    print("\n", prof.T.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?")
    ap.add_argument("--class-profiles", action="store_true")
    a = ap.parse_args()
    if a.class_profiles:
        class_profiles()
    elif a.image:
        print(json.dumps(extract(a.image)[0], indent=2))
    else:
        ap.print_help(sys.stderr)
