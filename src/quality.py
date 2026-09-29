"""Phase 2 per-fruit quality estimators that need no training: scale, size, colour
uniformity, surface defects, skin condition and weight. The two learned
estimators (quality grade, maturity stage) are CNNs trained with train.py;
grade.py puts everything together.

The defect and skin detectors are colour/texture rules, not trained models: no
public dataset labels date defects by type. They are checked indirectly
(`--check-defects`: defect area should rise from Grade-1 to Grade-3 in the
grading set) and should be tuned on dealer photos with labelled defects.

Usage:
  python src/quality.py <image>                  # quality features of the largest fruit
  python src/quality.py --fit-size               # fit + test size thresholds on the grading set
  python src/quality.py --check-defects          # defect/skin statistics by grade (sanity check)
  python src/quality.py --calibrate-weight w.csv # csv: image,weight_g[,mm_per_px] -> fitted WEIGHT_K
"""
import argparse
import json
import os
import sys

import cv2
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

import config
from features import color_features, load_rgb, segment_fruit, texture_features

SIZE_MODEL = config.OUTPUT_DIR / "quality" / "size_model.json"  # {"mm": {variety: {"metric", "cuts"}}}
SIZE_FOREST = config.OUTPUT_DIR / "quality" / "size_forest.joblib"
SIZE_FEATURES = ["sqrt_area_px", "length_px", "width_px"]
SIZES = ["Small", "Medium", "Large"]
# Fallback length thresholds (mm) when a variety has no fitted model: typical trade
# ranges for mid-sized dates. Replace with the dealer's own grading rules.
DEFAULT_SIZE_MM = {"metric": "length_mm", "cuts": [30.0, 38.0]}


# ---------------------------------------------------------------- scale & instances
def find_scale(rgb):
    """(mm per pixel, marker corners) from a printed ArUco marker (config.ARUCO_*), or (None, None)."""
    aruco = cv2.aruco
    det = aruco.ArucoDetector(aruco.getPredefinedDictionary(getattr(aruco, config.ARUCO_DICT)))
    corners, ids, _ = det.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    if ids is None:
        return None, None
    c = corners[0][0]
    side_px = np.mean([np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)])
    return float(config.ARUCO_SIDE_MM / side_px), c


def hide_marker(rgb, corners):
    """Paint the marker (and its white quiet zone) with the background colour so it is
    not segmented as a fruit."""
    if corners is None:
        return rgb
    centre = corners.mean(0)
    poly = np.int32(centre + (corners - centre) * 1.6)
    out = rgb.copy()
    border = np.concatenate([rgb[:6].reshape(-1, 3), rgb[-6:].reshape(-1, 3)])
    cv2.fillConvexPoly(out, poly, tuple(int(v) for v in np.median(border, 0)))
    return out


def _neck_groups(dist, peaks, neck_ratio=0.6):
    """Group distance-transform peaks that belong to the same fruit. Two peaks are
    separate fruits only if the fruit's half-width (the distance value) drops below
    neck_ratio x the smaller peak somewhere on the straight line between them.
    A single elongated date has a flat ridge, so its peaks stay together."""
    parent = list(range(len(peaks)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            (r0, c0), (r1, c1) = peaks[i], peaks[j]
            n = int(max(abs(r1 - r0), abs(c1 - c0))) + 1
            line = dist[np.linspace(r0, r1, n).round().astype(int), np.linspace(c0, c1, n).round().astype(int)]
            if line.min() >= neck_ratio * min(dist[r0, c0], dist[r1, c1]):
                parent[find(i)] = find(j)
    roots = sorted({find(i) for i in range(len(peaks))})
    return [roots.index(find(i)) + 1 for i in range(len(peaks))]


def fruit_instances(rgb, min_frac=0.15):
    """One mask per fruit. Touching fruits are split with a watershed on the distance
    transform, seeded with one marker per fruit (peaks separated by a neck)."""
    mask = segment_fruit(rgb)
    n, labels = cv2.connectedComponents(mask)
    masks = []
    for i in range(1, n):
        comp = (labels == i).astype(np.uint8) * 255
        dist = ndi.distance_transform_edt(comp)
        peaks = peak_local_max(dist, min_distance=max(3, int(dist.max() * 0.5)), threshold_abs=dist.max() * 0.4,
                               labels=comp > 0, exclude_border=False)
        groups = _neck_groups(dist, peaks) if len(peaks) > 1 else [1]
        if max(groups) == 1:
            masks.append(comp)
            continue
        markers = np.zeros(comp.shape, np.int32)
        for (r, c), g in zip(peaks, groups):
            markers[r, c] = g
        ws = watershed(-dist, markers, mask=comp > 0)
        masks += [(ws == j).astype(np.uint8) * 255 for j in range(1, ws.max() + 1)]
    if not masks:
        return []
    areas = [m.sum() for m in masks]
    masks = [m for m, a in zip(masks, areas) if a >= min_frac * max(areas)]
    return sorted(masks, key=lambda m: -m.sum())


def fruit_closeup(full_rgb, mask, px_scale, margin=0.12):
    """High-resolution view of one fruit for colour/defect/skin analysis: crop the
    original image around the fruit, resize to features.WORK_SIDE and re-segment."""
    from features import WORK_SIDE

    x, y, w, h = cv2.boundingRect(mask)
    m = int(max(w, h) * margin)
    x0, y0 = max(0, int((x - m) / px_scale)), max(0, int((y - m) / px_scale))
    x1 = min(full_rgb.shape[1], int((x + w + m) / px_scale))
    y1 = min(full_rgb.shape[0], int((y + h + m) / px_scale))
    crop = full_rgb[y0:y1, x0:x1]
    s = min(1.0, WORK_SIDE / max(crop.shape[:2]))
    crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else crop.copy()
    # keep only this fruit: other fruits inside the box are masked by the overview mask
    own = cv2.resize(mask[max(0, y - m):y + h + m, max(0, x - m):x + w + m], crop.shape[1::-1],
                     interpolation=cv2.INTER_NEAREST)
    own = cv2.dilate(own, np.ones((9, 9), np.uint8))
    close = segment_fruit(cv2.copyMakeBorder(crop, 20, 20, 20, 20, cv2.BORDER_REPLICATE))[20:-20, 20:-20]
    return crop, cv2.bitwise_and(close, own)


# ---------------------------------------------------------------- size & weight
def geometry(mask, mm_per_px=None, px_scale=1.0):
    """Length/width/area of the fruit. px_scale = working-image px per original-image px
    (features.load_rgb resizes to 512 px): pixel values are reported in original pixels,
    which is what the size model was fitted on. mm_per_px is per working-image pixel."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    (_, _), (a, b), _ = cv2.minAreaRect(c)
    L, W, area = max(a, b), min(a, b), cv2.contourArea(c)
    g = {"length_px": round(L / px_scale, 1), "width_px": round(W / px_scale, 1), "area_px": int(area / px_scale**2)}
    if mm_per_px:
        g.update(length_mm=round(L * mm_per_px, 1), width_mm=round(W * mm_per_px, 1),
                 area_mm2=round(area * mm_per_px**2, 1))
    return g


def size_class(geom, variety=None):
    """Small/Medium/Large. With a mm scale: the variety's mm cut points (DEFAULT_SIZE_MM
    until the dealer sets their own). Without: the variety's random forest fitted on
    the grading set, valid only for photos from that camera rig."""
    if "length_mm" in geom:
        cuts = json.loads(SIZE_MODEL.read_text()).get("mm", {}) if SIZE_MODEL.exists() else {}
        m = cuts.get(variety, DEFAULT_SIZE_MM)
        k = int(np.searchsorted(m["cuts"], geom[m["metric"]]))
        return {"size_class": SIZES[min(k, 2)], "method": f"{m['metric']} cut points", "cuts": m["cuts"]}
    forests = joblib.load(SIZE_FOREST) if SIZE_FOREST.exists() else {}
    if variety not in forests:
        return {"size_class": None, "note": "no scale reference and no pixel model for this variety"}
    f = forests[variety]
    if isinstance(f, str):  # variety with a single size class in the grading set
        return {"size_class": f, "method": "only class seen for this variety"}
    x = pd.DataFrame([[np.sqrt(geom["area_px"]), geom["length_px"], geom["width_px"]]], columns=SIZE_FEATURES)
    probs = f.predict_proba(x)[0]
    return {"size_class": str(f.classes_[probs.argmax()]), "confidence": round(float(probs.max()), 3),
            "method": "random forest on sqrt(area), length, width (grading-rig pixels)"}


def estimate_weight(geom, k=config.WEIGHT_K):
    """Ellipsoid-of-revolution weight: k * L * W^2 (mm -> g). Needs a scale."""
    if "length_mm" not in geom:
        return None
    return round(k * geom["length_mm"] * geom["width_mm"] ** 2, 2)


def fit_size():
    """Per-variety random forest on sqrt(area), length and width in pixels, trained on
    the grading set's train split and scored on its test split. (Chosen on the val
    split over logistic regression, gradient boosting and area-only cut points.)"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix

    df = pd.read_csv(config.QUALITY_DIR / "grading_index.csv")
    df["sqrt_area_px"] = np.sqrt(df.area_px)
    forests, report = {}, {}
    for variety, g in df.groupby("variety"):
        tr, te = g[g.split == "train"], g[g.split == "test"]
        if g["size"].nunique() == 1:
            forests[variety] = str(g["size"].iat[0])
            continue
        f = RandomForestClassifier(300, min_samples_leaf=5, random_state=config.SEED).fit(tr[SIZE_FEATURES], tr["size"])
        forests[variety] = f
        pred = f.predict(te[SIZE_FEATURES])
        labels = [s for s in SIZES if s in f.classes_]
        report[variety] = {"test_accuracy": round(float((pred == te["size"]).mean()), 3), "n_test": len(te),
                           "labels": labels, "confusion": confusion_matrix(te["size"], pred, labels=labels).tolist()}
    n = sum(r["n_test"] for r in report.values())
    report["overall_test_accuracy"] = round(sum(r["test_accuracy"] * r["n_test"] for r in report.values()) / n, 3)
    SIZE_FOREST.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(forests, SIZE_FOREST)
    if not SIZE_MODEL.exists():
        SIZE_MODEL.write_text(json.dumps({"mm": {}}, indent=2))
    (SIZE_MODEL.parent / "size_model_test.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


# ---------------------------------------------------------------- colour uniformity
def color_uniformity(rgb, mask):
    c = color_features(rgb, mask)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[mask > 0].astype(float)
    lab[:, 0] *= 100 / 255
    lab[:, 1:] -= 128
    de = np.linalg.norm(lab - np.median(lab, 0), axis=1)
    within = float((de < 10).mean())  # share of the skin within ΔE 10 of its median colour
    score = c["color_uniformity"]
    return {
        "uniformity_score": score,  # exp(-mean ΔE / 15): 1 = perfectly even colour
        "mean_deltaE_from_mean": c["color_spread_deltaE"],
        "share_within_deltaE10": round(within, 3),
        "lightness_std": round(float(lab[:, 0].std()), 2),
        "level": "uniform" if score >= 0.55 else "moderately uniform" if score >= 0.4 else "non-uniform",
        "color_name": c["color_name"],
        "dominant_colors": c["dominant_colors"],
    }


# ---------------------------------------------------------------- defects & skin
def _lab(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, a, b = lab[..., 0] * 100 / 255, lab[..., 1] - 128, lab[..., 2] - 128
    return L, a, b, np.hypot(a, b)


def _local_median(ch, mask, k):
    """Median of the fruit's own pixels around each point (background filled first)."""
    filled = np.where(mask, ch, np.median(ch[mask])).astype(np.float32)
    lo, hi = filled.min(), filled.max()
    u8 = np.uint8(255 * (filled - lo) / max(hi - lo, 1e-6))
    return cv2.medianBlur(u8, k).astype(np.float32) / 255 * (hi - lo) + lo


def _blobs(bin_mask, fruit_area, min_frac, max_frac=1.0):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bin_mask.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if not (min_frac * fruit_area <= area <= max_frac * fruit_area):
            continue
        comp = (lab == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = cnts[0]
        perim = max(cv2.arcLength(c, True), 1)
        (_, _), (w, h), _ = cv2.minAreaRect(c)
        out.append({"mask": comp > 0, "area": int(area), "circularity": 4 * np.pi * area / perim**2,
                    "elongation": max(w, h) / max(min(w, h), 1), "length": max(w, h),
                    "bbox": [int(v) for v in stats[i, :4]]})
    return out


def surface_defects(rgb, mask):
    """Rule-based candidates for cracks, black spots, sunburn, mold and insect holes.
    Thresholds are relative to the fruit's own colour so dark and light varieties
    are treated alike. Returns (summary dict, {defect: bool mask})."""
    fruit = cv2.erode(mask, np.ones((7, 7), np.uint8)) > 0  # skip the rim (shadows / background bleed)
    A = int(fruit.sum())
    if A < 500:
        return {"error": "fruit too small"}, {}
    L, a, b, C = _lab(rgb)
    side = max(rgb.shape[:2])
    k = (int(side * 0.08) | 1)  # neighbourhood ~8% of the image
    Lm, Cm = _local_median(L, fruit, k), _local_median(C, fruit, k)
    Lmed, Cmed = float(np.median(L[fruit])), float(np.median(C[fruit]))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    V = hsv[..., 2].astype(np.float32)
    Vf = V[fruit]
    # Highlights relative to the fruit's own brightness: on a dark glossy date a
    # reflection is grey, not white, and would otherwise read as mold.
    hl_thr = max(np.median(Vf) + 3 * (np.percentile(Vf, 75) - np.percentile(Vf, 25)), np.percentile(Vf, 93))
    specular = ((V >= hl_thr) | ((V > 215) & (hsv[..., 1] < 70))).astype(np.uint8)
    specular = cv2.dilate(specular, np.ones((7, 7), np.uint8)) > 0
    # Light defects are also ignored along the rim, where the skin curves away and catches light.
    rim = max(7, int(0.06 * np.sqrt(A))) | 1
    inner = cv2.erode(mask, np.ones((rim, rim), np.uint8)) > 0

    dark = fruit & (L < Lm - 12) & (L < Lmed - 10)
    dark = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    light = inner & ~specular & (L > 40)
    pale = light & (L > Lm + 12) & (C < Cm * 0.8)
    grey = light & (C < max(6.0, 0.45 * Cmed)) & (L > Lmed + 8)
    greenish = inner & (a < -4) & (b > 0) & (L > 20)

    found = {k: [] for k in ["crack", "black_spot", "insect_hole", "sunburn", "mold", "sugar_spot"]}
    for bl in _blobs(dark, A, 0.0004, 0.2):
        if bl["elongation"] > 4 and bl["length"] > 0.12 * np.sqrt(A):
            found["crack"].append(bl)
        elif bl["circularity"] > 0.65 and bl["area"] < 0.006 * A:
            found["insect_hole"].append(bl)
        elif bl["area"] >= 0.0015 * A:
            found["black_spot"].append(bl)
    # Light, desaturated regions: large smooth ones = sunburn, textured/greenish = mold,
    # many small granular ones = sugar spots (a skin condition, reported with skin).
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    local_std = np.sqrt(np.maximum(cv2.blur(gray**2, (7, 7)) - cv2.blur(gray, (7, 7)) ** 2, 0))
    tex_med = float(np.median(local_std[fruit]))
    for bl in _blobs(pale | grey | greenish, A, 0.0003, 0.6):
        m = bl["mask"]
        textured = local_std[m].mean() > 1.5 * tex_med
        green = greenish[m].mean() > 0.3
        if bl["area"] < 0.004 * A:
            found["sugar_spot"].append(bl)
        elif bl["elongation"] > 3:  # thin light strips are lit crease edges, not a defect
            continue
        elif green or (grey[m].mean() > 0.5 and textured):
            found["mold"].append(bl)
        elif bl["area"] >= 0.03 * A and not textured:
            found["sunburn"].append(bl)
        elif grey[m].mean() > 0.5:
            found["sugar_spot"].append(bl)

    masks, summary = {}, {}
    for name, blobs in found.items():
        m = np.zeros(mask.shape, bool)
        for bl in blobs:
            m |= bl["mask"]
        masks[name] = m
        summary[name] = {"count": len(blobs), "area_pct": round(100 * float(m.sum()) / A, 2)}
    # A trace amount is usually noise; these minimums decide "present".
    min_pct = {"crack": 0.3, "black_spot": 0.5, "insect_hole": 0.05, "sunburn": 3.0, "mold": 2.0, "sugar_spot": 0.3}
    for name, s in summary.items():
        s["present"] = s["area_pct"] >= min_pct[name]
    defects = {k: v for k, v in summary.items() if k != "sugar_spot"}
    return {
        "defects": defects,
        "defects_present": [k for k, v in defects.items() if v["present"]],
        "total_defect_area_pct": round(sum(v["area_pct"] for v in defects.values()), 2),
        "sugar_spots": summary["sugar_spot"],
    }, masks


def skin_condition(rgb, mask, sugar_spots=None):
    t = texture_features(rgb, mask)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    fruit = mask > 0
    V, S = hsv[..., 2][fruit], hsv[..., 1][fruit]
    # Gloss: bright, desaturated highlights relative to the fruit's own brightness
    # (an absolute V>225 test misses highlights on dark dates).
    thr = max(np.median(V) + 3.0 * (np.percentile(V, 75) - np.percentile(V, 25)), np.percentile(V, 95))
    highlight = (V >= thr) & (S < np.median(S))
    gloss = float(highlight.mean())
    contrast = float((V[highlight].mean() - np.median(V)) / 255) if highlight.any() else 0.0
    gloss_index = gloss * contrast * 100
    level = lambda v, lo, hi: "low" if v < lo else "medium" if v < hi else "high"
    out = {
        "wrinkle_index": t["wrinkle_index"],
        "wrinkling": t["wrinkling"],
        "gloss_index": round(gloss_index, 3),
        "gloss": level(gloss_index, 0.3, 1.0),
        "surface": t["surface"],
        "glcm_homogeneity": t["glcm_homogeneity"],
    }
    if sugar_spots is not None:
        out["sugar_spots"] = {**sugar_spots, "level": "none" if not sugar_spots["present"]
                              else level(sugar_spots["area_pct"], 1.0, 3.0)}
    return out


def maturity_by_colour(rgb, mask):
    """Rule-based ripeness from skin colour, for what colour can tell: Immature
    (Kimri) dates are green, Khalal fully yellow or red, and Rutab ripen from the
    tip so part of the skin is still yellow/red. A fully brown fruit is Rutab or
    Tamar; colour does not separate those (on Alhamdan Rutab vs Tamar photos a
    brown/wrinkle rule scored 38%), so the maturity CNN makes that call."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV_FULL)[mask > 0].astype(float)
    h, s, v = hsv[:, 0] * 360 / 255, hsv[:, 1] / 255, hsv[:, 2] / 255
    green = ((h >= 65) & (h < 160) & (s > 0.25) & (v > 0.2)).mean()
    khalal = ((((h >= 35) & (h < 65)) | (h < 12) | (h >= 340)) & (s > 0.45) & (v > 0.5)).mean()
    shares = {"green": round(float(green), 3), "khalal_yellow_red": round(float(khalal), 3)}
    if green > 0.5:
        stage = "Immature"
    elif khalal > 0.6:
        stage = "Khalal"
    elif khalal > 0.15:
        stage = "Rutab (partly ripened)"
    else:
        stage = "Rutab or Tamar (fully ripened)"
    return {"stage": stage, "skin_shares": shares}


def analyse_fruit(rgb, mask, mm_per_px=None, variety=None, px_scale=1.0, closeup=None):
    """rgb/mask: overview image (geometry). closeup: (rgb, mask) of the fruit at higher
    resolution for colour, defects and skin; defaults to the overview."""
    geom = geometry(mask, mm_per_px, px_scale)
    crgb, cmask = closeup if closeup is not None else (rgb, mask)
    defects, defect_masks = surface_defects(crgb, cmask)
    skin = skin_condition(crgb, cmask, defects.get("sugar_spots"))
    return {
        "geometry": geom,
        "size": size_class(geom, variety),
        "estimated_weight_g": estimate_weight(geom),
        "color_uniformity": color_uniformity(crgb, cmask),
        "surface_defects": defects,
        "skin": skin,
        "maturity_by_colour": maturity_by_colour(crgb, cmask),
    }, defect_masks


# ---------------------------------------------------------------- checks / calibration
def _grade_row(path):
    rgb = load_rgb(path)
    rgb = cv2.copyMakeBorder(rgb, 30, 30, 30, 30, cv2.BORDER_REPLICATE)
    mask = segment_fruit(rgb)
    d, _ = surface_defects(rgb, mask)
    if "error" in d:
        return None
    s = skin_condition(rgb, mask)
    row = {f"{k}_pct": v["area_pct"] for k, v in d["defects"].items()}
    row.update(total_defect_pct=d["total_defect_area_pct"], sugar_spot_pct=d["sugar_spots"]["area_pct"],
               wrinkle_index=s["wrinkle_index"], gloss_index=s["gloss_index"],
               color_uniformity=color_uniformity(rgb, mask)["uniformity_score"])
    return row


def check_defects():
    """No defect labels exist, so check the rules against the grade labels: a Grade-3
    fruit should show more defect area than a Grade-1 fruit of the same variety."""
    import concurrent.futures as cf

    df = pd.read_csv(config.QUALITY_DIR / "grading_index.csv")
    df = df[df.variety.isin(["Gajar", "Kupro"])]  # the two varieties with all three grades
    with cf.ProcessPoolExecutor() as pool:
        rows = list(pool.map(_grade_row, df.path, chunksize=8))
    feats = pd.DataFrame([r or {} for r in rows], index=df.index)
    df = pd.concat([df, feats], axis=1).dropna(subset=["total_defect_pct"])
    out = config.OUTPUT_DIR / "quality"
    out.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["out"]).assign(path=df.path.map(lambda p: os.path.relpath(p, config.ROOT))).to_csv(
        out / "defect_features_by_grade.csv", index=False)
    cols = [c for c in feats.columns]
    table = df.groupby(["variety", "grade"])[cols].mean().round(3)
    from scipy.stats import spearmanr

    rho = {c: {v: round(float(spearmanr(g.grade.str[-1].astype(int), g[c])[0]), 3)
               for v, g in df.groupby("variety")} for c in cols}
    (out / "defect_check.json").write_text(json.dumps(
        {"mean_by_grade": json.loads(table.to_json(orient="index")), "spearman_with_grade": rho}, indent=2))
    print(table.T.to_string(), "\n\nSpearman(feature, grade number):\n", pd.DataFrame(rho).to_string())


def calibrate_weight(csv_path):
    """Fit WEIGHT_K from fruits weighed on a scale: image,weight_g[,mm_per_px]."""
    df = pd.read_csv(csv_path)
    lw2 = []
    for r in df.itertuples():
        rgb = load_rgb(r.image)
        mmpp, corners = find_scale(rgb)
        if getattr(r, "mm_per_px", None):  # given per original pixel; load_rgb resized the image
            mmpp = r.mm_per_px * max(Image.open(r.image).size) / max(rgb.shape[:2])
        if not mmpp:
            raise SystemExit(f"{r.image}: no ArUco marker found and no mm_per_px given")
        g = geometry(fruit_instances(hide_marker(rgb, corners))[0], mmpp)
        lw2.append(g["length_mm"] * g["width_mm"] ** 2)
    lw2 = np.array(lw2)
    k = float((lw2 * df.weight_g).sum() / (lw2**2).sum())  # least squares through the origin
    err = np.abs(k * lw2 - df.weight_g) / df.weight_g
    print(json.dumps({"WEIGHT_K": k, "n": len(df), "mean_abs_pct_error": round(100 * float(err.mean()), 1)}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?")
    ap.add_argument("--mm-per-px", type=float, help="mm per pixel of the original image")
    ap.add_argument("--variety")
    ap.add_argument("--fit-size", action="store_true")
    ap.add_argument("--check-defects", action="store_true")
    ap.add_argument("--calibrate-weight")
    a = ap.parse_args()
    if a.fit_size:
        fit_size()
    elif a.check_defects:
        check_defects()
    elif a.calibrate_weight:
        calibrate_weight(a.calibrate_weight)
    elif a.image:
        rgb = load_rgb(a.image)
        px_scale = max(rgb.shape[:2]) / max(Image.open(a.image).size)
        mmpp, corners = find_scale(rgb)
        if a.mm_per_px:
            mmpp = a.mm_per_px / px_scale
        rgb = hide_marker(rgb, corners)
        masks = fruit_instances(rgb)
        print(json.dumps(analyse_fruit(rgb, masks[0], mmpp, a.variety, px_scale)[0], indent=2))
    else:
        ap.print_help(sys.stderr)
