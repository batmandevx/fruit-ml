"""Full Phase 2 quality report for a photo of one or more date fruits (a single fruit,
a tray, or the top of an open bag).

Per fruit: variety (Phase 1 model), maturity stage (CNN), quality grade 1-3 (CNN
+ Grad-CAM), size class, colour uniformity, surface defects, skin condition and
estimated weight. For the whole photo: fruit count, size-class counts and total
estimated weight.

Scale: a printed ArUco marker in the photo (config.ARUCO_*) or --mm-per-px.
Without a scale, size uses the pixel thresholds fitted on the grading rig and
weight is not estimated.
--bag-weight: the weight the scale shows for the fruits in the photo. It is
used to recalibrate the weight coefficient for this batch (k = bag weight /
sum of L*W^2), which also gives a per-fruit weight share.

Usage: python src/grade.py <image> [<image> ...] [--variety Al-majdool] [--mm-per-px 0.12] [--bag-weight 250]
Python: from grade import load_models, grade_image; grade_image("photo.jpg", load_models())
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches  # noqa: E402
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

import config
from explain import gradcam, overlay, upsample
from features import WORK_SIDE
from quality import analyse_fruit, find_scale, fruit_closeup, fruit_instances, hide_marker

DEFECT_COLORS = {"crack": (255, 0, 0), "black_spot": (255, 0, 255), "insect_hole": (0, 255, 255),
                 "sunburn": (255, 200, 0), "mold": (0, 255, 0), "sugar_spot": (255, 255, 255)}


def load_model(runs):
    """{"members": [(model, meta), ...], "class_names", "calib"} for one run or an ensemble."""
    runs = [runs] if isinstance(runs, str) else list(runs)
    members = []
    for run in runs:
        meta_path = config.MODEL_DIR / f"{run}_meta.json"  # written when training has finished
        if not meta_path.exists():
            print(f"note: model '{run}' not trained yet, skipped", file=sys.stderr)
            return None
        meta = json.loads(meta_path.read_text())
        if "timm" in meta:  # PyTorch foundation-model probe (probe.py); torch is only imported when used
            from probe import load_probe
            members.append((load_probe(run), meta))
        else:
            members.append((tf.keras.models.load_model(config.MODEL_DIR / f"{run}_best.keras"), meta))
    names = members[0][1]["class_names"]
    assert all(m["class_names"] == names for _, m in members), "ensemble members disagree on classes"
    calib_path = config.MODEL_DIR / f"{config.run_name(runs)}_calib.json"
    calib = json.loads(calib_path.read_text()) if calib_path.exists() else {}
    return {"members": members, "class_names": names, "calib": calib}


def fruit_crop(rgb, mask, margin=0.15):
    """Square crop around one fruit with the other fruits greyed out, padded like training."""
    x, y, w, h = cv2.boundingRect(mask)
    m = int(max(w, h) * margin)
    x0, y0, x1, y1 = max(0, x - m), max(0, y - m), min(rgb.shape[1], x + w + m), min(rgb.shape[0], y + h + m)
    crop = rgb[y0:y1, x0:x1].copy()
    keep = cv2.dilate(mask[y0:y1, x0:x1], np.ones((15, 15), np.uint8)) > 0
    bg = np.median(rgb[mask == 0], 0) if (mask == 0).any() else np.array([200, 200, 200])
    crop[~keep] = bg
    side = max(crop.shape[:2])
    out = np.empty((side, side, 3), np.uint8)
    out[:] = bg
    oy, ox = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
    out[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    return out


def classify(model_meta, crop):
    """Top class with temperature-calibrated probabilities (calibrate.py). A prediction
    below the model's review threshold is flagged for manual review. Also returns the
    first Keras member's input (for Grad-CAM) and the class index."""
    probs, xs = [], []
    for model, meta in model_meta["members"]:
        if "timm" in meta:
            from probe import probe_probs
            probs.append(probe_probs(model, crop).astype(np.float64))
            continue
        S = meta["img_size"]
        x = tf.constant(np.asarray(Image.fromarray(crop).resize((S, S), Image.BILINEAR), np.float32)[None])
        probs.append(model(x, training=False)[0].numpy().astype(np.float64))
        xs.append(x)
    probs = np.mean(probs, axis=0)
    calib = model_meta["calib"]
    T = calib.get("temperature", 1.0)
    z = np.log(np.clip(probs, 1e-12, 1)) / T
    probs = np.exp(z - z.max())
    probs /= probs.sum()
    i = int(probs.argmax())
    thr = calib.get("review_threshold")
    names = model_meta["class_names"]
    return {"label": names[i], "confidence": round(float(probs[i]), 3),
            "needs_review": bool(thr is not None and probs[i] < thr),
            "probs": {c: round(float(p), 3) for c, p in zip(names, probs)}}, (xs or [None])[0], i


def load_models(runs=None):
    """{"variety"|"grade"|"maturity": load_model(...) or None}; runs default to config.PRODUCTION_RUNS."""
    runs = {**config.PRODUCTION_RUNS, **(runs or {})}
    return {k: load_model(r) for k, r in runs.items()}


def load_image(path):
    try:
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    except FileNotFoundError:
        raise ValueError(f"image not found: {path}")
    except Exception as e:  # PIL.UnidentifiedImageError, truncated files, ...
        raise ValueError(f"cannot read image {path}: {e}")
    if min(img.size) < 64:
        raise ValueError(f"image too small ({img.width}x{img.height}); need at least 64 px")
    return img


def grade_image(image, models, variety_override=None, mm_per_px_arg=None, bag_weight=None, save=True):
    """Full quality report (dict) for one photo. Raises ValueError for unusable input."""
    if bag_weight is not None and bag_weight <= 0:
        raise ValueError("bag weight must be positive")
    if mm_per_px_arg is not None and mm_per_px_arg <= 0:
        raise ValueError("mm per pixel must be positive")
    register_heif_opener()
    full = load_image(image)
    px_scale = min(1.0, WORK_SIDE / max(full.size))  # working px per original px
    rgb = np.asarray(full.resize((round(full.width * px_scale), round(full.height * px_scale)), Image.LANCZOS))
    full = np.asarray(full)
    mm_per_px, marker = find_scale(rgb)
    if mm_per_px_arg:
        mm_per_px = mm_per_px_arg / px_scale
    rgb = hide_marker(rgb, marker)
    if marker is not None:
        full = hide_marker(full, marker / px_scale)

    masks = fruit_instances(rgb)
    if not masks:
        raise ValueError("no fruit found in the image")

    fruits, panels = [], []
    for n, mask in enumerate(masks, 1):
        close_rgb, close_mask = fruit_closeup(full, mask, px_scale)
        crop = fruit_crop(close_rgb, close_mask)
        rec = {"fruit": n}
        if variety_override:
            rec["variety"] = {"label": variety_override, "confidence": None}
        elif models["variety"]:
            rec["variety"] = classify(models["variety"], crop)[0]
        variety = rec.get("variety", {}).get("label")
        if models["maturity"]:
            rec["maturity"] = classify(models["maturity"], crop)[0]
        cam = None
        if models["grade"]:
            rec["grade"], x, gi = classify(models["grade"], crop)
            cam = upsample(gradcam(models["grade"]["members"][0][0], x, gi), x.shape[1])  # first member
            panels.append((x[0].numpy().astype(np.uint8), cam, f"#{n} {rec['grade']['label']}"))
        # The size model knows the grading-set varieties by their folder names.
        size_variety = variety if variety in config.GRADING_VARIETIES else None
        q, dmasks = analyse_fruit(rgb, mask, mm_per_px, size_variety, px_scale, (close_rgb, close_mask))
        rec.update(q)
        rec["_mask"], rec["_defects"], rec["_close"] = mask, dmasks, close_rgb
        fruits.append(rec)

    summary = {"image": str(image), "n_fruits": len(fruits),
               "scale_mm_per_px": round(mm_per_px, 4) if mm_per_px else None,
               "scale_source": "argument" if mm_per_px_arg else "aruco" if marker is not None else None}
    sizes = [f["size"].get("size_class") for f in fruits]
    summary["size_counts"] = {s: sizes.count(s) for s in sorted(set(sizes), key=str)}
    if mm_per_px:
        lw2 = np.array([f["geometry"]["length_mm"] * f["geometry"]["width_mm"] ** 2 for f in fruits])
        summary["estimated_total_weight_g"] = round(float(config.WEIGHT_K * lw2.sum()), 1)
        if bag_weight:
            k = bag_weight / lw2.sum()
            summary["bag_weight_g"] = bag_weight
            summary["calibrated_weight_k"] = round(float(k), 7)
            summary["model_vs_scale_error_pct"] = round(
                100 * (summary["estimated_total_weight_g"] - bag_weight) / bag_weight, 1)
            for f, v in zip(fruits, lw2):
                f["weight_from_bag_g"] = round(float(k * v), 2)
    elif bag_weight:
        summary["bag_weight_g"] = bag_weight
        summary["mean_weight_per_fruit_g"] = round(bag_weight / len(fruits), 2)
    for key in ("grade", "maturity", "variety"):
        labels = [f[key]["label"] for f in fruits if key in f]
        if labels:
            summary[f"{key}_counts"] = {l: labels.count(l) for l in sorted(set(labels))}
    summary["fruits_with_defects"] = sum(bool(f["surface_defects"].get("defects_present")) for f in fruits)
    summary["fruits_needing_review"] = [f["fruit"] for f in fruits
                                        if any(f.get(k, {}).get("needs_review") for k in ("variety", "grade", "maturity"))]
    report = {"summary": summary, "fruits": [{k: v for k, v in f.items() if not k.startswith("_")} for f in fruits]}
    if not save:
        return report

    out = config.OUTPUT_DIR / "quality" / "reports"
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(str(image)).stem
    annotated = rgb.copy()
    for f in fruits:
        cnts, _ = cv2.findContours(f["_mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, cnts, -1, (0, 200, 0), 2)
        x, y, _, _ = cv2.boundingRect(f["_mask"])
        cv2.putText(annotated, str(f["fruit"]), (x + 4, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(annotated, str(f["fruit"]), (x + 4, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    # Column 0: overview. Then per fruit (max 5): defect candidates (top), grade Grad-CAM (bottom).
    shown = fruits[:5]
    fig, axes = plt.subplots(2, 1 + len(shown), figsize=(4 * (1 + len(shown)), 8), squeeze=False)
    axes[0, 0].imshow(annotated)
    axes[0, 0].set_title(f"{len(fruits)} fruit(s)")
    axes[1, 0].legend(handles=[matplotlib.patches.Patch(facecolor=np.array(c) / 255, edgecolor="k", label=k)
                               for k, c in DEFECT_COLORS.items()], loc="center", title="defect candidates")
    cams = {int(t.split()[0][1:]): (img, cam) for img, cam, t in panels}
    for col, f in enumerate(shown, 1):
        img = f["_close"].copy()
        for name, m in f["_defects"].items():
            img[m] = (0.3 * img[m] + 0.7 * np.array(DEFECT_COLORS[name])).astype(np.uint8)
        axes[0, col].imshow(img)
        axes[0, col].set_title(f"#{f['fruit']} " + (", ".join(f["surface_defects"].get("defects_present", []))
                                                    or "no defects"), fontsize=9)
        if f["fruit"] in cams:
            axes[1, col].imshow(overlay(*cams[f["fruit"]]))
            g = f["grade"]
            axes[1, col].set_title(f"grade Grad-CAM: {g['label']} ({g['confidence']:.2f})", fontsize=9)
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / f"{stem}_quality.png", dpi=100)

    plt.close(fig)
    (out / f"{stem}_quality.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--variety", help="skip the variety model and use this variety")
    ap.add_argument("--mm-per-px", type=float, help="mm per pixel of the ORIGINAL image")
    ap.add_argument("--bag-weight", type=float, help="scale reading (g) for the fruits in the photo")
    ap.add_argument("--variety-run", nargs="+")
    ap.add_argument("--grade-run", nargs="+")
    ap.add_argument("--maturity-run", nargs="+")
    args = ap.parse_args()
    runs = {k: v for k, v in [("variety", args.variety_run), ("grade", args.grade_run),
                              ("maturity", args.maturity_run)] if v}
    models = load_models(runs)
    failed = 0
    for image in args.images:
        try:
            report = grade_image(image, models, args.variety, args.mm_per_px, args.bag_weight)
            print(json.dumps(report, indent=2))
        except ValueError as e:
            failed += 1
            print(json.dumps({"image": image, "error": str(e)}), file=sys.stderr)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
