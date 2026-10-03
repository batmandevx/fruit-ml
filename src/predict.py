"""Classify one date-fruit image and report what was extracted from it:
predicted variety (top-3), Grad-CAM heatmap, and colour/shape/texture features
compared with the predicted variety's typical profile.

The production model is an ensemble (config.PRODUCTION_RUNS["variety"]); the heatmap
is Grad-CAM of its Keras member for the ensemble's predicted class.

Usage: python src/predict.py <image> [--run effv2b0_260 ...]   (default: the production variety model)
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from explain import gradcam, overlay, upsample
from features import extract, load_rgb
from grade import classify, load_model

COMPARE = ["color.mean_hue_deg", "color.mean_brightness", "color.color_uniformity",
           "shape.aspect_ratio", "shape.circularity", "texture.glcm_homogeneity", "texture.wrinkle_index"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--run", nargs="+", default=config.PRODUCTION_RUNS["variety"], help="several = ensemble")
    args = ap.parse_args()

    ensemble = load_model(args.run)
    rgb = load_rgb(args.image)
    h, w = rgb.shape[:2]
    side = min(h, w)  # centre crop, same as training
    crop = rgb[(h - side) // 2 : (h + side) // 2, (w - side) // 2 : (w + side) // 2]
    res, x, i = classify(ensemble, crop)
    class_names = ensemble["class_names"]
    probs = np.array([res["probs"][c] for c in class_names])
    top = probs.argsort()[::-1][:3]
    pred = res["label"]

    feats, mask = extract(rgb)
    report = {
        "image": args.image,
        "predicted_variety": pred,
        "confidence": res["confidence"],
        "needs_review": res["needs_review"],
        "top3": [{"variety": class_names[i], "prob": round(float(probs[i]), 4)} for i in top],
        "features": feats,
    }
    prof_path = config.OUTPUT_DIR / "class_feature_profiles.csv"
    if prof_path.exists() and "error" not in feats:
        prof = pd.read_csv(prof_path, index_col=0)
        if pred in prof.index:
            flat = {f"{g}.{k}": v for g, d in feats.items() if isinstance(d, dict) for k, v in d.items()}
            report["vs_typical_" + pred] = {
                k: {"this_image": flat[k], "variety_median": float(prof.loc[pred, k])} for k in COMPARE if k in prof
            }

    out = config.OUTPUT_DIR / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    keras = next((m for m, meta in ensemble["members"] if "timm" not in meta), None)
    panels = [crop, mask]
    titles = ["input", "fruit mask (features)"]
    if keras is not None:  # x is that member's input
        img = x[0].numpy()
        panels = [img.astype(np.uint8), overlay(img, upsample(gradcam(keras, x, i), img.shape[0])), mask]
        titles = ["input", f"Grad-CAM → {pred} ({res['confidence']:.2f})", "fruit mask (features)"]
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    for ax, panel, title in zip(axes, panels, titles):
        ax.imshow(panel, cmap="gray" if panel is mask else None)
        ax.set_title(title), ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / f"{stem}_explained.png", dpi=110)
    (out / f"{stem}_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
