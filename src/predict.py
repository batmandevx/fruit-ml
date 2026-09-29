"""Classify one date-fruit image and report what was extracted from it:
predicted variety (top-3), Grad-CAM heatmap, and colour/shape/texture features
compared with the predicted variety's typical profile.

Usage: python src/predict.py <image> [--run effv2b0_260]   (default: the production variety model)
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

import config
from explain import gradcam, overlay, upsample
from features import extract, load_rgb

COMPARE = ["color.mean_hue_deg", "color.mean_brightness", "color.color_uniformity",
           "shape.aspect_ratio", "shape.circularity", "texture.glcm_homogeneity", "texture.wrinkle_index"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--run", default=config.PRODUCTION_RUNS["variety"][0])
    args = ap.parse_args()

    model = tf.keras.models.load_model(config.MODEL_DIR / f"{args.run}_best.keras")
    meta = json.loads((config.MODEL_DIR / f"{args.run}_meta.json").read_text())
    class_names = meta["class_names"]

    rgb = load_rgb(args.image)
    S = meta.get("img_size", config.IMG_SIZE)
    h, w = rgb.shape[:2]
    side = min(h, w)  # centre crop, same as training
    crop = rgb[(h - side) // 2 : (h + side) // 2, (w - side) // 2 : (w + side) // 2]
    img = np.asarray(Image.fromarray(crop).resize((S, S), Image.BILINEAR)).astype(np.float32)
    x = tf.constant(img[None])
    probs = model(x, training=False)[0].numpy().astype(np.float64)
    calib_path = config.MODEL_DIR / f"{args.run}_calib.json"  # temperature from calibrate.py
    if calib_path.exists():
        calib = json.loads(calib_path.read_text())
        z = np.log(np.clip(probs, 1e-12, 1)) / calib["temperature"]
        probs = np.exp(z - z.max())
        probs /= probs.sum()
    top = probs.argsort()[::-1][:3]
    pred = class_names[top[0]]

    feats, mask = extract(rgb)
    report = {
        "image": args.image,
        "predicted_variety": pred,
        "confidence": round(float(probs[top[0]]), 4),
        "needs_review": bool(calib_path.exists() and probs[top[0]] < (calib.get("review_threshold") or 0)),
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
    cam = upsample(gradcam(model, x, int(top[0])), S)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, panel, title in zip(axes, [img.astype(np.uint8), overlay(img, cam), mask],
                                ["input", f"Grad-CAM → {pred} ({probs[top[0]]:.2f})", "fruit mask (features)"]):
        ax.imshow(panel, cmap="gray" if panel is mask else None)
        ax.set_title(title), ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / f"{stem}_explained.png", dpi=110)
    (out / f"{stem}_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
