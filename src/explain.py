"""Explainability: Grad-CAM, Grad-CAM++ and Guided Grad-CAM, plus sanity checks.

Guided Grad-CAM = Grad-CAM x a fine-grained input-gradient map. Classic guided
backprop needs every ReLU's gradient overridden, which MobileNetV3's
hard-swish blocks don't allow cleanly, so the fine-grained map here is
SmoothGrad (positive input gradients averaged over noisy copies).

Sanity checks (per image, summarised over the test set):
  * Model-randomisation test (Adebayo et al., 2018): heatmaps from the trained
    model vs the same architecture with (a) a re-initialised head, (b) all
    weights re-initialised. Low Spearman correlation = the map depends on what
    the model learned, i.e. it is not just an edge detector.
  * Deletion test: confidence drop when the top-20% heatmap pixels are blurred
    out, vs blurring 20% random pixels. A faithful map causes a bigger drop.
  * Fruit-focus: share of heatmap energy that falls on fruit pixels
    (GrabCut mask). Low values mean the model looks at the background, which is a
    real risk here because each Zenodo class was shot in its own session.

Usage: python src/explain.py [--run mobilenetv3] [--per-class 2]
"""
import argparse
import json

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.stats import spearmanr

import config
from features import segment_fruit


def _split(model):
    return model.get_layer("backbone"), model.get_layer("head")


def _grads(model, x, class_idx):
    backbone, head = _split(model)
    with tf.GradientTape() as tape:
        fmap = backbone(x, training=False)
        tape.watch(fmap)
        score = head(fmap, training=False)[:, class_idx]
    return fmap[0].numpy(), tape.gradient(score, fmap)[0].numpy()


def _norm(cam):
    cam = np.maximum(cam, 0)
    return cam / cam.max() if cam.max() > 0 else cam


def gradcam(model, x, class_idx):
    fmap, g = _grads(model, x, class_idx)
    return _norm((fmap * g.mean((0, 1))).sum(-1))


def gradcam_pp(model, x, class_idx):
    fmap, g = _grads(model, x, class_idx)
    g2, g3 = g**2, g**3
    denom = 2 * g2 + fmap.sum((0, 1)) * g3
    alpha = g2 / np.where(denom != 0, denom, 1e-10)
    weights = (alpha * np.maximum(g, 0)).sum((0, 1))
    return _norm((fmap * weights).sum(-1))


def smoothgrad(model, x, class_idx, n=24, sigma=0.12):
    xs = tf.repeat(x, n, 0) + tf.random.normal((n, *x.shape[1:]), stddev=sigma * 255)
    with tf.GradientTape() as tape:
        tape.watch(xs)
        score = model(xs, training=False)[:, class_idx]
    g = tape.gradient(score, xs).numpy().mean(0)
    g = np.maximum(g, 0).sum(-1)
    return g / g.max() if g.max() > 0 else g


def upsample(cam, size):
    return cv2.resize(cam.astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC).clip(0, 1)


def overlay(img, cam, alpha=0.45):
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)[..., ::-1]
    return np.uint8((1 - alpha) * img + alpha * heat)


def randomized_copy(model, head_only):
    rnd = tf.keras.models.clone_model(model)  # fresh random initialisation
    if head_only:
        rnd.get_layer("backbone").set_weights(model.get_layer("backbone").get_weights())
    return rnd


def rank_corr(a, b):
    """Spearman correlation; a constant (vanished) map counts as uncorrelated."""
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return 0.0
    return float(spearmanr(a.ravel(), b.ravel())[0])


def deletion_drop(model, x, cam, class_idx, frac=0.2, rng=None):
    img = x[0].numpy()
    blurred = cv2.GaussianBlur(img, (0, 0), 12)
    p0 = float(model(x, training=False)[0, class_idx])
    k = int(frac * cam.size)
    top = np.zeros(cam.size, bool)
    top[np.argsort(cam.ravel())[-k:]] = True
    rand = np.zeros(cam.size, bool)
    rand[rng.choice(cam.size, k, replace=False)] = True
    out = []
    for m in (top, rand):
        m = m.reshape(cam.shape)[..., None]
        xm = np.where(m, blurred, img)[None]
        out.append(p0 - float(model(xm, training=False)[0, class_idx]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=config.BACKBONE)
    ap.add_argument("--per-class", type=int, default=2, help="example images per class in the gallery")
    args = ap.parse_args()

    model = tf.keras.models.load_model(config.MODEL_DIR / f"{args.run}_best.keras")
    meta = json.loads((config.MODEL_DIR / f"{args.run}_meta.json").read_text())
    class_names = meta["class_names"]
    rnd_head, rnd_all = randomized_copy(model, True), randomized_copy(model, False)
    out_dir = config.OUTPUT_DIR / "explainability"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.SEED)
    S = meta.get("img_size", config.IMG_SIZE)

    stats, gallery = [], []
    for ci, cname in enumerate(class_names):
        paths = sorted((config.DATASETS[meta.get("data", "variety")] / "test" / cname).glob("*.jpg"))
        for i, p in enumerate(paths):
            img = tf.keras.utils.img_to_array(
                tf.keras.utils.load_img(p, target_size=(S, S), keep_aspect_ratio=True)
            )
            x = tf.constant(img[None])
            probs = model(x, training=False)[0].numpy()
            pred = int(probs.argmax())
            cam = upsample(gradcam(model, x, pred), S)
            campp = upsample(gradcam_pp(model, x, pred), S)
            guided = cam * smoothgrad(model, x, pred)
            guided = guided / guided.max() if guided.max() > 0 else guided

            cam_rh = upsample(gradcam(rnd_head, x, pred), S)
            cam_ra = upsample(gradcam(rnd_all, x, pred), S)
            fruit = segment_fruit(img.astype(np.uint8)) > 0
            drop_cam, drop_rand = deletion_drop(model, x, cam, pred, rng=rng)
            stats.append({
                "file": p.name, "true": cname, "pred": class_names[pred], "conf": float(probs[pred]),
                "spearman_vs_random_head": rank_corr(cam, cam_rh),
                "spearman_vs_random_all": rank_corr(cam, cam_ra),
                "deletion_drop_cam": drop_cam, "deletion_drop_random": drop_rand,
                "cam_energy_on_fruit": float(cam[fruit].sum() / max(cam.sum(), 1e-9)),
                "fruit_area_fraction": float(fruit.mean()),
            })
            if i < args.per_class:
                gallery.append((img.astype(np.uint8), cam, campp, guided, cam_ra, stats[-1]))

    # Gallery: original | Grad-CAM | Grad-CAM++ | Guided Grad-CAM | randomised-model Grad-CAM
    cols = ["input", "Grad-CAM", "Grad-CAM++", "Guided Grad-CAM", "Grad-CAM (random weights)"]
    fig, axes = plt.subplots(len(gallery), 5, figsize=(15, 3.1 * len(gallery)))
    for r, (img, cam, campp, guided, cam_ra, s) in enumerate(gallery):
        panels = [img, overlay(img, cam), overlay(img, campp), guided, overlay(img, cam_ra)]
        for c, panel in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(panel, cmap="inferno" if c == 3 else None)
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=11)
        ok = "✓" if s["true"] == s["pred"] else "✗"
        axes[r, 0].set_ylabel(f"{s['true']}\n→ {s['pred']} {s['conf']:.2f} {ok}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / f"{args.run}_gradcam_gallery.png", dpi=90)

    keys = ["spearman_vs_random_head", "spearman_vs_random_all", "deletion_drop_cam",
            "deletion_drop_random", "cam_energy_on_fruit", "fruit_area_fraction"]
    summary = {k: round(float(np.mean([s[k] for s in stats])), 4) for k in keys}
    summary["verdict"] = {
        "depends_on_learned_weights": summary["spearman_vs_random_all"] < 0.5,
        "faithful_deletion": summary["deletion_drop_cam"] > summary["deletion_drop_random"],
        "focuses_on_fruit": summary["cam_energy_on_fruit"] > summary["fruit_area_fraction"],
    }
    (out_dir / f"{args.run}_sanity_checks.json").write_text(
        json.dumps({"summary": summary, "per_image": stats}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
