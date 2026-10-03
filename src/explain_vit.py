"""Explainability for the foundation-model probes (probe.py): the same maps and sanity
checks as explain.py, written to the same files, so the report treats both alike.

The probe is linear, so backbone + probe is one differentiable PyTorch model:
logits = W · standardise([CLS token, mean patch token]) + b. Grad-CAM runs on the
ViT's final patch tokens (a grid of image patches, like a CNN feature map):
channel weights = mean gradient over patches, map = ReLU(tokens · weights).

  * Grad-CAM, Grad-CAM++ (token version), Guided Grad-CAM (Grad-CAM x SmoothGrad)
  * Model-randomisation test: re-initialised probe weights / fully untrained backbone
  * Deletion test: confidence drop after blurring the top-20% heatmap pixels vs 20% random pixels
  * Fruit focus: share of heatmap energy on fruit pixels
  * Background removal: test accuracy with everything outside the fruit mask replaced by the median
    background colour. ViT patch tokens attend to the whole image, so Grad-CAM can light up background
    patches that encode the fruit; this check tells that apart from a real background shortcut.

Single view (no flip TTA), so the map explains exactly the forward pass it was computed on.

Usage: python src/explain_vit.py --run probe_dinov3_b [--per-class 2]
       python src/explain_vit.py --run probe_dinov3_b --background-only   # add only the background check
"""
import argparse
import json

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
from PIL import Image

import config
from explain import overlay, rank_corr, upsample
from features import remove_background, segment_fruit
from probe import DEVICE, MODELS, backbone, load_probe, to_tensor


class ProbeNet(torch.nn.Module):
    """Backbone + the probe's scaler and logistic regression as one float32 model."""

    def __init__(self, net, clf, random_head=False):
        super().__init__()
        scaler, lr = clf.named_steps["standardscaler"], clf.named_steps["logisticregression"]
        W = torch.tensor(lr.coef_, dtype=torch.float32)
        b = torch.tensor(lr.intercept_, dtype=torch.float32)
        if random_head:
            W, b = torch.randn_like(W) * W.std(), torch.zeros_like(b)
        self.net = net
        self.k = getattr(net, "num_prefix_tokens", 1)
        self.grid = tuple(net.patch_embed.grid_size)
        self.register_buffer("mu", torch.tensor(scaler.mean_, dtype=torch.float32))
        self.register_buffer("sd", torch.tensor(scaler.scale_, dtype=torch.float32))
        self.register_buffer("W", W)
        self.register_buffer("b", b)

    def head(self, tokens):
        f = torch.cat([tokens[:, 0], tokens[:, self.k:].mean(1)], 1)
        return ((f - self.mu) / self.sd) @ self.W.T + self.b

    def forward(self, x):
        return self.head(self.net.forward_features(x))


def token_cam(model, x, cls, plus=False):
    tokens = model.net.forward_features(x)
    grads = torch.autograd.grad(model.head(tokens)[0, cls], tokens)[0]
    A, G = tokens[0, model.k:].detach(), grads[0, model.k:]  # (patches, channels)
    if plus:
        g2, g3 = G**2, G**3
        denom = 2 * g2 + A.sum(0) * g3
        alpha = g2 / torch.where(denom != 0, denom, torch.full_like(denom, 1e-10))
        w = (alpha * G.clamp(min=0)).sum(0)
    else:
        w = G.mean(0)
    cam = (A @ w).clamp(min=0).reshape(model.grid).cpu().numpy()
    return cam / cam.max() if cam.max() > 0 else cam


def smoothgrad(model, x, cls, std, n=16, sigma=0.12):
    noise = torch.randn((n, *x.shape[1:]), device=x.device) * (sigma / torch.tensor(std, device=x.device).view(1, 3, 1, 1))
    xs = (x.repeat(n, 1, 1, 1) + noise).requires_grad_(True)
    g = torch.autograd.grad(model(xs)[:, cls].sum(), xs)[0].mean(0).clamp(min=0).sum(0).cpu().numpy()
    return g / g.max() if g.max() > 0 else g


@torch.no_grad()
def background_check(model, class_names, data, S, mean, std):
    """Test accuracy (single view) on original vs background-removed images."""
    hits = {"original": [], "background_removed": []}
    for ci, cname in enumerate(class_names):
        for p in sorted((config.DATASETS[data] / "test" / cname).glob("*.jpg")):
            img = np.asarray(Image.open(p).convert("RGB").resize((S, S), Image.BICUBIC))
            for key, im in (("original", img), ("background_removed", remove_background(img))):
                x = to_tensor(Image.fromarray(im), S, mean, std)[None].to(DEVICE)
                hits[key].append(int(model(x).argmax(1)) == ci)
    return {f"accuracy_{k}": round(float(np.mean(v)), 4) for k, v in hits.items()}


def add_background_check(summary, bg):
    summary.update(bg)
    summary["verdict"]["robust_to_background_removal"] = bg["accuracy_background_removed"] >= bg["accuracy_original"] - 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--background-only", action="store_true", help="add the background check to existing results")
    args = ap.parse_args()

    probe = load_probe(args.run)
    meta = json.loads((config.MODEL_DIR / f"{args.run}_meta.json").read_text())
    class_names = probe["class_names"]
    net, S, mean, std = backbone(probe["model"])
    net.float()
    model = ProbeNet(net, probe["clf"]).to(DEVICE).eval()
    rnd_head = ProbeNet(net, probe["clf"], random_head=True).to(DEVICE).eval()
    torch.manual_seed(config.SEED)
    name, _ = MODELS[probe["model"]]
    raw = timm.create_model(name, pretrained=False, num_classes=0, img_size=S).eval().to(DEVICE)
    rnd_all = ProbeNet(raw, probe["clf"], random_head=True).to(DEVICE).eval()
    tensor = lambda img: to_tensor(Image.fromarray(img), S, mean, std)[None].to(DEVICE)
    prob = lambda m, x: torch.softmax(m(x), 1)[0]
    out_dir = config.OUTPUT_DIR / "explainability"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.SEED)
    sanity_path = out_dir / f"{args.run}_sanity_checks.json"
    if args.background_only:
        sanity = json.loads(sanity_path.read_text())
        add_background_check(sanity["summary"], background_check(model, class_names, meta["data"], S, mean, std))
        sanity_path.write_text(json.dumps(sanity, indent=2))
        return print(json.dumps(sanity["summary"], indent=2))

    stats, gallery = [], []
    for ci, cname in enumerate(class_names):
        paths = sorted((config.DATASETS[meta["data"]] / "test" / cname).glob("*.jpg"))
        for i, p in enumerate(paths):
            img = np.asarray(Image.open(p).convert("RGB").resize((S, S), Image.BICUBIC))
            x = tensor(img)
            with torch.no_grad():
                probs = prob(model, x)
            pred = int(probs.argmax())
            cam = upsample(token_cam(model, x, pred), S)
            cam_rh = upsample(token_cam(rnd_head, x, pred), S)
            cam_ra = upsample(token_cam(rnd_all, x, pred), S)
            fruit = segment_fruit(img) > 0

            # deletion test: blur the top-20% heatmap pixels vs 20% random pixels
            blurred = cv2.GaussianBlur(img, (0, 0), 12)
            k = int(0.2 * cam.size)
            drops = []
            for idx in (np.argsort(cam.ravel())[-k:], rng.choice(cam.size, k, replace=False)):
                m = np.zeros(cam.size, bool)
                m[idx] = True
                with torch.no_grad():
                    drops.append(float(probs[pred] - prob(model, tensor(np.where(m.reshape(cam.shape)[..., None], blurred, img)))[pred]))
            stats.append({
                "file": p.name, "true": cname, "pred": class_names[pred], "conf": float(probs[pred]),
                "spearman_vs_random_head": rank_corr(cam, cam_rh),
                "spearman_vs_random_all": rank_corr(cam, cam_ra),
                "deletion_drop_cam": drops[0], "deletion_drop_random": drops[1],
                "cam_energy_on_fruit": float(cam[fruit].sum() / max(cam.sum(), 1e-9)),
                "fruit_area_fraction": float(fruit.mean()),
            })
            if i < args.per_class:
                campp = upsample(token_cam(model, x, pred, plus=True), S)
                guided = cam * upsample(smoothgrad(model, x, pred, std), S)
                guided = guided / guided.max() if guided.max() > 0 else guided
                gallery.append((img, cam, campp, guided, cam_ra, stats[-1]))
        print(f"{cname}: {len(paths)} images", flush=True)

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
    add_background_check(summary, background_check(model, class_names, meta["data"], S, mean, std))
    sanity_path.write_text(json.dumps({"summary": summary, "per_image": stats}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
