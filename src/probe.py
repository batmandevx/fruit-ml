"""Foundation-model probes (PyTorch + timm on the Apple GPU): frozen modern backbone
-> image embedding -> multinomial logistic regression.

On a few thousand images a linear probe on DINOv3 / DINOv2 / ConvNeXt-V2 / SigLIP features
is often as accurate as fine-tuning a smaller CNN, costs minutes instead of hours
and fits in 16 GB. Every image is embedded in its 4 flips (the TTA views of
evaluate.py): train views are extra samples, val/test probabilities are averaged.

  embeddings  outputs/features/<model>__<data>__<split>.npz  (cached)
  probe       outputs/models/<run>_probe.joblib   run = probe_<model> (variety) or <data>_probe_<model>
  metrics     outputs/eval/<run>_tta_<split>_metrics.json (+ probs, confusion matrix; see metrics.py)

The regularisation C is chosen on the validation log-loss. Several --model = an
ensemble (averaged probabilities).

--bg-aug (run <run>_bgaug): train on each image twice, as photographed and with the background
removed (features.remove_background), and choose C on the mean validation loss of both. A frozen
backbone also encodes the photo session (lighting, framing, background), which a probe can use as a
shortcut; the background-removed copies force it onto the fruit. --no-background scores a
background-removed split (metrics name <run>_nobg_tta) to measure that reliance.

Usage:
  python src/probe.py --data variety --model dinov2_b dinov2_l --split val
  python src/probe.py --data grade --model dinov2_l --split test
  python src/probe.py --data variety --model dinov3_b --bg-aug --split val [--no-background]
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
import timm
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from features import remove_background
from metrics import nll, write_metrics

# name -> (timm model, input size). Sizes keep the ViTs at <= 576 patch tokens so the
# embeddings of all three datasets take minutes on an M-series GPU.
MODELS = {
    "dinov3_l": ("vit_large_patch16_dinov3.lvd1689m", 336),
    "dinov3_b": ("vit_base_patch16_dinov3.lvd1689m", 336),
    "dinov2_b": ("vit_base_patch14_dinov2.lvd142m", 336),
    "dinov2_l": ("vit_large_patch14_dinov2.lvd142m", 336),
    "convnextv2_b": ("convnextv2_base.fcmae_ft_in22k_in1k_384", 384),
    "siglip_b": ("vit_base_patch16_siglip_384.webli", 384),
}
FEAT_DIR = config.OUTPUT_DIR / "features"
C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def to_tensor(img, size, mean, std):
    # Stored images are already square (cropped or padded), so a plain resize keeps the whole fruit.
    img = img.convert("RGB").resize((size, size), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255
    return (x - torch.tensor(mean).view(3, 1, 1)) / torch.tensor(std).view(3, 1, 1)


class Images(torch.utils.data.Dataset):
    def __init__(self, files, size, mean, std, no_background=False):
        self.files, self.args, self.no_background = files, (size, mean, std), no_background

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        if self.no_background:
            img = Image.fromarray(remove_background(np.asarray(img)))
        return to_tensor(img, *self.args)


def list_split(data, split):
    root = config.DATASETS[data] / split
    class_names = sorted(p.name for p in root.iterdir() if p.is_dir())
    files, labels = [], []
    for k, c in enumerate(class_names):
        for f in sorted((root / c).glob("*.jpg")):
            files.append(str(f))
            labels.append(k)
    return files, np.array(labels), class_names


_NETS = {}


def backbone(model):
    """(net, size, mean, std), loaded once per process (grade.py shares it across tasks)."""
    if model not in _NETS:
        name, size = MODELS[model]
        kw = {"img_size": size} if "patch" in name else {}
        net = timm.create_model(name, pretrained=True, num_classes=0, **kw).eval().to(DEVICE)
        _NETS[model] = (net, size, net.pretrained_cfg["mean"], net.pretrained_cfg["std"])
    return _NETS[model]


@torch.no_grad()
def embed_batch(net, x):
    """ViT: [CLS token, mean patch token] (the DINOv2 linear-probe recipe). CNN: pooled features."""
    f = net.forward_features(x)
    if f.ndim == 3:
        k = getattr(net, "num_prefix_tokens", 1)
        return torch.cat([f[:, 0], f[:, k:].mean(1)], 1)
    return net.forward_head(f, pre_logits=True)


def embed_views(net, x):
    """(batch, 4 flips, dim) embeddings; the flips are the TTA views of evaluate.py."""
    views = [x, x.flip(3), x.flip(2), x.flip(2).flip(3)]
    with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "mps"):
        return torch.stack([embed_batch(net, v).float() for v in views], 1).cpu().numpy()


def embeddings(model, data, split, no_background=False):
    path = FEAT_DIR / f"{model}__{data}__{split}{'__nobg' if no_background else ''}.npz"
    if path.exists():
        d = np.load(path)
        return d["x"], d["y"], list(d["files"]), list(d["class_names"])
    net, size, mean, std = backbone(model)
    files, y, class_names = list_split(data, split)
    loader = torch.utils.data.DataLoader(Images(files, size, mean, std, no_background), batch_size=32, num_workers=8,
                                         persistent_workers=False)
    out = []
    for i, x in enumerate(loader):
        out.append(embed_views(net, x.to(DEVICE)))
        if i % 20 == 0:
            print(f"  {model} {data}/{split}: {i * 32}/{len(files)}", flush=True)
    x = np.concatenate(out)  # (n, 4 views, dim)
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(path, x=x, y=y, files=np.array(files), class_names=np.array(class_names))
    return x, y, files, class_names


def predict(clf, x):
    n, v, d = x.shape
    return clf.predict_proba(x.reshape(n * v, d)).reshape(n, v, -1).mean(1)


def load_probe(run):
    return joblib.load(config.MODEL_DIR / f"{run}_probe.joblib")


def probe_probs(probe, rgb):
    """Class probabilities for one square uint8 RGB crop (grade.py / predict.py)."""
    net, size, mean, std = backbone(probe["model"])
    x = to_tensor(Image.fromarray(rgb), size, mean, std)[None].to(DEVICE)
    return predict(probe["clf"], embed_views(net, x))[0]


def run_name(data, model, bg_aug=False):
    return ("probe_" if data == "variety" else f"{data}_probe_") + model + ("_bgaug" if bg_aug else "")


def fit_probe(model, data, bg_aug=False):
    run = run_name(data, model, bg_aug)
    path = config.MODEL_DIR / f"{run}_probe.joblib"
    if path.exists():
        return joblib.load(path)
    xt, yt, _, class_names = embeddings(model, data, "train")
    xv, yv, _, _ = embeddings(model, data, "val")
    vals = [(xv, yv)]
    if bg_aug:
        xt = np.concatenate([xt, embeddings(model, data, "train", True)[0]])
        yt = np.concatenate([yt, yt])
        vals.append((embeddings(model, data, "val", True)[0], yv))
    n, v, d = xt.shape
    best = None
    for C in C_GRID:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
        clf.fit(xt.reshape(n * v, d), np.repeat(yt, v))
        scores = [(nll(p, y), (p.argmax(1) == y).mean()) for p, y in ((predict(clf, x), y) for x, y in vals)]
        score = float(np.mean([l for l, _ in scores]))
        print(f"  {run}: C={C:<6} val_loss={score:.4f} val_acc={' / '.join(f'{a:.4f}' for _, a in scores)}")
        if best is None or score < best[0]:
            best = (score, C, clf)
    loss, C, clf = best
    probe = {"clf": clf, "C": C, "class_names": class_names, "model": model, "timm": MODELS[model][0],
             "img_size": MODELS[model][1], "val_loss": loss, "bg_aug": bg_aug}
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(probe, path)
    meta = {k: v for k, v in probe.items() if k != "clf"}
    (config.MODEL_DIR / f"{run}_meta.json").write_text(json.dumps({"data": data, **meta}, indent=2))
    return probe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="variety", choices=sorted(config.DATASETS))
    ap.add_argument("--model", nargs="+", required=True, choices=sorted(MODELS))
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--bg-aug", action="store_true", help="also train on background-removed copies")
    ap.add_argument("--no-background", action="store_true", help="score the background-removed split")
    args = ap.parse_args()

    probs = []
    for model in args.model:
        probe = fit_probe(model, args.data, args.bg_aug)
        x, y, files, class_names = embeddings(model, args.data, args.split, args.no_background)
        assert class_names == probe["class_names"]
        probs.append(predict(probe["clf"], x))
    name = "+".join(run_name(args.data, m, args.bg_aug) for m in args.model) + ("_nobg" if args.no_background else "") + "_tta"
    write_metrics(name, args.split, y, np.mean(probs, 0), class_names, [Path(f).name for f in files])


if __name__ == "__main__":
    main()
