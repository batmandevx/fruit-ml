"""Full fine-tuning of a modern timm backbone (DINOv3 / DINOv2 / ConvNeXt-V2) on the
Apple GPU (PyTorch MPS), for tasks where a frozen probe is not enough (quality grade:
the cues are small surface defects).

Recipe: AdamW, layer-wise learning-rate decay (lower layers move less), cosine schedule
with warm-up, label smoothing, class-balanced loss, fp16 autocast, flips / rotation /
colour-jitter augmentation. The best epoch by validation loss is kept.

  weights   outputs/models/<run>_ft.pt        metrics  outputs/eval/<run>_tta_<split>_metrics.json

Usage:
  python src/finetune.py --data grade --model dinov3_b --run grade_ft_dinov3_b
  python src/finetune.py --run grade_ft_dinov3_b --eval test        # score a trained run once
"""
import argparse
import json
import math
import time

import numpy as np
import timm
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from PIL import Image

import config
from metrics import nll, write_metrics
from probe import DEVICE, MODELS, list_split


class Images(torch.utils.data.Dataset):
    def __init__(self, files, labels, transform):
        self.files, self.labels, self.transform = files, labels, transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.transform(Image.open(self.files[i]).convert("RGB")), int(self.labels[i])


def transforms(size, mean, std, train):
    base = [T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC, antialias=True)]
    if train:
        base += [T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                 T.RandomAffine(degrees=45, translate=(0.05, 0.05), scale=(0.9, 1.15)),
                 T.ColorJitter(0.15, 0.15, 0.1, 0.02)]
    return T.Compose(base + [T.PILToTensor(), T.ToDtype(torch.float32, scale=True), T.Normalize(mean, std)])


def build(model, n_classes, drop_path):
    name, size = MODELS[model]
    kw = {"img_size": size} if "patch" in name else {}
    net = timm.create_model(name, pretrained=True, num_classes=n_classes, drop_path_rate=drop_path, **kw)
    return net, size, net.pretrained_cfg["mean"], net.pretrained_cfg["std"]


def param_groups(net, lr, decay, weight_decay):
    """Layer-wise LR decay: the head gets lr, block i of n gets lr * decay^(n - i), the stem the lowest."""
    blocks = getattr(net, "blocks", None) or getattr(net, "stages", None)
    n = len(blocks)
    groups = {}
    for pname, p in net.named_parameters():
        if not p.requires_grad:
            continue
        if pname.startswith(("blocks.", "stages.")):
            depth = int(pname.split(".")[1]) + 1
        elif pname.startswith(("head", "fc_norm", "norm.", "head_norm")):
            depth = n + 1
        else:  # patch embed, cls / register tokens, position embeddings, stem
            depth = 0
        no_wd = p.ndim == 1 or pname.endswith(".bias") or "token" in pname or "pos_embed" in pname
        key = (depth, no_wd)
        if key not in groups:
            groups[key] = {"params": [], "lr_scale": decay ** (n + 1 - depth), "weight_decay": 0.0 if no_wd else weight_decay}
        groups[key]["params"].append(p)
    return list(groups.values())


@torch.no_grad()
def predict(net, loader):
    """Probabilities averaged over the 4 flips (same TTA as evaluate.py)."""
    net.eval()
    out = []
    for x, _ in loader:
        x = x.to(DEVICE)
        with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "mps"):
            p = [F.softmax(net(v).float(), 1) for v in (x, x.flip(3), x.flip(2), x.flip(2).flip(3))]
        out.append(torch.stack(p).mean(0).cpu().numpy())
    return np.concatenate(out)


def loader(files, labels, tf, shuffle, batch_size):
    return torch.utils.data.DataLoader(Images(files, labels, tf), batch_size=batch_size, shuffle=shuffle,
                                       num_workers=8, persistent_workers=True, drop_last=shuffle)


def evaluate(run, split):
    meta = json.loads((config.MODEL_DIR / f"{run}_meta.json").read_text())
    net, size, mean, std = build(meta["model"], len(meta["class_names"]), 0.0)
    net.load_state_dict(torch.load(config.MODEL_DIR / f"{run}_ft.pt", map_location="cpu"))
    net.to(DEVICE)
    files, y, class_names = list_split(meta["data"], split)
    assert class_names == meta["class_names"]
    probs = predict(net, loader(files, y, transforms(size, mean, std, False), False, meta["batch_size"]))
    write_metrics(f"{run}_tta", split, y, probs, class_names, files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=sorted(config.DATASETS))
    ap.add_argument("--model", choices=sorted(MODELS))
    ap.add_argument("--run", required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4, help="head LR; lower layers get lr * decay^depth")
    ap.add_argument("--layer-decay", type=float, default=0.75)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--drop-path", type=float, default=0.1)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--eval", choices=["val", "test"], help="only score an already trained run")
    args = ap.parse_args()
    if args.eval:
        return evaluate(args.run, args.eval)

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    tr_files, tr_y, class_names = list_split(args.data, "train")
    va_files, va_y, va_classes = list_split(args.data, "val")
    assert va_classes == class_names
    net, size, mean, std = build(args.model, len(class_names), args.drop_path)
    net.to(DEVICE)
    train_dl = loader(tr_files, tr_y, transforms(size, mean, std, True), True, args.batch_size)
    val_dl = loader(va_files, va_y, transforms(size, mean, std, False), False, args.batch_size)

    counts = np.bincount(tr_y, minlength=len(class_names))
    class_w = torch.tensor(len(tr_y) / (len(class_names) * counts), dtype=torch.float32, device=DEVICE)
    opt = torch.optim.AdamW(param_groups(net, args.lr, args.layer_decay, args.weight_decay), lr=args.lr)
    steps, warm = args.epochs * len(train_dl), len(train_dl)
    lr_at = lambda s: args.lr * (s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (steps - warm))))
    scaler = torch.amp.GradScaler("mps") if DEVICE == "mps" else None
    ckpt = config.MODEL_DIR / f"{args.run}_ft.pt"
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    hist, best, bad, step = [], None, 0, 0
    print(f"run={args.run} model={args.model} img={size} classes={class_names} train={len(tr_y)} val={len(va_y)}")
    for epoch in range(args.epochs):
        net.train()
        t0, tot, correct, seen = time.time(), 0.0, 0, 0
        for x, y in train_dl:
            for g in opt.param_groups:
                g["lr"] = lr_at(step) * g["lr_scale"]
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "mps"):
                logits = net(x)
            loss = F.cross_entropy(logits.float(), y, weight=class_w, label_smoothing=args.label_smoothing)
            opt.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            step += 1
            tot += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
        pv = predict(net, val_dl)
        val_loss, val_acc = nll(pv, va_y), float((pv.argmax(1) == va_y).mean())
        hist.append({"epoch": epoch + 1, "loss": tot / seen, "accuracy": correct / seen, "val_loss": val_loss,
                     "val_accuracy": val_acc})
        improved = best is None or val_loss < best[0]
        print(f"epoch {epoch + 1}/{args.epochs} loss={tot / seen:.4f} acc={correct / seen:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} {time.time() - t0:.0f}s{' *' if improved else ''}",
              flush=True)
        if improved:
            best, bad = (val_loss, val_acc, epoch + 1), 0
            torch.save(net.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop")
                break

    meta = {"run": args.run, "data": args.data, "model": args.model, "timm": MODELS[args.model][0], "img_size": size,
            "class_names": class_names, "batch_size": args.batch_size, "args": vars(args), "best_val_loss": best[0],
            "best_val_accuracy": best[1], "best_epoch": best[2], "history": hist}
    (config.MODEL_DIR / f"{args.run}_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {ckpt}  best epoch {best[2]} val_acc={best[1]:.4f} val_loss={best[0]:.4f}")
    del net
    evaluate(args.run, "val")


if __name__ == "__main__":
    main()
