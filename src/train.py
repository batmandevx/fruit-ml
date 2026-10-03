"""Two-stage transfer learning: (1) frozen backbone, train head; (2) fine-tune
the backbone (all or the top N layers) with a cosine learning-rate schedule.
The best checkpoint by validation loss across both stages is kept.

Usage:
  python src/train.py --backbone mobilenetv3
  python src/train.py --data grade --run grade_mnv3      # Phase 2 quality grade
  python src/train.py --backbone efficientnetv2s --img-size 384 --ft-layers 0 --ft-lr 1e-4 --run effv2s_384
  python src/train.py --backbone efficientnetv2b0 --img-size 260 --init-from effv2b0_260 --run effv2b0_260_v2
      # start from a trained run's backbone (new head, so the class list may differ)
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

import config
from data import get_datasets
from model import BACKBONES, build_model, unfreeze_top


def plot_history(hist, split_epoch, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key in zip(axes, ["loss", "accuracy"]):
        ax.plot(hist[key], label=f"train {key}")
        ax.plot(hist[f"val_{key}"], label=f"val {key}")
        ax.axvline(split_epoch - 0.5, color="gray", ls="--", label="start fine-tune")
        ax.set_xlabel("epoch")
        ax.set_title(key)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="variety", choices=sorted(config.DATASETS))
    ap.add_argument("--backbone", default=config.BACKBONE, choices=sorted(BACKBONES))
    ap.add_argument("--img-size", type=int, help="default: the backbone's native size")
    ap.add_argument("--run", help="artifact name (default: backbone name)")
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--head-epochs", type=int, default=config.HEAD_EPOCHS)
    ap.add_argument("--ft-epochs", type=int, default=config.FINETUNE_EPOCHS)
    ap.add_argument("--ft-layers", type=int, default=config.FINETUNE_LAST_N_LAYERS, help="0 = whole backbone")
    ap.add_argument("--ft-lr", type=float, default=config.FINETUNE_LR)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--init-from", help="run whose fine-tuned backbone weights to start from (same backbone and size)")
    ap.add_argument("--mixed-precision", action="store_true", help="float16 compute (GPU only)")
    args = ap.parse_args()
    img_size = args.img_size or BACKBONES[args.backbone][1]
    run = args.run or (args.backbone if args.data == "variety" else f"{args.data}_{args.backbone}")

    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    tf.keras.utils.set_random_seed(config.SEED)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train_ds, val_ds, _, class_names = get_datasets(img_size, args.batch_size, one_hot=True,
                                                        data_dir=config.DATASETS[args.data])
    labels = np.concatenate([y.numpy().argmax(1) for _, y in train_ds])
    weights = compute_class_weight("balanced", classes=np.arange(len(class_names)), y=labels)
    class_weight = dict(enumerate(weights))
    print(f"run={run} backbone={args.backbone} img={img_size} classes={class_names}")

    model = build_model(len(class_names), args.backbone, img_size)
    if args.init_from:
        src = tf.keras.models.load_model(config.MODEL_DIR / f"{args.init_from}_best.keras")
        model.get_layer("backbone").set_weights(src.get_layer("backbone").get_weights())
        print(f"backbone initialised from {args.init_from}")
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing)
    ckpt = config.MODEL_DIR / f"{run}_best.keras"

    print("\n=== Stage 1: frozen backbone ===")
    model.compile(tf.keras.optimizers.Adam(config.HEAD_LR), loss, metrics=["accuracy"])
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.head_epochs, class_weight=class_weight,
                   callbacks=[
                       tf.keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True),
                       tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
                       tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-6),
                   ])
    best_stage1 = min(h1.history["val_loss"])

    what = "whole backbone" if args.ft_layers <= 0 else f"last {args.ft_layers} backbone layers"
    print(f"\n=== Stage 2: fine-tune {what} ===")
    unfreeze_top(model, args.ft_layers)
    steps = args.ft_epochs * len(train_ds)
    schedule = tf.keras.optimizers.schedules.CosineDecay(
        0.0, steps, warmup_target=args.ft_lr, warmup_steps=max(1, int(0.1 * steps)), alpha=0.01)
    model.compile(tf.keras.optimizers.AdamW(schedule, weight_decay=1e-4), loss, metrics=["accuracy"])
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=args.ft_epochs, class_weight=class_weight,
                   callbacks=[
                       # only overwrite the stage-1 checkpoint if fine-tuning actually beats it
                       tf.keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True,
                                                          initial_value_threshold=best_stage1),
                       tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10),
                   ])

    best = tf.keras.models.load_model(ckpt)
    val_loss, val_acc = best.evaluate(val_ds, verbose=0)
    hist = {k: h1.history[k] + h2.history[k] for k in ("loss", "accuracy", "val_loss", "val_accuracy")}
    meta = {"run": run, "data": args.data, "backbone": args.backbone, "img_size": img_size, "class_names": class_names,
            "args": vars(args), "best_val_loss": val_loss, "best_val_accuracy": val_acc,
            "stage1_epochs": len(h1.history["loss"]), "history": hist}
    (config.MODEL_DIR / f"{run}_meta.json").write_text(json.dumps(meta, indent=2))
    (config.OUTPUT_DIR / "training_curves").mkdir(parents=True, exist_ok=True)
    plot_history(hist, len(h1.history["loss"]), config.OUTPUT_DIR / "training_curves" / f"{run}_training_curves.png")
    print(f"saved {ckpt}  best val_acc={val_acc:.4f} val_loss={val_loss:.4f}")


if __name__ == "__main__":
    main()
