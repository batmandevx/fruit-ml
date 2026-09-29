"""Evaluate trained model(s): loss, accuracy, F1, per-class report and confusion matrix.

  --tta          average predictions over 4 flips of each image
  --run a b ...  several runs = ensemble (averaged probabilities)
  --split val    score on validation instead of test (use this for model selection)

Usage: python src/evaluate.py --run mobilenetv3 [--tta] [--split test]
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix, f1_score

import config
from data import load_split

FLIPS = [lambda x: x, lambda x: tf.reverse(x, [2]), lambda x: tf.reverse(x, [1]), lambda x: tf.reverse(x, [1, 2])]


def load_run(run):
    model = tf.keras.models.load_model(config.MODEL_DIR / f"{run}_best.keras")
    meta = json.loads((config.MODEL_DIR / f"{run}_meta.json").read_text())
    return model, meta


def predict_probs(model, ds, tta):
    views = FLIPS if tta else FLIPS[:1]
    return np.mean([model.predict(ds.map(lambda x, y, f=f: (f(x), y)), verbose=0) for f in views], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", default=[config.BACKBONE])
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    args = ap.parse_args()

    probs, class_names, y_true, files = [], None, None, None
    for run in args.run:
        model, meta = load_run(run)
        ds = load_split(args.split, meta["img_size"], data_dir=config.DATASETS[meta.get("data", "variety")])
        assert class_names is None or meta["class_names"] == class_names
        class_names, files = meta["class_names"], ds.file_paths
        y_true = np.concatenate([y.numpy() for _, y in ds])
        probs.append(predict_probs(model, ds, args.tta))
        tf.keras.backend.clear_session()
    probs = np.mean(probs, axis=0)
    y_pred = probs.argmax(1)
    sources = np.array([Path(p).name.split("__")[0] for p in files])
    name = "+".join(args.run) + ("_tta" if args.tta else "")

    metrics = {
        "name": name,
        "split": args.split,
        "n": int(len(y_true)),
        "loss": float(tf.keras.losses.SparseCategoricalCrossentropy()(y_true, probs)),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "top2_accuracy": float(np.mean([t in p.argsort()[-2:] for t, p in zip(y_true, probs)])),
        # Filename prefix before "__" = photo source (variety data) or variety (grade/maturity data).
        "accuracy_by_source": {
            str(s): float(np.mean((y_pred == y_true)[sources == s])) for s in np.unique(sources)
        },
        "per_class": classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0),
        "misclassified": [
            {"file": Path(f).name, "true": class_names[t], "pred": class_names[p], "conf": round(float(pr[p]), 3)}
            for f, t, p, pr in zip(files, y_true, y_pred, probs) if t != p
        ],
    }
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3, zero_division=0))
    print(f"[{name} | {args.split}]", {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)})
    print("accuracy by source:", {k: round(v, 4) for k, v in metrics["accuracy_by_source"].items()})

    out = config.OUTPUT_DIR / "eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}_{args.split}_metrics.json").write_text(json.dumps(metrics, indent=2))
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title(f"{args.split} confusion matrix ({name})\nacc {metrics['accuracy']:.3f}, macro-F1 {metrics['f1_macro']:.3f}")
    fig.tight_layout()
    fig.savefig(out / f"{name}_{args.split}_confusion_matrix.png", dpi=120)


if __name__ == "__main__":
    main()
