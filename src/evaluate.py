"""Evaluate trained model(s): loss, accuracy, F1, per-class report and confusion matrix.

  --tta          average predictions over 4 flips of each image
  --run a b ...  several runs = ensemble (averaged probabilities)
  --split val    score on validation instead of test (use this for model selection)

Usage: python src/evaluate.py --run mobilenetv3 [--tta] [--split test]
"""
import argparse
import json

import numpy as np
import tensorflow as tf

import config
from data import load_split
from metrics import write_metrics

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
    name = "+".join(args.run) + ("_tta" if args.tta else "")
    write_metrics(name, args.split, y_true, np.mean(probs, axis=0), class_names, files)


if __name__ == "__main__":
    main()
