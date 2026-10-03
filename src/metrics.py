"""Classification metrics + confusion matrix for a set of predicted probabilities.

Framework-free, so the TensorFlow (evaluate.py) and PyTorch (probe.py) paths write
identical files. The probabilities are also saved (outputs/eval/probs/<name>_<split>.npz,
keyed by file name) so models from either framework can be ensembled afterwards.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix, f1_score

import config

EVAL_DIR = config.OUTPUT_DIR / "eval"
PROBS_DIR = EVAL_DIR / "probs"


def nll(probs, y):
    return float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-7, 1)).mean())


def write_metrics(name, split, y_true, probs, class_names, files):
    y_true, probs = np.asarray(y_true), np.asarray(probs)
    y_pred = probs.argmax(1)
    names = [Path(f).name for f in files]
    sources = np.array([n.split("__")[0] for n in names])
    metrics = {
        "name": name,
        "split": split,
        "n": int(len(y_true)),
        "loss": nll(probs, y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "top2_accuracy": float(np.mean([t in p.argsort()[-2:] for t, p in zip(y_true, probs)])),
        # Filename prefix before "__" = photo source (variety data) or variety (grade/maturity data).
        "accuracy_by_source": {
            str(s): float(np.mean((y_pred == y_true)[sources == s])) for s in np.unique(sources)
        },
        "per_class": classification_report(y_true, y_pred, labels=range(len(class_names)), target_names=class_names,
                                           output_dict=True, zero_division=0),
        "misclassified": [
            {"file": n, "true": class_names[t], "pred": class_names[p], "conf": round(float(pr[p]), 3)}
            for n, t, p, pr in zip(names, y_true, y_pred, probs) if t != p
        ],
    }
    print(classification_report(y_true, y_pred, labels=range(len(class_names)), target_names=class_names, digits=3,
                                zero_division=0))
    print(f"[{name} | {split}]", {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)})
    print("accuracy by source:", {k: round(v, 4) for k, v in metrics["accuracy_by_source"].items()})

    PROBS_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / f"{name}_{split}_metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez(PROBS_DIR / f"{name}_{split}.npz", probs=probs, y=y_true, files=np.array(names),
             class_names=np.array(class_names))
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    size = max(8, 0.5 * len(class_names))
    fig, ax = plt.subplots(figsize=(size + 1, size))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title(f"{split} confusion matrix ({name})\nacc {metrics['accuracy']:.3f}, macro-F1 {metrics['f1_macro']:.3f}")
    fig.tight_layout()
    fig.savefig(EVAL_DIR / f"{name}_{split}_confusion_matrix.png", dpi=120)
    plt.close(fig)
    return metrics


def load_probs(name, split):
    """Saved probabilities of a model/ensemble, sorted by file name."""
    d = np.load(PROBS_DIR / f"{name}_{split}.npz")
    order = np.argsort(d["files"])
    return d["probs"][order], d["y"][order], list(d["files"][order]), list(d["class_names"])
