"""Confidence calibration and a manual-review threshold for a trained classifier.

1. Temperature scaling (Guo et al., 2017): one scalar T fitted on the validation
   split so that softmax(log p / T) is calibrated. Reported as ECE before/after.
2. Selective prediction: the lowest calibrated-confidence threshold whose
   accuracy on the *validation* predictions it accepts reaches --target. Test
   accuracy on accepted images and the share sent to manual review ("coverage")
   are then measured once on the test split.

T and the threshold are written to outputs/models/<name>_calib.json (name = runs
joined by "+"); grade.py reads them. Several runs are calibrated as an ensemble.

Usage: python src/calibrate.py --run grade_mnv3 grade_effv2b0 [--target 0.98]
"""
import argparse
import json

import numpy as np
import tensorflow as tf
from scipy.optimize import minimize_scalar

import config
from data import load_split
from evaluate import load_run, predict_probs


def scale(probs, T):
    z = np.log(np.clip(probs, 1e-12, 1)) / T
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


def nll(probs, y):
    return float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1)).mean())


def ece(probs, y, bins=10):
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


def coverage_curve(probs, y):
    conf, ok = probs.max(1), probs.argmax(1) == y
    rows = []
    for t in [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98]:
        m = conf >= t
        rows.append({"threshold": t, "coverage": round(float(m.mean()), 3),
                     "accuracy": round(float(ok[m].mean()), 4) if m.any() else None})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--target", type=float, default=0.98, help="accuracy wanted on auto-accepted predictions")
    args = ap.parse_args()

    name = config.run_name(args.run)
    out = {"val": [[], None], "test": [[], None]}
    for run in args.run:
        model, meta = load_run(run)
        for split in ("val", "test"):
            ds = load_split(split, meta["img_size"], data_dir=config.DATASETS[meta.get("data", "variety")])
            out[split][1] = np.concatenate([b.numpy() for _, b in ds])
            out[split][0].append(predict_probs(model, ds, tta=True))
        tf.keras.backend.clear_session()
    out = {s: (np.mean(p, axis=0), y) for s, (p, y) in out.items()}

    pv, yv = out["val"]
    T = float(minimize_scalar(lambda t: nll(scale(pv, t), yv), bounds=(0.05, 10), method="bounded").x)
    cal = {s: scale(p, T) for s, (p, _) in out.items()}

    # lowest threshold reaching the target on val (more coverage = less manual work)
    conf_v, ok_v = cal["val"].max(1), cal["val"].argmax(1) == yv
    threshold = None
    for t in np.unique(conf_v):
        m = conf_v >= t
        if ok_v[m].mean() >= args.target:
            threshold = float(t)
            break
    pt, yt = cal["test"], out["test"][1]
    acc_t = pt.argmax(1) == yt
    accepted = pt.max(1) >= threshold if threshold is not None else np.zeros(len(yt), bool)

    report = {
        "run": name,
        "temperature": round(T, 4),
        "val": {"nll_before": round(nll(pv, yv), 4), "nll_after": round(nll(cal["val"], yv), 4),
                "ece_before": round(ece(pv, yv), 4), "ece_after": round(ece(cal["val"], yv), 4)},
        "test": {"accuracy": round(float(acc_t.mean()), 4),
                 "nll_before": round(nll(out["test"][0], yt), 4), "nll_after": round(nll(pt, yt), 4),
                 "ece_before": round(ece(out["test"][0], yt), 4), "ece_after": round(ece(pt, yt), 4)},
        "review": {
            "target_accuracy": args.target,
            "threshold": None if threshold is None else round(threshold, 4),
            "test_coverage": round(float(accepted.mean()), 4),
            "test_accuracy_on_accepted": round(float(acc_t[accepted].mean()), 4) if accepted.any() else None,
            "test_sent_to_review": int((~accepted).sum()),
            "n_test": int(len(yt)),
        },
        "test_coverage_curve": coverage_curve(pt, yt),
    }
    (config.MODEL_DIR / f"{name}_calib.json").write_text(json.dumps(
        {"runs": args.run, "temperature": report["temperature"], "review_threshold": report["review"]["threshold"],
         "target_accuracy": args.target}, indent=2))
    dest = config.OUTPUT_DIR / "eval" / f"{name}_calibration.json"
    dest.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "test_coverage_curve"}, indent=2))


if __name__ == "__main__":
    main()
