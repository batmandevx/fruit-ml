"""Average the saved probabilities of models from either framework (evaluate.py or
probe.py outputs in outputs/eval/probs/) and score the ensemble.

Runs are named as in evaluate.py / probe.py (their "<run>_tta" outputs are read); the
ensemble is written as "<run1>+<run2>_tta", the name evaluate.py uses for Keras ensembles.

Usage: python src/ensemble.py --run probe_dinov3_b effv2b0_260_v2 --split val [--weights 0.7 0.3]
"""
import argparse

import numpy as np

from metrics import load_probs, write_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--weights", nargs="+", type=float)
    args = ap.parse_args()
    weights = np.array(args.weights or [1.0] * len(args.run))
    assert len(weights) == len(args.run)

    probs, ref = [], None
    for n in args.run:
        p, y, files, classes = load_probs(f"{n}_tta", args.split)
        assert ref is None or (files == ref[1] and classes == ref[2]), f"{n}: different images or classes"
        ref = (y, files, classes)
        probs.append(p)
    avg = np.tensordot(weights / weights.sum(), np.array(probs), axes=1)
    tag = "" if args.weights is None else "_w" + "-".join(f"{w:g}" for w in weights)
    write_metrics("+".join(args.run) + tag + "_tta", args.split, ref[0], avg, ref[2], ref[1])


if __name__ == "__main__":
    main()
