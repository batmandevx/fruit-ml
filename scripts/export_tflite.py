"""Export trained Keras classifiers to TFLite (float16 weights) for mobile/edge use and
check that the TFLite model agrees with the Keras model on the test split.

Writes outputs/models/<run>.tflite and <run>_tflite.json (size, top-1 agreement,
max probability difference, CPU latency).

Usage: python scripts/export_tflite.py [--run effv2b0_260 grade_mnv3 ...] (default: config.PRODUCTION_RUNS)
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config  # noqa: E402
from data import load_split  # noqa: E402


def export(run, n_check=200):
    model = tf.keras.models.load_model(config.MODEL_DIR / f"{run}_best.keras")
    meta = json.loads((config.MODEL_DIR / f"{run}_meta.json").read_text())
    with tempfile.TemporaryDirectory() as d:
        model.export(d, format="tf_saved_model", verbose=False)
        conv = tf.lite.TFLiteConverter.from_saved_model(d)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.target_spec.supported_types = [tf.float16]
        blob = conv.convert()
    dest = config.MODEL_DIR / f"{run}.tflite"
    dest.write_bytes(blob)

    interp = tf.lite.Interpreter(model_content=blob)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    ds = load_split("test", meta["img_size"], batch_size=1, data_dir=config.DATASETS[meta.get("data", "variety")])
    agree, max_diff, times = [], 0.0, []
    for x, _ in ds.take(n_check):
        k = model(x, training=False).numpy()[0]
        interp.set_tensor(inp["index"], x.numpy().astype(np.float32))
        t0 = time.perf_counter()
        interp.invoke()
        times.append(time.perf_counter() - t0)
        t = interp.get_tensor(out["index"])[0]
        agree.append(k.argmax() == t.argmax())
        max_diff = max(max_diff, float(np.abs(k - t).max()))
    res = {"run": run, "tflite_mb": round(len(blob) / 1e6, 2),
           "keras_mb": round((config.MODEL_DIR / f"{run}_best.keras").stat().st_size / 1e6, 2),
           "n_checked": len(agree), "top1_agreement": round(float(np.mean(agree)), 4),
           "max_prob_diff": round(max_diff, 4), "cpu_latency_ms_median": round(1000 * float(np.median(times[5:])), 1)}
    (config.MODEL_DIR / f"{run}_tflite.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", default=[r for runs in config.PRODUCTION_RUNS.values() for r in runs])
    for r in ap.parse_args().run:
        export(r)
