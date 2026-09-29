"""Generate notebooks/colab_train.ipynb: a self-contained notebook that recreates this
project on a Colab GPU runtime (source files are embedded via %%writefile),
downloads both datasets, runs a backbone sweep and packs the results.

Re-run after changing anything in src/:  python scripts/build_colab_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EMBED = sorted((ROOT / "src").glob("*.py")) + [ROOT / "scripts" / "download_zenodo.py"]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


cells = [
    md("""
# Date fruit classifier: GPU training sweep (Colab)

1. **Select Kernel → Colab → New Colab Server → L4 or A100 GPU** (T4 works but is slower; **not G4**: TensorFlow cannot run on its Blackwell GPU).
2. **Run All.** Roughly 1–1.5 h on an L4: data download ~5 min, 5 training runs, evaluation, explainability.
3. At the end, download **`/content/results.zip`** (Colab sidebar → Files, or the Colab extension's
   *Download* command) and unzip it into the local `date-fruit-quality-ml/` folder.

Model selection uses the **validation** split only. The test split is scored once at the end.
"""),
    code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip -q install pillow-heif
import os, subprocess, time, json, glob, itertools
import tensorflow as tf, keras

# Run a shell command and print its output (only lines containing one of `grep`, if given).
# check=True stops the notebook (Run All) when the command fails.
def sh(cmd, grep=None, check=False):
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    for line in (out.stdout + out.stderr).splitlines():
        if grep is None or any(g in line for g in grep):
            print(line)
    if check and out.returncode != 0:
        raise RuntimeError(f"command failed (exit {out.returncode}): {cmd}")
    return out.returncode

print("TF", tf.__version__, "| Keras", keras.__version__, "| GPUs:", tf.config.list_physical_devices("GPU"))
assert tf.config.list_physical_devices("GPU"), "No GPU: Runtime → change runtime type → GPU"
# TensorFlow's CUDA kernels don't cover every GPU generation (e.g. Blackwell "G4" fails with
# cuLaunchKernel errors). Run a real op on the GPU now instead of failing mid-sweep.
try:
    with tf.device("/GPU:0"):
        _ = tf.reduce_sum(tf.cast(tf.equal(tf.range(8), 3), tf.float32) + tf.random.normal((256, 256)) @ tf.random.normal((256, 256)))
    print("GPU smoke test OK:", tf.config.experimental.get_device_details(tf.config.list_physical_devices("GPU")[0]))
except Exception as e:
    raise RuntimeError("TensorFlow cannot run on this GPU. Disconnect and pick an L4 or A100 runtime "
                       "(Colab: Remove Server, then New Colab Server -> L4/A100).") from e
"""),
    md("## Project source (generated from `src/`, do not edit here)"),
    code("%mkdir -p /content/date-fruit-quality-ml/src /content/date-fruit-quality-ml/scripts\n"
         "%cd /content/date-fruit-quality-ml"),
]
for f in EMBED:
    rel = f.relative_to(ROOT)
    cells.append(code(f"%%writefile {rel}\n" + f.read_text()))

cells += [
    md("## Data: Zenodo 4639543 + Kaggle *Date Fruit Image Dataset in Controlled Environment*"),
    code("""
if not os.path.exists("data/kaggle/alhamdan/Ajwa"):
    !mkdir -p data/kaggle data/raw
    !curl -sL -o data/kaggle/alhamdan_dates.zip https://www.kaggle.com/api/v1/datasets/download/wadhasnalhamdan/date-fruit-image-dataset-in-controlled-environment
    assert os.path.getsize("data/kaggle/alhamdan_dates.zip") > 1e9, "Kaggle download failed: download the dataset zip manually and upload it to data/kaggle/alhamdan_dates.zip"
    !unzip -q -o data/kaggle/alhamdan_dates.zip -d data/kaggle/alhamdan && rm data/kaggle/alhamdan_dates.zip
# Zenodo rate-limits downloads; the script resumes, so retry a few passes until all 478 files are there.
for attempt in range(4):
    if sh("python scripts/download_zenodo.py", grep=["complete", "FAILED"]) == 0:
        break
    time.sleep(30)
else:
    raise RuntimeError("Zenodo download incomplete - re-run this cell")
assert len(os.listdir("data/kaggle/alhamdan")) >= 9, "Kaggle data missing"
"""),
    code('sh("python src/prepare_data.py", check=True)'),
    md("""
## Backbone sweep
Every run: frozen-backbone head training, then **whole-backbone** fine-tuning with cosine LR,
label smoothing 0.1, mixed precision. Best checkpoint by validation loss.
"""),
    code("""
RUNS = [
    # run name,          backbone,           img, fine-tune LR
    ("mnv3_224",         "mobilenetv3",      224, 1e-4),
    ("effv2b0_260",      "efficientnetv2b0", 260, 1e-4),
    ("effv2s_384",       "efficientnetv2s",  384, 5e-5),
    ("convnext_t_288",   "convnext_tiny",    288, 5e-5),
    ("convnext_s_288",   "convnext_small",   288, 3e-5),
]
os.makedirs("outputs/logs", exist_ok=True)
for run, bb, img, lr in RUNS:
    if os.path.exists(f"outputs/models/{run}_meta.json"):
        print("skip (done):", run); continue
    t = time.time()
    log = f"outputs/logs/train_{run}.log"
    sh(f"python src/train.py --backbone {bb} --img-size {img} --run {run} --ft-layers 0 --ft-lr {lr} "
       f"--label-smoothing 0.1 --head-epochs 15 --ft-epochs 40 --mixed-precision > {log} 2>&1")
    print(f"{run}: {(time.time()-t)/60:.1f} min")
    sh(f"tail -1 {log}", grep=["saved", "Error"])  # full log: outputs/logs/train_<run>.log
assert any(os.path.exists(f"outputs/models/{r}_meta.json") for r, *_ in RUNS), "every training run failed - see outputs/logs/"
"""),
    md("## Model selection on the validation split (single models and ensembles, with flip-TTA)"),
    code("""
done = [r for r, *_ in RUNS if os.path.exists(f"outputs/models/{r}_meta.json")]
def val_score(runs):
    name = "+".join(runs) + "_tta"
    path = f"outputs/eval/{name}_val_metrics.json"
    if not os.path.exists(path):
        sh(f"python src/evaluate.py --run {' '.join(runs)} --tta --split val", grep=["Error"])
    m = json.load(open(path))
    return (m["accuracy"], m["f1_macro"], -m["loss"]), m
scores = {(r,): val_score([r]) for r in done}
top = sorted(scores, key=lambda k: scores[k][0], reverse=True)[:3]
for k in (2, 3):
    for combo in itertools.combinations([t[0] for t in top], k):
        scores[combo] = val_score(list(combo))
print(f"{'candidate':55s} val_acc  val_f1  val_loss")
for k in sorted(scores, key=lambda k: scores[k][0], reverse=True):
    m = scores[k][1]
    print(f"{'+'.join(k):55s} {m['accuracy']:.4f}  {m['f1_macro']:.4f}  {m['loss']:.4f}")
best = max(scores, key=lambda k: (scores[k][0], -len(k)))  # tie -> fewer models
best_single = max((k for k in scores if len(k) == 1), key=lambda k: scores[k][0])
print("\\nSELECTED:", "+".join(best), "| best single:", best_single[0])
json.dump({"selected": list(best), "best_single": best_single[0]}, open("outputs/selection.json", "w"))
"""),
    md("## Final test-set scores (touched once)"),
    code("""
for runs in dict.fromkeys(["mnv3_224", best_single[0], "+".join(best)]):
    sh(f"python src/evaluate.py --run {runs.replace('+', ' ')} --tta --split test", grep=["] {", "by source", "Error"])
"""),
    md("## Explainability + feature profiles + example prediction for the best single model"),
    code("""
run = best_single[0]
sh(f"python src/explain.py --run {run} --per-class 1 > outputs/logs/explain.log 2>&1; tail -14 outputs/logs/explain.log")
sh("python src/features.py --class-profiles > outputs/logs/features.log 2>&1; tail -3 outputs/logs/features.log")
ex = sorted(glob.glob("data/processed/test/Al-ajwa/*.jpg"))[0]
sh(f'python src/predict.py "{ex}" --run {run} 2>&1 | grep -A12 predicted_variety')
"""),
    code("""
!cd /content/date-fruit-quality-ml && zip -qr /content/results.zip outputs data/splits.csv
!ls -lh /content/results.zip
try:  # optional copy to Drive if it was mounted (Colab extension: "Mount Google Drive")
    if os.path.isdir("/content/drive/MyDrive"):
        !cp /content/results.zip /content/drive/MyDrive/date_fruit_results.zip && echo "copied to Drive"
except Exception as e:
    print(e)
"""),
]

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = ROOT / "notebooks" / "colab_train.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({len(cells)} cells, {len(EMBED)} embedded files)")
