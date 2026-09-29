# Date Fruit Variety Classification & Quality Grading

Computer-vision pipeline for a date-fruit dealer, in two phases:

1. **Variety classification.** A transfer-learned CNN recognises 10 date varieties, explains its decision with
   Grad-CAM, and reports the colour, shape and texture of the fruit.
2. **Quality grading.** From one photo of a fruit, a tray or an open bag, it estimates for each fruit the quality grade,
   size class, ripeness stage, colour uniformity, surface defects, skin condition and weight.

📄 **Full evaluation report:** [`reports/model_report.pdf`](reports/model_report.pdf) (or [`.html`](reports/model_report.html)).
It covers every metric, confusion matrix, calibration curve and Grad-CAM check.

## Results (held-out test sets)

| Component | Model | Accuracy | Macro-F1 | Loss | With manual review of low-confidence photos |
|---|---|---|---|---|---|
| Variety (10 classes) | EfficientNetV2-B0 | **97.5%** | 0.972 | 0.196 | **99.2%** on the 95% auto-accepted |
| Quality grade 1–3 | MobileNetV3 + EfficientNetV2-B0 ensemble | **84.5%** | 0.813 | 0.371 | **97.6%** on the 47% auto-accepted |
| Size class | Random forest on fruit outline | **80.3%** | – | – | – |
| Ripeness stage | EfficientNetV2-B0 | **67.8%** on an unseen variety · **100%** on single studio fruits | 0.667 | 0.881 | not reliable on new varieties |

- Model choice, calibration temperature and review thresholds were fixed on **validation** splits. Each test set was scored once.
- The ripeness test set is a whole variety the model never saw. Each ripeness stage was photographed in its own session,
  so a random split would leak the session's lighting into the test set.
- Colour uniformity, defects, skin condition and weight are measurement rules. No labelled data exists for them yet
  (see [Limitations](#limitations)).

## What it outputs

`python src/grade.py photo.jpg` → `outputs/quality/reports/photo_quality.{json,png}`. For each fruit:

| Field | How |
|---|---|
| `variety` | CNN, with calibrated `confidence` and a `needs_review` flag |
| `grade` (Grade-1/2/3) | CNN ensemble + Grad-CAM heatmap |
| `maturity` (Immature/Khalal/Rutab/Tamar) | CNN; `maturity_by_colour` gives a colour-based second opinion |
| `size` (Small/Medium/Large) | Outline geometry. In mm when a printed ArUco marker is in the photo |
| `color_uniformity` | CIELAB ΔE spread of the skin, dominant colours |
| `surface_defects` | Cracks, black spots, insect holes, sunburn, mold: area % and count |
| `skin` | Wrinkling, gloss, sugar spots |
| `estimated_weight_g` | Ellipsoid volume × density (needs mm scale); `--bag-weight` recalibrates from a scale reading |

The summary gives fruit count, counts per size, grade and ripeness stage, total estimated weight, and which fruits need review.

## Repository layout

```
src/                     library + CLI entry points (run from the repo root)
  config.py              paths, class lists, hyper-parameters, PRODUCTION_RUNS (models used in production)
  prepare_data.py        Phase 1 dataset: map/crop/resize, stratified 70/15/15 split
  prepare_quality.py     Phase 2 datasets: grade + maturity splits, fruit measurements
  data.py, model.py      tf.data pipeline with augmentation; backbone + head model
  train.py               frozen-backbone head training, then fine-tuning   (--data variety|grade|maturity)
  evaluate.py            accuracy, F1, loss, confusion matrix, TTA, ensembles
  calibrate.py           temperature scaling + manual-review threshold
  explain.py             Grad-CAM, Grad-CAM++, Guided Grad-CAM + sanity checks
  features.py            fruit segmentation; colour / shape / texture descriptors
  quality.py             size, colour uniformity, defects, skin, maturity-by-colour, weight
  predict.py             Phase 1: variety + heatmap + colour/shape/texture for one photo
  grade.py               Phase 2: full quality report for a photo (Python API + CLI)
scripts/
  download_zenodo.py     Phase 1 Zenodo images
  download_maturity.py   Phase 2 maturity crops from Zenodo (HTTP range requests, ~125 MB instead of 40 GB)
  export_tflite.py       TFLite float16 export + parity check against Keras
  build_report.py        reports/model_report.{html,pdf} from the metric files
  build_colab_notebook.py  regenerates notebooks/colab_train.ipynb
tests/                   pytest suite (geometry, splitting, scale, defects, end-to-end, bad input)
notebooks/               self-contained Colab GPU training notebook
reports/                 evaluation report (HTML + PDF)
outputs/
  models/                production models (.keras + .tflite), metadata, calibration
  eval/                  metrics JSON + confusion matrices for every evaluated model
  explainability/        Grad-CAM galleries + sanity-check results
  quality/               size model, defect-vs-grade check
  training_curves/       loss/accuracy curves
  class_feature_profiles.csv   colour/shape/texture per variety
data/                    datasets (not in git; see below)
```

## Quick start

Python 3.12. On Apple Silicon the lock file installs TensorFlow 2.18 with the Metal GPU plugin:

```bash
uv venv --python 3.12 .venv-metal
uv pip install --python .venv-metal/bin/python -r requirements-lock.txt   # macOS arm64, exact versions
# other platforms: uv pip install -r requirements.txt   (TensorFlow 2.16–2.20)
source .venv-metal/bin/activate
```

The production models are in `outputs/models/`, so inference works right after cloning:

```bash
python src/grade.py photo.jpg                          # quality report for every fruit in the photo
python src/grade.py tray.jpg --bag-weight 250          # + recalibrate weight from the scale reading (g)
python src/grade.py fruit.jpg --variety Gajar          # skip variety model (e.g. for grading-set varieties)
python src/predict.py fruit.jpg                        # Phase 1: variety, Grad-CAM, colour/shape/texture
python -m pytest tests -q                              # 16 tests
```

```python
import sys; sys.path.insert(0, "src")
from grade import load_models, grade_image

models = load_models()                       # loads config.PRODUCTION_RUNS once
report = grade_image("photo.jpg", models)    # dict; raises ValueError for unusable input
```

**For sizes in mm and weight,** print a 30 mm ArUco marker (`DICT_4X4_50`, any id) and place it next to the fruit.
It's detected automatically, or you can pass `--mm-per-px`. Without a marker, size uses pixel models that are only valid for the
grading-set camera rig, and weight is not estimated.

## Reproducing everything

### 1. Data (≈8.5 GB, not in git)

| Dataset | Used for | Put it in |
|---|---|---|
| [Zenodo 4639543](https://zenodo.org/records/4639543): date fruit images | Phase 1 varieties (web photos) | `python scripts/download_zenodo.py` → `data/raw/` |
| [Date Fruit Image Dataset in Controlled Environment](https://www.kaggle.com/datasets/wadhasnalhamdan/date-fruit-image-dataset-in-controlled-environment) (Kaggle) | Phase 1 varieties (studio photos); Rutab/Tamar single fruits | unzip so that `data/kaggle/alhamdan/<Variety>/` exists |
| [Date Fruit Dataset for Inspection and Grading](https://data.mendeley.com/datasets/s5zfvsw5kv/3) (Mendeley, [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10801328/)) | Phase 2 grade + size | unzip `Date Fruit/*` so that `data/kaggle/grading/<Variety>/<Size>/<Grade>/` exists |
| [Zenodo 10143465](https://zenodo.org/records/10143465): Moroccan date maturity dataset | Phase 2 ripeness | `python scripts/download_maturity.py` → `data/maturity/crops/` |

The Zenodo 4639543 record describes 27 classes / 2,000+ images, but only 478 images of about 10 varieties are published.
That is why the Kaggle studio set was added.

### 2. Train, evaluate, report

```bash
# Phase 1: variety
python src/prepare_data.py
python src/train.py --backbone efficientnetv2b0 --img-size 260 --run effv2b0_260 --ft-layers 0 --ft-lr 1e-4 \
    --label-smoothing 0.1 --head-epochs 15 --ft-epochs 40
python src/features.py --class-profiles

# Phase 2: grade, maturity, size
python src/prepare_quality.py
for spec in "grade 40" "maturity 30"; do             # dataset, fine-tune epochs
  set -- ${=spec}                                    # zsh; in bash: set -- $spec
  python src/train.py --data $1 --run $1_mnv3 --ft-layers 0 --ft-lr 1e-4 --label-smoothing 0.1 --head-epochs 15 --ft-epochs $2
  python src/train.py --data $1 --backbone efficientnetv2b0 --run $1_effv2b0 --ft-layers 0 --ft-lr 1e-4 \
      --label-smoothing 0.1 --head-epochs 15 --ft-epochs $2
done
python src/quality.py --fit-size
python src/quality.py --check-defects

# Select on validation, then score test once, calibrate, explain, export
python src/evaluate.py --run grade_mnv3 grade_effv2b0 --tta --split val     # compare candidates / ensembles
python src/evaluate.py --run grade_mnv3 grade_effv2b0 --tta --split test
python src/calibrate.py --run grade_mnv3 grade_effv2b0 --target 0.99
python src/explain.py --run grade_mnv3
python scripts/export_tflite.py
python scripts/build_report.py
```

Set `PRODUCTION_RUNS` in `src/config.py` to the models you selected. On an Apple M-series GPU, one EfficientNetV2-B0 run
takes about 40–80 minutes. `notebooks/colab_train.ipynb` runs the Phase 1 backbone sweep on a Colab GPU.

## Method notes

- **Transfer learning:** ImageNet backbone frozen while a new head trains, then the whole backbone is fine-tuned with AdamW,
  a cosine schedule and label smoothing 0.1. Keras has no EfficientNet-Lite, so EfficientNetV2-B0 and MobileNetV3 are used.
- **Explainability:** Grad-CAM, Grad-CAM++ and Guided Grad-CAM (Grad-CAM × SmoothGrad; MobileNetV3's hard-swish rules out
  classic guided backprop). Each has three sanity checks: a model-randomisation test, a deletion test and fruit focus.
  The variety model, the ripeness model and the grade ensemble's MobileNetV3 member pass all three.
- **Calibration:** temperature scaling on validation. Predictions under the validation-chosen threshold are flagged `needs_review`.
- **Fruit splitting:** a distance-transform watershed splits blobs only where the outline narrows (a "neck"), so touching fruits
  are separated but a single elongated date is not. All 451 grading test images are counted as exactly one fruit.
- **TFLite:** float16 models are 6–12 MB, agree 100% with Keras on 200 test images each, and take 14–70 ms per image on CPU.

## Limitations

- **No dealer photos yet.** All results are on public datasets. Barhi, Mabroum, Khasab, Helwa, Maktoumi, Bowytha, Kelas and
  Rashudia (from the spec's list) are not in any public dataset.
- **Ripeness:** public data has only orchard bunches with one label per photo session, so a "Rutab" bunch still holds
  Khalal-coloured fruit. Grad-CAM shows the model looks at the fruit, not at the protective bags. On a new variety its
  confidence is not reliable, so it can't route fruits to review.
- **Defects, skin and weight** are transparent rules without labelled data. Defect area rises with grade on Gajar
  (Spearman +0.53) but only weakly on Kupro. Mold is the least reliable. The weight coefficient and mm size cut-offs are
  untested against weighed fruit.
- **Grade and size labels are subjective.** Most grade errors are between Grade 1 and 2. Gajar Small and Medium overlap in size.

**To reach field accuracy:**
- dealer photos of each variety (one fixed setup)
- about 100 single fruits per ripeness stage
- a few hundred photos with defects marked
- 20–30 fruits weighed on a scale and photographed with the marker
