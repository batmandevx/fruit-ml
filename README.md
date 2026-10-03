# Date Fruit Variety Classification & Quality Grading

Computer-vision pipeline for a date-fruit dealer, in two phases:

1. **Variety classification.** An ensemble of a DINOv3 foundation-model probe and a fine-tuned CNN recognises 17 date
   varieties, explains its decision with Grad-CAM, and reports the colour, shape and texture of the fruit.
2. **Quality grading.** From one photo of a fruit, a tray or an open bag, it estimates for each fruit the quality grade,
   size class, ripeness stage, colour uniformity, surface defects, skin condition and weight.

📄 **Full evaluation report:** [`reports/model_report.pdf`](reports/model_report.pdf) (or [`.html`](reports/model_report.html)).
It covers every metric, confusion matrix, calibration curve and Grad-CAM check.

## Results (held-out test sets)

| Component | Model | Accuracy | Macro-F1 | Loss | With manual review of low-confidence photos |
|---|---|---|---|---|---|
| Variety (17 classes) | DINOv3 ViT-B probe + EfficientNetV2-B0 ensemble | **99.5%** | 0.994 | 0.102 | **99.5%** on the 99.8% auto-accepted |
| Quality grade 1–3 | MobileNetV3 + EfficientNetV2-B0 ensemble | **84.5%** | 0.813 | 0.371 | **97.6%** on the 47% auto-accepted |
| Size class | Random forest on fruit outline | **80.3%** | – | – | – |
| Ripeness stage | DINOv3 ViT-B probe + EfficientNetV2-B0 ensemble | **70.0%** on an unseen variety · **100%** on single studio fruits | 0.696 | 0.727 | not reliable on new varieties |

- Model choice, calibration temperature and review thresholds were fixed on **validation** splits. Each test set was scored once.
- **Variety:** 17 classes: the earlier 10 plus Al-masyihia, Al-shagra, SagaiIRAQ and the four grading-set varieties
  (Aseel, Fasli Toto, Gajar, Kupro). The ensemble makes 2 errors on 411 test photos; the EfficientNetV2-B0 alone scored
  96.1%. The DINOv3 member is a frozen self-supervised ViT with a logistic-regression head (`src/probe.py`).
- **Background shortcut, found and fixed:** the first DINOv3 probe dropped from 99% to 82% when the background was removed
  (the CNN: 97% → 96%), because the frozen backbone also encodes each variety's photo session. Its Grad-CAM showed heat on the
  background. The production probe is trained on every image twice, as photographed and with the background removed
  (`--bg-aug`): 99.3% on test photos, **98.8% with the background removed**.
- **Grade:** DINOv3 probes (80–83% on validation) and a full DINOv3-B fine-tune (87.4%) did not beat the CNN ensemble
  (89.8%), so it is unchanged. Three model families stopping at 87–90% points to the subjective Grade 1/2 boundary.
- **Ripeness:** the ensemble improves the unseen-variety test from 67.8% to 70.0%. A leave-one-variety-out check in
  train+val gives 55–76% per held-out variety, so a stronger backbone does not fix new-variety ripeness; per-fruit
  stage labels would.
- The ripeness test set is a whole variety the model never saw. Each ripeness stage was photographed in its own session,
  so a random split would leak the session's lighting into the test set.
- Colour uniformity, defects, skin condition and weight are measurement rules. No labelled data exists for them yet
  (see [Limitations](#limitations)).

## What it outputs

`python src/grade.py photo.jpg` → `outputs/quality/reports/photo_quality.{json,png}`. For each fruit:

| Field | How |
|---|---|
| `variety` | DINOv3 probe + CNN ensemble, with calibrated `confidence` and a `needs_review` flag |
| `grade` (Grade-1/2/3) | CNN ensemble + Grad-CAM heatmap |
| `maturity` (Immature/Khalal/Rutab/Tamar) | DINOv3 probe + CNN ensemble; `maturity_by_colour` gives a colour-based second opinion |
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
  evaluate.py            accuracy, F1, loss, confusion matrix, TTA, ensembles (Keras models)
  probe.py               PyTorch/timm foundation-model probes (DINOv3, DINOv2, ConvNeXt-V2, SigLIP) on the Apple GPU
  finetune.py            PyTorch full fine-tuning of those backbones (layer-wise LR decay, fp16)
  ensemble.py            ensembles across frameworks from saved probabilities
  metrics.py             shared metrics / confusion matrix / saved probabilities
  calibrate.py           temperature scaling + manual-review threshold
  explain.py             Grad-CAM, Grad-CAM++, Guided Grad-CAM + sanity checks (Keras models)
  explain_vit.py         the same maps and checks for the ViT probes (Grad-CAM on patch tokens)
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
  models/                production models (.keras + .tflite, probe .joblib), metadata, calibration
  features/              cached foundation-model embeddings (not in git)
  eval/                  metrics JSON + confusion matrices for every evaluated model
  explainability/        Grad-CAM galleries + sanity-check results
  quality/               size model, defect-vs-grade check
  training_curves/       loss/accuracy curves
  class_feature_profiles.csv   colour/shape/texture per variety
data/                    datasets (not in git; see below)
```

## Quick start

Python 3.12. On Apple Silicon the lock file installs TensorFlow 2.18 with the Metal GPU plugin and PyTorch 2.14 (MPS):

```bash
uv venv --python 3.12 .venv-metal
uv pip install --python .venv-metal/bin/python -r requirements-lock.txt   # macOS arm64, exact versions
# other platforms: uv pip install -r requirements.txt   (TensorFlow 2.16–2.20)
source .venv-metal/bin/activate
```

The production models are in `outputs/models/`, so inference works right after cloning. The first run downloads the
DINOv3 ViT-B weights (~350 MB) from Hugging Face; set `HF_HUB_OFFLINE=1` afterwards to skip the online check.

```bash
python src/grade.py photo.jpg                          # quality report for every fruit in the photo
python src/grade.py tray.jpg --bag-weight 250          # + recalibrate weight from the scale reading (g)
python src/grade.py fruit.jpg --variety Gajar          # skip the variety model when the variety is known
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
# Phase 1: variety (prepare_quality.py first: the variety set borrows grading-set crops)
python src/prepare_quality.py
python src/prepare_data.py
python src/train.py --backbone efficientnetv2b0 --img-size 260 --run effv2b0_260 --ft-layers 0 --ft-lr 1e-4 \
    --label-smoothing 0.1 --head-epochs 15 --ft-epochs 40                       # earlier 10-class model
python src/train.py --backbone efficientnetv2b0 --img-size 260 --init-from effv2b0_260 --run effv2b0_260_v2 \
    --ft-layers 0 --ft-lr 5e-5 --label-smoothing 0.1 --head-epochs 15 --ft-epochs 50   # production, 17 classes
python src/features.py --class-profiles

# Foundation-model probes (PyTorch, Apple GPU): embeddings are cached in outputs/features/
for d in variety grade maturity; do
  for m in dinov3_b dinov3_l dinov2_l; do python src/probe.py --data $d --model $m --split val; done
done
python src/finetune.py --data grade --model dinov3_b --run grade_ft_dinov3_b   # full fine-tune, ~2 h on an M5

# Phase 2: grade, maturity, size
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
# mixed PyTorch + Keras ensembles: score each member (evaluate.py / probe.py), then combine the saved probabilities
python src/evaluate.py --run effv2b0_260_v2 --tta --split val
python src/ensemble.py --run probe_dinov3_b_bgaug effv2b0_260_v2 --split val      # then --split test once
python src/calibrate.py --run probe_dinov3_b_bgaug effv2b0_260_v2 --from-probs --target 0.99
python src/probe.py --data variety --model dinov3_b --bg-aug --split val        # + --no-background: robustness
python src/explain_vit.py --run probe_dinov3_b_bgaug
python scripts/export_tflite.py
python scripts/build_report.py
```

Set `PRODUCTION_RUNS` in `src/config.py` to the models you selected. On an Apple M-series GPU, one EfficientNetV2-B0 run
takes about 40–80 minutes; a DINOv3 probe takes minutes once the weights are downloaded. `notebooks/colab_train.ipynb` runs
the Phase 1 backbone sweep on a Colab GPU. ConvNeXt does not train with TensorFlow on the Mac: Keras compiles its grouped
convolutions with XLA, which the Metal plugin lacks.

## Method notes

- **Transfer learning:** ImageNet backbone frozen while a new head trains, then the whole backbone is fine-tuned with AdamW,
  a cosine schedule and label smoothing 0.1. Keras has no EfficientNet-Lite, so EfficientNetV2-B0 and MobileNetV3 are used.
- **Foundation-model probes:** a frozen DINOv3 ViT-B/16 (self-supervised on 1.7 B images) at 336 px embeds each image as
  [CLS token, mean patch token] in 4 flips; a class-balanced logistic regression is trained on the embeddings, with C chosen
  on validation loss. The variety probe also trains on background-removed copies of each image (see Results). On a few thousand images this beats fine-tuning a small CNN and costs minutes. Variety and ripeness
  share one DINOv3 pass at inference.
- **Explainability:** Grad-CAM, Grad-CAM++ and Guided Grad-CAM (Grad-CAM × SmoothGrad; MobileNetV3's hard-swish rules out
  classic guided backprop). For the DINOv3 probes, Grad-CAM runs on the ViT's final patch tokens (`explain_vit.py`).
  Each has three sanity checks: a model-randomisation test, a deletion test and fruit focus. The probes add a fourth:
  test accuracy with the background removed, which catches photo-session shortcuts that Grad-CAM only hints at.
- **Calibration:** temperature scaling on validation. Predictions under the validation-chosen threshold are flagged `needs_review`.
- **Fruit splitting:** a distance-transform watershed splits blobs only where the outline narrows (a "neck"), so touching fruits
  are separated but a single elongated date is not. All 451 grading test images are counted as exactly one fruit.
- **TFLite:** float16 models are 6–12 MB, agree 100% with Keras on 200 test images each, and take 14–70 ms per image on CPU.
  The DINOv3 probe members run in PyTorch and have no mobile export yet.

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
