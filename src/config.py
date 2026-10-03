"""Central configuration for the date fruit project (Phase 1: variety, Phase 2: quality)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"  # Zenodo 4639543, flat folder
ALHAMDAN_DIR = ROOT / "data" / "kaggle" / "alhamdan"  # Kaggle "Date Fruit Image Dataset in Controlled Environment"
GRADING_DIR = ROOT / "data" / "kaggle" / "grading"  # variety/size/grade dataset, for Phase 2
PROCESSED_DIR = ROOT / "data" / "processed"
SPLITS_CSV = ROOT / "data" / "splits.csv"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
MATURITY_SRC_DIR = ROOT / "data" / "maturity"  # Zenodo 10143465 crops, from scripts/download_maturity.py
QUALITY_DIR = ROOT / "data" / "quality"  # Phase 2 splits, from src/prepare_quality.py

# Image-classification datasets (train.py --data): name -> <dir>/<split>/<class>/*.jpg
DATASETS = {
    "variety": PROCESSED_DIR,
    "grade": QUALITY_DIR / "grade",
    "maturity": QUALITY_DIR / "maturity",
}

# Zenodo filename prefix -> class name used by the doc.
# The Zenodo record actually publishes 478 images across ~10 prefixes, not the
# 27 classes / 2000+ images its description claims. "Sawr" (7 images) and
# "IMG_*" (unlabelled) are never used.
ZENODO_MAP = {
    "NbotAli": "NbotAli",
    "SequeeIRAQ": "SagaiIRAQ",
    "Skari": "Al-skari",
    "skri_magrosh": "Al-skri_magrosh",
    "Muraaya": "Al-muraaya",
    "sequee": "Al-sagai",
    "masyihia": "Al-masyihia",
    "Shagra": "Al-shagra",
}
# Alhamdan folder -> class name. "Rutab" is left out: in the doc Rutab is a
# ripeness stage (Khalal/Rutab/Tamar), not a variety.
ALHAMDAN_MAP = {
    "Ajwa": "Al-ajwa",
    "Medjool": "Al-majdool",
    "Sokari": "Al-skari",
    "Sugaey": "Al-sagai",
    "Nabtat Ali": "NbotAli",
    "Meneifi": "Al-meneifi",
    "Shaishe": "Al-shaishe",
    "Galaxy": "Galaxy",
}

# The 17 varieties trained. First five are on the doc's preferred list; then the
# best-supplied remaining web/studio classes, then the four grading-set varieties
# (named as in data/kaggle/grading so grade.py picks their pixel size model).
GRADING_VARIETIES = ["Aseel", "Fasli Toto", "Gajar", "Kupro"]
CLASSES = [
    "Al-skari", "Al-ajwa", "Al-majdool", "Al-sagai", "Al-muraaya",
    "NbotAli", "Al-meneifi", "Al-shaishe", "Galaxy", "Al-skri_magrosh",
    "Al-masyihia", "Al-shagra", "SagaiIRAQ",
] + GRADING_VARIETIES
# Grading-set photos per variety used for the variety model (Gajar alone has 1,310;
# all come from one camera rig, so more would only unbalance the classes).
GRADING_PER_VARIETY = 200

# Longest side of images stored in data/processed (originals go up to ~9 MB).
STORE_MAX_SIDE = 640
IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42
SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}

# "mobilenetv3" (MobileNetV3Large) or "efficientnet" (EfficientNetB0;
# EfficientNet-Lite has no official Keras application, B0 is its closest counterpart).
BACKBONE = "mobilenetv3"
HEAD_EPOCHS = 25
FINETUNE_EPOCHS = 30
HEAD_LR = 1e-3
FINETUNE_LR = 3e-5
FINETUNE_LAST_N_LAYERS = 60

# ---------------- Phase 2: quality ----------------
MATURITY_STAGES = ["Immature", "Khalal", "Rutab", "Tamar"]
# Moroccan variety held out entirely for the maturity test split (orchard sessions
# are one per stage, so a random split would leak session lighting into test).
MATURITY_TEST_VARIETY = "kholt"
# Printed ArUco marker (cv2.aruco DICT_4X4_50) placed next to the fruit gives the mm/px scale.
ARUCO_DICT = "DICT_4X4_50"
ARUCO_SIDE_MM = 30.0
# Weight = k * length * width^2 (mm -> g). An ellipsoid of revolution has V = pi/6 * L * W^2;
# with date flesh density ~1.3 g/cm^3 that gives k ~ 6.8e-4. Recalibrate per variety with
# `python src/quality.py --calibrate-weight <csv>` or a bag weight (see grade.py --bag-weight).
WEIGHT_K = 6.8e-4

# Models used by grade.py (chosen on the validation split; see reports/model_report.html).
# A list of several runs = an ensemble (averaged probabilities).
PRODUCTION_RUNS = {
    # DINOv3-B probe (probe.py, PyTorch) + EfficientNetV2-B0; one DINOv3 pass serves variety and maturity.
    # The variety probe is trained with background-removed copies (--bg-aug): without them it relied on photo-session cues.
    "variety": ["probe_dinov3_b_bgaug", "effv2b0_260_v2"],
    "grade": ["grade_mnv3", "grade_effv2b0"],
    "maturity": ["maturity_probe_dinov3_b", "maturity_effv2b0"],
}


def run_name(runs):
    """File-name key of a model or ensemble, as written by evaluate.py / calibrate.py."""
    return "+".join([runs] if isinstance(runs, str) else runs)
