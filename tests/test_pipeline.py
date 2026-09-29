"""Smoke and regression tests for the inference pipeline.

Run: .venv-metal/bin/python -m pytest tests -q
Tests that need trained models or datasets skip themselves when those are missing.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
from quality import (estimate_weight, find_scale, fruit_instances, geometry, hide_marker,  # noqa: E402
                     maturity_by_colour, size_class, surface_defects)

GRADING_SAMPLE = next(iter(sorted((config.GRADING_DIR / "Aseel" / "Large" / "Grade-1").glob("*.jpg"))), None)


def ellipse_image(centres, axes=(90, 50), bg=(230, 228, 222), colour=(90, 45, 25), size=(600, 800)):
    img = np.full((*size, 3), bg, np.uint8)
    for c in centres:
        cv2.ellipse(img, c, axes, 0, 0, 360, colour, -1)
    return img


def with_marker(img, side_px=150, at=(20, 20)):
    mk = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.ARUCO_DICT)),
                                       3, side_px)
    y, x = at
    img[y:y + side_px + 40, x:x + side_px + 40] = 255
    img[y + 20:y + 20 + side_px, x + 20:x + 20 + side_px] = mk[..., None]
    return img


def test_geometry_of_known_ellipse():
    img = ellipse_image([(400, 300)], axes=(100, 40))
    (mask,) = fruit_instances(img)
    g = geometry(mask, mm_per_px=0.5)
    assert g["length_px"] == pytest.approx(200, rel=0.05)
    assert g["width_px"] == pytest.approx(80, rel=0.08)
    assert g["length_mm"] == pytest.approx(100, rel=0.05)


def test_counts_separate_and_touching_fruits():
    assert len(fruit_instances(ellipse_image([(200, 300), (560, 300)]))) == 2
    touching = ellipse_image([(300, 300), (470, 300)], axes=(90, 50))  # overlap ~10 px
    assert len(fruit_instances(touching)) == 2


def test_single_elongated_fruit_is_not_split():
    assert len(fruit_instances(ellipse_image([(400, 300)], axes=(200, 45)))) == 1


def test_aruco_scale_and_marker_is_not_a_fruit():
    img = with_marker(ellipse_image([(500, 350)]))
    mmpp, corners = find_scale(img)
    assert mmpp == pytest.approx(config.ARUCO_SIDE_MM / 150, rel=0.03)
    assert len(fruit_instances(hide_marker(img, corners))) == 1


def test_no_marker_gives_no_scale():
    assert find_scale(ellipse_image([(400, 300)])) == (None, None)


def test_weight_formula():
    g = {"length_mm": 40.0, "width_mm": 25.0}
    assert estimate_weight(g) == pytest.approx(config.WEIGHT_K * 40 * 25**2, abs=0.01)
    assert estimate_weight({"length_px": 1}) is None


def test_size_class_mm_default_cuts():
    assert size_class({"length_mm": 25.0, "area_mm2": 1})["size_class"] == "Small"
    assert size_class({"length_mm": 35.0, "area_mm2": 1})["size_class"] == "Medium"
    assert size_class({"length_mm": 45.0, "area_mm2": 1})["size_class"] == "Large"


def test_unknown_variety_without_scale_has_no_size():
    assert size_class({"length_px": 1, "width_px": 1, "area_px": 1}, "Unknown")["size_class"] is None


def test_clean_fruit_has_no_defects():
    img = ellipse_image([(400, 300)], axes=(150, 80))
    mask = fruit_instances(img)[0]
    d, _ = surface_defects(img, mask)
    assert d["defects_present"] == []


def test_dark_spot_is_detected():
    img = ellipse_image([(400, 300)], axes=(150, 80), colour=(150, 90, 50))
    mask = fruit_instances(img)[0]
    cv2.ellipse(img, (380, 290), (18, 12), 0, 0, 360, (30, 18, 10), -1)
    d, _ = surface_defects(img, mask)
    assert "black_spot" in d["defects_present"] or "insect_hole" in d["defects_present"]


def test_colour_rule_stages():
    green = ellipse_image([(400, 300)], colour=(90, 160, 60))
    yellow = ellipse_image([(400, 300)], colour=(235, 200, 40))
    brown = ellipse_image([(400, 300)], colour=(80, 40, 20))
    stage = lambda im: maturity_by_colour(im, fruit_instances(im)[0])["stage"]
    assert stage(green) == "Immature"
    assert stage(yellow) == "Khalal"
    assert stage(brown).startswith("Rutab or Tamar")


# ---------------------------------------------------------------- end to end (needs models)
def _models_ready():
    return all((config.MODEL_DIR / f"{r}_meta.json").exists() for runs in config.PRODUCTION_RUNS.values() for r in runs)


@pytest.fixture(scope="module")
def models():
    if not _models_ready():
        pytest.skip("production models not trained")
    from grade import load_models

    return load_models()


def test_grade_image_end_to_end(models, tmp_path):
    from grade import grade_image

    img = with_marker(ellipse_image([(300, 350), (600, 350)], size=(700, 900)), at=(20, 700))
    path = tmp_path / "tray.jpg"
    Image.fromarray(img).save(path)
    r = grade_image(path, models, bag_weight=50, save=False)
    s = r["summary"]
    assert s["n_fruits"] == 2 and s["scale_source"] == "aruco"
    assert s["model_vs_scale_error_pct"] is not None
    for f in r["fruits"]:
        for key in ("variety", "grade", "maturity"):
            probs = f[key]["probs"]
            assert abs(sum(probs.values()) - 1) < 0.01
            assert f[key]["label"] in probs
        assert f["size"]["size_class"] in {"Small", "Medium", "Large"}
    json.dumps(r)  # report must be JSON-serialisable


@pytest.mark.skipif(GRADING_SAMPLE is None, reason="grading dataset not present")
def test_grade_real_fruit(models):
    from grade import grade_image

    r = grade_image(GRADING_SAMPLE, models, variety_override="Aseel", save=False)
    assert r["summary"]["n_fruits"] == 1
    assert r["fruits"][0]["size"]["size_class"] == "Large"


@pytest.mark.parametrize("bad", ["/no/such/file.jpg", __file__])
def test_bad_input_raises_value_error(models, bad):
    from grade import grade_image

    with pytest.raises(ValueError):
        grade_image(bad, models, save=False)


def test_blank_image_has_no_fruit(models, tmp_path):
    from grade import grade_image

    path = tmp_path / "blank.jpg"
    Image.fromarray(np.full((400, 400, 3), 230, np.uint8)).save(path)
    with pytest.raises(ValueError, match="no fruit"):
        grade_image(path, models, save=False)
