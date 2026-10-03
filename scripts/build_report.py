"""Build reports/model_report.html (self-contained, images embedded) and
reports/model_report.pdf (headless Chrome) from the metric files in outputs/.

Every number in the report is read from a file written by evaluate.py,
calibrate.py, explain.py, export_tflite.py, quality.py or pytest, so re-running
this after retraining refreshes the whole report.

Usage: python scripts/build_report.py [--no-pdf]
"""
import argparse
import base64
import datetime as dt
import html
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import config  # noqa: E402

EVAL, EXPL, MODELS, QUAL = (config.OUTPUT_DIR / d for d in ("eval", "explainability", "models", "quality"))
OUT = ROOT / "reports"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

TASKS = {
    "variety": {"title": "Phase 1 · Variety classification",
                "candidates": ["mnv3_v2", "effv2b0_260_v2", "effv2b0_260_v2+mnv3_v2", "probe_dinov3_b", "probe_dinov3_l",
                               "probe_dinov2_l", "probe_dinov3_b+effv2b0_260_v2", "probe_dinov3_l+probe_dinov2_l+effv2b0_260_v2",
                               "probe_dinov3_b_bgaug", "probe_dinov3_b_bgaug+effv2b0_260_v2"],
                "test_note": "Random stratified 70/15/15 split of 17 varieties (Zenodo web photos, Alhamdan studio photos, and "
                             f"{config.GRADING_PER_VARIETY} grading-set photos per grading variety, kept in their grade split). "
                             "Images of the original 10 varieties keep their earlier split.",
                "extra": "<p class='note'><b>Background shortcut found and fixed.</b> The first DINOv3 probe scored 99.0 % on "
                         "validation but only 82.1 % once the background was removed (the CNN: 96.8 → 95.6 % on test): the frozen "
                         "backbone also encodes each variety's photo session (lighting, framing), and the probe used it. Its Grad-CAM "
                         "showed the same, with heat on the background around the fruit. Training the probe on every image twice, "
                         "as photographed and with the background removed (<code>--bg-aug</code>), keeps 99.0 % on original "
                         "photos and reaches 98.8 % without background. It was chosen over the slightly higher but shortcut-prone "
                         "ensemble (99.8 % vs 99.5 % validation) on this check, before the test set was scored.</p>"},
    "grade": {"title": "Phase 2 · Quality grade (1–3)",
              "candidates": ["grade_mnv3", "grade_effv2b0", "grade_mnv3+grade_effv2b0", "grade_probe_dinov3_b",
                             "grade_probe_dinov3_l", "grade_ft_dinov3_b", "grade_probe_dinov3_b+grade_mnv3+grade_effv2b0",
                             "grade_ft_dinov3_b+grade_mnv3+grade_effv2b0"],
              "test_note": "Stratified split of the Kaggle grading set (4 varieties). Aseel and Fasli Toto only have Grade-1, "
                           "so the informative scores are Gajar and Kupro.",
              "extra": "<p class='note'>Modern backbones did not beat the CNN ensemble here: a frozen DINOv3 probe (80–83 %) "
                       "misses the small surface defects that separate the grades, and fully fine-tuning DINOv3-B at 336 px "
                       "(layer-wise LR decay, fp16 on the Apple GPU) reached 87.4 % on validation vs 89.8 %; ensembling it "
                       "with the CNNs did not help either. Three different model families plateauing at 87–90 % points to the "
                       "subjective Grade-1/Grade-2 boundary in the labels as the limit.</p>"},
    "maturity": {"title": "Phase 2 · Maturity stage",
                 "candidates": ["maturity_mnv3", "maturity_effv2b0", "maturity_mnv3+maturity_effv2b0", "maturity_probe_dinov3_b",
                                "maturity_probe_dinov3_l", "maturity_probe_dinov2_l", "maturity_probe_dinov3_b+maturity_effv2b0",
                                "maturity_probe_dinov3_l+maturity_probe_dinov2_l+maturity_effv2b0"],
                 "test_note": "Test = the whole Moroccan variety Kholt, never seen in training (each stage was shot in one "
                              "session, so a random split would leak session lighting), plus held-out Alhamdan single fruits.",
                 "extra": "<p class='note'>Validation (seen varieties, ~95–97 %) does not predict accuracy on a new variety "
                          "(~65–70 %). A leave-one-variety-out check inside train+val gives the DINOv3-B probe 55–76 % on each "
                          "held-out Moroccan variety, the same level as the CNN on the test variety: a stronger backbone does "
                          "not solve new-variety ripeness, per-fruit stage labels would. On studio photos of single fruits "
                          "(Alhamdan, Rutab vs Tamar) it is 100 %.</p>"},
}
BACKBONE_NAMES = {"mobilenetv3": "MobileNetV3-Large 224", "efficientnetv2b0": "EfficientNetV2-B0 260",
                  "dinov3_b": "DINOv3 ViT-B/16 336", "dinov3_l": "DINOv3 ViT-L/16 336", "dinov2_l": "DINOv2 ViT-L/14 336"}


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def img(path, alt, cls="fig"):
    if not path.exists():
        return f'<p class="muted">[{html.escape(alt)}: not generated]</p>'
    import io

    from PIL import Image

    im = Image.open(path).convert("RGB")
    im.thumbnail((1400, 4000))  # keeps the report small enough to email
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<img class="{cls}" alt="{html.escape(alt)}" src="data:image/jpeg;base64,{b64}">'


def pct(v, d=1):
    return "–" if v is None else f"{100 * v:.{d}f}%"


def num(v, d=3):
    return "–" if v is None else f"{v:.{d}f}"


def table(head, rows, cls=""):
    th = "".join(f"<th>{h}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>'


def run_meta(run):
    return load(MODELS / f"{run}_meta.json") or {}


def model_label(name):
    parts = name.split("+")
    labels = []
    for r in parts:
        m = run_meta(r)
        if "timm" in m:  # PyTorch: frozen probe (probe.py) or full fine-tune (finetune.py)
            labels.append(BACKBONE_NAMES.get(m["model"], m["model"]) + (" fine-tuned" if "history" in m else " probe")
                          + (" (background-augmented)" if m.get("bg_aug") else ""))
        else:
            labels.append(BACKBONE_NAMES.get(m.get("backbone"), r))
    return labels[0] if len(labels) == 1 else "Ensemble: " + " + ".join(labels)


def model_section(task, spec):
    members = config.PRODUCTION_RUNS[task]
    run = config.run_name(members)
    meta = run_meta(members[0])
    test = load(EVAL / f"{run}_tta_test_metrics.json")
    if test is None:
        return f"<section><h2>{spec['title']}</h2><p class='muted'>Not evaluated yet.</p></section>", None
    cal = load(EVAL / f"{run}_calibration.json")
    sanities = [(m, load(EXPL / f"{m}_sanity_checks.json")) for m in members]
    sanities = [(m, s) for m, s in sanities if s]
    tfls = [t for t in (load(MODELS / f"{m}_tflite.json") for m in members) if t]

    # model selection on the validation split
    sel_rows = []
    for c in spec["candidates"]:
        v = load(EVAL / f"{c}_tta_val_metrics.json")
        if v is None:
            continue
        t = load(EVAL / f"{c}_tta_test_metrics.json")
        sel_rows.append([("<b>" if c == run else "") + model_label(c) + (" ✔ selected</b>" if c == run else ""),
                         pct(v["accuracy"]), num(v["f1_macro"]), num(v["loss"]),
                         pct(t["accuracy"]) if t else "not run"])
    selection = table(["Candidate", "Val accuracy", "Val macro-F1", "Val loss", "Test accuracy (for disclosure)"],
                      sel_rows) if sel_rows else ""

    pc = test["per_class"]
    classes = meta.get("class_names", [k for k in pc if k not in ("accuracy", "macro avg", "weighted avg")])
    cls_rows = [[c, num(pc[c]["precision"]), num(pc[c]["recall"]), num(pc[c]["f1-score"]), int(pc[c]["support"])]
                for c in classes if c in pc]
    src_rows = [[html.escape(k), pct(v)] for k, v in test["accuracy_by_source"].items()]

    cal_html = ""
    if cal:
        r = cal["review"]
        curve = table(["Confidence ≥", "Auto-accepted", "Accuracy on accepted"],
                      [[num(c["threshold"], 2), pct(c["coverage"]), pct(c["accuracy"])] for c in cal["test_coverage_curve"]],
                      "compact")
        worse = cal["test"]["ece_after"] > cal["test"]["ece_before"]
        warn = ("<p class='note'><b>Calibration does not transfer to the unseen test variety.</b> Temperature scaling fitted on "
                "validation (seen varieties) makes test calibration worse, and the review threshold does not reach its target: "
                "on a new variety the model is confidently wrong, so its confidence cannot be used to route fruits to review.</p>"
                if worse else "")
        cal_html = warn + f"""
        <h3>Calibration and manual-review threshold</h3>
        <p>Temperature scaling (T = {cal['temperature']}) fitted on validation. Expected calibration error on test
        {num(cal['test']['ece_before'])} → <b>{num(cal['test']['ece_after'])}</b>; test NLL {num(cal['test']['nll_before'])} →
        {num(cal['test']['nll_after'])}. The review threshold ({num(r['threshold'])}) is the lowest confidence whose
        <i>validation</i> accuracy reaches {pct(r['target_accuracy'], 0)}; on test it auto-accepts
        <b>{pct(r['test_coverage'])}</b> of images at <b>{pct(r['test_accuracy_on_accepted'])}</b> accuracy and sends
        {r['test_sent_to_review']} of {r['n_test']} to manual review.</p>
        <div class="two"><div>{curve}</div></div>"""

    sanity_html = ""
    ok = lambda b: '<span class="pass">pass</span>' if b else '<span class="fail">fail</span>'
    for m, sanity in sanities:
        s, v = sanity["summary"], sanity["summary"]["verdict"]
        member = f" · ensemble member {model_label(m)}" if len(members) > 1 else ""
        sanity_html += f"""
        <h3>Explainability: Grad-CAM sanity checks (test split){member}</h3>
        {table(["Check", "Result", "Verdict"], [
            ["Heatmap vs fully randomised model (Spearman, low = map depends on learning)", num(s['spearman_vs_random_all']), ok(v['depends_on_learned_weights'])],
            ["Confidence drop when top-20% heatmap pixels are blurred vs random 20%", f"{num(s['deletion_drop_cam'])} vs {num(s['deletion_drop_random'])}", ok(v['faithful_deletion'])],
            ["Share of heatmap energy on fruit pixels vs fruit share of image", f"{pct(s['cam_energy_on_fruit'])} vs {pct(s['fruit_area_fraction'])}", ok(v['focuses_on_fruit'])],
        ] + ([["Test accuracy with the background removed vs as photographed (single view)",
               f"{pct(s['accuracy_background_removed'])} vs {pct(s['accuracy_original'])}", ok(v['robust_to_background_removal'])]]
             if "accuracy_background_removed" in s else []))}
        {img(EXPL / f"{m}_gradcam_gallery.png", "Grad-CAM gallery", "fig wide")}
        <p class="caption">Columns: input · Grad-CAM · Grad-CAM++ · Guided Grad-CAM (Grad-CAM × SmoothGrad) · Grad-CAM of a randomised
        model.{" For the ViT, Grad-CAM runs on the final patch tokens (explain_vit.py)." if "timm" in run_meta(m) else ""}</p>"""

    bag = next((EXPL / f"{m}_bagged_examples.png" for m in members if (EXPL / f"{m}_bagged_examples.png").exists()),
               EXPL / "none.png")
    if task == "maturity" and bag.exists():
        sanity_html += f"""
        <h3>Does the model look at the protective bags?</h3>
        {img(bag, "Grad-CAM on bagged Kholt bunches", "fig wide")}
        <p>Grad-CAM on bagged Rutab/Tamar bunches of the unseen test variety: the heatmaps fall on the fruit visible through the
        netting, not on the net, so the bags are not being used as a shortcut. Most of these examples are misclassified,
        and the reason is visible: labels are per photo session, so a "Rutab" bunch still holds many green/yellow Khalal-coloured
        fruits, and a red "Tamar" bunch looks like another variety's earlier stage. Per-fruit stage labels on the dealer's own
        varieties are what would fix this.</p>"""
    tfl_html = ""
    if tfls:
        tfl_html = "<h3>TFLite export (float16)</h3>" + table(
            ["Model", "TFLite MB", "Keras MB", "Top-1 agreement with Keras", "Max prob. diff", "Median CPU latency"],
            [[model_label(t["run"]), t["tflite_mb"], t["keras_mb"], f"{pct(t['top1_agreement'])} (n={t['n_checked']})",
              t["max_prob_diff"], f"{t['cpu_latency_ms_median']} ms"] for t in tfls])

    body = f"""
    <section id="{task}">
      <h2>{spec['title']}</h2>
      <p class="muted">{spec['test_note']}</p>
      <div class="kpis">
        <div><span>{pct(test['accuracy'])}</span>test accuracy</div>
        <div><span>{num(test['f1_macro'])}</span>macro-F1</div>
        <div><span>{num(test['f1_weighted'])}</span>weighted-F1</div>
        <div><span>{num(test['loss'])}</span>test loss (cross-entropy)</div>
        <div><span>{pct(test['top2_accuracy'])}</span>top-2 accuracy</div>
        <div><span>{test['n']}</span>test images</div>
      </div>
      <h3>Model selection</h3>
      {selection}
      <p>Selected on validation: <b>{model_label(run)}</b> (highest validation accuracy among candidates that pass the explainability sanity checks; ties go to the smaller model).
      CNNs: ImageNet weights, frozen-backbone head training then whole-backbone fine-tuning (AdamW, cosine schedule, label
      smoothing 0.1). Probes (PyTorch, Apple GPU): frozen self-supervised DINOv3/DINOv2 backbone, [CLS, mean patch]
      embedding, logistic regression with C chosen on validation loss. Ensembles average probabilities. Test scores use
      4-flip test-time augmentation. The test column is shown for transparency only and was not used to choose.</p>
      {spec.get('extra', '')}
      <div class="two">
        <div><h3>Per-class results (test)</h3>{table(["Class", "Precision", "Recall", "F1", "n"], cls_rows)}
             <h3>Accuracy by source</h3>{table(["Source / variety", "Accuracy"], src_rows)}</div>
        <div><h3>Confusion matrix (test)</h3>{img(EVAL / f"{run}_tta_test_confusion_matrix.png", "confusion matrix")}</div>
      </div>
      {cal_html}
      {sanity_html}
      {tfl_html}
    </section>"""
    summary = {"task": task, "title": spec["title"], "model": model_label(run),
               "n": test["n"], "acc": test["accuracy"], "f1": test["f1_macro"], "loss": test["loss"],
               "cov": cal["review"]["test_coverage"] if cal else None,
               "acc_acc": cal["review"]["test_accuracy_on_accepted"] if cal else None}
    return body, summary


def size_section():
    rep = load(QUAL / "size_model_test.json")
    if not rep:
        return "", None
    rows, cms = [], []
    for v, r in rep.items():
        if v == "overall_test_accuracy":
            continue
        rows.append([v, pct(r["test_accuracy"]), r["n_test"]])
        cm = table([""] + [f"pred {l}" for l in r["labels"]],
                   [[f"true {l}"] + row for l, row in zip(r["labels"], r["confusion"])], "compact")
        cms.append(f"<div><h4>{v}</h4>{cm}</div>")
    rows.append(["Kupro", "only Large in dataset", "–"])
    body = f"""
    <section id="size"><h2>Phase 2 · Size class (Small / Medium / Large)</h2>
      <div class="kpis"><div><span>{pct(rep['overall_test_accuracy'])}</span>test accuracy (3 varieties with 3 sizes)</div></div>
      <p>Per-variety random forest on the measured fruit outline: √area, length and width. Chosen on the validation split
      over logistic regression, gradient boosting and area-only cut points (val 0.839 vs 0.805–0.833). Size classes are relative to
      the variety (a Small Gajar is longer than a Large Aseel). Pixel models only hold for the grading-set camera rig;
      with an ArUco marker in the photo, size is measured in mm and classified with mm cut points.</p>
      {table(["Variety", "Test accuracy", "n"], rows)}
      <div class="grid3">{''.join(cms)}</div>
      <p class="note">Gajar Small and Medium overlap almost completely in measured size (median √area 347 vs 355 px), so the
      labels themselves cap accuracy for Gajar; more model capacity does not help (all four methods: 0.66–0.71).</p>
    </section>"""
    return body, {"task": "size", "title": "Phase 2 · Size class", "model": "Random forest on outline geometry",
                  "n": sum(r["n_test"] for k, r in rep.items() if k != "overall_test_accuracy"),
                  "acc": rep["overall_test_accuracy"], "f1": None, "loss": None, "cov": None, "acc_acc": None}


def profiles_section():
    """Phase 1 spec item: colour / shape / texture extracted per predicted variety."""
    import pandas as pd

    path = config.OUTPUT_DIR / "class_feature_profiles.csv"
    if not path.exists():
        return ""
    df = pd.read_csv(path, index_col=0)
    cols = [("color.color_name", "Colour", None), ("color.mean_hue_deg", "Hue °", 0), ("color.mean_brightness", "Brightness", 2),
            ("color.mean_saturation", "Saturation", 2), ("color.color_uniformity", "Colour uniformity", 2),
            ("shape.shape", "Shape", None), ("shape.aspect_ratio", "Aspect ratio", 2), ("shape.circularity", "Circularity", 2),
            ("shape.solidity", "Solidity", 3), ("texture.glcm_homogeneity", "GLCM homogeneity", 3),
            ("texture.glcm_contrast", "GLCM contrast", 2), ("texture.lbp_entropy", "LBP entropy", 2),
            ("texture.wrinkle_index", "Wrinkle index", 3)]
    cols = [c for c in cols if c[0] in df]
    rows = [[v] + [df.loc[v, c] if d is None else f"{df.loc[v, c]:.{d}f}" for c, _, d in cols] for v in df.index]
    return f"""
    <section id="profiles"><h2>Phase 1 · Colour, shape and texture per variety</h2>
      <p>Extracted from the segmented fruit (GrabCut / plain-background mask) of every training image; the table shows each
      variety's median. <code>python src/predict.py &lt;image&gt;</code> reports the same features for one photo next to the
      predicted variety's typical values, with the Grad-CAM heatmap. Shape values are scale-free ratios; hue is the circular mean
      of skin pixels; wrinkle index is the edge density inside the fruit.</p>
      {table(["Variety"] + [h for _, h, _ in cols], rows, "compact")}
    </section>"""


def rules_section():
    chk = load(QUAL / "defect_check.json")
    rows = ""
    if chk:
        feats = ["crack_pct", "black_spot_pct", "insect_hole_pct", "sunburn_pct", "mold_pct", "total_defect_pct",
                 "sugar_spot_pct", "wrinkle_index", "gloss_index", "color_uniformity"]
        mb = chk["mean_by_grade"]
        keys = list(mb.keys())
        head = ["Feature"] + [html.escape(k.strip("()").replace("'", "")) for k in keys] + ["ρ Gajar", "ρ Kupro"]
        rho = chk["spearman_with_grade"]
        fmt = lambda v: "–" if v is None or v != v else f"{v:+.2f}"
        rows = table(head, [[f] + [num(mb[k][f], 3) for k in keys] + [fmt(rho[f].get("Gajar")), fmt(rho[f].get("Kupro"))]
                            for f in feats], "compact")
    return f"""
    <section id="rules"><h2>Phase 2 · Colour, defects, skin, weight (measurement rules)</h2>
      <p>No public dataset labels date defects, skin condition or weight, so these features are computed with transparent
      colour/texture rules on the segmented fruit rather than trained models. They cannot be scored with accuracy yet.
      As a sanity check, they are compared with the grade labels of the two varieties that have all three grades: defect
      area should rise from Grade-1 to Grade-3 (positive Spearman ρ with the grade number).</p>
      {rows}
      <h3>What each feature measures</h3>
      {table(["Feature", "Method", "Validation status"], [
        ["Colour uniformity", "Mean CIELAB ΔE of skin pixels from their mean → score exp(−ΔE/15); share within ΔE 10; dominant colours (k-means)", "Deterministic measurement"],
        ["Cracks / black spots / insect holes", "Pixels ≥12 L* darker than their local neighbourhood; blobs split by elongation and circularity", "ρ with grade +0.31 / +0.48 / +0.59 on Gajar; weak on Kupro"],
        ["Sunburn / mold", "Light, desaturated regions away from the rim and specular highlights; mold = compact, textured or greenish", "Not validated; mold threshold raised to 2 % after false positives on glossy Kupro"],
        ["Wrinkling", "Edge density inside the fruit", "Rises with grade on Kupro (+0.35), not Gajar (−0.25)"],
        ["Gloss", "Share and contrast of highlights relative to the fruit's own brightness", "Falls with grade on both (−0.31, −0.23)"],
        ["Sugar spots", "Small light, desaturated granular blobs", "Weak (+0.09 / +0.28)"],
        ["Maturity by colour", "Green share → Immature; yellow/red share → Khalal / partly ripened Rutab", "Cannot split Rutab from Tamar (38 % on Alhamdan), so it only complements the CNN"],
        ["Weight", "Ellipsoid of revolution: k·L·W² with k = 6.8×10⁻⁴ g/mm³ (ρ ≈ 1.3 g/cm³); needs an ArUco marker or mm/px", "Untested on weighed fruit. A bag's scale reading recalibrates k per batch (--bag-weight)"],
      ])}
    </section>"""


def readiness_section():
    tests = (config.OUTPUT_DIR / "tests.txt").read_text().strip() if (config.OUTPUT_DIR / "tests.txt").exists() else "not run"
    return f"""
    <section id="production"><h2>Production readiness</h2>
      {table(["Item", "Status"], [
        ["Inference API", "<code>grade.load_models()</code> + <code>grade.grade_image(path)</code> → JSON report; CLI <code>python src/grade.py img1 img2 …</code>"],
        ["Automated tests", html.escape(tests)],
        ["Input validation", "Missing/unreadable/tiny images, images without fruit and invalid scale or bag weight raise <code>ValueError</code>; the CLI reports each failure and exits non-zero"],
        ["Confidence handling", "Temperature-calibrated probabilities; predictions below the review threshold are flagged <code>needs_review</code> per fruit"],
        ["Multiple fruits", "Touching fruits are split at their neck (watershed); 451/451 grading test images counted as exactly one fruit"],
        ["Pinned environment", "<code>requirements-lock.txt</code> (Python 3.12, TensorFlow 2.18.1 + tensorflow-metal)"],
        ["Edge deployment", "TFLite float16 export of the Keras models with a parity check (<code>scripts/export_tflite.py</code>). "
                            "The DINOv3 probe members run in PyTorch (Apple GPU or CPU); they have no mobile export yet"],
        ["Reproducibility", "Fixed seeds, split files saved (<code>data/splits.csv</code>, <code>data/quality/*_index.csv</code>), every number in this report regenerated from metric files"],
      ])}
    </section>"""


def limitations_section():
    return """
    <section id="limits"><h2>Limitations and what is needed before field use</h2>
      <ol>
        <li><b>No dealer photos yet.</b> All scores are on public datasets. Eight of the requested varieties (Barhi, Mabroum, Khasab,
        Khasab Khalal, Helwa, Helwa Macnooz, Maktoumi, Bowytha, Kelas, Rashudia) are not in any public dataset. Accuracy on the dealer's
        own camera setup must be measured on a labelled sample of their photos before relying on it.</li>
        <li><b>Maturity data is orchard bunches.</b> Immature/Khalal/Rutab/Tamar labels are per bunch photo; many Rutab/Tamar bunches are
        inside protective bags. No public single-fruit Khalal images exist.</li>
        <li><b>Defects, skin and weight are unvalidated rules.</b> They need a few hundred dealer photos with marked defects, and
        20–30 fruits weighed on a scale and photographed with the marker.</li>
        <li><b>Grade and size labels are subjective.</b> Most grade errors are between adjacent grades (1↔2); Gajar size classes overlap in
        measured size.</li>
        <li><b>99 % accuracy</b> is reached only on auto-accepted predictions (see the review thresholds); the remaining images go to a
        person. Headline accuracies above are over all test images.</li>
      </ol>
    </section>"""


CSS = """
:root{--ink:#1d2433;--muted:#5b6475;--line:#d9dde5;--bg:#ffffff;--soft:#f4f6f9;--acc:#8a4b12;--ok:#1e7b3a;--bad:#b3261e}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:1080px;margin:0 auto;padding:32px 20px 60px}
h1{font-size:28px;margin:0 0 4px}h2{font-size:21px;margin:36px 0 8px;padding-top:12px;border-top:2px solid var(--line)}
h3{font-size:15px;margin:20px 0 6px}h4{margin:8px 0 4px;font-size:13px}
.muted,.caption{color:var(--muted)}.caption{font-size:12px;margin-top:2px}
table{border-collapse:collapse;width:100%;margin:6px 0 10px;font-size:13px}
th,td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}
th{background:var(--soft);font-weight:600}table.compact td,table.compact th{padding:3px 6px;font-size:12px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin:12px 0}
.kpis div{background:var(--soft);border-radius:8px;padding:10px 12px;color:var(--muted);font-size:12px}
.kpis span{display:block;font-size:22px;font-weight:650;color:var(--ink)}
.two{display:grid;grid-template-columns:1fr 1fr;gap:20px}.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
.fig{max-width:100%;border:1px solid var(--line);border-radius:6px}.fig.wide{width:100%}
.pass{color:var(--ok);font-weight:600}.fail{color:var(--bad);font-weight:600}
.note{background:#fbf6ef;border-left:3px solid var(--acc);padding:8px 12px}
code{font-size:12px;background:var(--soft);padding:1px 4px;border-radius:3px}
@media (max-width:760px){.two,.grid3{grid-template-columns:1fr}}
@media print{main{padding:0}h2{break-before:page}section#summary h2{break-before:auto}.fig.wide{max-height:none}table,img{break-inside:avoid}}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()
    sections, summaries = [], []
    for task, spec in TASKS.items():
        body, s = model_section(task, spec)
        sections.append(body)
        summaries.append(s)
        if task == "variety":
            sections.append(profiles_section())
        if task == "grade":
            b, s2 = size_section()
            sections.append(b)
            summaries.append(s2)
    sections += [rules_section(), readiness_section(), limitations_section()]

    rows = [[s["title"], s["model"], s["n"], f"<b>{pct(s['acc'])}</b>", num(s["f1"]), num(s["loss"]),
             "–" if s["cov"] is None else f"{pct(s['acc_acc'])} on {pct(s['cov'], 0)} auto-accepted"]
            for s in summaries if s]
    summary = f"""
    <section id="summary"><h2>Summary of results (held-out test sets)</h2>
      {table(["Component", "Model", "Test n", "Accuracy", "Macro-F1", "Loss", "Accuracy with manual review of low-confidence"], rows)}
      <p class="muted">Model choice, calibration temperature and review thresholds were all fixed on validation splits; each test set was used
      once for the numbers shown. Loss is categorical cross-entropy of the uncalibrated model with test-time augmentation.
      Colour uniformity, defects, skin and weight are measurement rules without labelled data (section 6).</p>
    </section>"""

    today = dt.date.today().isoformat()
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Date Fruit Model Report</title><style>{CSS}</style></head>
<body><main>
<h1>Date Fruit Variety &amp; Quality Models: Evaluation Report</h1>
<p class="muted">Generated {today} from the metric files in <code>outputs/</code> · models: {", ".join(config.run_name(r) for r in config.PRODUCTION_RUNS.values())}</p>
{summary}
{''.join(sections)}
</main></body></html>"""
    OUT.mkdir(exist_ok=True)
    html_path = OUT / "model_report.html"
    html_path.write_text(page)
    print(f"wrote {html_path}")
    if not args.no_pdf:
        pdf = OUT / "model_report.pdf"
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={pdf}", html_path.as_uri()], check=True, capture_output=True, timeout=180)
        print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
