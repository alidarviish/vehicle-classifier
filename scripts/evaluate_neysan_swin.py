"""One-time, inference-only Neysan evaluation of the final Swin-Tiny model `swin_t_ft_aug`.

Run ONCE, from the repository root, only after approval:
    python scripts/evaluate_neysan_swin.py --visibility-tags <tags.csv> --val-baseline <val_predictions.csv>

Neysan is a subtype of vanet (docs/DATA_DECISIONS.md), not a ninth class. The 8-class model cannot
identify the subtype; every Neysan image is expected to be predicted as `vanet`. Nothing here trains,
tunes the threshold or changes the model, labels, splits, Test or the historical ResNet outputs (13_*).

Inputs:
- decisions/neysan_eval.csv (621 rows), verified with the checks of scripts/evaluate_neysan.py.
- --visibility-tags: private Gate-1 annotation CSV (columns image_path, lighting, front_visibility),
  covering exactly the 337 confirmed Neysan images not from v1/test. Kept outside the repository;
  only its file name and SHA256 are recorded.
- --val-baseline: existing per-image Swin validation predictions (columns true_label, pred_label,
  confidence, checkpoint_sha256; 659 rows), used only for its 26 vanet rows. Kept outside the
  repository; only its file name and SHA256 are recorded.
- reports/experiments/24_swin_final_test_predictions.csv: read only for its 50 vanet rows, reported
  descriptively and labelled non-pristine. Test is not run.

Order:
1. Stops if an output already exists.
2. Verifies the Neysan manifest, the two private inputs, the Swin checkpoint SHA256, that
   src/predict.py points at that checkpoint with threshold 0.95, and the class order.
   If any check fails it stops before any prediction and writes nothing.
3. Loads the model once and calls src.predict.predict_image once per row. A forward hook keeps the
   logits of that same call, so full-precision probabilities are available without a second pass.
4. Writes reports/experiments/25_swin_neysan_predictions.csv and
   reports/experiments/25_swin_neysan_evaluation.md (mode "x": never overwrites).
"""

import argparse
import csv
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# model-independent helpers of the historical evaluators (no checkpoint or threshold used from them)
from scripts.evaluate_test import CLASSES, read_manifest, sha256_of, write_csv  # noqa: E402
from scripts.evaluate_neysan import (ALLOWED_DUPLICATE_PAIR, NEYSAN_MANIFEST, NEYSAN_REVIEWS,  # noqa: E402
                                     SPLIT_MANIFEST, TEST_MANIFEST, git_snapshot, quartiles,
                                     verify_neysan_manifest)

CHECKPOINT = REPO_ROOT / "checkpoints" / "swin_t_ft_aug_best.pt"
CHECKPOINT_SHA256 = "f03bacde0a155461ea4aede0b813b8688d6ed17973d5ad19b0d2c1b913ff64db"
EXPECTED_ARCHITECTURE = "Swin-Tiny"
EXPECTED_THRESHOLD = 0.95
EXPECTED_MODEL_LABEL = "vanet"
OUT_DIR = REPO_ROOT / "reports" / "experiments"
OUT_CSV = OUT_DIR / "25_swin_neysan_predictions.csv"
OUT_REPORT = OUT_DIR / "25_swin_neysan_evaluation.md"
TEST_PREDICTIONS = OUT_DIR / "24_swin_final_test_predictions.csv"
HISTORICAL = [OUT_DIR / n for n in ("12_final_test_predictions.csv", "12_final_test_evaluation.md",
                                    "13_neysan_predictions.csv", "13_neysan_unclean_analysis.md",
                                    "24_swin_final_test_predictions.csv", "24_swin_final_test_evaluation.md")]

SUBGROUPS = [("CONFIRMED_NOT_V1TEST", "Confirmed Neysan, not from v1/test", 337),
             ("CONFIRMED_V1TEST", "Confirmed Neysan from v1/test", 34),
             ("N1", "N1 (folder label neysan, not individually reviewed)", 250)]
VIS_GROUPS = [("DAY_CLEAR", "DAY / CLEAR"), ("DAY_PARTIAL_OR_NOT_OBSERVABLE", "Day, partial / not observable"),
              ("NIGHT", "Night"), ("INFRARED", "Infrared"), ("LOW_LIGHT_UNCLEAR", "Low light / unclear")]
OUTCOMES = [("silent_acceptance", "predicted vanet, not flagged (silent acceptance)"),
            ("sent_to_review", "predicted vanet, flagged (sent to review)"),
            ("confident_misclassification", "predicted non-vanet, not flagged (confident misclassification)"),
            ("error_caught", "predicted non-vanet, flagged (error caught by review)")]


# ---------------------------------------------------------------- pure helpers (no torch)

def subgroup_of(m):
    if m["basis"] == "CONFIRMED_NEYSAN":
        return "CONFIRMED_V1TEST" if m["origin_split"] == "test" else "CONFIRMED_NOT_V1TEST"
    if m["basis"] == "neysan_folder_label":
        return "N1"
    raise ValueError(f"unknown basis {m['basis']!r} for {m['image_path']}")


def vis_group_of(lighting, visibility):
    if lighting == "DAY":
        return "DAY_CLEAR" if visibility == "CLEAR" else "DAY_PARTIAL_OR_NOT_OBSERVABLE"
    if lighting in ("NIGHT", "INFRARED", "LOW_LIGHT_UNCLEAR"):
        return lighting
    raise ValueError(f"unknown lighting tag {lighting!r}")


def outcome_of(is_vanet, flagged):
    return {(True, False): "silent_acceptance", (True, True): "sent_to_review",
            (False, False): "confident_misclassification", (False, True): "error_caught"}[(is_vanet, flagged)]


def load_visibility_tags(path, manifest_rows):
    """{image_path: vis_group} for exactly the CONFIRMED_NOT_V1TEST rows; raises ValueError otherwise."""
    rows = read_manifest(path)
    for col in ("image_path", "lighting", "front_visibility"):
        if not rows or col not in rows[0]:
            raise ValueError(f"visibility tags: missing column {col!r}")
    wanted = {m["image_path"] for m in manifest_rows if subgroup_of(m) == "CONFIRMED_NOT_V1TEST"}
    tags = {r["image_path"]: r for r in rows if r["image_path"] in wanted}
    if set(tags) != wanted:
        raise ValueError(f"visibility tags cover {len(tags)} of {len(wanted)} confirmed-not-v1/test images")
    return {p: vis_group_of(r["lighting"], r["front_visibility"]) for p, r in tags.items()}


def load_val_baseline(path):
    """The 26 validation vanet rows of the existing Swin validation predictions."""
    rows = read_manifest(path)
    for col in ("true_label", "pred_label", "confidence", "checkpoint_sha256"):
        if not rows or col not in rows[0]:
            raise ValueError(f"validation baseline: missing column {col!r}")
    if len(rows) != 659:
        raise ValueError(f"validation baseline: expected 659 rows, found {len(rows)}")
    if {r["checkpoint_sha256"] for r in rows} != {CHECKPOINT_SHA256}:
        raise ValueError("validation baseline: rows are not from the Swin final checkpoint")
    van = [r for r in rows if r["true_label"] == EXPECTED_MODEL_LABEL]
    if len(van) != 26:
        raise ValueError(f"validation baseline: expected 26 vanet rows, found {len(van)}")
    return [{"predicted_class": r["pred_label"], "confidence": float(r["confidence"]),
             "needs_review": float(r["confidence"]) < EXPECTED_THRESHOLD} for r in van]


def load_test_vanet(path):
    rows = [r for r in read_manifest(path) if r["true_label"] == EXPECTED_MODEL_LABEL]
    if len(rows) != 50:
        raise ValueError(f"Swin Test predictions: expected 50 vanet rows, found {len(rows)}")
    return [{"predicted_class": r["predicted_class"], "confidence": float(r["confidence"]),
             "needs_review": r["needs_review"] == "True"} for r in rows]


def build_rows(manifest_rows, predictions, full_probs, vis):
    """One output row per manifest row; full_probs: list of 8-float lists in CLASSES order."""
    out = []
    for i, (m, p, fp) in enumerate(zip(manifest_rows, predictions, full_probs), start=1):
        conf = max(fp)
        sg = subgroup_of(m)
        row = {"index": i, "path": m["image_path"], "sha256": m["sha256"], "basis": m["basis"],
               "origin_split": m["origin_split"], "source": m["source"], "subgroup": sg,
               "visibility_group": vis.get(m["image_path"], "") if sg == "CONFIRMED_NOT_V1TEST" else "",
               "expected_model_label": EXPECTED_MODEL_LABEL, "predicted_class": p["predicted_class"],
               "predicted_is_vanet": p["predicted_class"] == EXPECTED_MODEL_LABEL,
               "confidence": f"{p['confidence']:.4f}", "confidence_full": repr(conf),
               "needs_review": p["needs_review"],
               "outcome": outcome_of(p["predicted_class"] == EXPECTED_MODEL_LABEL, p["needs_review"])}
        row.update({f"p_{c}": repr(v) for c, v in zip(CLASSES, fp)})
        out.append(row)
    return out


def integrity_checks(manifest_rows, rows, predictions, threshold):
    probs = [[float(r[f"p_{c}"]) for c in CLASSES] for r in rows]
    confs = [float(r["confidence_full"]) for r in rows]
    by_path = {r["path"]: r for r in rows}
    a, b = sorted(ALLOWED_DUPLICATE_PAIR)
    return {
        "one prediction per manifest row (621)": len(rows) == len(manifest_rows) == 621,
        "no duplicate inference rows (unique paths)": len({r["path"] for r in rows}) == len(rows),
        "prediction order / path mapping matches the manifest":
            [r["path"] for r in rows] == [m["image_path"] for m in manifest_rows],
        "predicted_class is one of the 8 classes": all(r["predicted_class"] in CLASSES for r in rows),
        "probabilities in [0, 1] and sum to 1 (+-1e-5)":
            all(all(0.0 <= v <= 1.0 for v in p) and abs(sum(p) - 1.0) < 1e-5 for p in probs),
        "argmax of probabilities equals predicted_class":
            all(CLASSES[p.index(max(p))] == r["predicted_class"] for p, r in zip(probs, rows)),
        "confidence_full equals the maximum probability": all(c == max(p) for c, p in zip(confs, probs)),
        "rounded confidence equals predict_image output":
            all(round(c, 4) == pr["confidence"] for c, pr in zip(confs, predictions)),
        "rounded full probabilities equal predict_image probabilities":
            all(all(round(v, 4) == pr["probabilities"][c] for c, v in zip(CLASSES, p)) for p, pr in zip(probs, predictions)),
        f"needs_review exactly equals confidence_full < {threshold}":
            all(r["needs_review"] == (c < threshold) for r, c in zip(rows, confs)),
        "known duplicate pair got identical predictions":
            by_path[a]["predicted_class"] == by_path[b]["predicted_class"]
            and by_path[a]["confidence_full"] == by_path[b]["confidence_full"],
    }


def stats(rows):
    n = len(rows)
    c = [float(r["confidence_full"]) if "confidence_full" in r else float(r["confidence"]) for r in rows]
    q = quartiles(c)
    flagged = sum(bool(r["needs_review"]) for r in rows)
    vanet = sum(r["predicted_class"] == EXPECTED_MODEL_LABEL for r in rows)
    return {"n": n, "vanet": vanet, "flagged": flagged, "classes": Counter(r["predicted_class"] for r in rows),
            "median": q[1] if q else None, "q1": q[2] if q else None, "q3": q[3] if q else None,
            "min": min(c) if c else None,
            "outcomes": Counter(outcome_of(r["predicted_class"] == EXPECTED_MODEL_LABEL, bool(r["needs_review"]))
                                for r in rows)}


def pct(k, n):
    return f"{k}/{n} ({100 * k / n:.1f}%)" if n else "0/0"


def f4(x):
    return "n/a" if x is None else f"{x:.4f}"


def render_report(info, rows, checks, val_rows, test_rows):
    th = info["threshold"]
    sub = {k: [r for r in rows if r["subgroup"] == k] for k, _, _ in SUBGROUPS}
    L = ["# Neysan evaluation: `swin_t_ft_aug` (inference only)", "",
         "Written by `scripts/evaluate_neysan_swin.py`. Neysan is a subtype of `vanet`, not a ninth class;",
         "the expected 8-class output for every image is `vanet`. Nothing in this report was used to change",
         "the model, the threshold or the final Test result.", "",
         "## Run", "",
         f"- Run identifier: `{info['run_id']}`", f"- Timestamp (UTC): {info['timestamp']}",
         f"- Code: git commit `{info['git_head']}`" + (" (working tree had uncommitted changes)" if info["git_dirty"] else ""),
         f"- Checkpoint: `{info['checkpoint']}` ({info['architecture']}, epoch {info['epoch']}), SHA256 `{info['checkpoint_sha256']}`",
         "- Inference: `src/predict.py` (`load_model`, `predict_image`), one call per row; full-precision",
         "  probabilities taken from the logits of that same call (forward hook)",
         f"- Inference transform: {info['transform']} (no training augmentation)",
         f"- `needs_review` threshold: {th:.2f} (confidence < threshold; flag only, nothing rejected)",
         f"- Neysan manifest: `decisions/neysan_eval.csv`, SHA256 `{info['neysan_sha256']}`",
         f"- Visibility tags (private, Gate 1): `{info['vis_name']}`, SHA256 `{info['vis_sha256']}`",
         f"- Validation baseline (private): `{info['val_name']}`, SHA256 `{info['val_sha256']}`", "",
         "## A. Subgroups", "",
         "Reported separately; not pooled unless a row says so.", ""]
    L += [f"- {label}: {len(sub[k])} rows (expected {exp})" for k, label, exp in SUBGROUPS]
    a, b = sorted(ALLOWED_DUPLICATE_PAIR)
    L += [f"- Known duplicate pair (same SHA256): `{a}` (confirmed, not v1/test) and `{b}` (N1).",
          "  Each copy is counted in its own subgroup; pooled figures cover 621 rows = 620 unique images.", "",
          "## B. Predictions per subgroup", "",
          "| Subgroup | n | " + " | ".join(CLASSES) + " | vanet rate | conf. median | Q1 | Q3 | min | needs_review |",
          "|---|---|" + "---|" * len(CLASSES) + "---|---|---|---|---|---|"]
    for k, label, _ in SUBGROUPS:
        s = stats(sub[k])
        L.append(f"| {label} | {s['n']} | " + " | ".join(str(s["classes"][c]) for c in CLASSES)
                 + f" | {pct(s['vanet'], s['n'])} | {f4(s['median'])} | {f4(s['q1'])} | {f4(s['q3'])} | {f4(s['min'])}"
                 + f" | {pct(s['flagged'], s['n'])} |")
    L += ["", "## C. Review outcomes", "",
          "| Subgroup | n | " + " | ".join(lbl for _, lbl in OUTCOMES) + " |", "|---|---|" + "---|" * len(OUTCOMES)]
    for k, label, _ in SUBGROUPS + [("ALL", "Pooled, all 621 rows (620 unique images)", None)]:
        rs = rows if k == "ALL" else sub[k]
        s = stats(rs)
        L.append(f"| {label} | {s['n']} | " + " | ".join(pct(s["outcomes"][o], s["n"]) for o, _ in OUTCOMES) + " |")
    L += ["", "Silent acceptance is the correct 8-class answer (`vanet`) given without a review flag; it is not a",
          "classification error. It means the Neysan image is not distinguishable from an ordinary vanet output.", "",
          "## D. Confirmed Neysan, not from v1/test, by Gate-1 visibility / lighting", "",
          "| Group | n | vanet rate | conf. median | min | needs_review | " + " | ".join(lbl for _, lbl in OUTCOMES) + " |",
          "|---|---|---|---|---|---|" + "---|" * len(OUTCOMES)]
    for k, label in VIS_GROUPS:
        rs = [r for r in sub["CONFIRMED_NOT_V1TEST"] if r["visibility_group"] == k]
        s = stats(rs)
        L.append(f"| {label} | {s['n']} | {pct(s['vanet'], s['n'])} | {f4(s['median'])} | {f4(s['min'])} | "
                 f"{pct(s['flagged'], s['n'])} | " + " | ".join(pct(s["outcomes"][o], s["n"]) for o, _ in OUTCOMES) + " |")
    vs, ts = stats(val_rows), stats(test_rows)
    L += ["", "## E. Ordinary-vanet baseline", "",
          "| Set | n | vanet rate | conf. median | Q1 | Q3 | min | needs_review |", "|---|---|---|---|---|---|---|---|",
          f"| Validation vanet (main baseline) | {vs['n']} | {pct(vs['vanet'], vs['n'])} | {f4(vs['median'])} | {f4(vs['q1'])} | "
          f"{f4(vs['q3'])} | {f4(vs['min'])} | {pct(vs['flagged'], vs['n'])} |",
          f"| Test vanet (non-pristine, descriptive only) | {ts['n']} | {pct(ts['vanet'], ts['n'])} | {f4(ts['median'])} | "
          f"{f4(ts['q1'])} | {f4(ts['q3'])} | {f4(ts['min'])} | {pct(ts['flagged'], ts['n'])} |", "",
          "Validation confidences are stored with 6 decimals and Test confidences with 4; their flags are",
          f"recomputed (validation) or taken as recorded (Test). Validation rows within 1e-6 of {th}: "
          f"{sum(abs(r['confidence'] - th) < 1e-6 for r in val_rows)}.", "",
          "Flag rate: Neysan subgroup vs validation vanet (raw counts; difference in percentage points):", ""]
    for k, label, _ in SUBGROUPS:
        s = stats(sub[k])
        d = 100 * (s["flagged"] / s["n"] - vs["flagged"] / vs["n"]) if s["n"] and vs["n"] else float("nan")
        L.append(f"- {label}: {s['flagged']}/{s['n']} vs {vs['flagged']}/{vs['n']} -> {d:+.1f} points")
    L += ["", "## F. Integrity checks", ""]
    L += [f"- {name}: {'PASS' if ok else 'FAIL'}" for name, ok in checks.items()]
    L += ["", "## G. Limitations", "",
          "- Neysan is a vanet subtype, not a ninth class. The model cannot identify the Neysan subtype.",
          "- Silent acceptance (predicted `vanet`, not flagged) is not an 8-class classification error.",
          "- N1 images were not individually reviewed; their labels may be noisy.",
          "- The validation baseline has 26 images, so the flag-rate difference is imprecise.",
          "- This evaluation cannot predict the result on the mentor-held Test set.",
          "- The set is 100% Neysan, so it cannot estimate Neysan prevalence, or the share of flagged images that",
          "  are Neysan, in a mixed stream.",
          "- It does not establish general calibration or generalisation to other cameras or conditions.",
          f"- Per-image predictions: `{OUT_CSV.relative_to(REPO_ROOT).as_posix()}`.", ""]
    return "\n".join(L)


# ---------------------------------------------------------------- the one-time run

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--visibility-tags", required=True, type=Path)
    ap.add_argument("--val-baseline", required=True, type=Path)
    args = ap.parse_args(argv)
    for p in (OUT_CSV, OUT_REPORT):
        if p.exists():
            sys.exit(f"refusing to run: {p.relative_to(REPO_ROOT)} already exists (run only once)")

    # ---- checks, before any prediction ----
    from scripts.build_statuses import load_statuses
    from src.dataset import DEFAULT_CONFIG, load_source_roots, resolve_image_path
    roots = load_source_roots(DEFAULT_CONFIG)
    manifest = read_manifest(NEYSAN_MANIFEST)
    statuses = load_statuses()
    not_neysan = [r for f in NEYSAN_REVIEWS for r in read_manifest(f) if r["decision"] == "NOT_NEYSAN"]
    try:
        files = verify_neysan_manifest(manifest, lambda p: resolve_image_path(p, roots), read_manifest(TEST_MANIFEST),
                                       read_manifest(SPLIT_MANIFEST),
                                       [r for r in statuses if r["status"] == "excluded"], not_neysan)
        counts = Counter(subgroup_of(m) for m in manifest)
        if dict(counts) != {k: n for k, _, n in SUBGROUPS}:
            raise ValueError(f"subgroup counts {dict(counts)} differ from {[(k, n) for k, _, n in SUBGROUPS]}")
        vis = load_visibility_tags(args.visibility_tags, manifest)
        val_rows = load_val_baseline(args.val_baseline)
        test_rows = load_test_vanet(TEST_PREDICTIONS)
    except ValueError as e:
        sys.exit(str(e))
    guarded = {"checkpoint": CHECKPOINT, "neysan_eval.csv": NEYSAN_MANIFEST, "test_frozen.csv": TEST_MANIFEST,
               "split_manifest.csv": SPLIT_MANIFEST, **{p.name: p for p in HISTORICAL}}
    sha_before = {k: sha256_of(p) for k, p in guarded.items()}
    if sha_before["checkpoint"] != CHECKPOINT_SHA256:
        sys.exit(f"checkpoint SHA256 mismatch: {sha_before['checkpoint']} (expected {CHECKPOINT_SHA256})")
    import src.predict as P
    if Path(P.FINAL_CHECKPOINT).resolve() != CHECKPOINT.resolve():
        sys.exit(f"src/predict.py FINAL_CHECKPOINT is {P.FINAL_CHECKPOINT}, expected {CHECKPOINT}")
    if P.NEEDS_REVIEW_THRESHOLD != EXPECTED_THRESHOLD:
        sys.exit(f"needs_review threshold is {P.NEEDS_REVIEW_THRESHOLD}, expected {EXPECTED_THRESHOLD}")
    if list(P.CLASSES) != CLASSES:
        sys.exit(f"class order differs: {P.CLASSES}")
    head, dirty, status_before = git_snapshot()
    print(f"verified: {len(manifest)} Neysan rows, tags {len(vis)}, validation vanet {len(val_rows)}, "
          f"Test vanet {len(test_rows)}, checkpoint, threshold, class order")

    # ---- the single inference pass ----
    import torch
    model, eval_transform, checkpoint = P.load_model(CHECKPOINT)   # checks architecture and class_to_idx
    if checkpoint["architecture"] != EXPECTED_ARCHITECTURE:
        sys.exit(f"architecture is {checkpoint['architecture']}, expected {EXPECTED_ARCHITECTURE}")
    captured = []
    hook = model.register_forward_hook(lambda m, i, o: captured.append(o.detach()))
    started = datetime.now(timezone.utc)
    predictions, full_probs = [], []
    for f, m in zip(files, manifest):
        captured.clear()
        predictions.append(P.predict_image(f, model, eval_transform, path=m["image_path"]))
        if len(captured) != 1:
            sys.exit(f"expected one forward pass for {m['image_path']}, got {len(captured)}")
        full_probs.append([float(v) for v in torch.softmax(captured[0], dim=1)[0]])
    hook.remove()

    rows = build_rows(manifest, predictions, full_probs, vis)
    checks = integrity_checks(manifest, rows, predictions, P.NEEDS_REVIEW_THRESHOLD)
    checks["visibility group present for all 337 confirmed-not-v1/test rows"] = \
        sum(bool(r["visibility_group"]) for r in rows) == 337
    sha_after = {k: sha256_of(p) for k, p in guarded.items()}
    checks["checkpoint, manifests and historical outputs (12_*, 13_*, 24_*) unchanged"] = sha_after == sha_before
    info = {"run_id": "swin_neysan_" + started.strftime("%Y%m%dT%H%M%SZ"),
            "timestamp": started.strftime("%Y-%m-%d %H:%M:%S"), "git_head": head, "git_dirty": dirty,
            "checkpoint": CHECKPOINT.relative_to(REPO_ROOT).as_posix(), "checkpoint_sha256": sha_before["checkpoint"],
            "architecture": checkpoint["architecture"], "epoch": checkpoint["epoch"],
            "transform": " -> ".join(repr(t) for t in eval_transform.transforms).replace("\n", " "),
            "threshold": P.NEEDS_REVIEW_THRESHOLD, "neysan_sha256": sha_before["neysan_eval.csv"],
            "vis_name": args.visibility_tags.name, "vis_sha256": sha256_of(args.visibility_tags),
            "val_name": args.val_baseline.name, "val_sha256": sha256_of(args.val_baseline)}
    write_csv(rows, OUT_CSV)
    with open(OUT_REPORT, "x", encoding="utf-8") as fh:
        fh.write(render_report(info, rows, checks, val_rows, test_rows))
    _, _, status_after = git_snapshot()
    new = {OUT_CSV.relative_to(REPO_ROOT).as_posix(), OUT_REPORT.relative_to(REPO_ROOT).as_posix()}
    if status_before is not None and status_after is not None and status_after - status_before != new:
        print(f"WARNING: unexpected changed files: {sorted(status_after - status_before - new)}")
    s = stats(rows)
    print(f"pooled 621 rows: vanet {s['vanet']} | needs_review {s['flagged']} | "
          f"integrity {'PASS' if all(checks.values()) else 'FAIL'}")
    print(f"written: {OUT_REPORT.relative_to(REPO_ROOT)}, {OUT_CSV.relative_to(REPO_ROOT)}")
    if not all(checks.values()):
        sys.exit("integrity checks failed; see the report")


if __name__ == "__main__":
    main()
