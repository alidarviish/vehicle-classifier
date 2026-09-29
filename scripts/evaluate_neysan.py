"""One-time, inference-only evaluation of the final model on the Neysan evaluation set.

Run ONCE, from the repository root, only after approval:
    python scripts/evaluate_neysan.py

Neysan is a subtype of vanet (docs/DATA_DECISIONS.md), not a ninth class. The model has 8 classes;
every Neysan image is expected to be predicted as `vanet` (expected_model_label = vanet is the
existing taxonomy mapping, not a new label). The model never saw Neysan images in training,
validation or Test, so this describes behaviour on an unseen vanet subtype.

Order:
1. Stops if an output already exists.
2. Verifies decisions/neysan_eval.csv (the only input; no folder scanning): 621 rows,
   371 CONFIRMED_NEYSAN + 250 neysan_folder_label, folder labels vanet / neysan, no duplicate path,
   only the one known duplicate-SHA256 pair, every file exists with the manifest SHA256, and no
   path / SHA256 overlap with Test, train/val, excluded or NOT_NEYSAN images.
3. Verifies the final checkpoint SHA256, the 0.90 threshold and the class order, and records the
   SHA256 of the checkpoint, Test manifest, split manifest and Neysan manifest.
   If any check fails, it stops before any prediction and writes nothing.
4. Loads the model once (src.predict.load_model, eval mode) and runs src.predict.predict_image for
   every row. No training, no optimizer, no gradients, nothing saved to the checkpoint.
5. Writes reports/experiments/13_neysan_predictions.csv and
   reports/experiments/13_neysan_unclean_analysis.md (descriptive only: no precision, F1, macro
   metric or confusion matrix, because the set contains a single expected class).

Nothing here tunes the threshold or changes the model, labels, splits or Test.
"""

import csv
import statistics
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# shared helpers of the Test evaluation (no inference logic in them)
from scripts.evaluate_test import (CHECKPOINT, CHECKPOINT_SHA256, CLASSES, EXPECTED_THRESHOLD,  # noqa: E402
                                   read_manifest, sha256_of, write_csv)

NEYSAN_MANIFEST = REPO_ROOT / "decisions" / "neysan_eval.csv"
TEST_MANIFEST = REPO_ROOT / "decisions" / "test_frozen.csv"
SPLIT_MANIFEST = REPO_ROOT / "data" / "split_manifest.csv"
NEYSAN_REVIEWS = [REPO_ROOT / "decisions" / "neysan_review.csv", REPO_ROOT / "decisions" / "neysan_test_review.csv"]
EXPECTED_ROWS = 621
EXPECTED_BASIS = {"CONFIRMED_NEYSAN": 371, "neysan_folder_label": 250}
EXPECTED_FOLDER_LABELS = {"vanet", "neysan"}
ALLOWED_DUPLICATE_PAIR = {"v1/train/vanet/214844236.jpg", "v1/unclean/neysan/214844236.jpg"}
EXPECTED_MODEL_LABEL = "vanet"
OUT_DIR = REPO_ROOT / "reports" / "experiments"
OUT_CSV = OUT_DIR / "13_neysan_predictions.csv"
OUT_REPORT = OUT_DIR / "13_neysan_unclean_analysis.md"
GUARDED = {"checkpoint": CHECKPOINT, "test_frozen.csv": TEST_MANIFEST,
           "split_manifest.csv": SPLIT_MANIFEST, "neysan_eval.csv": NEYSAN_MANIFEST}


def verify_neysan_manifest(rows, resolve, test_rows, split_rows, excluded_rows, not_neysan_rows):
    """All pre-prediction data checks. Returns the local file paths; raises ValueError on any failure."""
    problems = []
    if len(rows) != EXPECTED_ROWS:
        problems.append(f"expected {EXPECTED_ROWS} rows, found {len(rows)}")
    basis = Counter(r["basis"] for r in rows)
    if dict(basis) != EXPECTED_BASIS:
        problems.append(f"expected basis counts {EXPECTED_BASIS}, found {dict(basis)}")
    folders = set(r["folder_label"] for r in rows)
    if folders != EXPECTED_FOLDER_LABELS:
        problems.append(f"expected folder labels {sorted(EXPECTED_FOLDER_LABELS)}, found {sorted(folders)}")
    paths = [r["image_path"] for r in rows]
    if len(set(paths)) != len(paths):
        problems.append(f"{len(paths) - len(set(paths))} duplicate paths")
    by_sha = {}
    for r in rows:
        by_sha.setdefault(r["sha256"], set()).add(r["image_path"])
    dup_groups = [p for p in by_sha.values() if len(p) > 1]
    if dup_groups != [ALLOWED_DUPLICATE_PAIR]:
        problems.append(f"expected exactly one duplicate-SHA256 group {sorted(ALLOWED_DUPLICATE_PAIR)}, "
                        f"found {[sorted(g) for g in dup_groups]}")
    for name, other in [("Test (test_frozen.csv)", test_rows), ("train/val (split_manifest.csv)", split_rows),
                        ("excluded", excluded_rows), ("NOT_NEYSAN", not_neysan_rows)]:
        p = set(paths) & {r["image_path"] for r in other}
        s = set(by_sha) & {r["sha256"] for r in other}
        if p or s:
            problems.append(f"overlap with {name}: {len(p)} paths, {len(s)} SHA256 values")
    files, missing, mismatch = [], [], []
    for r in rows:
        f = Path(resolve(r["image_path"]))
        files.append(f)
        if not f.is_file():
            missing.append(r["image_path"])
        elif sha256_of(f) != r["sha256"]:
            mismatch.append(r["image_path"])
    if missing:
        problems.append(f"{len(missing)} files not found, first: {missing[:3]}")
    if mismatch:
        problems.append(f"{len(mismatch)} files whose SHA256 differs from the manifest, first: {mismatch[:3]}")
    if problems:
        raise ValueError("Neysan manifest verification failed:\n- " + "\n- ".join(problems))
    return files


def prediction_rows(manifest_rows, predictions):
    """One output row per manifest row, in manifest order."""
    rows = []
    for i, (m, p) in enumerate(zip(manifest_rows, predictions), start=1):
        row = {"index": i, "path": m["image_path"], "sha256": m["sha256"], "source": m["source"],
               "origin_split": m["origin_split"], "folder_label": m["folder_label"], "basis": m["basis"],
               "expected_model_label": EXPECTED_MODEL_LABEL, "predicted_class": p["predicted_class"],
               "predicted_is_vanet": p["predicted_class"] == EXPECTED_MODEL_LABEL,
               "confidence": f"{p['confidence']:.4f}", "needs_review": p["needs_review"]}
        row.update({f"p_{c}": f"{p['probabilities'][c]:.4f}" for c in CLASSES})
        rows.append(row)
    return rows


def post_checks(manifest_rows, out_rows, threshold):
    return {
        f"{EXPECTED_ROWS} predictions, one per manifest row": len(out_rows) == len(manifest_rows) == EXPECTED_ROWS,
        "prediction order / path mapping matches the manifest":
            [r["path"] for r in out_rows] == [m["image_path"] for m in manifest_rows],
        "predicted_class is one of the 8 classes": all(r["predicted_class"] in CLASSES for r in out_rows),
        "confidence in [0, 1]": all(0.0 <= float(r["confidence"]) <= 1.0 for r in out_rows),
        # needs_review is decided on the unrounded confidence; a rounded 0.9000 can be either side
        f"needs_review == (confidence < {threshold:.2f})":
            all(r["needs_review"] == (float(r["confidence"]) < threshold) or r["confidence"] == f"{threshold:.4f}"
                for r in out_rows),
    }


def quartiles(values):
    """(mean, median, Q1, Q3) with statistics.quantiles(method='inclusive'); None if empty."""
    if not values:
        return None
    if len(values) == 1:
        v = values[0]
        return v, v, v, v
    q1, med, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return statistics.mean(values), med, q1, q3


def summarize(rows):
    n = len(rows)
    vanet = sum(r["predicted_is_vanet"] for r in rows)
    review = sum(r["needs_review"] for r in rows)
    return {"n": n, "vanet": vanet, "non_vanet": n - vanet, "vanet_recall": vanet / n if n else 0.0,
            "needs_review": review,
            "review_x": {(v, nr): sum(r["predicted_is_vanet"] == v and r["needs_review"] == nr for r in rows)
                         for v in (True, False) for nr in (True, False)},
            "classes": Counter(r["predicted_class"] for r in rows)}


def unique_by_sha(rows):
    seen, out = set(), []
    for r in rows:
        if r["sha256"] not in seen:
            seen.add(r["sha256"])
            out.append(r)
    return out


def render_report(info, out_rows, checks, threshold):
    f4 = lambda x: f"{x:.4f}"
    pct = lambda a, b: f"{a} ({a / b:.1%})" if b else "0"
    groups = [("All 621 rows", out_rows),
              ("CONFIRMED_NEYSAN (371, human-confirmed)", [r for r in out_rows if r["basis"] == "CONFIRMED_NEYSAN"]),
              ("neysan_folder_label (250, policy N1, not human-confirmed)",
               [r for r in out_rows if r["basis"] == "neysan_folder_label"]),
              ("Unique SHA256 images (620)", unique_by_sha(out_rows))]
    S = {name: summarize(rows) for name, rows in groups}
    L = ["# Neysan / unclean analysis (inference only)", "",
         "Written by `scripts/evaluate_neysan.py`. Descriptive analysis of the final model on the separate",
         "Neysan evaluation set. It makes no decision about the model, the threshold, labels or Test.", "",
         "## Scope", "",
         "- Neysan is an unseen subtype of `vanet`, not a ninth class. The model has 8 classes and never saw",
         "  Neysan images in training, validation or Test.",
         f"- `expected_model_label` = `{EXPECTED_MODEL_LABEL}` for every row: the existing taxonomy mapping",
         "  (Neysan -> vanet), not a new label.",
         "- The test set was not used in this analysis; no Test image is in the Neysan set (checked by path",
         "  and SHA256 before prediction).",
         f"- The `needs_review` threshold ({threshold:.2f}) was chosen on validation and was not tuned on Neysan.",
         "- The 250 `neysan_folder_label` images (policy N1) are not human-confirmed; only the 371",
         "  `CONFIRMED_NEYSAN` images were individually reviewed.",
         "- No precision, F1, macro metric or confusion matrix is reported: the set has a single expected",
         "  class and no negatives, so those numbers would be trivial or undefined. The share predicted as",
         "  `vanet` (vanet recall on Neysan) is the only supervised number, and it is not comparable with the",
         "  validation or Test macro metrics.", "",
         "## Run", "",
         f"- Run identifier: `{info['run_id']}`",
         f"- Timestamp (UTC): {info['timestamp']}",
         f"- Code: git commit `{info['git_head']}`" + (" (working tree had uncommitted changes)"
                                                     if info["git_dirty"] else ""),
         f"- Checkpoint: `{info['checkpoint']}` (SHA256 verified: `{info['sha_before']['checkpoint']}`)",
         f"- Inference: `src/predict.py` (`load_model`, `predict_image`), unchanged; transform: {info['transform']}",
         f"- Input: `decisions/neysan_eval.csv` only ({len(out_rows)} rows, every file's SHA256 verified)", "",
         "## Predicted vanet vs non-vanet", "",
         "| Group | Images | Predicted vanet | Predicted non-vanet | Vanet recall | needs_review |",
         "|---|---|---|---|---|---|"]
    for name, s in S.items():
        L.append(f"| {name} | {s['n']} | {s['vanet']} | {s['non_vanet']} | {f4(s['vanet_recall'])} | "
                 f"{pct(s['needs_review'], s['n'])} |")
    L += ["", "The N1 row is a policy-based, non-human-confirmed result. The duplicate pair",
          f"`{'` and `'.join(sorted(ALLOWED_DUPLICATE_PAIR))}` is byte-identical and therefore gets the same",
          "prediction; the manifest keeps both rows (621), and the unique-SHA256 row counts it once (620).", "",
          "## Prediction distribution over the 8 classes", "",
          "| Group | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * len(CLASSES)]
    for name, s in S.items():
        L.append(f"| {name} | " + " | ".join(str(s["classes"].get(c, 0)) for c in CLASSES) + " |")
    L += ["", f"## needs_review (confidence < {threshold:.2f}) by prediction", "",
          "| Group | vanet, flagged | vanet, not flagged | non-vanet, flagged | non-vanet, not flagged |",
          "|---|---|---|---|---|"]
    for name, s in S.items():
        x = s["review_x"]
        L.append(f"| {name} | {x[(True, True)]} | {x[(True, False)]} | {x[(False, True)]} | {x[(False, False)]} |")
    L += ["", "## Confidence (highest softmax probability)", "",
          "| Group | Images | Mean | Median | Q1 | Q3 |", "|---|---|---|---|---|---|"]
    for label, rows in [("All 621 rows", out_rows),
                        ("Predicted vanet", [r for r in out_rows if r["predicted_is_vanet"]]),
                        ("Predicted non-vanet", [r for r in out_rows if not r["predicted_is_vanet"]]),
                        ("CONFIRMED_NEYSAN", [r for r in out_rows if r["basis"] == "CONFIRMED_NEYSAN"]),
                        ("neysan_folder_label (N1)", [r for r in out_rows if r["basis"] == "neysan_folder_label"])]:
        q = quartiles([float(r["confidence"]) for r in rows])
        L.append(f"| {label} | {len(rows)} | " + (" | ".join(f4(v) for v in q) if q else "- | - | - | -") + " |")
    L += ["", "Quartiles: `statistics.quantiles(n=4, method='inclusive')` on the 4-decimal confidences.", "",
          "## By source / origin", "",
          "| Source / origin | Basis | Images | Predicted vanet | Predicted non-vanet | needs_review |",
          "|---|---|---|---|---|---|"]
    keys = sorted({(r["source"], r["origin_split"], r["basis"]) for r in out_rows})
    for src, orig, basis in keys:
        rows = [r for r in out_rows if (r["source"], r["origin_split"], r["basis"]) == (src, orig, basis)]
        s = summarize(rows)
        L.append(f"| {src}/{orig} | {basis} | {s['n']} | {s['vanet']} | {s['non_vanet']} | {s['needs_review']} |")
    L += ["", "## Integrity", ""]
    L += [f"- {k}: {'PASS' if v else 'FAIL'}" for k, v in checks.items()]
    L += ["", "SHA256 of the guarded files before and after inference:", "",
          "| File | Before | After |", "|---|---|---|"]
    L += [f"| {k} | `{info['sha_before'][k]}` | `{info['sha_after'][k]}` |" for k in GUARDED]
    L += ["", f"- Files created by this run: `{OUT_CSV.relative_to(REPO_ROOT).as_posix()}` and this report.", ""]
    return "\n".join(L)


def git_snapshot():
    """(short HEAD, dirty?, set of untracked/modified paths) or (unknown, None, None) without git."""
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=REPO_ROOT,
                                capture_output=True, text=True, check=True).stdout.splitlines()
        return head, bool(status), {line[3:] for line in status}
    except (OSError, subprocess.CalledProcessError):
        return "unknown", None, None


def main():
    for p in (OUT_CSV, OUT_REPORT):
        if p.exists():
            sys.exit(f"refusing to run: {p.relative_to(REPO_ROOT)} already exists (run only once)")

    # ---- checks, before any prediction ----
    from scripts.build_statuses import load_statuses
    from src.dataset import DEFAULT_CONFIG, load_source_roots, resolve_image_path
    roots = load_source_roots(DEFAULT_CONFIG)
    rows = read_manifest(NEYSAN_MANIFEST)
    statuses = load_statuses()
    not_neysan = [r for f in NEYSAN_REVIEWS for r in read_manifest(f) if r["decision"] == "NOT_NEYSAN"]
    try:
        files = verify_neysan_manifest(rows, lambda p: resolve_image_path(p, roots), read_manifest(TEST_MANIFEST),
                                       read_manifest(SPLIT_MANIFEST),
                                       [r for r in statuses if r["status"] == "excluded"], not_neysan)
    except ValueError as e:
        sys.exit(str(e))
    sha_before = {k: sha256_of(p) for k, p in GUARDED.items()}
    if sha_before["checkpoint"] != CHECKPOINT_SHA256:
        sys.exit(f"checkpoint SHA256 mismatch: {sha_before['checkpoint']} (expected {CHECKPOINT_SHA256})")
    import src.predict as P
    if P.NEEDS_REVIEW_THRESHOLD != EXPECTED_THRESHOLD:
        sys.exit(f"needs_review threshold is {P.NEEDS_REVIEW_THRESHOLD}, expected {EXPECTED_THRESHOLD}")
    if list(P.CLASSES) != CLASSES:
        sys.exit(f"class order differs: {P.CLASSES}")
    head, dirty, status_before = git_snapshot()
    print(f"Neysan manifest, checkpoint, threshold and class order verified ({len(rows)} rows)")

    # ---- the single inference pass (model loaded once, eval mode, no gradients) ----
    model, eval_transform, checkpoint = P.load_model(CHECKPOINT)
    started = datetime.now(timezone.utc)
    predictions = [P.predict_image(f, model, eval_transform, path=m["image_path"]) for f, m in zip(files, rows)]

    out_rows = prediction_rows(rows, predictions)
    threshold = P.NEEDS_REVIEW_THRESHOLD
    checks = post_checks(rows, out_rows, threshold)
    sha_after = {k: sha256_of(p) for k, p in GUARDED.items()}
    checks["checkpoint, test_frozen.csv, split_manifest.csv, neysan_eval.csv unchanged"] = sha_after == sha_before
    info = {"run_id": "neysan_eval_" + started.strftime("%Y%m%dT%H%M%SZ"),
            "timestamp": started.strftime("%Y-%m-%d %H:%M:%S"), "git_head": head, "git_dirty": dirty,
            "checkpoint": CHECKPOINT.relative_to(REPO_ROOT).as_posix(),
            "transform": " -> ".join(repr(t) for t in eval_transform.transforms).replace("\n", " "),
            "sha_before": sha_before, "sha_after": sha_after}

    write_csv(out_rows, OUT_CSV)
    new_paths = {OUT_CSV.relative_to(REPO_ROOT).as_posix(), OUT_REPORT.relative_to(REPO_ROOT).as_posix()}
    _, _, status_mid = git_snapshot()
    if status_before is not None and status_mid is not None:
        checks["only the two outputs are new (CSV now, report next)"] = \
            status_mid - status_before == {OUT_CSV.relative_to(REPO_ROOT).as_posix()}
    with open(OUT_REPORT, "x", encoding="utf-8") as f:
        f.write(render_report(info, out_rows, checks, threshold))
    _, _, status_after = git_snapshot()
    if status_before is not None and status_after is not None and status_after - status_before != new_paths:
        print(f"WARNING: unexpected changed files: {sorted(status_after - status_before - new_paths)}")
        checks["no other file changed"] = False

    s = summarize(out_rows)
    print(f"predicted vanet {s['vanet']}/{s['n']} | needs_review {s['needs_review']} | "
          f"integrity {'PASS' if all(checks.values()) else 'FAIL'}")
    print(f"written: {OUT_CSV.relative_to(REPO_ROOT)}, {OUT_REPORT.relative_to(REPO_ROOT)}")
    if not all(checks.values()):
        sys.exit("integrity checks failed; see the report / messages above")


if __name__ == "__main__":
    main()
