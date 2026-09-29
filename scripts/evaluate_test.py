"""Final, one-time evaluation of the selected model on the frozen Test set.

Run this ONCE, from the repository root, only after the protocol has been approved:
    python scripts/evaluate_test.py

What it does, in this order:
1. Refuses to start if the output files already exist (Test is evaluated only once).
2. Verifies the Test manifest decisions/test_frozen.csv (the only Test source; no folder scanning):
   exactly 400 rows, 50 per class, only the 8 classes, no duplicate path, no duplicate SHA256,
   every file exists and its actual SHA256 equals the manifest SHA256.
3. Verifies the SHA256 of the frozen final checkpoint and the needs_review threshold (0.90).
   If any check in 2 or 3 fails, it stops BEFORE any prediction.
4. Predicts every Test image with the unchanged inference code of src/predict.py
   (load_model / predict_image: Resize 224x224 -> ToTensor -> ImageNet normalization, softmax over
   the 8 classes, argmax, needs_review = confidence < 0.90). No training augmentation.
5. Writes reports/experiments/12_final_test_predictions.csv (one row per manifest row, same order)
   and reports/experiments/12_final_test_evaluation.md (summary, metrics, integrity checks).

Read-only for the dataset, the manifest and the checkpoint. Neysan images are not used.
"""

import csv
import hashlib
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TEST_MANIFEST = REPO_ROOT / "decisions" / "test_frozen.csv"
CHECKPOINT = REPO_ROOT / "checkpoints" / "resnet224_ft_aug_best.pt"
CHECKPOINT_SHA256 = "c26280e97f7324ee1ec85d7bf78a58ea0b4d2acb4cf3e975dbd52316366281a1"
EXPECTED_THRESHOLD = 0.90
CLASSES = ["ambulance", "autobus", "kamyun", "kamyunet", "minibus", "savari", "taxi", "vanet"]
TEST_TOTAL = 400
TEST_PER_CLASS = 50
OUT_DIR = REPO_ROOT / "reports" / "experiments"
OUT_CSV = OUT_DIR / "12_final_test_predictions.csv"
OUT_REPORT = OUT_DIR / "12_final_test_evaluation.md"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def verify_manifest(rows, resolve, total=TEST_TOTAL, per_class=TEST_PER_CLASS, classes=CLASSES):
    """Check the manifest rows; returns the local file paths. Raises ValueError on any failure."""
    problems = []
    if len(rows) != total:
        problems.append(f"expected {total} rows, found {len(rows)}")
    counts = Counter(r["label"] for r in rows)
    if set(counts) != set(classes) or any(counts[c] != per_class for c in classes):
        problems.append(f"expected {per_class} per class for {classes}, found {dict(counts)}")
    paths = [r["image_path"] for r in rows]
    shas = [r["sha256"] for r in rows]
    if len(set(paths)) != len(paths):
        problems.append(f"{len(paths) - len(set(paths))} duplicate paths")
    if len(set(shas)) != len(shas):
        problems.append(f"{len(shas) - len(set(shas))} duplicate SHA256 values")
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
        raise ValueError("Test manifest verification failed:\n- " + "\n- ".join(problems))
    return files


def compute_metrics(y_true, y_pred, classes=CLASSES):
    """Confusion matrix (rows = true, columns = predicted), per-class and macro metrics."""
    idx = {c: i for i, c in enumerate(classes)}
    k = len(classes)
    matrix = [[0] * k for _ in range(k)]
    for t, p in zip(y_true, y_pred):
        matrix[idx[t]][idx[p]] += 1
    per_class = {}
    for i, c in enumerate(classes):
        tp = matrix[i][i]
        predicted = sum(matrix[r][i] for r in range(k))
        support = sum(matrix[i])
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[c] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    correct = sum(matrix[i][i] for i in range(k))
    total = len(y_true)
    return {
        "matrix": matrix, "per_class": per_class, "total": total, "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "macro_precision": sum(v["precision"] for v in per_class.values()) / k,
        "macro_recall": sum(v["recall"] for v in per_class.values()) / k,
        "macro_f1": sum(v["f1"] for v in per_class.values()) / k,
    }


def prediction_rows(manifest_rows, predictions, classes=CLASSES):
    """One output row per manifest row, in manifest order."""
    rows = []
    for i, (m, p) in enumerate(zip(manifest_rows, predictions), start=1):
        row = {"index": i, "path": m["image_path"], "sha256": m["sha256"], "basis": m.get("basis", ""),
               "true_label": m["label"], "predicted_class": p["predicted_class"],
               "correct": p["predicted_class"] == m["label"], "confidence": f"{p['confidence']:.4f}",
               "needs_review": p["needs_review"]}
        row.update({f"p_{c}": f"{p['probabilities'][c]:.4f}" for c in classes})
        rows.append(row)
    return rows


def integrity_checks(manifest_rows, out_rows):
    """Post-run checks written into the report; every value must be True."""
    return {
        "every manifest row has exactly one prediction": len(out_rows) == len(manifest_rows),
        f"{TEST_TOTAL} rows evaluated": len(out_rows) == TEST_TOTAL,
        "no duplicate path in the prediction CSV": len({r["path"] for r in out_rows}) == len(out_rows),
        "prediction order / path mapping matches the manifest":
            [r["path"] for r in out_rows] == [m["image_path"] for m in manifest_rows],
    }


def review_counts(out_rows):
    flagged = [r for r in out_rows if r["needs_review"]]
    wrong = [r for r in out_rows if not r["correct"]]
    return {"needs_review": len(flagged),
            "correct_flagged": sum(r["correct"] for r in flagged),
            "incorrect_flagged": sum(not r["correct"] for r in flagged),
            "unflagged_errors": sum(not r["needs_review"] for r in wrong)}


def git_state():
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                               capture_output=True, text=True, check=True).stdout.strip() != ""
        return head, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", None


def write_csv(rows, path):
    with open(path, "x", newline="", encoding="utf-8") as f:   # "x": never overwrite
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_report(info, metrics, counts, checks, classes=CLASSES):
    f4 = lambda x: f"{x:.4f}"
    m = metrics
    lines = [
        "# Final Test evaluation",
        "",
        "One-time evaluation of the selected model on the frozen Test set, written by",
        "`scripts/evaluate_test.py`.",
        "",
        "## Run",
        "",
        f"- Run identifier: `{info['run_id']}`",
        f"- Timestamp (UTC): {info['timestamp']}",
        f"- Code: git commit `{info['git_head']}`"
        + (" (working tree had uncommitted changes)" if info["git_dirty"] else "" if info["git_dirty"] is False else " (state unknown)"),
        f"- Checkpoint: `{info['checkpoint']}`",
        f"- Checkpoint SHA256: `{info['checkpoint_sha256']}` (verified before prediction)",
        f"- Checkpoint epoch: {info['epoch']} (stored validation macro F1 {f4(info['val_f1'])})",
        f"- Inference: `src/predict.py` (`load_model`, `predict_image`), unchanged",
        f"- Inference transform: {info['transform']} (no training augmentation)",
        f"- `needs_review` threshold: {info['threshold']:.2f} (confidence < threshold -> needs_review; flag only, nothing rejected)",
        f"- Test source: `decisions/test_frozen.csv` only ({info['manifest_rows']} rows, SHA256 of every file verified before prediction)",
        f"- Test manifest SHA256: `{info['manifest_sha256']}`",
        "",
        "## Results",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total | {m['total']} |",
        f"| Correct | {m['correct']} |",
        f"| Incorrect | {m['total'] - m['correct']} |",
        f"| Accuracy | {f4(m['accuracy'])} |",
        f"| Macro precision | {f4(m['macro_precision'])} |",
        f"| Macro recall | {f4(m['macro_recall'])} |",
        f"| Macro F1 | {f4(m['macro_f1'])} |",
        "",
        "Per-class metrics:",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|",
    ]
    lines += [f"| {c} | {f4(v['precision'])} | {f4(v['recall'])} | {f4(v['f1'])} | {v['support']} |"
              for c, v in m["per_class"].items()]
    lines += ["", "Confusion matrix (rows = true class, columns = predicted class):", "",
              "| true \\ predicted | " + " | ".join(classes) + " |",
              "|---|" + "---|" * len(classes)]
    lines += [f"| {c} | " + " | ".join(str(x) for x in row) + " |" for c, row in zip(classes, m["matrix"])]
    lines += ["", "## needs_review", "",
              "| | Count |", "|---|---|",
              f"| needs_review (confidence < {info['threshold']:.2f}) | {counts['needs_review']} |",
              f"| correct predictions flagged | {counts['correct_flagged']} |",
              f"| incorrect predictions flagged | {counts['incorrect_flagged']} |",
              f"| errors not flagged | {counts['unflagged_errors']} |",
              "", "## Integrity checks", ""]
    lines += [f"- {name}: {'PASS' if ok else 'FAIL'}" for name, ok in checks.items()]
    lines += ["", "## Scope", "",
              "- The Test manifest `decisions/test_frozen.csv` was the only Test source; no folder was scanned.",
              "- Test was not used for model selection (`10_final_model_selection.md`) or for the",
              "  `needs_review` threshold (`11_needs_review_threshold.md`); both were decided on validation.",
              "- Neysan evaluation was not part of this run.",
              f"- Per-image predictions: `{OUT_CSV.relative_to(REPO_ROOT).as_posix()}`.", ""]
    return "\n".join(lines)


def main():
    for p in (OUT_CSV, OUT_REPORT):
        if p.exists():
            sys.exit(f"refusing to run: {p.relative_to(REPO_ROOT)} already exists (Test is evaluated only once)")

    # --- verification, before any prediction ---
    from src.dataset import DEFAULT_CONFIG, load_source_roots, resolve_image_path
    roots = load_source_roots(DEFAULT_CONFIG)
    manifest_rows = read_manifest(TEST_MANIFEST)
    try:
        files = verify_manifest(manifest_rows, lambda p: resolve_image_path(p, roots))
    except ValueError as e:
        sys.exit(str(e))
    print(f"Test manifest verified: {len(manifest_rows)} rows, {TEST_PER_CLASS} per class, all SHA256 match")

    checkpoint_sha = sha256_of(CHECKPOINT)
    if checkpoint_sha != CHECKPOINT_SHA256:
        sys.exit(f"checkpoint SHA256 mismatch: {checkpoint_sha} (expected {CHECKPOINT_SHA256})")
    import src.predict as P
    if P.NEEDS_REVIEW_THRESHOLD != EXPECTED_THRESHOLD:
        sys.exit(f"needs_review threshold is {P.NEEDS_REVIEW_THRESHOLD}, expected {EXPECTED_THRESHOLD}")
    if [c for c in P.CLASSES] != CLASSES:
        sys.exit(f"class order differs: {P.CLASSES}")
    print("checkpoint SHA256 and threshold verified")

    # --- prediction with the unchanged inference code ---
    model, eval_transform, checkpoint = P.load_model(CHECKPOINT)
    started = datetime.now(timezone.utc)
    predictions = [P.predict_image(f, model, eval_transform, path=m["image_path"])
                   for f, m in zip(files, manifest_rows)]

    out_rows = prediction_rows(manifest_rows, predictions)
    metrics = compute_metrics([r["true_label"] for r in out_rows], [r["predicted_class"] for r in out_rows])
    counts = review_counts(out_rows)
    checks = integrity_checks(manifest_rows, out_rows)
    head, dirty = git_state()
    info = {
        "run_id": "final_test_" + started.strftime("%Y%m%dT%H%M%SZ"),
        "timestamp": started.strftime("%Y-%m-%d %H:%M:%S"),
        "git_head": head, "git_dirty": dirty,
        "checkpoint": CHECKPOINT.relative_to(REPO_ROOT).as_posix(), "checkpoint_sha256": checkpoint_sha,
        "epoch": checkpoint["epoch"], "val_f1": checkpoint["val_f1"],
        "transform": " -> ".join(repr(t) for t in eval_transform.transforms).replace("\n", " "),
        "threshold": P.NEEDS_REVIEW_THRESHOLD,
        "manifest_rows": len(manifest_rows), "manifest_sha256": sha256_of(TEST_MANIFEST),
    }
    write_csv(out_rows, OUT_CSV)
    with open(OUT_REPORT, "x", encoding="utf-8") as f:
        f.write(render_report(info, metrics, counts, checks))

    print(f"accuracy {metrics['accuracy']:.4f} | macro F1 {metrics['macro_f1']:.4f} | "
          f"needs_review {counts['needs_review']} | integrity {'PASS' if all(checks.values()) else 'FAIL'}")
    print(f"written: {OUT_REPORT.relative_to(REPO_ROOT)}, {OUT_CSV.relative_to(REPO_ROOT)}")
    if not all(checks.values()):
        sys.exit("integrity checks failed; see the report")


if __name__ == "__main__":
    main()
