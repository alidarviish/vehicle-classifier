"""Second evaluation of the frozen Test set, for the reopened final model `swin_t_ft_aug`.

Run this ONCE, from the repository root, only after the protocol has been approved:
    python scripts/evaluate_test_swin.py

Context: the frozen Test set was first evaluated once for the previous final model
`resnet224_ft_aug` (scripts/evaluate_test.py, reports/experiments/12_*). The final model selection
was later reopened on validation only (decisions/REOPEN_FINAL_MODEL_SELECTION.md) and
`swin_t_ft_aug` was selected; its needs_review threshold 0.95 was chosen on validation only
(reports/experiments/23_swin_needs_review_threshold.md). This run is therefore the SECOND use of
the same frozen Test set: a documented protocol deviation, not a pristine held-out estimate. Its
result is not used for model selection or for the threshold.

What it does, in this order:
1. Refuses to start if its own output files already exist.
2. Records the SHA256 of the historical ResNet Test outputs (12_*) so they can be checked unchanged.
3. Verifies the Test manifest decisions/test_frozen.csv with the same checks as evaluate_test.py
   (400 rows, 50 per class, 8 classes, no duplicate path or SHA256, every file's SHA256 matches),
   and that the manifest itself has its recorded SHA256.
4. Verifies the Swin checkpoint SHA256, that src/predict.py uses this checkpoint, the threshold 0.95
   and the class order. If any check in 3 or 4 fails, it stops BEFORE any prediction.
5. Predicts every Test image with src/predict.py (load_model / predict_image, Swin-Tiny).
6. Writes reports/experiments/24_swin_final_test_predictions.csv and
   reports/experiments/24_swin_final_test_evaluation.md (never overwrites; mode "x").

Read-only for the dataset, the manifest, the checkpoints and the historical outputs. Neysan is not
used. scripts/evaluate_test.py and scripts/evaluate_neysan.py are not changed; only their
model-independent helpers are imported.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# model-independent helpers of the historical Test evaluation (no model, checkpoint or threshold in them)
from scripts.evaluate_test import (CLASSES, TEST_MANIFEST, TEST_PER_CLASS, compute_metrics,  # noqa: E402
                                   git_state, integrity_checks, prediction_rows, read_manifest,
                                   render_report, review_counts, sha256_of, verify_manifest, write_csv)

CHECKPOINT = REPO_ROOT / "checkpoints" / "swin_t_ft_aug_best.pt"
CHECKPOINT_SHA256 = "f03bacde0a155461ea4aede0b813b8688d6ed17973d5ad19b0d2c1b913ff64db"
EXPECTED_ARCHITECTURE = "Swin-Tiny"
EXPECTED_THRESHOLD = 0.95
TEST_MANIFEST_SHA256 = "e1727eea750e73a4924d79fbb5fb810af8f8a05d822754c51ade8376e0622540"
OUT_DIR = REPO_ROOT / "reports" / "experiments"
OUT_CSV = OUT_DIR / "24_swin_final_test_predictions.csv"
OUT_REPORT = OUT_DIR / "24_swin_final_test_evaluation.md"
HISTORICAL_OUTPUTS = [OUT_DIR / "12_final_test_predictions.csv", OUT_DIR / "12_final_test_evaluation.md"]


def hashes(paths):
    """SHA256 of each existing file (None if missing); used to show historical files are unchanged."""
    return {p.name: sha256_of(p) if p.is_file() else None for p in paths}


def render_swin_report(info, metrics, counts, checks):
    """Own header and scope; the Results, needs_review and Integrity sections reuse render_report."""
    base = render_report(info, metrics, counts, checks)
    if "## Results" not in base or "## Scope" not in base:
        raise RuntimeError("render_report layout changed; cannot reuse its result sections")
    body = "## Results" + base.split("## Results", 1)[1].split("## Scope", 1)[0]
    head = [
        "# Final Test evaluation: `swin_t_ft_aug` (second use of the frozen Test set)",
        "",
        "Written by `scripts/evaluate_test_swin.py`.",
        "",
        "## Protocol status",
        "",
        "- The frozen Test set was first evaluated once for the previous final model `resnet224_ft_aug`",
        "  (`12_final_test_evaluation.md`). Those results are historical and unchanged.",
        "- The final model selection was reopened on validation only and `swin_t_ft_aug` was selected",
        "  (`decisions/REOPEN_FINAL_MODEL_SELECTION.md`). The `needs_review` threshold 0.95 was chosen on",
        "  validation only (`23_swin_needs_review_threshold.md`).",
        "- This run is the second use of the same frozen Test set. It is a documented protocol deviation:",
        "  the result is not a pristine held-out estimate.",
        "- This Test result is not used for model selection or for the threshold.",
        "",
        "## Run",
        "",
        f"- Run identifier: `{info['run_id']}`",
        f"- Timestamp (UTC): {info['timestamp']}",
        f"- Code: git commit `{info['git_head']}`"
        + (" (working tree had uncommitted changes)" if info["git_dirty"] else "" if info["git_dirty"] is False else " (state unknown)"),
        f"- Checkpoint: `{info['checkpoint']}` ({info['architecture']})",
        f"- Checkpoint SHA256: `{info['checkpoint_sha256']}` (verified before prediction)",
        f"- Checkpoint epoch: {info['epoch']} (stored validation macro F1 {info['val_f1']:.4f})",
        "- Inference: `src/predict.py` (`load_model`, `predict_image`)",
        f"- Inference transform: {info['transform']} (no training augmentation)",
        f"- `needs_review` threshold: {info['threshold']:.2f} (confidence < threshold -> needs_review; flag only, nothing rejected)",
        f"- Test source: `decisions/test_frozen.csv` only ({info['manifest_rows']} rows, SHA256 of every file verified before prediction)",
        f"- Test manifest SHA256: `{info['manifest_sha256']}`",
        "",
    ]
    scope = [
        "## Scope",
        "",
        "- The Test manifest `decisions/test_frozen.csv` was the only Test source; no folder was scanned.",
        "- The historical outputs `12_final_test_predictions.csv` and `12_final_test_evaluation.md` were",
        "  only hashed, before and after this run.",
        "- Neysan evaluation was not part of this run.",
        f"- Per-image predictions: `{OUT_CSV.relative_to(REPO_ROOT).as_posix()}`.",
        "",
    ]
    return "\n".join(head) + body + "\n".join(scope)


def main():
    for p in (OUT_CSV, OUT_REPORT):
        if p.exists():
            sys.exit(f"refusing to run: {p.relative_to(REPO_ROOT)} already exists (this evaluation runs only once)")
    historical_before = hashes(HISTORICAL_OUTPUTS)

    # --- verification, before any prediction ---
    manifest_sha = sha256_of(TEST_MANIFEST)
    if manifest_sha != TEST_MANIFEST_SHA256:
        sys.exit(f"Test manifest SHA256 mismatch: {manifest_sha} (expected {TEST_MANIFEST_SHA256})")
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
    if Path(P.FINAL_CHECKPOINT).resolve() != CHECKPOINT.resolve():
        sys.exit(f"src/predict.py FINAL_CHECKPOINT is {P.FINAL_CHECKPOINT}, expected {CHECKPOINT}")
    if P.NEEDS_REVIEW_THRESHOLD != EXPECTED_THRESHOLD:
        sys.exit(f"needs_review threshold is {P.NEEDS_REVIEW_THRESHOLD}, expected {EXPECTED_THRESHOLD}")
    if list(P.CLASSES) != CLASSES:
        sys.exit(f"class order differs: {P.CLASSES}")
    print("Swin checkpoint SHA256, predict.py checkpoint, threshold and class order verified")

    # --- prediction with the current inference code ---
    model, eval_transform, checkpoint = P.load_model(CHECKPOINT)   # load_model checks architecture and class_to_idx
    if checkpoint["architecture"] != EXPECTED_ARCHITECTURE:
        sys.exit(f"architecture is {checkpoint['architecture']}, expected {EXPECTED_ARCHITECTURE}")
    started = datetime.now(timezone.utc)
    predictions = [P.predict_image(f, model, eval_transform, path=m["image_path"])
                   for f, m in zip(files, manifest_rows)]

    out_rows = prediction_rows(manifest_rows, predictions)
    metrics = compute_metrics([r["true_label"] for r in out_rows], [r["predicted_class"] for r in out_rows])
    counts = review_counts(out_rows)
    checks = integrity_checks(manifest_rows, out_rows)
    checks.update({
        "Test manifest SHA256 unchanged after prediction": sha256_of(TEST_MANIFEST) == manifest_sha,
        "Swin checkpoint SHA256 unchanged after prediction": sha256_of(CHECKPOINT) == CHECKPOINT_SHA256,
        "historical ResNet Test outputs (12_*) unchanged": hashes(HISTORICAL_OUTPUTS) == historical_before,
    })
    head, dirty = git_state()
    info = {
        "run_id": "swin_final_test_" + started.strftime("%Y%m%dT%H%M%SZ"),
        "timestamp": started.strftime("%Y-%m-%d %H:%M:%S"),
        "git_head": head, "git_dirty": dirty,
        "checkpoint": CHECKPOINT.relative_to(REPO_ROOT).as_posix(), "checkpoint_sha256": checkpoint_sha,
        "architecture": checkpoint["architecture"],
        "epoch": checkpoint["epoch"], "val_f1": checkpoint["val_f1"],
        "transform": " -> ".join(repr(t) for t in eval_transform.transforms).replace("\n", " "),
        "threshold": P.NEEDS_REVIEW_THRESHOLD,
        "manifest_rows": len(manifest_rows), "manifest_sha256": manifest_sha,
    }
    write_csv(out_rows, OUT_CSV)
    with open(OUT_REPORT, "x", encoding="utf-8") as f:
        f.write(render_swin_report(info, metrics, counts, checks))

    print(f"accuracy {metrics['accuracy']:.4f} | macro F1 {metrics['macro_f1']:.4f} | "
          f"needs_review {counts['needs_review']} | integrity {'PASS' if all(checks.values()) else 'FAIL'}")
    print(f"written: {OUT_REPORT.relative_to(REPO_ROOT)}, {OUT_CSV.relative_to(REPO_ROOT)}")
    if not all(checks.values()):
        sys.exit("integrity checks failed; see the report")


if __name__ == "__main__":
    main()
