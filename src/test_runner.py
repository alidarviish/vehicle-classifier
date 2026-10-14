"""Test runner for the final model on a folder of new images (for example the mentors' test set).

Start it with:  python run_test.py --input path/to/TEST

Steps:
  1. find_images     find every image under --input; the class folder above an image is its true label
  2. load the model  the frozen final checkpoint (SHA256 checked) and the Neysan detector
  3. predict         8-class prediction for every image; if the Neysan detector says "Neysan", the
                     output becomes "unknown"
  4. metrics         accuracy, precision, recall, F1, confusion matrix (overall and per test group)
  5. save            CSV files, confusion matrix picture, metrics.json and summary.txt

The runner only reads the input images and writes only into the output folder. Its results are not
used to change the model, the threshold or anything else.
"""

import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset import CLASSES  # noqa: E402  the 8 classes, in the model's index order

# ------------------------------------------------------------------------------------------- settings

# final model: Swin-Tiny v3 (src/train_v3.py, run v3_final)
# previous final model: swin_t_ft_aug_best.pt, sha256 f03bacde0a155461ea4aede0b813b8688d6ed17973d5ad19b0d2c1b913ff64db
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "v3_final_best.pt"
FINAL_CHECKPOINT_SHA256 = "bb8b159a2b1c42ab92cd1b1c33020ac32550baebe5200a20071525e3279d3848"

# Neysan detector: built by scripts/build_neysan_detector_v3.py for the model above
# previous detector: neysan_detector.json, sha256 1f4106cbcef2c9ad3d7b5b64e017ede1f4ce03ecbdc3fb9527af0a26a845e3c7
DEFAULT_DETECTOR = REPO_ROOT / "checkpoints" / "neysan_detector_v3.json"
DETECTOR_SHA256 = "69bc046a9c4655d8790e128afa60e8446c255a922523833a5cf03c1d3f57cf33"

UNKNOWN = "unknown"                    # output for an image the Neysan detector rejects
COLUMNS = CLASSES + [UNKNOWN]          # columns of the confusion matrix (predicted values)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
CSV_ENCODING = "utf-8-sig"             # with BOM, so Excel shows non-English file names correctly


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------ 1. find images

def find_images(input_dir, output_dir):
    """Return (images, skipped).

    images:  one dict per image file: path, image_path (relative), true_label (or None), group
    skipped: one dict per other file, with the reason
    The true label is the nearest folder above the image whose name is a class (not case-sensitive).
    The group is the folder part above that class folder, e.g. "test2" for test2/vanet/x.jpg.
    """
    class_names = {c.lower(): c for c in CLASSES}
    images, skipped = [], []
    for root, dirs, files in os.walk(input_dir):
        root = Path(root)
        dirs.sort()
        # never read our own output folder, or an earlier results folder inside the input
        is_output = root.resolve() == Path(output_dir).resolve()
        is_old_results = root != Path(input_dir) and {"predictions.csv", "summary.txt"} <= set(files)
        if is_output or is_old_results:
            dirs[:] = []
            if is_old_results:
                skipped.append({"image_path": root.relative_to(input_dir).as_posix() + "/",
                                "reason": "earlier results folder (not scanned)"})
            continue
        for name in sorted(files):
            path = root / name
            rel = path.relative_to(input_dir)
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                skipped.append({"image_path": rel.as_posix(), "reason": "not an image file"})
                continue
            folders = list(rel.parts[:-1])
            label_at = [i for i, f in enumerate(folders) if f.strip().lower() in class_names]
            if label_at:
                i = label_at[-1]
                true_label, group = class_names[folders[i].strip().lower()], "/".join(folders[:i]) or "."
            else:
                true_label, group = None, "/".join(folders) or "."
            images.append({"path": path, "image_path": rel.as_posix(), "true_label": true_label, "group": group})
    return images, skipped


# ------------------------------------------------------------------------------------ 2. the models

class FinalModelPredictor:
    """The frozen 8-class model, loaded with src/predict.load_model (same code as the project)."""

    def __init__(self, checkpoint, device="auto"):
        import torch
        from src.predict import load_model
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("ERROR: --device cuda was requested but no GPU is available. Use --device cpu or auto.")
        self.torch = torch
        self.device = torch.device(device)
        model, self.transform, ckpt = load_model(checkpoint)   # checks the architecture and class order
        self.model = model.to(self.device).eval()
        # keep the 768-number feature vector that goes into the last layer (used by the Neysan detector)
        self._features = {}
        self.model.net.head.register_forward_hook(lambda m, inp, out: self._features.__setitem__("f", inp[0]))
        self.last_features = []
        self.info = {"architecture": ckpt.get("architecture"), "epoch": ckpt.get("epoch"), "device": str(self.device)}

    def __call__(self, rgb_images):
        """List of RGB images -> list of 8 probabilities per image (CLASSES order)."""
        x = self.torch.stack([self.transform(im) for im in rgb_images]).to(self.device)
        with self.torch.inference_mode():
            probs = self.torch.softmax(self.model(x).float(), dim=1).cpu()
        self.last_features = self._features["f"].float().cpu().tolist()
        return probs.tolist()


class NeysanDetector:
    """Logistic regression: p(Neysan) = sigmoid(sum(((feature - mean) / scale) * coef) + intercept)."""

    def __init__(self, path, allow_other=False, checkpoint_sha=FINAL_CHECKPOINT_SHA256):
        raw = Path(path).read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        if self.sha256 != DETECTOR_SHA256 and not allow_other:
            raise SystemExit(f"ERROR: Neysan detector SHA256 {self.sha256} is not the accepted file ({DETECTOR_SHA256}).")
        d = json.loads(raw.decode("utf-8"))
        if d.get("checkpoint_sha256") != checkpoint_sha:   # a detector only fits the model it was built for
            raise SystemExit("ERROR: the Neysan detector was not built for this model checkpoint.")
        self.mean, self.scale, self.coef = d["mean"], d["scale"], d["coef"]
        self.intercept, self.threshold = float(d["intercept"]), float(d["threshold"])
        if not len(self.mean) == len(self.scale) == len(self.coef) == d["feature_dim"]:
            raise SystemExit("ERROR: the Neysan detector file is damaged (sizes do not match).")

    def score(self, feature):
        z = self.intercept + sum((f - m) / s * c for f, m, s, c in zip(feature, self.mean, self.scale, self.coef))
        return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))   # sigmoid, no overflow


# ------------------------------------------------------------------------------------ 4. metrics

def compute_metrics(rows):
    """Same definitions as the project's evaluation (scripts/evaluate_test.compute_metrics).

    "unknown" is an extra predicted column: it is a miss for the image's true class.
    Macro precision / recall / F1 average over the 8 classes.
    """
    matrix = [[0] * len(COLUMNS) for _ in CLASSES]           # rows = true class, columns = predicted
    for r in rows:
        matrix[CLASSES.index(r["true_label"])][COLUMNS.index(r["predicted_class"])] += 1
    per_class = {}
    for i, c in enumerate(CLASSES):
        tp = matrix[i][i]
        predicted = sum(matrix[j][i] for j in range(len(CLASSES)))
        support = sum(matrix[i])
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[c] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    correct = sum(matrix[i][i] for i in range(len(CLASSES)))
    n = len(rows)
    return {"images": n, "correct": correct, "wrong": n - correct,
            "accuracy": correct / n if n else 0.0,
            "macro_precision": sum(v["precision"] for v in per_class.values()) / len(CLASSES),
            "macro_recall": sum(v["recall"] for v in per_class.values()) / len(CLASSES),
            "macro_f1": sum(v["f1"] for v in per_class.values()) / len(CLASSES),
            "needs_review": sum(r["needs_review"] for r in rows),
            "unknown": sum(r["predicted_class"] == UNKNOWN for r in rows),
            "per_class": per_class,
            "confusion_matrix": {"rows_true": CLASSES, "columns_predicted": COLUMNS, "matrix": matrix}}


# ------------------------------------------------------------------------------------ 5. save

def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding=CSV_ENCODING) as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def save_confusion_matrix(m, out_dir):
    """confusion_matrix.csv and confusion_matrix.png (counts, and row percentages = recall)."""
    matrix = m["confusion_matrix"]["matrix"]
    with open(out_dir / "confusion_matrix.csv", "w", newline="", encoding=CSV_ENCODING) as f:
        w = csv.writer(f)
        w.writerow(["true \\ predicted"] + COLUMNS)
        for c, line in zip(CLASSES, matrix):
            w.writerow([c] + line)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    counts = np.array(matrix, dtype=float)
    totals = counts.sum(axis=1, keepdims=True)
    share = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, data, title, fmt in ((axes[0], counts, "counts", "{:.0f}"), (axes[1], share, "share of each true class", "{:.2f}")):
        ax.imshow(data, cmap="Blues", vmin=0, vmax=max(data.max(), 1e-9))
        ax.set_xticks(range(len(COLUMNS)), COLUMNS, rotation=45, ha="right")
        ax.set_yticks(range(len(CLASSES)), CLASSES)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true (folder)")
        ax.set_title(title)
        for i in range(len(CLASSES)):
            for j in range(len(COLUMNS)):
                if counts[i, j] > 0:
                    ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center", fontsize=8,
                            color="white" if data[i, j] > 0.6 * data.max() else "black")
    fig.suptitle(f"{m['images']} images, accuracy {m['accuracy']:.4f}, macro F1 {m['macro_f1']:.4f}")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=130)
    plt.close(fig)


def summary_text(info, counts, overall, per_group):
    det = info["neysan_detector"]
    lines = ["VEHICLE CLASSIFIER - TEST RUN SUMMARY", "=" * 60,
             f"input folder     : {info['input']}",
             f"model            : {info['model']}, sha256 {info['checkpoint_sha256'][:16]}..., device {info['device']}",
             f"needs_review     : confidence < {info['needs_review_threshold']} (a flag only; the prediction is kept)",
             "neysan detector  : " + (f"ON - p(Neysan) >= {det['threshold']} -> unknown" if det["enabled"]
                                      else f"OFF ({det['reason']}) - plain 8-class output"),
             "",
             f"images found     : {counts['images']}   (with class folder {counts['labeled']}, without {counts['unlabeled']})",
             f"images failed    : {counts['failed']}   (see failed_images.csv)",
             f"other files      : {counts['skipped']}   (see skipped_files.csv)",
             f"needs_review     : {counts['needs_review']}   (see needs_review.csv)",
             f"unknown (Neysan) : {counts['unknown']}   (see unknown_predictions.csv)", ""]
    if overall:
        lines += ["RESULTS (images with a class folder)", "-" * 60,
                  f"correct / wrong  : {overall['correct']} / {overall['wrong']}   (see wrong_predictions.csv)",
                  f"accuracy         : {overall['accuracy']:.4f}",
                  f"macro precision  : {overall['macro_precision']:.4f}",
                  f"macro recall     : {overall['macro_recall']:.4f}",
                  f"macro F1         : {overall['macro_f1']:.4f}",
                  "(an 'unknown' output counts as wrong for its folder class)", "",
                  "class          precision  recall   f1       images"]
        for c in CLASSES:
            v = overall["per_class"][c]
            lines.append(f"  {c:12s} {v['precision']:.4f}     {v['recall']:.4f}   {v['f1']:.4f}   {v['support']}")
        if len(per_group) > 1:
            lines += ["", "PER TEST GROUP", "-" * 60]
            for g, m in per_group.items():
                lines.append(f"  {g:16s} images {m['images']:5d}  accuracy {m['accuracy']:.4f}  "
                             f"macro F1 {m['macro_f1']:.4f}  unknown {m['unknown']}")
    else:
        lines.append("No class folders found: predictions only, no metrics.")
    if info["warnings"]:
        lines += ["", "WARNINGS", "-" * 60] + [f"  - {w}" for w in info["warnings"]]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------ main

def run(input_dir, output_dir, checkpoint=DEFAULT_CHECKPOINT, batch_size=32, device="auto",
        overwrite=False, allow_other_checkpoint=False, predictor=None,
        neysan_detector=DEFAULT_DETECTOR, use_neysan_detector=True, allow_other_detector=False,
        review_threshold=None, report_images=True):
    start = time.time()
    input_dir, output_dir = Path(input_dir).expanduser(), Path(output_dir).expanduser()

    # ---- checks before any work
    if not input_dir.is_dir():
        raise SystemExit(f"ERROR: input folder not found: {input_dir}")
    if output_dir.resolve() == input_dir.resolve():
        raise SystemExit("ERROR: --output must be a different folder than --input")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise SystemExit(f"ERROR: output folder is not empty: {output_dir}\n       choose another --output or add --overwrite")
    if batch_size < 1:
        raise SystemExit("ERROR: --batch-size must be at least 1")

    # ---- 2. load the 8-class model (predictor can be replaced by a fake one in tests)
    if predictor is None:
        checkpoint = Path(checkpoint).expanduser()
        if not checkpoint.is_file():
            raise SystemExit(f"ERROR: checkpoint not found: {checkpoint}\n       pass the model file with --checkpoint")
        ckpt_sha = sha256_of(checkpoint)
        if ckpt_sha != FINAL_CHECKPOINT_SHA256 and not allow_other_checkpoint:
            raise SystemExit(f"ERROR: checkpoint SHA256 {ckpt_sha} is not the final model ({FINAL_CHECKPOINT_SHA256}).")
        predictor = FinalModelPredictor(checkpoint, device)
    else:
        ckpt_sha = getattr(predictor, "sha256", "test-predictor")
    from src.predict import NEEDS_REVIEW_THRESHOLD, needs_review   # the one place the default threshold is defined
    if review_threshold is not None:   # only to re-run an earlier model with its own threshold
        NEEDS_REVIEW_THRESHOLD = float(review_threshold)

    # ---- 2. load the Neysan detector
    detector, det_info = None, {"enabled": False, "reason": "--no-neysan-detector"}
    if use_neysan_detector:
        det_path = Path(neysan_detector).expanduser()
        if det_path.is_file():
            detector = NeysanDetector(det_path, allow_other_detector, ckpt_sha)
            det_info = {"enabled": True, "sha256": detector.sha256, "threshold": detector.threshold}
        elif Path(neysan_detector) == DEFAULT_DETECTOR:
            print(f"WARNING: Neysan detector not found ({det_path}); plain 8-class output", flush=True)
            det_info = {"enabled": False, "reason": "detector file not found"}
        else:
            raise SystemExit(f"ERROR: Neysan detector not found: {det_path}")

    # ---- 1. find the images
    images, skipped = find_images(input_dir, output_dir)
    print(f"found {len(images)} images and {len(skipped)} other files", flush=True)

    # ---- 3. predict, in batches
    rows, failed = [], []

    def predict_batch(batch):
        """batch = list of (image dict, RGB image). Adds one row per image to rows."""
        try:
            probs_list = predictor([rgb for _, rgb in batch])
            feats_list = list(predictor.last_features)
        except Exception as e:   # e.g. GPU out of memory: try the images one by one
            print(f"WARNING: batch failed ({type(e).__name__}); retrying one image at a time", flush=True)
            probs_list, feats_list = [], []
            for im, rgb in batch:
                try:
                    probs_list.append(predictor([rgb])[0])
                    feats_list.append(predictor.last_features[0])
                except Exception as e2:
                    probs_list.append(None)
                    feats_list.append(None)
                    failed.append({"image_path": im["image_path"], "error": f"{type(e2).__name__}: {e2}"[:300]})
        for (im, _), probs, feats in zip(batch, probs_list, feats_list):
            if probs is None:
                continue
            best = max(range(len(CLASSES)), key=lambda i: probs[i])
            model_class, confidence = CLASSES[best], float(probs[best])
            score = detector.score(feats) if detector else None
            final = UNKNOWN if score is not None and score >= detector.threshold else model_class
            rows.append({"image_path": im["image_path"], "group": im["group"],
                         "true_label": im["true_label"] or "",
                         "predicted_class": final, "model_class": model_class,
                         "neysan_score": "" if score is None else f"{score:.6f}",
                         "confidence": f"{confidence:.6f}",
                         "needs_review": needs_review(confidence, NEEDS_REVIEW_THRESHOLD),
                         "correct": "" if im["true_label"] is None else final == im["true_label"],
                         **{f"p_{c}": f"{p:.6f}" for c, p in zip(CLASSES, probs)}})

    batch = []
    for n, im in enumerate(images, 1):
        try:
            with Image.open(im["path"]) as img:
                rgb = img.convert("RGB")         # same conversion as src/predict.py
        except Exception as e:
            failed.append({"image_path": im["image_path"], "error": f"{type(e).__name__}: {e}"[:300]})
        else:
            batch.append((im, rgb))
            if len(batch) == batch_size:
                predict_batch(batch)
                batch = []
        if n % 200 == 0:
            print(f"  {n}/{len(images)} images", flush=True)
    if batch:
        predict_batch(batch)

    # ---- 4. metrics (only images that have a class folder)
    labeled = [r for r in rows if r["true_label"]]
    overall = compute_metrics(labeled) if labeled else None
    groups = defaultdict(list)
    for r in labeled:
        groups[r["group"]].append(r)
    per_group = {g: compute_metrics(rs) for g, rs in sorted(groups.items())}

    warnings = []
    if failed:
        warnings.append(f"{len(failed)} image(s) could not be read; they are not in the metrics (failed_images.csv)")
    if labeled and len(labeled) < len(rows):
        warnings.append(f"{len(rows) - len(labeled)} image(s) have no class folder; metrics use only the {len(labeled)} others")
    if overall:
        missing = [c for c in CLASSES if overall["per_class"][c]["support"] == 0]
        if missing:
            warnings.append(f"no images for {missing}; macro scores still average over all 8 classes")

    # ---- 5. save
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["image_path", "group", "true_label", "predicted_class", "model_class", "neysan_score",
              "confidence", "needs_review", "correct"] + [f"p_{c}" for c in CLASSES]
    write_csv(output_dir / "predictions.csv", rows, fields)
    write_csv(output_dir / "wrong_predictions.csv", [r for r in labeled if not r["correct"]], fields)
    write_csv(output_dir / "needs_review.csv", [r for r in rows if r["needs_review"]], fields)
    write_csv(output_dir / "unknown_predictions.csv", [r for r in rows if r["predicted_class"] == UNKNOWN], fields)
    write_csv(output_dir / "failed_images.csv", failed, ["image_path", "error"])
    write_csv(output_dir / "skipped_files.csv", skipped, ["image_path", "reason"])
    if overall:
        save_confusion_matrix(overall, output_dir)
    if labeled and report_images:   # per-test folders with pictures of errors, needs_review and unknown
        from src.test_report import save_group_reports
        save_group_reports(labeled, groups, input_dir, output_dir, compute_metrics, save_confusion_matrix, write_csv, fields)

    counts = {"images": len(images), "labeled": len(labeled), "unlabeled": len(rows) - len(labeled),
              "failed": len(failed), "skipped": len(skipped),
              "needs_review": sum(r["needs_review"] for r in rows),
              "unknown": sum(r["predicted_class"] == UNKNOWN for r in rows)}
    info = {"input": str(input_dir.resolve()), "output": str(output_dir.resolve()),
            "model": f"{predictor.info['architecture']} (epoch {predictor.info['epoch']})",
            "device": predictor.info["device"], "checkpoint_sha256": ckpt_sha,
            "needs_review_threshold": NEEDS_REVIEW_THRESHOLD, "neysan_detector": det_info,
            "counts": counts, "warnings": warnings, "overall": overall,
            "per_group": per_group if len(per_group) > 1 else None,
            "runtime_seconds": round(time.time() - start, 1)}
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    text = summary_text(info, counts, overall, per_group)
    (output_dir / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text, flush=True)
    return info
