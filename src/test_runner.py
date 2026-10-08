"""Test / inference runner for the final model, for a folder of new images (e.g. the mentors' Test set).

Run it through the thin command-line wrapper in the repository root:
    python run_test.py --input path/to/TEST

What it does:
- finds every file under --input (recursively); image files are predicted, every other file is
  listed in skipped_files.csv (nothing is skipped silently);
- reads the ground truth from the folder names: the nearest folder (below --input) whose name is
  one of the 8 classes, e.g. TEST/ambulance/x.jpg or test2/ambulance/x.jpg. Images with no such
  folder are predicted without a label;
- predicts with the frozen final model exactly as src/predict.py does: the same checkpoint loader,
  the same evaluation transform recorded in the checkpoint (Resize 224x224 -> ToTensor -> ImageNet
  normalization, no augmentation), the same class order and the same needs_review rule (0.95);
- images that cannot be opened are listed in failed_images.csv; the run continues;
- evaluation mode (ground truth found): metrics with the same metric code as the project's recorded
  Test evaluation (scripts/evaluate_test.compute_metrics), per test group if there are several;
  prediction-only mode (no ground truth): predictions only, no metrics.

The runner is for final inference / evaluation only. Its results must not be used to choose a model,
a threshold or any setting. It only reads the checkpoint and the input images; it writes only into
the output folder.
"""

import csv
import hashlib
import json
import os
import platform
import sys
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset import CLASSES  # noqa: E402  the 8 classes; index = position (also checked against the checkpoint)

# SHA256 of the frozen final checkpoint swin_t_ft_aug_best.pt (epoch 17), as recorded in the README and
# reports/experiments/22-25. The runner refuses any other file unless --allow-other-checkpoint is given.
FINAL_CHECKPOINT_SHA256 = "f03bacde0a155461ea4aede0b813b8688d6ed17973d5ad19b0d2c1b913ff64db"
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "swin_t_ft_aug_best.pt"

# Neysan rejection (added 2026-10-08): a logistic regression on the 768-number feature vector of the
# frozen model (the input of its last layer). It is applied to every image; if it says Neysan, the output
# becomes UNKNOWN, whatever the 8-class prediction. The 8-class model itself is unchanged.
# Built and accepted with fixed rules in colab/neysan_detector_colab.ipynb (see docs/NEYSAN_REJECTION.md).
UNKNOWN = "unknown"
DEFAULT_DETECTOR = REPO_ROOT / "checkpoints" / "neysan_detector.json"
DETECTOR_SHA256 = "1f4106cbcef2c9ad3d7b5b64e017ede1f4ce03ecbdc3fb9527af0a26a845e3c7"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
CSV_ENCODING = "utf-8-sig"   # UTF-8 with BOM: opens correctly in Excel, also with non-English file names
CLASS_LOOKUP = {c.lower(): c for c in CLASSES}


# ----------------------------------------------------------------------------------------------- files

def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def discover(input_dir, exclude_dir=None):
    """Walk input_dir. Returns (images, skipped, empty_dirs).

    images:  dicts with path, rel_path, true_label (or None), group and label_note
    skipped: dicts with rel_path and reason (every non-image file)
    """
    input_dir = Path(input_dir)
    exclude_dir = Path(exclude_dir).resolve() if exclude_dir else None
    images, skipped, empty_dirs = [], [], []
    for root, dirs, files in os.walk(input_dir):
        root_path = Path(root)
        if exclude_dir is not None and root_path.resolve() == exclude_dir:
            dirs[:] = []
            continue
        if root_path != input_dir and {"predictions.csv", "summary.txt"} <= set(files):
            # an earlier results folder of this runner inside the input: never treat its PNG as test data
            skipped.append({"image_path": root_path.relative_to(input_dir).as_posix() + "/",
                            "reason": "earlier results folder of run_test.py (not scanned)"})
            dirs[:] = []
            continue
        dirs.sort()
        if not dirs and not files:
            empty_dirs.append(root_path.relative_to(input_dir).as_posix() or ".")
        for name in sorted(files):
            path = root_path / name
            rel = path.relative_to(input_dir)
            rel_posix = rel.as_posix()
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                reason = "hidden/system file" if name.startswith(".") or name.lower() in ("thumbs.db", "desktop.ini") \
                    else f"not an image extension ({path.suffix or 'no extension'})"
                skipped.append({"image_path": rel_posix, "reason": reason})
                continue
            folders = list(rel.parts[:-1])
            label_positions = [i for i, part in enumerate(folders) if part.strip().lower() in CLASS_LOOKUP]
            if label_positions:
                i = label_positions[-1]   # the nearest class folder above the file
                true_label = CLASS_LOOKUP[folders[i].strip().lower()]
                group = "/".join(folders[:i]) or "."
                note = "several class names in path; nearest folder used" if len(label_positions) > 1 else ""
            else:
                true_label, group, note = None, "/".join(folders) or ".", ""
            images.append({"path": path, "image_path": rel_posix, "true_label": true_label,
                           "group": group, "label_note": note})
    return images, skipped, empty_dirs


def unrecognized_folders(images):
    """Folder names directly holding unlabeled images (to warn about misspelled class folders)."""
    names = Counter(Path(im["image_path"]).parent.name for im in images if im["true_label"] is None)
    names.pop("", None)
    return dict(names)


def open_image(path):
    """Open an image fully and return (RGB image, original mode, (width, height)).

    The conversion is the same as src/predict.predict_image (PIL .convert("RGB")): grayscale is
    repeated over 3 channels, the alpha channel of RGBA / LA / transparent palette images is dropped.
    Only 16/32-bit integer and float images (which .convert("RGB") would clip) are first scaled to 8 bit.
    Any error (missing, truncated, corrupt, unsupported) is raised to the caller.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        with Image.open(path) as img:
            img.load()
            mode, size = img.mode, img.size
            if mode in ("I", "I;16", "I;16B", "I;16L", "I;16N", "F"):
                hi = img.getextrema()[1]
                scale = 255.0 / 65535.0 if hi > 255 else 1.0   # 16-bit range -> 8-bit
                img = img.convert("I").point(lambda v: v * scale).convert("L")
            rgb = img.convert("RGB")
    return rgb, mode, size


# ----------------------------------------------------------------------------------------------- model

def resolve_device(requested):
    import torch
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("ERROR: --device cuda was requested but no CUDA GPU is available. Use --device cpu or auto.")
    return torch.device(requested)


class FinalModelPredictor:
    """The frozen final model, loaded with src/predict.load_model (no second copy of the loading code)."""

    def __init__(self, checkpoint, device="auto"):
        import torch
        from src.predict import load_model
        self.torch = torch
        self.device = resolve_device(device)
        model, eval_transform, ckpt = load_model(checkpoint)   # checks architecture and class mapping
        self.model = model.to(self.device).eval()
        self.transform = eval_transform
        self.last_features = []   # 768-number input of the last layer, filled by the hook below
        self._feats = {}
        self.model.net.head.register_forward_hook(lambda m, inp, out: self._feats.__setitem__("f", inp[0]))
        self.info = {"architecture": ckpt.get("architecture"), "epoch": ckpt.get("epoch"),
                     "class_to_idx": ckpt.get("class_to_idx"), "transform": {k: ckpt["transform"][k] for k in
                                                                            ("resize", "normalize_mean", "normalize_std")},
                     "device": str(self.device), "torch": torch.__version__}

    def __call__(self, rgb_images):
        """List of PIL RGB images -> list of 8 class probabilities each (float, in CLASSES order)."""
        x = self.torch.stack([self.transform(im) for im in rgb_images]).to(self.device)
        with self.torch.inference_mode():
            probs = self.torch.softmax(self.model(x).float(), dim=1).cpu()
        self.last_features = self._feats["f"].float().cpu().tolist()
        return probs.tolist()


class NeysanDetector:
    """p(Neysan) = sigmoid(((feature - mean) / scale) . coef + intercept); rejected when p >= threshold."""

    def __init__(self, path, allow_other=False):
        raw = Path(path).read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        if self.sha256 != DETECTOR_SHA256 and not allow_other:
            raise SystemExit(f"ERROR: Neysan detector SHA256 {self.sha256} is not the accepted file ({DETECTOR_SHA256}).")
        d = json.loads(raw.decode("utf-8"))
        if d.get("checkpoint_sha256") != FINAL_CHECKPOINT_SHA256:
            raise SystemExit("ERROR: the Neysan detector was not built for the final checkpoint.")
        self.mean, self.scale, self.coef = d["mean"], d["scale"], d["coef"]
        self.intercept, self.threshold = float(d["intercept"]), float(d["threshold"])
        if not (len(self.mean) == len(self.scale) == len(self.coef) == d["feature_dim"]):
            raise SystemExit("ERROR: the Neysan detector file is damaged (sizes do not match).")

    def score(self, feature):
        import math
        if len(feature) != len(self.coef):
            raise ValueError(f"feature size {len(feature)} != {len(self.coef)}")
        z = self.intercept + sum(((f - m) / s) * c for f, m, s, c in zip(feature, self.mean, self.scale, self.coef))
        return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))


# ----------------------------------------------------------------------------------------------- outputs

def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding=CSV_ENCODING) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_metrics_with_unknown(y_true, y_pred):
    """Same definitions as scripts/evaluate_test.compute_metrics (macro over the 8 classes), plus an extra
    "unknown" column: an UNKNOWN prediction is a miss for its true class and a hit for no class."""
    cols = CLASSES + [UNKNOWN]
    k = len(CLASSES)
    matrix = [[0] * len(cols) for _ in range(k)]
    for t, p in zip(y_true, y_pred):
        matrix[CLASSES.index(t)][cols.index(p)] += 1
    per_class = {}
    for i, c in enumerate(CLASSES):
        tp = matrix[i][i]
        predicted = sum(matrix[r][i] for r in range(k))
        support = sum(matrix[i])
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[c] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    correct = sum(matrix[i][i] for i in range(k))
    total = len(y_true)
    return {"matrix": matrix, "columns": cols, "per_class": per_class, "total": total, "correct": correct,
            "accuracy": correct / total if total else 0.0,
            "macro_precision": sum(v["precision"] for v in per_class.values()) / k,
            "macro_recall": sum(v["recall"] for v in per_class.values()) / k,
            "macro_f1": sum(v["f1"] for v in per_class.values()) / k}


def metrics_for(rows):
    y_true, y_pred = [r["true_label"] for r in rows], [r["predicted_class"] for r in rows]
    if UNKNOWN in y_pred:
        m = compute_metrics_with_unknown(y_true, y_pred)
    else:
        from scripts.evaluate_test import compute_metrics   # same metric definition as the recorded Test evaluation
        m = compute_metrics(y_true, y_pred, CLASSES)
        m["columns"] = list(CLASSES)
    present = [c for c in CLASSES if m["per_class"][c]["support"] > 0]
    out = {"images": m["total"], "correct": m["correct"], "wrong": m["total"] - m["correct"],
           "accuracy": m["accuracy"], "macro_precision": m["macro_precision"],
           "macro_recall": m["macro_recall"], "macro_f1": m["macro_f1"],
           "needs_review": sum(r["needs_review"] for r in rows),
           "classes_in_ground_truth": present, "per_class": m["per_class"],
           "unknown_predictions": sum(p == UNKNOWN for p in y_pred),
           "confusion_matrix": {"rows_true": CLASSES, "columns_predicted": m["columns"], "matrix": m["matrix"]}}
    if len(present) < len(CLASSES):   # the project macro average is over all 8 classes
        k = len(present)
        out["macro_over_present_classes"] = {
            key: sum(m["per_class"][c][name] for c in present) / k
            for key, name in (("macro_precision", "precision"), ("macro_recall", "recall"), ("macro_f1", "f1"))}
    return out


def classification_report_text(rows):
    from sklearn.metrics import classification_report
    return classification_report([r["true_label"] for r in rows], [r["predicted_class"] for r in rows],
                                 labels=CLASSES, digits=4, zero_division=0)


def save_confusion_png(matrix, path, title, columns=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    columns = columns or CLASSES
    m = np.array(matrix, dtype=float)
    rows = m.sum(axis=1, keepdims=True)
    norm = np.divide(m, rows, out=np.zeros_like(m), where=rows > 0)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, data, sub, fmt in ((axes[0], m, "counts", "{:.0f}"), (axes[1], norm, "row-normalized (recall on diagonal)", "{:.2f}")):
        ax.imshow(data, cmap="Blues", vmin=0, vmax=data.max() if data.max() > 0 else 1)
        ax.set_xticks(range(len(columns)), columns, rotation=45, ha="right")
        ax.set_yticks(range(len(CLASSES)), CLASSES)
        ax.set_xlabel("predicted class")
        ax.set_ylabel("true class")
        ax.set_title(sub)
        for i in range(len(CLASSES)):
            for j in range(len(columns)):
                if m[i, j] > 0:
                    ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center", fontsize=8,
                            color="white" if data[i, j] > 0.6 * data.max() else "black")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fmt(x):
    return f"{x:.4f}"


def summary_text(info, overall, per_group, counts, warnings_list):
    lines = ["VEHICLE CLASSIFIER - TEST RUN SUMMARY", "=" * 60,
             f"mode               : {info['mode']}",
             f"input folder       : {info['input']}",
             f"output folder      : {info['output']}",
             f"model              : {info['model']['architecture']} (epoch {info['model']['epoch']}), checkpoint sha256 {info['checkpoint_sha256'][:16]}...",
             f"preprocessing      : Resize {tuple(info['model']['transform']['resize'])} -> ToTensor -> ImageNet normalization (no augmentation)",
             f"needs_review rule  : confidence < {info['needs_review_threshold']} (a flag for human review; the prediction is kept)",
             "neysan detector    : " + (f"ON - {info['neysan_detector']['rule']} (sha256 {info['neysan_detector']['sha256'][:16]}...)"
                                        if info['neysan_detector']['enabled'] else
                                        f"OFF ({info['neysan_detector'].get('reason', '')}) - 8-class output only"),
             f"device             : {info['model']['device']}", "",
             "FILES", "-" * 60,
             f"files found        : {counts['files_found']}",
             f"images predicted   : {counts['predicted']}",
             f"images failed      : {counts['failed']}   (see failed_images.csv)",
             f"non-image files    : {counts['skipped']}   (see skipped_files.csv)",
             f"with ground truth  : {counts['labeled']}",
             f"without label      : {counts['unlabeled']}",
             f"needs_review       : {counts['needs_review']}   (see needs_review.csv)",
             f"unknown (Neysan)   : {counts['unknown']}   (see unknown_predictions.csv)", ""]
    if overall:
        lines += ["RESULTS (images with ground truth that were read successfully)", "-" * 60,
                  f"images             : {overall['images']}",
                  f"correct / wrong    : {overall['correct']} / {overall['wrong']}   (see wrong_predictions.csv)",
                  f"accuracy           : {fmt(overall['accuracy'])}",
                  f"macro precision    : {fmt(overall['macro_precision'])}",
                  f"macro recall       : {fmt(overall['macro_recall'])}",
                  f"macro F1           : {fmt(overall['macro_f1'])}",
                  f"needs_review       : {overall['needs_review']}",
                  f"unknown            : {overall['unknown_predictions']}   (counted as wrong for their folder class)"]
        if "macro_over_present_classes" in overall:
            mp = overall["macro_over_present_classes"]
            lines.append(f"macro over the {len(overall['classes_in_ground_truth'])} classes present: "
                         f"P {fmt(mp['macro_precision'])}, R {fmt(mp['macro_recall'])}, F1 {fmt(mp['macro_f1'])}")
        lines.append("")
        lines += ["per class            precision  recall     f1         support"]
        for c in CLASSES:
            v = overall["per_class"][c]
            lines.append(f"  {c:18s} {fmt(v['precision']):10s} {fmt(v['recall']):10s} {fmt(v['f1']):10s} {v['support']}")
        lines.append("")
        if len(per_group) > 1:
            lines += ["PER TEST GROUP", "-" * 60]
            for g, m in per_group.items():
                lines.append(f"  {g:20s} images {m['images']:5d}  accuracy {fmt(m['accuracy'])}  macro F1 {fmt(m['macro_f1'])}  needs_review {m['needs_review']}  unknown {m['unknown_predictions']}")
            lines.append("")
    else:
        lines += ["No ground-truth class folders were found: predictions only, no metrics.", ""]
    if warnings_list:
        lines += ["WARNINGS", "-" * 60] + [f"  - {w}" for w in warnings_list] + [""]
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------- run

def run(input_dir, output_dir, checkpoint=DEFAULT_CHECKPOINT, batch_size=32, device="auto",
        overwrite=False, allow_other_checkpoint=False, predictor=None,
        neysan_detector=DEFAULT_DETECTOR, use_neysan_detector=True, allow_other_detector=False):
    t0 = time.time()
    input_dir = Path(input_dir).expanduser()
    output_dir = Path(output_dir).expanduser()
    if not input_dir.is_dir():
        raise SystemExit(f"ERROR: input folder not found: {input_dir}")
    if output_dir.resolve() == input_dir.resolve():
        raise SystemExit("ERROR: --output must be a different folder than --input")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise SystemExit(f"ERROR: output folder is not empty: {output_dir}\n"
                         "       choose another --output or add --overwrite")
    if batch_size < 1:
        raise SystemExit("ERROR: --batch-size must be at least 1")

    # model first: a wrong checkpoint should stop the run before any work
    if predictor is None:
        checkpoint = Path(checkpoint).expanduser()
        if not checkpoint.is_file():
            raise SystemExit(f"ERROR: checkpoint not found: {checkpoint}\n"
                             "       pass the path of swin_t_ft_aug_best.pt with --checkpoint")
        ckpt_sha = sha256_of(checkpoint)
        if ckpt_sha != FINAL_CHECKPOINT_SHA256 and not allow_other_checkpoint:
            raise SystemExit(f"ERROR: checkpoint SHA256 {ckpt_sha} is not the final model "
                             f"({FINAL_CHECKPOINT_SHA256}). The file may be incomplete or a different model.")
        predictor = FinalModelPredictor(checkpoint, device)
    else:
        ckpt_sha = getattr(predictor, "sha256", "test-predictor")
    model_info = predictor.info

    # Neysan detector: every image is checked; if it says Neysan -> UNKNOWN (whatever the 8-class prediction)
    detector, detector_info = None, {"enabled": False}
    if use_neysan_detector:
        det_path = Path(neysan_detector).expanduser()
        if det_path.is_file():
            detector = NeysanDetector(det_path, allow_other=allow_other_detector)
            detector_info = {"enabled": True, "path": str(det_path.resolve()), "sha256": detector.sha256,
                             "threshold": detector.threshold,
                             "rule": f"p(Neysan) >= {detector.threshold} -> {UNKNOWN} (any 8-class prediction)"}
        elif Path(neysan_detector) == DEFAULT_DETECTOR:
            print(f"WARNING: Neysan detector not found ({det_path}); running the 8-class model only", flush=True)
            detector_info = {"enabled": False, "reason": f"file not found: {det_path}"}
        else:
            raise SystemExit(f"ERROR: Neysan detector not found: {det_path}")
    else:
        detector_info = {"enabled": False, "reason": "--no-neysan-detector"}

    from src.predict import NEEDS_REVIEW_THRESHOLD, needs_review   # the single place the threshold is defined

    images, skipped, empty_dirs = discover(input_dir, exclude_dir=output_dir)
    files_found = len(images) + len(skipped)
    print(f"found {files_found} files: {len(images)} images, {len(skipped)} other files", flush=True)
    if not images:
        print("WARNING: no image files found", flush=True)

    rows, failed = [], []
    batch_imgs, batch_meta = [], []

    def flush_batch():
        if not batch_imgs:
            return
        try:
            all_probs = predictor(batch_imgs)
            all_feats = list(predictor.last_features) if detector else [None] * len(all_probs)
        except Exception as e:   # e.g. GPU out of memory: retry the batch one image at a time
            print(f"WARNING: batch failed ({type(e).__name__}: {e}); retrying one image at a time", flush=True)
            all_probs, all_feats = [], []
            for im, meta in zip(batch_imgs, batch_meta):
                try:
                    all_probs.append(predictor([im])[0])
                    all_feats.append(predictor.last_features[0] if detector else None)
                except Exception as e2:
                    all_probs.append(None)
                    all_feats.append(None)
                    failed.append({"image_path": meta["image_path"], "stage": "inference",
                                   "error_type": type(e2).__name__, "error_message": str(e2)[:300]})
        for probs, feat, meta in zip(all_probs, all_feats, batch_meta):
            if probs is None:
                continue
            best = max(range(len(CLASSES)), key=lambda i: probs[i])
            confidence = float(probs[best])
            model_class = CLASSES[best]
            pred, neysan_score = model_class, ""
            if detector:
                sc = detector.score(feat)
                neysan_score = f"{sc:.6f}"
                if sc >= detector.threshold:
                    pred = UNKNOWN
            row = {"image_path": meta["image_path"], "group": meta["group"],
                   "true_label": meta["true_label"] or "", "predicted_class": pred,
                   "model_class": model_class, "neysan_score": neysan_score,
                   "confidence": f"{confidence:.6f}",
                   "needs_review": needs_review(confidence, NEEDS_REVIEW_THRESHOLD),   # on the unrounded value
                   "correct": "" if meta["true_label"] is None else pred == meta["true_label"],
                   "image_mode": meta["mode"], "width": meta["size"][0], "height": meta["size"][1],
                   "label_note": meta["label_note"]}
            row.update({f"p_{c}": f"{float(p):.6f}" for c, p in zip(CLASSES, probs)})
            rows.append(row)
        batch_imgs.clear()
        batch_meta.clear()

    for n, im in enumerate(images, 1):
        try:
            rgb, mode, size = open_image(im["path"])
        except Exception as e:
            failed.append({"image_path": im["image_path"], "stage": "open",
                           "error_type": type(e).__name__, "error_message": str(e)[:300]})
        else:
            batch_imgs.append(rgb)
            batch_meta.append({**im, "mode": mode, "size": size})
            if len(batch_imgs) >= batch_size:
                flush_batch()
        if n % 200 == 0:
            print(f"  {n}/{len(images)} images processed", flush=True)
    flush_batch()

    # ---------------- outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["image_path", "group", "true_label", "predicted_class", "model_class", "neysan_score",
              "confidence", "needs_review", "correct",
              "image_mode", "width", "height", "label_note"] + [f"p_{c}" for c in CLASSES]
    write_csv(output_dir / "predictions.csv", rows, fields)
    write_csv(output_dir / "needs_review.csv", [r for r in rows if r["needs_review"]], fields)
    write_csv(output_dir / "unknown_predictions.csv", [r for r in rows if r["predicted_class"] == UNKNOWN], fields)
    write_csv(output_dir / "failed_images.csv", failed, ["image_path", "stage", "error_type", "error_message"])
    write_csv(output_dir / "skipped_files.csv", skipped, ["image_path", "reason"])

    labeled = [r for r in rows if r["true_label"]]
    unlabeled = [r for r in rows if not r["true_label"]]
    mode = "evaluation" if labeled and not unlabeled else "prediction-only" if not labeled else "evaluation (partial)"

    warn = []
    if failed:
        warn.append(f"{len(failed)} image(s) could not be read or predicted; they are NOT in the metrics (failed_images.csv)")
    if labeled and unlabeled:
        warn.append(f"{len(unlabeled)} image(s) have no class folder; metrics use only the {len(labeled)} labeled images")
    odd = unrecognized_folders(images)
    if labeled and odd:
        warn.append(f"images in folders that are not class names: {odd}")
    multi = sum(1 for im in images if im["label_note"])
    if multi:
        warn.append(f"{multi} image path(s) contain several class folder names; the nearest one was used as label")
    if empty_dirs:
        warn.append(f"empty folders: {empty_dirs[:10]}{' ...' if len(empty_dirs) > 10 else ''}")

    overall, per_group = None, {}
    if labeled:
        overall = metrics_for(labeled)
        groups = defaultdict(list)
        for r in labeled:
            groups[r["group"]].append(r)
        per_group = {g: metrics_for(rs) for g, rs in sorted(groups.items())}
        if len(overall["classes_in_ground_truth"]) < len(CLASSES):
            missing = [c for c in CLASSES if c not in overall["classes_in_ground_truth"]]
            warn.append(f"classes with no ground-truth images: {missing}; the main macro scores average over all 8 "
                        "classes (as in the project), macro_over_present_classes is also reported")
        write_csv(output_dir / "wrong_predictions.csv", [r for r in labeled if not r["correct"]], fields)
        (output_dir / "classification_report.txt").write_text(
            "Classification report (images with ground truth that were read successfully)\n\n"
            + classification_report_text(labeled), encoding="utf-8")
        cm_cols = overall["confusion_matrix"]["columns_predicted"]
        save_confusion_png(overall["confusion_matrix"]["matrix"], output_dir / "confusion_matrix.png",
                           f"Confusion matrix - {overall['images']} images, accuracy {overall['accuracy']:.4f}, "
                           f"macro F1 {overall['macro_f1']:.4f}", columns=cm_cols)
        with open(output_dir / "confusion_matrix.csv", "w", newline="", encoding=CSV_ENCODING) as f:
            w = csv.writer(f)
            w.writerow(["true \\ predicted"] + cm_cols)
            for c, line in zip(CLASSES, overall["confusion_matrix"]["matrix"]):
                w.writerow([c] + line)

    counts = {"files_found": files_found, "predicted": len(rows), "failed": len(failed), "skipped": len(skipped),
              "labeled": len(labeled), "unlabeled": len(unlabeled), "needs_review": sum(r["needs_review"] for r in rows),
              "unknown": sum(r["predicted_class"] == UNKNOWN for r in rows)}
    info = {"mode": mode, "input": str(input_dir.resolve()), "output": str(output_dir.resolve()),
            "checkpoint_sha256": ckpt_sha, "model": model_info, "classes": CLASSES,
            "needs_review_threshold": NEEDS_REVIEW_THRESHOLD, "neysan_detector": detector_info,
            "batch_size": batch_size,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": platform.python_version(), "platform": platform.platform(),
            "runtime_seconds": None, "counts": counts, "warnings": warn}
    if overall:
        metrics = {**info, "overall": overall}
        if len(per_group) > 1:
            metrics["per_group"] = per_group
        info_out = metrics
    else:
        info_out = info
    info_out["runtime_seconds"] = round(time.time() - t0, 1)
    with open(output_dir / ("metrics.json" if overall else "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(info_out, f, indent=2, ensure_ascii=False)
    text = summary_text(info, overall, per_group, counts, warn)
    (output_dir / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text, flush=True)
    return info_out
