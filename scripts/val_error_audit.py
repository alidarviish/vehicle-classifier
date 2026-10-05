"""Validation error and confidence audit of one finished run (VALIDATION split only).

Read-only analysis: no training, no new checkpoint, no change to labels, splits or images.
Only the "val" rows of data/split_manifest.csv are read; the Test and Neysan files are never opened.
The confidence bins and thresholds used here are for analysis only. They do not change the
production needs_review threshold in src/predict.py, and no temperature is stored or used anywhere.

Run from the repository root, after scripts/analyze_baseline.py has written the run's confusion matrix:
    python scripts/val_error_audit.py --run-name resnet224_ft_aug

Reads:
    checkpoints/<run>_best.pt                   the best checkpoint of the run (read only)
    data/split_manifest.csv                     only the "val" rows
    configs/local_paths.json                    local image folders (or --config)
    reports/analysis/<run>/confusion_matrix.csv reference for the reproduction check
Writes (generated, ignored by git) in reports/analysis/<run>/val_audit/:
    val_predictions.csv       one row per validation image
    per_class_audit.csv       per-class counts, metrics and error confidence
    confusion_pairs.csv       every true -> predicted error pair
    confidence_bins.csv       reliability table (15 equal-width and 10 equal-mass bins)
    threshold_sweep.csv       flagging by confidence and by margin
    calibration_summary.json  ECE, MCE, NLL, Brier, AUROC, AURC, temperature-scaling diagnostic
    run_meta.json             run, checkpoint, hashes, versions
    audit_summary.md          short factual summary
    conf_hist_correct_vs_wrong.png, reliability.png, risk_coverage.png   (unless --no-plots)

Nothing is written if a check fails: validation row count, dataset order and labels, class mapping,
probabilities, confusion matrix different from the saved one, or the checkpoint / split manifest /
saved confusion matrix changed during the run.
"""

import argparse
import csv
import hashlib
import importlib
import json
import math
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))  # so "from src..." works with "python scripts/val_error_audit.py"

EXPECTED_VAL_ROWS = 659
HIGH_CONF, MEDIUM_CONF = 0.90, 0.70              # analysis bins: high >= 0.90, medium 0.70-0.90, low < 0.70
SENSITIVITY_THRESHOLDS = (0.80, 0.90, 0.95, 0.99)
EQUAL_WIDTH_BINS = 15
EQUAL_MASS_BINS = 10
CV_FOLDS = 5
LOG_BETA_RANGE = (math.log(0.05), math.log(20.0))  # search range of 1/T for the temperature diagnostic
PROB_SUM_TOLERANCE = 1e-6
WILSON_Z = 1.959963984540054                       # two-sided 95%

ARCHITECTURES = {                                  # checkpoint "architecture" -> (module, builder)
    "ResNet18": ("src.resnet", "build_resnet18"),
    "EfficientNet-B0": ("src.efficientnet", "build_efficientnet_b0"),
    "ConvNeXt-Tiny": ("src.convnext", "build_convnext_tiny"),
    "Swin-Tiny": ("src.swin", "build_swin_tiny"),
}

OUTPUT_FILES = ("val_predictions.csv", "per_class_audit.csv", "confusion_pairs.csv", "confidence_bins.csv",
                "threshold_sweep.csv", "calibration_summary.json", "run_meta.json", "audit_summary.md",
                "conf_hist_correct_vs_wrong.png", "reliability.png", "risk_coverage.png")


# ---------- small helpers ----------

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fmt(value, digits=6):
    return "" if value is None else f"{value:.{digits}f}"


def mean_or_none(values):
    return statistics.fmean(values) if values else None


def ratio(num, den):
    return num / den if den else None


def log_softmax(logits):
    m = max(logits)
    lse = m + math.log(sum(math.exp(v - m) for v in logits))
    return [v - lse for v in logits]


def conf_bin(confidence):
    if confidence >= HIGH_CONF:
        return "high"
    return "medium" if confidence >= MEDIUM_CONF else "low"


def wilson(k, n, z=WILSON_Z):
    """95% Wilson score interval of k successes in n trials."""
    if n == 0:
        return None, None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def git_commit():
    try:
        out = subprocess.run(["git", "--no-optional-locks", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=True, timeout=30)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def run_paths(run_name):
    if not run_name or run_name in (".", "..") or any(c in run_name for c in "/\\:"):
        raise ValueError(f"invalid run name: {run_name!r}")
    checkpoint = REPO_ROOT / "checkpoints" / f"{run_name}_best.pt"
    analysis_dir = REPO_ROOT / "reports" / "analysis" / run_name
    return checkpoint, analysis_dir / "confusion_matrix.csv", analysis_dir / "val_audit"


def read_saved_matrix(path, classes):
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if rows[0][1:] != list(classes) or [r[0] for r in rows[1:]] != list(classes):
        raise ValueError(f"class order in {path.name} differs from the project classes")
    return [[int(v) for v in r[1:]] for r in rows[1:]]


# ---------- model and inference (read only) ----------

def build_model(checkpoint, classes):
    """Build the network of the checkpoint (no download) and its evaluation transform, never augmentation."""
    from src.train import BASE_TRANSFORM, resnet_transform

    arch = checkpoint["architecture"]
    if arch == "BaselineCNN":
        from src.model import BaselineCNN
        model = BaselineCNN(num_classes=len(classes), image_size=checkpoint["image_size"],
                            dropout=checkpoint["dropout"], pooling=checkpoint.get("pooling", "max"))
        eval_transform = BASE_TRANSFORM
    elif arch in ARCHITECTURES:
        module_name, builder = ARCHITECTURES[arch]
        model = getattr(importlib.import_module(module_name), builder)(num_classes=len(classes), pretrained=False)
        recorded = checkpoint["transform"]
        eval_transform = resnet_transform(recorded["resize"][0], recorded["normalize_mean"], recorded["normalize_std"])
    else:
        raise ValueError(f"unexpected architecture: {arch}")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, eval_transform


def run_inference(model, dataset, device):
    """Logits (float64 lists), argmax predictions and labels, in dataset order."""
    import torch
    from torch.utils.data import DataLoader

    from src.train import BATCH_SIZE

    model.to(device)
    model.eval()
    logits, preds, labels = [], [], []
    with torch.no_grad():
        for images, y in DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False):
            out = model(images.to(device)).float().cpu()
            preds.extend(out.argmax(dim=1).tolist())
            logits.extend(out.double().tolist())
            labels.extend(y.tolist())
    return logits, preds, labels


# ---------- per-sample records ----------

def build_records(rows, labels, preds, logits, classes):
    """One dict per validation image; checks labels, argmax and probabilities."""
    idx = {c: i for i, c in enumerate(classes)}
    if not (len(rows) == len(labels) == len(preds) == len(logits)):
        raise RuntimeError("rows, labels, predictions and logits have different lengths")
    records = []
    for i, (row, y, p, lg) in enumerate(zip(rows, labels, preds, logits)):
        if len(lg) != len(classes) or not all(math.isfinite(v) for v in lg):
            raise RuntimeError(f"row {i}: invalid logits")
        if idx[row["label"]] != y:
            raise RuntimeError(f"row {i}: dataset label differs from the manifest label")
        logp = log_softmax(lg)
        probs = [math.exp(v) for v in logp]
        if not all(0.0 <= q <= 1.0 for q in probs) or abs(sum(probs) - 1.0) > PROB_SUM_TOLERANCE:
            raise RuntimeError(f"row {i}: probabilities are not valid")
        order = sorted(range(len(classes)), key=lambda k: (-lg[k], k))
        if order[0] != p:
            raise RuntimeError(f"row {i}: argmax of the logits differs from the model prediction")
        second = order[1]
        records.append({
            "index": i, "image_path": row["image_path"], "sha256": row["sha256"],
            "true": y, "pred": p, "true_label": classes[y], "pred_label": classes[p],
            "folder_label": row["folder_label"], "label_corrected": row["label"] != row["folder_label"],
            "source": row["source"], "origin_split": row["origin_split"], "dup_group": row["dup_group"],
            "correct": y == p, "confidence": probs[p], "p_true": probs[y], "true_rank": order.index(y) + 1,
            "second_label": classes[second], "p_second": probs[second], "margin": probs[p] - probs[second],
            "entropy": -sum(math.exp(v) * v for v in logp), "logp_true": logp[y],
            "probs": probs, "logits": lg, "conf_bin": conf_bin(probs[p]),
        })
    return records


# ---------- analysis (pure Python, no torch) ----------

def confusion_matrix(records, n_classes):
    m = [[0] * n_classes for _ in range(n_classes)]
    for r in records:
        m[r["true"]][r["pred"]] += 1
    return m


def per_class_rows(records, classes):
    out = []
    for c, name in enumerate(classes):
        own = [r for r in records if r["true"] == c]
        predicted = [r for r in records if r["pred"] == c]
        tp = sum(r["correct"] for r in own)
        wrong = [r for r in own if not r["correct"]]
        precision, recall = ratio(tp, len(predicted)) or 0.0, ratio(tp, len(own)) or 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        lo, hi = wilson(tp, len(own))
        targets = Counter(r["pred"] for r in wrong)
        top = min(targets, key=lambda k: (-targets[k], k)) if targets else None
        row = {
            "class": name, "support": len(own), "n_pred": len(predicted), "TP": tp,
            "FN": len(own) - tp, "FP": len(predicted) - tp,
            "precision": precision, "recall": recall, "F1": f1,
            "recall_wilson95_lo": lo, "recall_wilson95_hi": hi,
            "errors_high": sum(r["conf_bin"] == "high" for r in wrong),
            "errors_medium": sum(r["conf_bin"] == "medium" for r in wrong),
            "errors_low": sum(r["conf_bin"] == "low" for r in wrong),
            "top_confusion": classes[top] if top is not None else "",
            "top_confusion_count": targets[top] if top is not None else 0,
            "top2_recall": ratio(sum(r["true_rank"] <= 2 for r in own), len(own)),
            "mean_conf_correct": mean_or_none([r["confidence"] for r in own if r["correct"]]),
            "mean_conf_wrong": mean_or_none([r["confidence"] for r in wrong]),
        }
        for t in SENSITIVITY_THRESHOLDS:
            row[f"wrong_conf_ge_{t:.2f}"] = sum(r["confidence"] >= t for r in wrong)
        out.append(row)
    return out


def confusion_pair_rows(records, classes):
    pairs = {}
    for r in records:
        if not r["correct"]:
            pairs.setdefault((r["true"], r["pred"]), []).append(r)
    rows = []
    for (t, p), items in pairs.items():
        confs = [r["confidence"] for r in items]
        rows.append({"true_label": classes[t], "predicted_label": classes[p], "count": len(items),
                     "n_high": sum(r["conf_bin"] == "high" for r in items),
                     "n_medium": sum(r["conf_bin"] == "medium" for r in items),
                     "n_low": sum(r["conf_bin"] == "low" for r in items),
                     "conf_mean": statistics.fmean(confs), "conf_median": statistics.median(confs),
                     "conf_max": max(confs), "margin_median": statistics.median(r["margin"] for r in items),
                     "_order": (t, p)})
    rows.sort(key=lambda r: (-r["count"], -r["n_high"], r["_order"]))
    for r in rows:
        del r["_order"]
    return rows


def reliability_bins(pairs, scheme):
    """pairs: list of (confidence, correct, index). Returns the bins of one scheme."""
    if scheme == "equal_width":
        groups = [[] for _ in range(EQUAL_WIDTH_BINS)]
        for item in pairs:
            groups[min(int(item[0] * EQUAL_WIDTH_BINS), EQUAL_WIDTH_BINS - 1)].append(item)
        edges = [(k / EQUAL_WIDTH_BINS, (k + 1) / EQUAL_WIDTH_BINS) for k in range(EQUAL_WIDTH_BINS)]
    else:
        ordered = sorted(pairs, key=lambda x: (x[0], x[2]))
        n, groups, start = len(ordered), [], 0
        for k in range(EQUAL_MASS_BINS):
            size = n // EQUAL_MASS_BINS + (1 if k < n % EQUAL_MASS_BINS else 0)
            groups.append(ordered[start:start + size])
            start += size
        edges = [(g[0][0], g[-1][0]) if g else (None, None) for g in groups]
    bins = []
    for (lo, hi), g in zip(edges, groups):
        acc = ratio(sum(x[1] for x in g), len(g))
        mc = mean_or_none([x[0] for x in g])
        bins.append({"scheme": scheme, "bin_lo": lo, "bin_hi": hi, "n": len(g), "n_correct": sum(x[1] for x in g),
                     "accuracy": acc, "mean_conf": mc, "gap": None if acc is None else acc - mc})
    return bins


def ece_mce(bins, total):
    used = [b for b in bins if b["n"]]
    ece = sum(b["n"] / total * abs(b["gap"]) for b in used)
    return ece, max(abs(b["gap"]) for b in used)


def auroc(correct_scores, wrong_scores):
    """P(score of a correct prediction > score of a wrong one), ties count 1/2."""
    if not correct_scores or not wrong_scores:
        return None
    wins = sum((c > w) + 0.5 * (c == w) for c in correct_scores for w in wrong_scores)
    return wins / (len(correct_scores) * len(wrong_scores))


def risk_coverage(records):
    """Most confident first (ties by index). Returns [(coverage, risk)] for k = 1..n and the AURC."""
    ordered = sorted(records, key=lambda r: (-r["confidence"], r["index"]))
    curve, errors = [], 0
    for k, r in enumerate(ordered, 1):
        errors += not r["correct"]
        curve.append((k / len(ordered), errors / k))
    return curve, statistics.fmean(risk for _, risk in curve)


def threshold_rows(records):
    """Flag if confidence < t (the rule of src/predict.needs_review), and margin at matched flag counts."""
    n = len(records)
    margins = sorted((r["margin"], r["index"]) for r in records)

    def row(score, rule, threshold, matched, flagged_set):
        flagged = [r for r in records if r["index"] in flagged_set]
        kept = n - len(flagged)
        kept_correct = sum(r["correct"] for r in records) - sum(r["correct"] for r in flagged)
        return {"score": score, "rule": rule, "threshold": threshold, "matched_confidence_threshold": matched,
                "flagged": len(flagged), "errors_caught": sum(not r["correct"] for r in flagged),
                "correct_flagged": sum(r["correct"] for r in flagged),
                "errors_not_flagged": sum(not r["correct"] for r in records) - sum(not r["correct"] for r in flagged),
                "coverage": kept / n, "selective_accuracy": ratio(kept_correct, kept)}

    rows, margin_thresholds = [], []
    for t in SENSITIVITY_THRESHOLDS:
        conf_flag = {r["index"] for r in records if r["confidence"] < t}
        rows.append(row("confidence", "flag if confidence < threshold", t, None, conf_flag))
        k = len(conf_flag)
        m_thr = math.inf if k >= n else margins[k][0]      # flag if margin < m_thr -> k images (fewer on ties)
        margin_flag = {r["index"] for r in records if r["margin"] < m_thr}
        margin_thresholds.append({"matched_confidence_threshold": t, "target_flagged": k,
                                  "margin_threshold": None if math.isinf(m_thr) else m_thr,
                                  "flagged": len(margin_flag)})
        rows.append(row("margin", "flag if margin < threshold", None if math.isinf(m_thr) else m_thr, t, margin_flag))
    return rows, margin_thresholds


def nll_at(records, beta):
    total = 0.0
    for r in records:
        scaled = [beta * v for v in r["logits"]]
        total -= log_softmax(scaled)[r["true"]]
    return total / len(records)


def golden_min(f, lo, hi, iters=120):
    g = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return (a + b) / 2


def temperature_cv(records, n_classes):
    """Diagnostic only: stratified k-fold inside validation, fit 1/T on k-1 folds, evaluate on the held-out fold.
    Folds: per true class, images in index order, fold = position % k (deterministic, no random numbers).
    Returns held-out NLL and ECE with T = 1 and with the fitted T; the fitted values are not returned."""
    fold_of = {}
    for c in range(n_classes):
        for pos, r in enumerate(sorted((r for r in records if r["true"] == c), key=lambda r: r["index"])):
            fold_of[r["index"]] = pos % CV_FOLDS
    held_base, held_scaled, above_one, at_bound = [], [], 0, 0
    for fold in range(CV_FOLDS):
        fit = [r for r in records if fold_of[r["index"]] != fold]
        test = [r for r in records if fold_of[r["index"]] == fold]
        u = golden_min(lambda x: nll_at(fit, math.exp(x)), *LOG_BETA_RANGE)
        beta = math.exp(u)
        above_one += beta < 1.0                             # T = 1 / beta > 1: softer probabilities fit better
        at_bound += min(abs(u - LOG_BETA_RANGE[0]), abs(u - LOG_BETA_RANGE[1])) < 1e-3
        for r in test:
            logp = log_softmax([beta * v for v in r["logits"]])
            held_base.append((r["confidence"], r["correct"], r["index"], r["logp_true"]))
            held_scaled.append((math.exp(logp[r["pred"]]), r["correct"], r["index"], logp[r["true"]]))
    n = len(records)

    def summary(items):
        ece, _ = ece_mce(reliability_bins([x[:3] for x in items], "equal_width"), n)
        return -statistics.fmean(x[3] for x in items), ece

    nll_1, ece_1 = summary(held_base)
    nll_t, ece_t = summary(held_scaled)
    return {"method": f"temperature scaling, stratified {CV_FOLDS}-fold CV inside validation (diagnostic only)",
            "fold_rule": "per true class, images in index order, fold = position % folds",
            "heldout_nll_T1": nll_1, "heldout_nll_Tcv": nll_t,
            "heldout_ece_equal_width_T1": ece_1, "heldout_ece_equal_width_Tcv": ece_t,
            "folds_with_fitted_T_above_1": above_one, "folds_with_fit_at_search_bound": at_bound,
            "note": "argmax is unchanged by a temperature, so accuracy is unchanged; the fitted T is not stored "
                    "and is not used by src/predict.py"}


def calibration_summary(records, bins_w, bins_m):
    n, k = len(records), len(records[0]["probs"])
    ece_w, mce_w = ece_mce(bins_w, n)
    ece_m, mce_m = ece_mce(bins_m, n)
    correct = [r for r in records if r["correct"]]
    wrong = [r for r in records if not r["correct"]]
    _, aurc = risk_coverage(records)
    brier = statistics.fmean(sum((p - (j == r["true"])) ** 2 for j, p in enumerate(r["probs"])) for r in records)
    return {
        "split": "val", "n": n, "n_correct": len(correct), "n_wrong": len(wrong),
        "ece_equal_width_15": ece_w, "mce_equal_width_15": mce_w,
        "ece_equal_mass_10": ece_m, "mce_equal_mass_10": mce_m,
        "nll": -statistics.fmean(r["logp_true"] for r in records),
        "brier": brier, "brier_definition": f"sum over the {k} classes of (p - onehot)^2, mean over images",
        "auroc_correct_vs_wrong": {
            "confidence": auroc([r["confidence"] for r in correct], [r["confidence"] for r in wrong]),
            "margin": auroc([r["margin"] for r in correct], [r["margin"] for r in wrong]),
            "negative_entropy": auroc([-r["entropy"] for r in correct], [-r["entropy"] for r in wrong]),
            "orientation": "higher score = more likely correct; 0.5 = no separation",
        },
        "aurc": aurc, "aurc_definition": "mean selective risk over coverage k/n, most confident first, ties by index",
        "temperature_diagnostic": temperature_cv(records, k),
    }


# ---------- writing ----------

def write_csv(path, rows, columns):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({c: (fmt(r[c]) if isinstance(r[c], float) else ("" if r[c] is None else r[c]))
                        for c in columns})


def prediction_rows(records, classes):
    out = []
    for r in records:
        row = {k: r[k] for k in ("index", "image_path", "sha256", "true_label", "folder_label")}
        row["label_corrected"] = int(r["label_corrected"])
        row.update({k: r[k] for k in ("source", "origin_split", "dup_group", "pred_label")})
        row["correct"] = int(r["correct"])
        row.update({k: f"{r[k]:.10f}" for k in ("confidence", "p_true")})
        row["true_rank"] = r["true_rank"]
        row["second_label"] = r["second_label"]
        row.update({k: f"{r[k]:.10f}" for k in ("p_second", "margin", "entropy")})
        row.update({f"p_{c}": f"{p:.10f}" for c, p in zip(classes, r["probs"])})
        row.update({f"logit_{c}": f"{v:.6f}" for c, v in zip(classes, r["logits"])})
        row["conf_bin"] = r["conf_bin"]
        out.append(row)
    return out


def summary_markdown(meta, records, per_class, pairs, sweep, cal, checks):
    n = len(records)
    wrong = [r for r in records if not r["correct"]]
    macro = {k: statistics.fmean(r[k] for r in per_class) for k in ("precision", "recall", "F1")}
    below = [r for r in sorted(per_class, key=lambda r: r["F1"]) if r["F1"] < macro["F1"]]
    L = [f"# Validation error and confidence audit: `{meta['run_name']}`", "",
         "Validation split only. Test and Neysan were not read. The confidence bins "
         f"(high >= {HIGH_CONF:.2f}, medium {MEDIUM_CONF:.2f}-{HIGH_CONF:.2f}, low < {MEDIUM_CONF:.2f}) are for "
         "analysis only and do not change the production threshold.", "",
         f"- Architecture: {meta['architecture']}, best epoch {meta['best_epoch']}",
         f"- Checkpoint SHA256: `{meta['checkpoint_sha256']}`",
         f"- Validation images: {n}",
         f"- Accuracy: {(n - len(wrong)) / n:.4f} ({n - len(wrong)}/{n})",
         f"- Macro precision / recall / F1: {macro['precision']:.4f} / {macro['recall']:.4f} / {macro['F1']:.4f}",
         f"- Wrong predictions: {len(wrong)} (high {sum(r['conf_bin'] == 'high' for r in wrong)}, "
         f"medium {sum(r['conf_bin'] == 'medium' for r in wrong)}, low {sum(r['conf_bin'] == 'low' for r in wrong)})",
         "", "## Wrong predictions with confidence >= threshold", "",
         "| Threshold | Wrong with confidence >= threshold |", "|---|---|"]
    L += [f"| {t:.2f} | {sum(r['confidence'] >= t for r in wrong)} |" for t in SENSITIVITY_THRESHOLDS]
    L += ["", "## Main confusions (true -> predicted)", "",
          "| True | Predicted | Count | High | Medium | Low | Median confidence |", "|---|---|---|---|---|---|---|"]
    L += [f"| {p['true_label']} | {p['predicted_label']} | {p['count']} | {p['n_high']} | {p['n_medium']} | "
          f"{p['n_low']} | {p['conf_median']:.4f} |" for p in pairs[:10]]
    L += ["", "## Per class", "",
          "| Class | Support | TP | FN | FP | Precision | Recall (95% Wilson) | F1 | Wrong high/medium/low | Top confusion |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in per_class:
        ci = "" if r["recall_wilson95_lo"] is None else f" ({r['recall_wilson95_lo']:.3f}-{r['recall_wilson95_hi']:.3f})"
        top = f"{r['top_confusion']} ({r['top_confusion_count']})" if r["top_confusion"] else "-"
        L.append(f"| {r['class']} | {r['support']} | {r['TP']} | {r['FN']} | {r['FP']} | {r['precision']:.4f} | "
                 f"{r['recall']:.4f}{ci} | {r['F1']:.4f} | {r['errors_high']}/{r['errors_medium']}/{r['errors_low']} | {top} |")
    L += ["", f"Classes with F1 below the macro F1 ({macro['F1']:.4f}): "
          + (", ".join(f"{r['class']} ({r['F1']:.4f}, support {r['support']})" for r in below) or "none"),
          "", "## Flagging (validation)", "",
          "| Score | Threshold | Matched to confidence | Flagged | Errors caught | Correct flagged | Coverage | Selective accuracy |",
          "|---|---|---|---|---|---|---|---|"]
    for s in sweep:
        L.append(f"| {s['score']} | {fmt(s['threshold'], 4) or 'all'} | {fmt(s['matched_confidence_threshold'], 2) or '-'} | "
                 f"{s['flagged']} | {s['errors_caught']} | {s['correct_flagged']} | {s['coverage']:.4f} | "
                 f"{fmt(s['selective_accuracy'], 4) or '-'} |")
    a = cal["auroc_correct_vs_wrong"]
    td = cal["temperature_diagnostic"]
    L += ["", "## Calibration (validation diagnostics)", "",
          f"- ECE: {cal['ece_equal_width_15']:.4f} (15 equal-width bins), {cal['ece_equal_mass_10']:.4f} (10 equal-mass bins)",
          f"- MCE: {cal['mce_equal_width_15']:.4f} (equal-width), {cal['mce_equal_mass_10']:.4f} (equal-mass)",
          f"- NLL {cal['nll']:.4f}, Brier {cal['brier']:.4f}, AURC {cal['aurc']:.4f}",
          f"- AUROC correct vs wrong: confidence {fmt(a['confidence'], 4) or '-'}, margin {fmt(a['margin'], 4) or '-'}, "
          f"negative entropy {fmt(a['negative_entropy'], 4) or '-'}",
          f"- Temperature diagnostic ({CV_FOLDS}-fold CV inside validation, not stored): held-out NLL "
          f"{td['heldout_nll_T1']:.4f} -> {td['heldout_nll_Tcv']:.4f}, held-out ECE {td['heldout_ece_equal_width_T1']:.4f} "
          f"-> {td['heldout_ece_equal_width_Tcv']:.4f}; fitted T above 1 in {td['folds_with_fitted_T_above_1']} of "
          f"{CV_FOLDS} folds", "",
          "## Checks", ""]
    L += [f"- {name}: PASS" for name in checks]
    return "\n".join(L) + "\n"


def save_plots(out_dir, records, bins_w, curve, run_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    correct = [r["confidence"] for r in records if r["correct"]]
    wrong = [r["confidence"] for r in records if not r["correct"]]
    edges = [k / 20 for k in range(21)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(correct, bins=edges, histtype="step", linewidth=1.8, label=f"correct (n={len(correct)})")
    ax.hist(wrong, bins=edges, histtype="step", linewidth=1.8, label=f"wrong (n={len(wrong)})")
    ax.set_yscale("log")
    ax.set_xlabel("confidence (max softmax)")
    ax.set_ylabel("images (log scale)")
    ax.set_title(f"{run_name}: validation confidence, correct vs wrong")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "conf_hist_correct_vs_wrong.png", dpi=150)
    plt.close(fig)

    used = [b for b in bins_w if b["n"]]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1, label="perfect calibration")
    ax.plot([b["mean_conf"] for b in used], [b["accuracy"] for b in used], marker="o", label="validation")
    for b in used:
        ax.annotate(str(b["n"]), (b["mean_conf"], b["accuracy"]), textcoords="offset points", xytext=(4, -10), fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("mean confidence in bin")
    ax.set_ylabel("accuracy in bin")
    ax.set_title(f"{run_name}: reliability ({EQUAL_WIDTH_BINS} equal-width bins)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "reliability.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot([c for c, _ in curve], [r for _, r in curve])
    ax.set_xlabel("coverage (most confident first)")
    ax.set_ylabel("error rate among covered images")
    ax.set_title(f"{run_name}: risk-coverage (validation)")
    fig.tight_layout()
    fig.savefig(out_dir / "risk_coverage.png", dpi=150)
    plt.close(fig)


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(
        description="Validation-only error and confidence audit of one finished run (read only; "
                    "Test and Neysan are never read; no checkpoint is written).")
    parser.add_argument("--run-name", required=True,
                        help="reads checkpoints/<run>_best.pt and reports/analysis/<run>/confusion_matrix.csv")
    parser.add_argument("--config", default=None, help="local raw-data paths (JSON), default configs/local_paths.json")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"), help="inference device (default cpu)")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace existing audit outputs in reports/analysis/<run>/val_audit/")
    parser.add_argument("--no-plots", action="store_true", help="skip the three PNG charts")
    args = parser.parse_args()

    checkpoint_path, saved_matrix_path, out_dir = run_paths(args.run_name)
    manifest_path = REPO_ROOT / "data" / "split_manifest.csv"
    for p in (checkpoint_path, saved_matrix_path, manifest_path):
        if not p.is_file():
            sys.exit(f"missing file: {p.relative_to(REPO_ROOT).as_posix()}")
    existing = [name for name in OUTPUT_FILES if (out_dir / name).exists()]
    if existing and not args.overwrite:
        sys.exit(f"outputs already exist in {out_dir.relative_to(REPO_ROOT).as_posix()} ({existing[0]}, ...); "
                 "use --overwrite to replace them")

    import torch
    import torchvision

    from src.dataset import CLASS_TO_IDX, CLASSES, DEFAULT_CONFIG, VehicleDataset, read_split_rows

    if args.device == "cuda" and not torch.cuda.is_available():
        sys.exit("--device cuda requested but CUDA is not available")

    guarded = {"checkpoint": checkpoint_path, "split_manifest.csv": manifest_path,
               "saved confusion_matrix.csv": saved_matrix_path}
    sha_before = {k: sha256_file(p) for k, p in guarded.items()}
    saved_matrix = read_saved_matrix(saved_matrix_path, CLASSES)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["class_to_idx"] != CLASS_TO_IDX:
        sys.exit(f"class mapping in checkpoint differs: {checkpoint['class_to_idx']}")
    model, eval_transform = build_model(checkpoint, CLASSES)

    rows = read_split_rows(manifest_path, "val")                    # only the "val" rows
    if len(rows) != EXPECTED_VAL_ROWS:
        sys.exit(f"expected {EXPECTED_VAL_ROWS} validation rows, found {len(rows)}")
    dataset = VehicleDataset("val", eval_transform, manifest_path, args.config or DEFAULT_CONFIG)
    if dataset.image_paths != [r["image_path"] for r in rows]:
        sys.exit("dataset order differs from the manifest order")

    print(f"{args.run_name}: {checkpoint['architecture']}, epoch {checkpoint['epoch']}, "
          f"{len(rows)} validation images, device {args.device}")
    logits, preds, labels = run_inference(model, dataset, args.device)

    sha_after = {k: sha256_file(p) for k, p in guarded.items()}
    changed = [k for k in guarded if sha_before[k] != sha_after[k]]
    if changed:
        sys.exit(f"files changed during the audit: {changed}")
    if len(preds) != EXPECTED_VAL_ROWS:
        sys.exit(f"expected {EXPECTED_VAL_ROWS} predictions, got {len(preds)}")
    records = build_records(rows, labels, preds, logits, CLASSES)
    if confusion_matrix(records, len(CLASSES)) != saved_matrix:
        sys.exit("re-predicted confusion matrix differs from the saved confusion_matrix.csv - stopping")
    checks = [f"{EXPECTED_VAL_ROWS} validation rows and {EXPECTED_VAL_ROWS} predictions",
              "dataset order and labels equal to the manifest",
              "argmax of the logits equal to the model prediction",
              f"probabilities in [0, 1], sum of the {len(CLASSES)} classes within {PROB_SUM_TOLERANCE} of 1",
              "confusion matrix identical to the saved confusion_matrix.csv",
              "checkpoint, split_manifest.csv and saved confusion_matrix.csv unchanged (SHA256 before = after)"]
    print("checks passed:\n  " + "\n  ".join(checks))

    per_class = per_class_rows(records, CLASSES)
    pairs = confusion_pair_rows(records, CLASSES)
    triples = [(r["confidence"], r["correct"], r["index"]) for r in records]
    bins_w = reliability_bins(triples, "equal_width")
    bins_m = reliability_bins(triples, "equal_mass")
    sweep, margin_thresholds = threshold_rows(records)
    cal = calibration_summary(records, bins_w, bins_m)
    curve, _ = risk_coverage(records)
    meta = {
        "run_name": args.run_name, "architecture": checkpoint["architecture"],
        "checkpoint": checkpoint_path.relative_to(REPO_ROOT).as_posix(),
        "checkpoint_sha256": sha_before["checkpoint"],
        "split_manifest": manifest_path.relative_to(REPO_ROOT).as_posix(),
        "split_manifest_sha256": sha_before["split_manifest.csv"],
        "best_epoch": checkpoint["epoch"], "stored_val_f1": checkpoint.get("val_f1"),
        "transform_recorded_in_checkpoint": checkpoint.get("transform"),
        "eval_transform": repr(eval_transform), "augmentation_at_evaluation": "none",
        "git_commit": git_commit(), "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__, "device": args.device,
        "n_validation": len(records), "classes": list(CLASSES),
        "confidence_bins": {"high": f">= {HIGH_CONF}", "medium": f">= {MEDIUM_CONF} and < {HIGH_CONF}",
                            "low": f"< {MEDIUM_CONF}", "use": "analysis only"},
        "sensitivity_thresholds": list(SENSITIVITY_THRESHOLDS),
        "margin_threshold_rule": "for each confidence threshold t, the margin threshold flags the same number of "
                                 "images as 'confidence < t' (the k-th smallest margin; fewer on ties), so both "
                                 "scores are compared at the same review cost",
        "margin_thresholds": margin_thresholds,
        "test_or_neysan_read": False,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    pred_rows = prediction_rows(records, CLASSES)
    write_csv(out_dir / "val_predictions.csv", pred_rows, list(pred_rows[0]))
    write_csv(out_dir / "per_class_audit.csv", per_class, list(per_class[0]))
    pair_columns = ["true_label", "predicted_label", "count", "n_high", "n_medium", "n_low",
                    "conf_mean", "conf_median", "conf_max", "margin_median"]
    write_csv(out_dir / "confusion_pairs.csv", pairs, pair_columns)
    write_csv(out_dir / "confidence_bins.csv", bins_w + bins_m,
              ["scheme", "bin_lo", "bin_hi", "n", "n_correct", "accuracy", "mean_conf", "gap"])
    write_csv(out_dir / "threshold_sweep.csv", sweep, list(sweep[0]))
    (out_dir / "calibration_summary.json").write_text(json.dumps(cal, indent=2) + "\n", encoding="utf-8")
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")
    (out_dir / "audit_summary.md").write_text(summary_markdown(meta, records, per_class, pairs, sweep, cal, checks),
                                              encoding="utf-8")
    if not args.no_plots:
        save_plots(out_dir, records, bins_w, curve, args.run_name)
    print(f"outputs written to {out_dir.relative_to(REPO_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
