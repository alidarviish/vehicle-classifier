"""Occlusion audit, Phase B1 (VALIDATION split only): does the model rely disproportionately on front/cab
information (H1), and does that explain its errors (H2)?

Read-only analysis: no training, no new checkpoint, no change to labels, splits, images or code.
Only the "val" rows of data/split_manifest.csv are read; the Test and Neysan files are never opened.
Sliding-window maps (Phase B2) are not part of this script.

Protocol (approved design, fixed before the run):
- Masking in the model's input space: raw image -> Resize(224x224) -> ToTensor -> MASK -> Normalize -> model.
- Mask families on the 224x224 image (y0:y1, x0:x1):
    F      front/cab proxy     128:224, 32:192
    C      cargo/body proxy      0:96,  32:192
    side   background strips   0:224, 0:32 and 0:224, 192:224   (14336 px vs 15360 px for F and C)
    full   whole image         0:224, 0:224
    random 20 rectangles of 96x160 per image (15360 px = F = C), top-left uniform, seed 42 + image index
    random_side 20 masks of two non-overlapping full-height 32 px strips (14336 px = side), same generator
    (all random coordinates are saved in occlusion_config.json)
- Fills: mean (ImageNet channel mean = 0 after Normalize), blur (Gaussian sigma 8, kernel 49, the blurred image
  is copied into the region), img_mean (the image's own mean colour).
- rel_drop = (z_pred_orig - z_pred_masked) / (z_pred_orig - z_pred_full) for the same fill; undefined when the
  denominator is <= 0. R = rel_drop_F - rel_drop_C.
- null_pct: percentile of the logit drop of F / C (among the 20 random rectangles) and of side (among the 20
  random_side masks), for the same image and fill; each null has exactly the area of the region it is compared with.
- H1 on the correct images of kamyun, kamyunet, ambulance, vanet; H2 on the three main pairs only
  (vanet -> savari is descriptive). The decision rules are in H1_RULES / H2_RULES below and in the summary.

Run from the repository root, after scripts/val_error_audit.py has written reports/analysis/<run>/val_audit/:
    python scripts/occlusion_audit.py
    python scripts/occlusion_audit.py --proxy-qc <region_proxy_qc.csv>   # optional pilot QC flags

Writes (generated, ignored by git) in reports/analysis/<run>/occlusion_audit/:
    occlusion_config.json, occlusion_region_results_B1.csv, occlusion_main_table_B1.csv, occlusion_h2_B1.csv,
    occlusion_summary_B1.md
Nothing is written if an integrity check fails.
"""

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))  # so "from src..." and "from scripts..." work

from scripts.val_error_audit import git_commit, log_softmax, sha256_file  # noqa: E402

DEFAULT_RUN = "resnet224_ft_aug"
EXPECTED_VAL_ROWS = 659
IMAGE_SIZE = 224
LOGIT_TOLERANCE = 1e-3

BANDS = {                                    # family -> list of (y0, y1, x0, x1) on the 224x224 input
    "F": [(128, 224, 32, 192)],
    "C": [(0, 96, 32, 192)],
    "side": [(0, 224, 0, 32), (0, 224, 192, 224)],
    "full": [(0, 224, 0, 224)],
}
RANDOM_N, RANDOM_H, RANDOM_W, SEED = 20, 96, 160, 42
SIDE_STRIP_W = 32                                        # random side null: two full-height strips of this width


def mask_area(rects):
    return sum((y1 - y0) * (x1 - x0) for y0, y1, x0, x1 in rects)


AREA = {fam: mask_area(rects) for fam, rects in BANDS.items()}   # F 15360, C 15360, side 14336, full 50176
if not (AREA["F"] == AREA["C"] == RANDOM_H * RANDOM_W and AREA["side"] == 2 * SIDE_STRIP_W * IMAGE_SIZE):
    raise AssertionError(f"random null masks are not equal-area with F / C / side: {AREA}")
FILLS = ("mean", "blur", "img_mean")
BLUR_SIGMA = 8.0
BLUR_KERNEL = 2 * math.ceil(3 * BLUR_SIGMA) + 1          # 49
FORWARD_BATCH = 32

PAIRS = [("kamyun", "kamyunet", "main"), ("kamyunet", "kamyun", "main"),
         ("ambulance", "vanet", "main"), ("vanet", "savari", "descriptive")]
H1_CLASSES = ("kamyun", "kamyunet", "ambulance", "vanet")
BOOTSTRAP_N, BOOTSTRAP_SEED = 10_000, 42

# thresholds of the approved design (do not change without a new design review)
MIN_FILLS = 2                 # "in at least 2 of 3 fills"
MIN_MAIN_PAIRS = 2            # "in at least 2 of the 3 main pairs"
NULL_F_MIN = 90.0             # median null_pct(F) >= 90
NULL_C_MAX = 75.0             # median null_pct(C) <= 75
NULL_SIDE_MAX = 75.0          # "side stays within the random null": median null_pct(side) <= 75 (same bound as C)
H2_MIN_WRONG_ABOVE_75 = 4     # at least 4 of the 6 wrong images above the 75th percentile of the correct R
QC_NO_CARGO_MAX = 0.30        # B1 unclear if more than 30% of the pilot QC flags are no_cargo_in_frame

H1_RULES = {
    "strengthened_per_fill": "median R > 0 and bootstrap CI lower bound > 0 and median null_pct(F) >= 90 and "
                             "median null_pct(C) <= 75 and median null_pct(side) <= 75",
    "weakened_per_fill": "bootstrap CI lower bound <= 0 (CI includes 0 or is negative) or median null_pct(F) < 90 "
                         "(F not above null) or median(rel_drop_side - rel_drop_F) >= 0 (side as damaging as F)",
    "decision": "population: correct images of kamyun, kamyunet, ambulance, vanet pooled (class results reported "
                "separately first); strengthened if >= 2 of 3 fills are strengthened and < 2 weakened; weakened if "
                ">= 2 of 3 fills are weakened and < 2 strengthened; otherwise unclear",
}
H2_RULES = {
    "support_per_pair_fill": "delta_R > 0 and >= 4 wrong images above the 75th percentile of the correct "
                             "same-class R and the pair-margin condition: for pair A -> B, d_margin_pair = "
                             "(z_A - z_B with mask C, same fill) - (z_A - z_B unmasked); median of abs(d_margin_pair) "
                             "over the wrong images (true A, predicted B) < median of abs(d_margin_pair) over the "
                             "correct images of class A (masking C changes the wrong images less)",
    "weak_per_pair_fill": "bootstrap CI lower bound of delta_R <= 0 (delta_R ~ 0 or negative, CI includes 0) and "
                          "< 4 wrong images above the 75th percentile (no concentration)",
    "pair_status": "supports if support in >= 2 of 3 fills; weakens if weak in >= 2 of 3 fills; otherwise unclear",
    "decision": "only the three main pairs; strengthened (descriptive) if >= 2 pairs support; weakened if >= 2 pairs "
                "weaken; otherwise unresolved. vanet -> savari is reported but has no weight.",
    "note": "with n = 6 wrong images per pair the bootstrap CI is descriptive evidence, not a significance test",
}
B1_RULE = ("clear if H1 is strengthened or weakened AND H2 is strengthened or weakened AND (when --proxy-qc is "
           "given) at most 30% of the QC flags are no_cargo_in_frame; otherwise unclear")

OUTPUT_FILES = ("occlusion_config.json", "occlusion_region_results_B1.csv", "occlusion_main_table_B1.csv",
                "occlusion_h2_B1.csv", "occlusion_summary_B1.md")


# ---------- small helpers ----------

def med(values):
    import numpy as np
    return float(np.median(values)) if len(values) else None


def pct_rank(x, ref):
    """Percentile of x within ref (0-100), ties count 1/2."""
    return 100.0 * (sum(r < x for r in ref) + 0.5 * sum(r == x for r in ref)) / len(ref)


def bootstrap_median_ci(values, seed=BOOTSTRAP_SEED):
    import numpy as np
    if not values:
        return None, None
    a = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    meds = np.median(a[rng.integers(0, len(a), (BOOTSTRAP_N, len(a)))], axis=1)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_median_diff_ci(a_values, b_values, seed=BOOTSTRAP_SEED):
    """95% bootstrap CI of median(a) - median(b); a and b resampled independently."""
    import numpy as np
    if not a_values or not b_values:
        return None, None
    a, b = np.asarray(a_values, dtype=float), np.asarray(b_values, dtype=float)
    rng = np.random.default_rng(seed)
    ma = np.median(a[rng.integers(0, len(a), (BOOTSTRAP_N, len(a)))], axis=1)
    mb = np.median(b[rng.integers(0, len(b), (BOOTSTRAP_N, len(b)))], axis=1)
    lo, hi = np.percentile(ma - mb, [2.5, 97.5])
    return float(lo), float(hi)


def rate(flags):
    return sum(flags) / len(flags) if flags else None


def cell(value, digits=6):
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return value


def write_csv(path, rows, columns):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({c: cell(r.get(c)) for c in columns})


def random_rectangles(index):
    """Deterministic random null masks of one image (depends only on SEED and the image index).
    Returns (rects, side_masks):
      rects      20 rectangles of 96x160 = 15360 px, the same area as F and C (null of F and C)
      side_masks 20 masks of two non-overlapping full-height strips of 32 px = 14336 px, the same area and shape
                 as side (null of side); drawn after the 20 rectangles from the same generator."""
    rng = random.Random(SEED * 1_000_003 + index)
    rects = []
    for _ in range(RANDOM_N):
        y0 = rng.randint(0, IMAGE_SIZE - RANDOM_H)
        x0 = rng.randint(0, IMAGE_SIZE - RANDOM_W)
        rects.append((y0, y0 + RANDOM_H, x0, x0 + RANDOM_W))
    side_masks = []
    for _ in range(RANDOM_N):
        while True:
            xa = rng.randint(0, IMAGE_SIZE - SIDE_STRIP_W)
            xb = rng.randint(0, IMAGE_SIZE - SIDE_STRIP_W)
            if abs(xa - xb) >= SIDE_STRIP_W:               # no overlap, so the area is exactly 2 x 32 x 224
                break
        side_masks.append([(0, IMAGE_SIZE, x, x + SIDE_STRIP_W) for x in sorted((xa, xb))])
    if any(mask_area([r]) != AREA["F"] for r in rects) or any(mask_area(m) != AREA["side"] for m in side_masks):
        raise AssertionError("random null mask with a different area")
    return rects, side_masks


def mask_list(index):
    """(family, mask_id, rectangles) of one image, in a fixed order."""
    rects, side_masks = random_rectangles(index)
    masks = [(fam, fam, r) for fam, r in BANDS.items()]
    masks += [("random", f"random_{j:02d}", [r]) for j, r in enumerate(rects)]
    masks += [("random_side", f"random_side_{j:02d}", m) for j, m in enumerate(side_masks)]
    return masks


# ---------- inputs and integrity ----------

def load_inputs(run_name):
    from src.dataset import CLASSES, DEFAULT_MANIFEST, read_split_rows

    checkpoint = REPO_ROOT / "checkpoints" / f"{run_name}_best.pt"
    audit_dir = REPO_ROOT / "reports" / "analysis" / run_name / "val_audit"
    audit_csv, audit_meta = audit_dir / "val_predictions.csv", audit_dir / "run_meta.json"
    for p in (checkpoint, audit_csv, audit_meta, DEFAULT_MANIFEST):
        if not p.is_file():
            sys.exit(f"missing file: {p.relative_to(REPO_ROOT).as_posix()}")
    with open(audit_csv, encoding="utf-8", newline="") as f:
        audit = list(csv.DictReader(f))
    split_val = read_split_rows(DEFAULT_MANIFEST, "val")          # only the "val" rows
    if len(audit) != EXPECTED_VAL_ROWS or len(split_val) != EXPECTED_VAL_ROWS:
        sys.exit(f"expected {EXPECTED_VAL_ROWS} validation rows (audit {len(audit)}, split {len(split_val)})")
    split_by_path = {r["image_path"]: r for r in split_val}
    for r in audit:
        s = split_by_path.get(r["image_path"])
        if s is None or s["sha256"] != r["sha256"] or s["label"] != r["true_label"]:
            sys.exit(f"val_predictions.csv row {r['index']} does not match the validation split")
    missing = [f"logit_{c}" for c in CLASSES if f"logit_{c}" not in audit[0]]
    if missing:
        sys.exit(f"val_predictions.csv lacks columns {missing}")
    meta = json.loads(audit_meta.read_text(encoding="utf-8"))
    return checkpoint, audit_csv, audit, meta, DEFAULT_MANIFEST


def raw_image_hashes(audit, roots):
    from src.dataset import resolve_image_path
    return {r["image_path"]: sha256_file(resolve_image_path(r["image_path"], roots)) for r in audit}


def split_transform(eval_transform):
    """(Resize + ToTensor, Normalize) of the checkpoint's evaluation transform; stops if it is not that shape."""
    from torchvision import transforms as T
    steps = list(getattr(eval_transform, "transforms", []))
    if (len(steps) != 3 or not isinstance(steps[0], T.Resize) or not isinstance(steps[1], T.ToTensor)
            or not isinstance(steps[2], T.Normalize)):
        sys.exit(f"unexpected evaluation transform: {eval_transform!r}")
    size = steps[0].size
    if (list(size) if isinstance(size, (list, tuple)) else [size]) != [IMAGE_SIZE, IMAGE_SIZE]:
        sys.exit(f"evaluation transform must resize to {IMAGE_SIZE}x{IMAGE_SIZE}, got {size}")
    return T.Compose(steps[:2]), steps[2]


# ---------- occlusion (torch) ----------

def occlude_image(model, x01, masks, normalize):
    """Logits (float64) of the unmasked image and of every mask x fill. Returns (keys, logits array)."""
    import torch
    import torchvision.transforms.functional as TF

    fills = {
        "mean": torch.tensor(normalize.mean, dtype=x01.dtype).view(3, 1, 1).expand_as(x01),
        "blur": TF.gaussian_blur(x01, kernel_size=[BLUR_KERNEL, BLUR_KERNEL], sigma=[BLUR_SIGMA, BLUR_SIGMA]),
        "img_mean": x01.mean(dim=(1, 2)).view(3, 1, 1).expand_as(x01),
    }
    keys, batch = [("orig", "orig", "none")], [x01]
    for fill in FILLS:
        for family, mask_id, rects in masks:
            m = x01.clone()
            for y0, y1, x0, x1 in rects:
                m[:, y0:y1, x0:x1] = fills[fill][:, y0:y1, x0:x1]
            keys.append((family, mask_id, fill))
            batch.append(m)
    out = []
    with torch.no_grad():
        for s in range(0, len(batch), FORWARD_BATCH):
            out.append(model(normalize(torch.stack(batch[s:s + FORWARD_BATCH]))).double())
    return keys, torch.cat(out).numpy()


def image_fill_metrics(rec, fill):
    """F / C / side / full metrics of one image and fill (predicted class = the unmasked prediction)."""
    k = rec["pred"]
    z0 = rec["orig"]
    den = z0[k] - rec["masked"][("full", "full", fill)][k]
    rand = [z0[k] - rec["masked"][("random", f"random_{j:02d}", fill)][k] for j in range(RANDOM_N)]
    rand_side = [z0[k] - rec["masked"][("random_side", f"random_side_{j:02d}", fill)][k] for j in range(RANDOM_N)]
    null_of = {"F": rand, "C": rand, "side": rand_side}          # equal-area null of each region
    out = {"den": den, "random_dz": rand, "random_side_dz": rand_side}
    for fam in BANDS:
        z = rec["masked"][(fam, fam, fill)]
        dz = z0[k] - z[k]
        out[fam] = {"dz": dz, "rel": dz / den if den > 0 else None,
                    "null_pct": pct_rank(dz, null_of[fam]) if fam in null_of else None,
                    "argmax": int(max(range(len(z)), key=lambda i: (z[i], -i))), "z": z}
    return out


def region_result_rows(rec, classes, fill_cache):
    idx = {c: i for i, c in enumerate(classes)}
    primary = {a: b for a, b, _ in PAIRS}
    true_c, pred_c = classes[rec["true"]], classes[rec["pred"]]
    pair = f"{true_c}->{primary[true_c]}" if true_c in primary else ""
    a, b = (idx[true_c], idx[primary[true_c]]) if pair else (None, None)
    k, z0 = rec["pred"], rec["orig"]
    p0 = math.exp(log_softmax(list(z0))[k])
    rows = []
    for fill in FILLS:
        fm = fill_cache[(rec["index"], fill)]
        for family, mask_id, rects in rec["masks"]:
            z = rec["masked"][(family, mask_id, fill)]
            pm = math.exp(log_softmax(list(z))[k])
            dz = float(z0[k] - z[k])
            arg = int(max(range(len(z)), key=lambda i: (z[i], -i)))
            row = {
                "index": rec["index"], "image_path": rec["image_path"], "true_label": true_c, "pred_label": pred_c,
                "correct": rec["correct"], "family": family, "mask_id": mask_id, "fill": fill,
                "rects": ";".join(f"{y0}:{y1},{x0}:{x1}" for y0, y1, x0, x1 in rects),
                "area": sum((y1 - y0) * (x1 - x0) for y0, y1, x0, x1 in rects),
                "z_pred_orig": float(z0[k]), "z_pred_masked": float(z[k]), "dz_pred": dz,
                "p_pred_orig": p0, "p_pred_masked": pm, "dp_pred": p0 - pm,
                "z_pred_full": float(z0[k] - fm["den"]),
                "rel_drop": dz / fm["den"] if fm["den"] > 0 else None,
                "masked_pred_label": classes[arg], "flip": arg != k,
                "flip_to_true": (arg == rec["true"]) if not rec["correct"] else None,
                "null_pct": fm[family]["null_pct"] if family in ("F", "C", "side") else None,
                "pair": pair,
            }
            if pair:
                m0, m1 = float(z0[a] - z0[b]), float(z[a] - z[b])
                row.update({"margin_pair_orig": m0, "margin_pair_masked": m1, "d_margin_pair": m1 - m0})
            rows.append(row)
    return rows


# ---------- analysis ----------

def group_region_stats(recs, fill_cache, fill, region, alt_idx=None, margin=None):
    """Statistics of one group of images for one region (F / C / side) and fill."""
    ms = [(r, fill_cache[(r["index"], fill)]) for r in recs]
    rels = [m[region]["rel"] for _, m in ms if m[region]["rel"] is not None]
    R = [m["F"]["rel"] - m["C"]["rel"] for _, m in ms if m["F"]["rel"] is not None and m["C"]["rel"] is not None]
    lo, hi = bootstrap_median_ci(R)
    stats = {"n": len(recs), "n_rel_undefined": len(recs) - len(rels), "median_rel_drop": med(rels),
             "median_R": med(R), "R_ci_lo": lo, "R_ci_hi": hi,
             "median_null_pct": med([m[region]["null_pct"] for _, m in ms]),
             "flip_rate": rate([m[region]["argmax"] != r["pred"] for r, m in ms]),
             "flip_to_true_rate": rate([m[region]["argmax"] == r["true"] for r, m in ms if not r["correct"]]),
             "flip_to_alt_rate": (rate([m[region]["argmax"] == alt_idx for r, m in ms if r["correct"]])
                                  if alt_idx is not None else None)}
    if margin is not None:
        a, b = margin
        stats["median_d_margin_pair"] = med([float((m[region]["z"][a] - m[region]["z"][b])
                                                   - (r["orig"][a] - r["orig"][b])) for r, m in ms])
    return stats


def h1_analysis(records, classes, fill_cache):
    """Per class and pooled (all4) statistics of the correct images; decision on the pooled population."""
    table, per_fill = [], {}
    for cls in H1_CLASSES + ("all4",):
        members = H1_CLASSES if cls == "all4" else (cls,)
        pop = [r for r in records if r["correct"] and classes[r["true"]] in members]
        for fill in FILLS:
            stats = {reg: group_region_stats(pop, fill_cache, fill, reg) for reg in ("F", "C", "side")}
            ms = [fill_cache[(r["index"], fill)] for r in pop]
            side_minus_f = med([m["side"]["rel"] - m["F"]["rel"] for m in ms
                                if m["side"]["rel"] is not None and m["F"]["rel"] is not None])
            f = stats["F"]
            strengthened = (f["median_R"] is not None and f["median_R"] > 0 and f["R_ci_lo"] > 0
                            and stats["F"]["median_null_pct"] >= NULL_F_MIN
                            and stats["C"]["median_null_pct"] <= NULL_C_MAX
                            and stats["side"]["median_null_pct"] <= NULL_SIDE_MAX)
            weakened = (f["R_ci_lo"] is None or f["R_ci_lo"] <= 0 or stats["F"]["median_null_pct"] < NULL_F_MIN
                        or (side_minus_f is not None and side_minus_f >= 0))
            per_fill[(cls, fill)] = {"strengthened": strengthened, "weakened": weakened,
                                     "median_side_minus_F": side_minus_f, "stats": stats}
            for reg, s in stats.items():
                table.append({"analysis": "H1", "pair_or_class": cls, "weight": "main", "group": "correct",
                              "region": reg, "fill": fill, **s})
    status = {}
    for cls in H1_CLASSES + ("all4",):
        n_s = sum(per_fill[(cls, f)]["strengthened"] for f in FILLS)
        n_w = sum(per_fill[(cls, f)]["weakened"] for f in FILLS)
        status[cls] = ("strengthened" if n_s >= MIN_FILLS and n_w < MIN_FILLS else
                       "weakened" if n_w >= MIN_FILLS and n_s < MIN_FILLS else "unclear")
    return table, per_fill, status


def h2_analysis(records, classes, fill_cache):
    import numpy as np
    idx = {c: i for i, c in enumerate(classes)}
    table, h2_rows, pair_status = [], [], {}
    for a_name, b_name, weight in PAIRS:
        a, b = idx[a_name], idx[b_name]
        pair = f"{a_name}->{b_name}"
        groups = {"wrong": [r for r in records if r["true"] == a and r["pred"] == b],
                  "correct_same": [r for r in records if r["true"] == a and r["correct"]],
                  "correct_opposite": [r for r in records if r["true"] == b and r["correct"]]}
        alt = {"wrong": None, "correct_same": b, "correct_opposite": a}
        n_support = n_weak = 0
        for fill in FILLS:
            for g, recs in groups.items():
                for reg in ("F", "C"):
                    table.append({"analysis": "H2", "pair_or_class": pair, "weight": weight, "group": g,
                                  "region": reg, "fill": fill,
                                  **group_region_stats(recs, fill_cache, fill, reg, alt[g], (a, b))})

            def r_values(recs):
                out = []
                for r in recs:
                    m = fill_cache[(r["index"], fill)]
                    if m["F"]["rel"] is not None and m["C"]["rel"] is not None:
                        out.append((r, m["F"]["rel"] - m["C"]["rel"]))
                return out

            def abs_dmargin_c(recs):
                vals = []
                for r in recs:
                    z = fill_cache[(r["index"], fill)]["C"]["z"]
                    vals.append(abs(float((z[a] - z[b]) - (r["orig"][a] - r["orig"][b]))))
                return med(vals)

            rw, rc = r_values(groups["wrong"]), r_values(groups["correct_same"])
            ro = r_values(groups["correct_opposite"])
            rc_vals = [v for _, v in rc]
            delta = (med([v for _, v in rw]) - med(rc_vals)) if rw and rc else None
            lo, hi = bootstrap_median_diff_ci([v for _, v in rw], rc_vals)
            q75 = float(np.percentile(rc_vals, 75)) if rc_vals else None
            q90 = float(np.percentile(rc_vals, 90)) if rc_vals else None
            above75 = int(sum(bool(v > q75) for _, v in rw)) if rc_vals else 0
            above90 = int(sum(bool(v > q90) for _, v in rw)) if rc_vals else 0
            mw, mc = abs_dmargin_c(groups["wrong"]), abs_dmargin_c(groups["correct_same"])
            margin_ok = mw is not None and mc is not None and mw < mc
            support = delta is not None and delta > 0 and above75 >= H2_MIN_WRONG_ABOVE_75 and margin_ok
            weak = lo is not None and lo <= 0 and above75 < H2_MIN_WRONG_ABOVE_75
            n_support += support
            n_weak += weak
            h2_rows.append({"row_type": "pair_fill", "pair": pair, "weight": weight, "fill": fill,
                            "n_wrong": len(groups["wrong"]), "n_wrong_R": len(rw),
                            "n_correct_same": len(groups["correct_same"]), "n_correct_same_R": len(rc),
                            "n_correct_opposite": len(groups["correct_opposite"]), "n_correct_opposite_R": len(ro),
                            "median_R_wrong": med([v for _, v in rw]), "median_R_correct_same": med(rc_vals),
                            "median_R_correct_opposite": med([v for _, v in ro]),
                            "delta_R": delta, "delta_R_ci_lo": lo, "delta_R_ci_hi": hi,
                            "correct_R_q75": q75, "correct_R_q90": q90,
                            "n_wrong_above_q75": above75, "n_wrong_above_q90": above90,
                            "median_abs_d_margin_C_wrong": mw, "median_abs_d_margin_C_correct_same": mc,
                            "margin_C_condition": margin_ok, "support": support, "weak": weak})
            for r, v in rw:
                h2_rows.append({"row_type": "wrong_image", "pair": pair, "weight": weight, "fill": fill,
                                "index": r["index"], "image_path": r["image_path"], "R": v,
                                "pct_in_correct_R": pct_rank(v, rc_vals) if rc_vals else None,
                                "above_q75": bool(v > q75) if rc_vals else None,
                                "above_q90": bool(v > q90) if rc_vals else None})
        pair_status[pair] = {"weight": weight, "n_support_fills": n_support, "n_weak_fills": n_weak,
                             "status": ("supports" if n_support >= MIN_FILLS else
                                        "weakens" if n_weak >= MIN_FILLS else "unclear")}
    main = [p for p, s in pair_status.items() if s["weight"] == "main"]
    n_sup = sum(pair_status[p]["status"] == "supports" for p in main)
    n_wk = sum(pair_status[p]["status"] == "weakens" for p in main)
    h2 = ("strengthened (descriptive)" if n_sup >= MIN_MAIN_PAIRS else
          "weakened" if n_wk >= MIN_MAIN_PAIRS else "unresolved")
    return table, h2_rows, pair_status, h2


def read_proxy_qc(path):
    if path is None:
        return None
    with open(path, encoding="utf-8", newline="") as f:
        flags = [r.get("flag", "").strip() for r in csv.DictReader(f)]
    flags = [x for x in flags if x]
    if not flags:
        sys.exit(f"no QC flags found in {path}")
    return {"n_flags": len(flags), "n_no_cargo_in_frame": flags.count("no_cargo_in_frame"),
            "share_no_cargo_in_frame": flags.count("no_cargo_in_frame") / len(flags)}


# ---------- summary ----------

def summary_markdown(meta, h1_fill, h1_status, pair_status, h2_status, b1, qc, checks, undefined):
    f = lambda v, d=4: "-" if v is None else f"{v:.{d}f}"  # noqa: E731
    L = [f"# Occlusion audit B1: `{meta['run_name']}`", "",
         "Validation split only (659 images). Test and Neysan were not read. No training; the checkpoint, the data "
         "and the code were not changed. Occlusion measures the sensitivity of the predicted-class logit to hiding "
         "a region; it does not show what the model understands.", "",
         f"- **B1 status: {b1}**",
         f"- **H1 (front/cab reliance, correct images of {', '.join(H1_CLASSES)} pooled): {h1_status['all4']}**",
         f"- **H2 (reliance explains the errors, three main pairs): {h2_status}**", "",
         f"B1 rule: {B1_RULE}.", "",
         "## Setup", "",
         f"- Checkpoint `{meta['checkpoint']}` (epoch {meta['best_epoch']}), SHA256 `{meta['checkpoint_sha256']}`",
         "- Masking on the transformed 224x224 input (after Resize, before Normalize)",
         "- Regions (y0:y1, x0:x1): F 128:224, 32:192 (15360 px); C 0:96, 32:192 (15360 px); side 0:224, 0:32 + "
         "0:224, 192:224 (14336 px); full 0:224, 0:224",
         "- Equal-area nulls per image (seed 42 + index, coordinates in occlusion_config.json): for F and C, 20 random "
         "96x160 rectangles (15360 px); for side, 20 random pairs of non-overlapping full-height 32 px strips "
         "(14336 px)",
         f"- Fills: mean (ImageNet mean), blur (Gaussian sigma {BLUR_SIGMA}, kernel {BLUR_KERNEL}), img_mean",
         f"- Bootstrap: {BOOTSTRAP_N} resamples, seed {BOOTSTRAP_SEED}; percentile CI (2.5, 97.5)",
         f"- rel_drop undefined (z_pred_orig - z_pred_full <= 0): {undefined} image x fill cases, excluded from "
         "rel_drop and R", "",
         "## H1: correct images, per fill", "",
         f"Rules: strengthened per fill = {H1_RULES['strengthened_per_fill']}; weakened per fill = "
         f"{H1_RULES['weakened_per_fill']}; {H1_RULES['decision']}.", "",
         "| Population | Fill | n | median R | R 95% CI | null_pct F | null_pct C | null_pct side | "
         "median(side - F) | flip F | flip C | strengthened | weakened |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for cls in H1_CLASSES + ("all4",):
        for fill in FILLS:
            e = h1_fill[(cls, fill)]
            s = e["stats"]
            L.append(f"| {cls} | {fill} | {s['F']['n']} | {f(s['F']['median_R'])} | "
                     f"{f(s['F']['R_ci_lo'])} to {f(s['F']['R_ci_hi'])} | {f(s['F']['median_null_pct'], 1)} | "
                     f"{f(s['C']['median_null_pct'], 1)} | {f(s['side']['median_null_pct'], 1)} | "
                     f"{f(e['median_side_minus_F'])} | {f(s['F']['flip_rate'], 3)} | {f(s['C']['flip_rate'], 3)} | "
                     f"{e['strengthened']} | {e['weakened']} |")
    L += ["", "Status per class (reported before the pooled decision): "
          + ", ".join(f"{c}: {h1_status[c]}" for c in H1_CLASSES) + f"; pooled all4: {h1_status['all4']}", "",
          "## H2: wrong vs correct in the same class", "",
          f"Rules: support per pair and fill = {H2_RULES['support_per_pair_fill']}; weak per pair and fill = "
          f"{H2_RULES['weak_per_pair_fill']}; {H2_RULES['pair_status']}; {H2_RULES['decision']} "
          f"Note: {H2_RULES['note']}.", "",
          "| Pair | Weight | Fills with support | Fills weak | Pair status |", "|---|---|---|---|---|"]
    L += [f"| {p} | {s['weight']} | {s['n_support_fills']} | {s['n_weak_fills']} | {s['status']} |"
          for p, s in pair_status.items()]
    L += ["", "Per pair and fill values are in `occlusion_h2_B1.csv`; group statistics in "
          "`occlusion_main_table_B1.csv`.", "",
          "## Pilot region-proxy QC", "",
          ("not applied (no --proxy-qc file given); the B1 status does not include this condition" if qc is None else
           f"{qc['n_no_cargo_in_frame']} of {qc['n_flags']} flags are no_cargo_in_frame "
           f"({qc['share_no_cargo_in_frame']:.1%}; limit {QC_NO_CARGO_MAX:.0%})"), "",
          "## Integrity checks", ""]
    L += [f"- {c}: PASS" for c in checks]
    L += ["", "## Limitations", "",
          "- F and C are fixed geometric bands that only approximate the vehicle front and cargo; they are not "
          "annotated parts. In close-up or rotated images C may not contain cargo.",
          "- The transform resizes to 224x224 without keeping the aspect ratio; the bands refer to that input.",
          "- Masks create out-of-distribution inputs; three fills and the random/side controls limit, but do not "
          "remove, this effect.",
          "- H2 rests on 6 wrong images per main pair (2 for vanet -> savari, descriptive only); the bootstrap CI is "
          "descriptive, not a significance test.",
          "- One model, one checkpoint, validation only; the checkpoint was selected on this validation split.",
          "- Sliding-window maps (B2) were not computed."]
    return "\n".join(L) + "\n"


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(
        description="Occlusion audit Phase B1 on the validation split (read only; Test and Neysan are never read).")
    parser.add_argument("--run-name", default=DEFAULT_RUN,
                        help="reads checkpoints/<run>_best.pt and reports/analysis/<run>/val_audit/")
    parser.add_argument("--config", default=None, help="local raw-data paths (JSON), default configs/local_paths.json")
    parser.add_argument("--proxy-qc", default=None,
                        help="optional CSV with a 'flag' column from the pilot region-proxy QC")
    parser.add_argument("--overwrite", action="store_true", help="replace existing B1 outputs")
    args = parser.parse_args()

    out_dir = REPO_ROOT / "reports" / "analysis" / args.run_name / "occlusion_audit"
    existing = [n for n in OUTPUT_FILES if (out_dir / n).exists()]
    if existing and not args.overwrite:
        sys.exit(f"outputs already exist in {out_dir.relative_to(REPO_ROOT).as_posix()}; use --overwrite")
    qc = read_proxy_qc(args.proxy_qc)

    import torch
    import torchvision
    from PIL import Image

    from scripts.val_error_audit import build_model
    from src.dataset import CLASS_TO_IDX, CLASSES, DEFAULT_CONFIG, load_source_roots, resolve_image_path

    torch.manual_seed(SEED)
    checkpoint_path, audit_csv, audit, audit_meta, manifest = load_inputs(args.run_name)
    roots = load_source_roots(args.config or DEFAULT_CONFIG)

    guarded = {"checkpoint": checkpoint_path, "split_manifest.csv": manifest, "val_predictions.csv": audit_csv}
    sha_before = {k: sha256_file(p) for k, p in guarded.items()}
    if audit_meta.get("checkpoint_sha256") != sha_before["checkpoint"]:
        sys.exit("the validation audit was made with a different checkpoint (SHA256 differs) - stopping")
    raw_before = raw_image_hashes(audit, roots)
    bad = [r["image_path"] for r in audit if raw_before[r["image_path"]] != r["sha256"]]
    if bad:
        sys.exit(f"{len(bad)} raw images differ from val_predictions.csv, first: {bad[:3]}")

    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if ck["class_to_idx"] != CLASS_TO_IDX:
        sys.exit("class mapping in checkpoint differs")
    model, eval_transform = build_model(ck, CLASSES)
    model.to("cpu")
    model.eval()
    to_x01, normalize = split_transform(eval_transform)

    records, fill_cache, max_diff = [], {}, 0.0
    for n, r in enumerate(audit, 1):
        index = int(r["index"])
        with Image.open(resolve_image_path(r["image_path"], roots)) as img:
            x01 = to_x01(img.convert("RGB"))                     # Resize + ToTensor, in memory
        masks = mask_list(index)
        keys, logits = occlude_image(model, x01, masks, normalize)
        audit_logits = [float(r[f"logit_{c}"]) for c in CLASSES]
        diff = max(abs(float(logits[0][j]) - audit_logits[j]) for j in range(len(CLASSES)))
        pred = int(max(range(len(CLASSES)), key=lambda j: (logits[0][j], -j)))
        if diff > LOGIT_TOLERANCE or CLASSES[pred] != r["pred_label"]:
            sys.exit(f"#{index}: unmasked logits differ from the validation audit (max abs diff {diff:.2e}) - stopping")
        max_diff = max(max_diff, diff)
        rec = {"index": index, "image_path": r["image_path"], "true": CLASS_TO_IDX[r["true_label"]], "pred": pred,
               "correct": r["true_label"] == r["pred_label"], "orig": logits[0], "masks": masks,
               "masked": {key: logits[i] for i, key in enumerate(keys) if i > 0}}
        records.append(rec)
        for fill in FILLS:
            fill_cache[(index, fill)] = image_fill_metrics(rec, fill)
        if n % 50 == 0 or n == len(audit):
            print(f"  {n}/{len(audit)} images")

    sha_after = {k: sha256_file(p) for k, p in guarded.items()}
    changed = [k for k in guarded if sha_before[k] != sha_after[k]]
    if changed:
        sys.exit(f"files changed during the audit: {changed} - stopping")
    raw_after = raw_image_hashes(audit, roots)
    if raw_after != raw_before:
        sys.exit("raw images changed during the audit - stopping")
    checks = [f"{EXPECTED_VAL_ROWS} validation rows; val_predictions.csv paths, SHA256 and labels equal to the "
              "split manifest (val rows only)",
              "validation audit made with the same checkpoint (run_meta.json checkpoint_sha256)",
              f"raw-image SHA256 equal to val_predictions.csv for all {len(raw_before)} images, before and after",
              f"unmasked logits equal to val_predictions.csv (max abs diff {max_diff:.2e} <= {LOGIT_TOLERANCE}) and "
              "the same predicted class for every image",
              "checkpoint, split_manifest.csv and val_predictions.csv unchanged (SHA256 before = after)"]
    print("integrity checks passed")

    undefined = sum(fill_cache[k]["den"] <= 0 for k in fill_cache)
    region_rows = [row for rec in records for row in region_result_rows(rec, CLASSES, fill_cache)]
    h1_table, h1_fill, h1_status = h1_analysis(records, CLASSES, fill_cache)
    h2_table, h2_rows, pair_status, h2_status = h2_analysis(records, CLASSES, fill_cache)
    qc_fail = qc is not None and qc["share_no_cargo_in_frame"] > QC_NO_CARGO_MAX
    b1 = ("clear" if h1_status["all4"] in ("strengthened", "weakened")
          and h2_status in ("strengthened (descriptive)", "weakened") and not qc_fail else "unclear")

    meta = {
        "phase": "B1", "run_name": args.run_name, "architecture": ck["architecture"], "best_epoch": ck["epoch"],
        "checkpoint": checkpoint_path.relative_to(REPO_ROOT).as_posix(),
        "checkpoint_sha256": sha_before["checkpoint"], "split_manifest_sha256": sha_before["split_manifest.csv"],
        "val_predictions_sha256": sha_before["val_predictions.csv"],
        "transform_recorded_in_checkpoint": ck.get("transform"), "eval_transform": repr(eval_transform),
        "masking_space": "after Resize(224x224) and ToTensor, before Normalize",
        "bands": {k: [list(r) for r in v] for k, v in BANDS.items()},
        "areas_px": AREA,
        "random": {"n": RANDOM_N, "height": RANDOM_H, "width": RANDOM_W, "area_px": RANDOM_H * RANDOM_W,
                   "null_of": "F and C (equal area)",
                   "seed_rule": f"random.Random({SEED} * 1000003 + image index)",
                   "rectangles": {str(rec["index"]): [list(m[2][0]) for m in rec["masks"] if m[0] == "random"]
                                  for rec in records}},
        "random_side": {"n": RANDOM_N, "strip_width": SIDE_STRIP_W, "strips_per_mask": 2,
                        "area_px": 2 * SIDE_STRIP_W * IMAGE_SIZE, "null_of": "side (equal area and shape)",
                        "rule": "two non-overlapping full-height strips at random x, drawn after the 20 rectangles "
                                "from the same generator (same seed rule)",
                        "masks": {str(rec["index"]): [[list(r) for r in m[2]] for m in rec["masks"]
                                                      if m[0] == "random_side"] for rec in records}},
        "fills": {"mean": "ImageNet channel mean from the checkpoint transform",
                  "blur": f"torchvision gaussian_blur, sigma {BLUR_SIGMA}, kernel {BLUR_KERNEL}, blurred image "
                          "copied into the region", "img_mean": "per-image mean colour of the 224x224 input"},
        "metrics": {"rel_drop": "(z_pred_orig - z_pred_masked) / (z_pred_orig - z_pred_full), same fill; undefined "
                                "if the denominator <= 0", "R": "rel_drop_F - rel_drop_C",
                    "null_pct": "percentile of the logit drop among the 20 equal-area random masks of the same image "
                                "and fill (F and C: random 96x160 rectangles; side: random_side strips; ties 1/2)",
                    "margin_pair": "z_A - z_B for the pair (A -> B)",
                    "d_margin_pair": "(z_A - z_B) masked - (z_A - z_B) unmasked, same image and fill"},
        "pairs": [{"from": a, "to": b, "weight": w} for a, b, w in PAIRS], "h1_classes": list(H1_CLASSES),
        "bootstrap": {"resamples": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED, "ci": "percentile 2.5 / 97.5"},
        "thresholds": {"min_fills": MIN_FILLS, "min_main_pairs": MIN_MAIN_PAIRS, "null_F_min": NULL_F_MIN,
                       "null_C_max": NULL_C_MAX, "null_side_max": NULL_SIDE_MAX,
                       "h2_min_wrong_above_q75": H2_MIN_WRONG_ABOVE_75, "qc_no_cargo_max": QC_NO_CARGO_MAX},
        "rules": {"H1": H1_RULES, "H2": H2_RULES, "B1": B1_RULE},
        "results": {"B1": b1, "H1": h1_status, "H2": h2_status, "H2_pairs": pair_status,
                    "rel_drop_undefined_cases": undefined, "proxy_qc": qc},
        "integrity_checks": checks, "git_commit": git_commit(), "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__, "device": "cpu", "n_validation": len(records),
        "test_or_neysan_read": False,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    region_cols = ["index", "image_path", "true_label", "pred_label", "correct", "family", "mask_id", "fill",
                   "rects", "area", "z_pred_orig", "z_pred_masked", "dz_pred", "p_pred_orig", "p_pred_masked",
                   "dp_pred", "z_pred_full", "rel_drop", "masked_pred_label", "flip", "flip_to_true", "null_pct",
                   "pair", "margin_pair_orig", "margin_pair_masked", "d_margin_pair"]
    write_csv(out_dir / "occlusion_region_results_B1.csv", region_rows, region_cols)
    main_cols = ["analysis", "pair_or_class", "weight", "group", "region", "fill", "n", "n_rel_undefined",
                 "median_rel_drop", "median_R", "R_ci_lo", "R_ci_hi", "median_null_pct", "flip_rate",
                 "flip_to_true_rate", "flip_to_alt_rate", "median_d_margin_pair"]
    write_csv(out_dir / "occlusion_main_table_B1.csv", h1_table + h2_table, main_cols)
    h2_cols = ["row_type", "pair", "weight", "fill", "n_wrong", "n_wrong_R", "n_correct_same", "n_correct_same_R",
               "n_correct_opposite", "n_correct_opposite_R", "median_R_wrong", "median_R_correct_same",
               "median_R_correct_opposite", "delta_R", "delta_R_ci_lo", "delta_R_ci_hi", "correct_R_q75",
               "correct_R_q90", "n_wrong_above_q75", "n_wrong_above_q90", "median_abs_d_margin_C_wrong",
               "median_abs_d_margin_C_correct_same", "margin_C_condition", "support", "weak",
               "index", "image_path", "R", "pct_in_correct_R", "above_q75", "above_q90"]
    write_csv(out_dir / "occlusion_h2_B1.csv", h2_rows, h2_cols)
    (out_dir / "occlusion_config.json").write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")
    (out_dir / "occlusion_summary_B1.md").write_text(
        summary_markdown(meta, h1_fill, h1_status, pair_status, h2_status, b1, qc, checks, undefined),
        encoding="utf-8")
    print(f"B1: {b1} | H1: {h1_status['all4']} | H2: {h2_status}")
    print(f"outputs written to {out_dir.relative_to(REPO_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
