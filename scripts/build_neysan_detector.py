"""Build the Neysan detector used by run_test.py (docs/NEYSAN_REJECTION.md).

    python scripts/build_neysan_detector.py

Images are read like everywhere else in the project: train / validation from data/split_manifest.csv,
Neysan from decisions/neysan_eval.csv, local folders from configs/local_paths.json.

Nothing here trains or changes the 8-class model. Steps:
  1. load the frozen final model and take the 768-number feature vector of every image
     (the input of the model's last layer)
  2. check the pipeline: the 8-class validation result must be 640 / 659 correct
  3. split the confirmed Neysan images (by SHA256, seed 42): 70% train, 15% calibration, 15% holdout
  4. train a logistic regression "Neysan / not Neysan" on the training images + 70% of the Neysan images
  5. measure it with the FIXED threshold 0.5 (not tuned) and save it as JSON

Acceptance rules, fixed before the first run:
  - held-out confirmed Neysan rejected >= 80%
  - validation images (659 ordinary images) rejected <= 6 (1%)

The result is written to a NEW file (default checkpoints/neysan_detector_rerun.json). The accepted
detector checkpoints/neysan_detector.json is never overwritten; a re-run on another machine (CPU vs
GPU) can differ in the last decimals and therefore has a different SHA256.
"""

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.dataset import (CLASSES, DEFAULT_CONFIG, DEFAULT_MANIFEST, load_source_roots,  # noqa: E402
                         read_split_rows, resolve_image_path)
from src.test_runner import (DEFAULT_CHECKPOINT, DEFAULT_DETECTOR, FINAL_CHECKPOINT_SHA256,  # noqa: E402
                             NeysanDetector, sha256_of)

NEYSAN_MANIFEST = REPO_ROOT / "decisions" / "neysan_eval.csv"
NEYSAN_MANIFEST_SHA256 = "1e8cad9bbca206740f78dda72da6318c8851d6f56ec17281317048308576086a"
THRESHOLD = 0.5   # fixed in advance, not tuned
SEED = 42


def main():
    ap = argparse.ArgumentParser(description="Build the Neysan detector (logistic regression on model features).")
    ap.add_argument("--output", default=str(DEFAULT_DETECTOR.with_name("neysan_detector_rerun.json")))
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from src.predict import load_model

    out = Path(args.output)
    if out.resolve() == DEFAULT_DETECTOR.resolve():
        raise SystemExit("ERROR: never overwrite the accepted detector; choose another --output")
    if out.exists():
        raise SystemExit(f"ERROR: output exists already: {out}")

    # ---- checks on the inputs
    if sha256_of(DEFAULT_CHECKPOINT) != FINAL_CHECKPOINT_SHA256:
        raise SystemExit("ERROR: checkpoints/swin_t_ft_aug_best.pt is not the final model")
    if sha256_of(NEYSAN_MANIFEST) != NEYSAN_MANIFEST_SHA256:
        raise SystemExit("ERROR: decisions/neysan_eval.csv was changed")
    roots = load_source_roots(DEFAULT_CONFIG)
    train = [(resolve_image_path(r["image_path"], roots), r["label"]) for r in read_split_rows(DEFAULT_MANIFEST, "train")]
    val = [(resolve_image_path(r["image_path"], roots), r["label"]) for r in read_split_rows(DEFAULT_MANIFEST, "val")]
    with open(NEYSAN_MANIFEST, encoding="utf-8-sig", newline="") as f:
        ny = list(csv.DictReader(f))
    if (len(train), len(val), len(ny)) != (2633, 659, 621):
        raise SystemExit(f"ERROR: expected 2633 / 659 / 621 images, found {len(train)} / {len(val)} / {len(ny)}")

    # ---- 1. model features
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform, _ = load_model(DEFAULT_CHECKPOINT)
    model = model.to(device).eval()
    feats = {}
    model.net.head.register_forward_hook(lambda m, inp, o: feats.__setitem__("f", inp[0]))

    def embed(open_fns, name):
        F, P = [], []
        for i in range(0, len(open_fns), args.batch_size):
            x = torch.stack([transform(fn().convert("RGB")) for fn in open_fns[i:i + args.batch_size]]).to(device)
            with torch.inference_mode():
                logits = model(x)
            F.append(feats["f"].float().cpu())
            P.append(torch.softmax(logits.float(), 1).cpu())
            print(f"\r  {name}: {min(i + args.batch_size, len(open_fns))}/{len(open_fns)}", end="", flush=True)
        print()
        return torch.cat(F).numpy(), torch.cat(P).numpy()

    print(f"device: {device}  (CPU takes a while)")
    t0 = time.time()
    Ftr, _ = embed([lambda p=p: Image.open(p) for p, _ in train], "train")
    Fva, Pva = embed([lambda p=p: Image.open(p) for p, _ in val], "val")
    Fny, _ = embed([lambda r=r: Image.open(resolve_image_path(r["image_path"], roots)) for r in ny], "neysan")
    print(f"features done in {time.time() - t0:.0f} s")

    # ---- 2. pipeline check
    yva = np.array([CLASSES.index(c) for _, c in val])
    val_correct = int((Pva.argmax(1) == yva).sum())
    if val_correct != 640:
        raise SystemExit(f"ERROR: 8-class validation correct {val_correct}/659, expected 640 - pipeline mismatch")

    # ---- 3. split the confirmed Neysan images by SHA256 (seed 42)
    conf = [i for i, r in enumerate(ny) if r["basis"] == "CONFIRMED_NEYSAN"]
    n1 = [i for i, r in enumerate(ny) if r["basis"] != "CONFIRMED_NEYSAN"]
    shas = sorted({ny[i]["sha256"] for i in conf})
    random.Random(SEED).shuffle(shas)
    a, b = int(0.70 * len(shas)), int(0.85 * len(shas))
    part = {s: ("train" if k < a else "calib" if k < b else "holdout") for k, s in enumerate(shas)}
    ny_tr = [i for i in conf if part[ny[i]["sha256"]] == "train"]
    ny_ca = [i for i in conf if part[ny[i]["sha256"]] == "calib"]
    ny_ho = [i for i in conf if part[ny[i]["sha256"]] == "holdout"]
    train_shas = {ny[i]["sha256"] for i in ny_tr}
    n1_check = [i for i in n1 if ny[i]["sha256"] not in train_shas]   # no copy of a training image

    # ---- 4. train
    X = np.concatenate([Ftr, Fny[ny_tr]])
    y = np.r_[np.zeros(len(Ftr)), np.ones(len(ny_tr))]
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(scaler.transform(X), y)

    def rejected(F):
        return clf.predict_proba(scaler.transform(F))[:, 1] >= THRESHOLD

    # ---- 5. measure with the fixed threshold
    rej_val = rejected(Fva)
    ho, ca, n1r = rejected(Fny[ny_ho]).mean(), rejected(Fny[ny_ca]).mean(), rejected(Fny[n1_check]).mean()
    correct_kept = int(((Pva.argmax(1) == yva) & ~rej_val).sum())
    passed = ho >= 0.80 and rej_val.sum() <= 6

    print("\n===== RESULT =====")
    print(f"8-class validation correct          : {val_correct}/659")
    print(f"Neysan split train / calib / holdout : {len(ny_tr)} / {len(ny_ca)} / {len(ny_ho)}")
    print(f"holdout Neysan rejected              : {ho:.3f}  ({int(round(ho * len(ny_ho)))}/{len(ny_ho)})   rule >= 0.80")
    print(f"calibration Neysan rejected          : {ca:.3f}  ({int(round(ca * len(ny_ca)))}/{len(ny_ca)})")
    print(f"N1 Neysan (not reviewed) rejected    : {n1r:.3f}  (n={len(n1_check)})")
    print(f"validation images rejected           : {int(rej_val.sum())}/659   rule <= 6")
    print(f"validation correct after rejection   : {correct_kept}/659")
    print("DECISION:", "PASS" if passed else "FAIL")

    det = {"type": "logistic_regression_on_swin_head_input", "threshold": THRESHOLD, "feature_dim": int(Ftr.shape[1]),
           "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(), "coef": clf.coef_[0].tolist(),
           "intercept": float(clf.intercept_[0]), "checkpoint_sha256": FINAL_CHECKPOINT_SHA256,
           "trained_on": {"negatives": "train split 2633", "positives": f"confirmed Neysan {len(ny_tr)} (seed 42, 70%)"},
           "results": {"holdout_rejected": ho, "calib_rejected": ca, "n1_rejected": n1r,
                       "val_rejected": int(rej_val.sum()), "val_correct_after_rejection": correct_kept},
           "decision": "PASS" if passed else "FAIL"}
    text = json.dumps(det)
    out.write_text(text)
    print(f"saved: {out}  sha256 {hashlib.sha256(text.encode()).hexdigest()}")

    # ---- compare with the accepted detector (same decision for how many images?)
    if DEFAULT_DETECTOR.is_file():
        accepted = NeysanDetector(DEFAULT_DETECTOR)
        F_all = np.concatenate([Fva, Fny])
        old = np.array([accepted.score(f) >= accepted.threshold for f in F_all.tolist()])
        new = rejected(F_all)
        print(f"same decision as the accepted detector: {int((old == new).sum())}/{len(F_all)} images (validation + Neysan)")
    print("===== END =====")


if __name__ == "__main__":
    main()
