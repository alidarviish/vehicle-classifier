"""Build the Neysan detector and choose the needs_review threshold for the v3 model.

    python scripts/build_neysan_detector_v3.py

Same method as scripts/archive/build_neysan_detector.py (docs/NEYSAN_REJECTION.md), now for the v3 model
(checkpoints/v3_final_best.pt) and dataset v3 (data/dataset_v3). Nothing here trains the 8-class model,
and the test folders are never read.

Steps:
  1. take the 768-number feature vector of every train, validation and Neysan image from the v3 model
  2. needs_review threshold on validation: the lowest of 0.70 / 0.80 / 0.90 / 0.95 that flags at least
     half of the validation errors (the same rule as before)
  3. split the trusted Neysan images by SHA256 (seed 42): 70% train, 15% calibration, 15% holdout
     trusted = confirmed Neysan from our vanet folders + Neysan images from the external data;
     the N1 folder images (never reviewed one by one) are only an extra check
  4. logistic regression "Neysan / not Neysan" on all train images + 70% of the trusted Neysan
  5. measure with the FIXED threshold 0.5 and save checkpoints/neysan_detector_v3.json

Acceptance rules, fixed before the first run:
  - held-out trusted Neysan rejected >= 80%
  - validation images (755 ordinary images) rejected <= 7 (1%)
"""
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.dataset import CLASSES  # noqa: E402

CHECKPOINT = REPO / "checkpoints" / "v3_final_best.pt"
OUTPUT = REPO / "checkpoints" / "neysan_detector_v3.json"
DATA = REPO / "data" / "dataset_v3"
THRESHOLD = 0.5                        # fixed in advance, not tuned
REVIEW_CANDIDATES = [0.70, 0.80, 0.90, 0.95]
SEED = 42
BATCH = 32


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def images(folder):
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})


def main():
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from src.predict import load_model

    if OUTPUT.exists():
        raise SystemExit(f"ERROR: {OUTPUT} exists already")
    ckpt_sha = sha256(CHECKPOINT)
    print("model:", CHECKPOINT.name, "sha256", ckpt_sha)

    train = [(p, p.parent.name) for p in images(DATA / "train")]
    val = [(p, p.parent.name) for p in images(DATA / "val")]
    neysan = images(DATA / "neysan")
    print(f"train {len(train)} | val {len(val)} | neysan {len(neysan)}")

    # ---- 1. features (input of the model's last layer)
    model, transform, _ = load_model(CHECKPOINT)
    model.eval()
    feats = {}
    model.net.head.register_forward_hook(lambda m, inp, o: feats.__setitem__("f", inp[0]))

    def embed(paths, name):
        F, P = [], []
        for i in range(0, len(paths), BATCH):
            x = torch.stack([transform(Image.open(p).convert("RGB")) for p in paths[i:i + BATCH]])
            with torch.inference_mode():
                logits = model(x)
            F.append(feats["f"].float())
            P.append(torch.softmax(logits.float(), 1))
            print(f"\r  {name}: {min(i + BATCH, len(paths))}/{len(paths)}", end="", flush=True)
        print()
        return torch.cat(F).numpy(), torch.cat(P).numpy()

    Ftr, _ = embed([p for p, _ in train], "train")
    Fva, Pva = embed([p for p, _ in val], "val")
    Fny, _ = embed(neysan, "neysan")

    # ---- 2. 8-class result and needs_review threshold on validation
    yva = np.array([CLASSES.index(c) for _, c in val])
    pred, conf = Pva.argmax(1), Pva.max(1)
    wrong = pred != yva
    print(f"\n8-class validation correct: {int((~wrong).sum())}/{len(val)}  ({int(wrong.sum())} errors)")
    print(f"{'threshold':>9} {'flagged':>8} {'errors caught':>14} {'correct flagged':>16}")
    review = None
    for t in REVIEW_CANDIDATES:
        flagged = conf < t
        caught = int((flagged & wrong).sum())
        print(f"{t:9.2f} {int(flagged.sum()):8d} {caught:9d}/{int(wrong.sum()):<4d} {int((flagged & ~wrong).sum()):16d}")
        if review is None and caught >= wrong.sum() / 2:
            review = t
    print("needs_review threshold (lowest that flags >= half of the errors):", review)

    # ---- 3. split the trusted Neysan images by SHA256
    n1 = [i for i, p in enumerate(neysan) if "_unclean_neysan_" in p.name]
    trusted = [i for i in range(len(neysan)) if i not in set(n1)]
    shas = [sha256(p) for p in neysan]
    uniq = sorted({shas[i] for i in trusted})
    random.Random(SEED).shuffle(uniq)
    a, b = int(0.70 * len(uniq)), int(0.85 * len(uniq))
    part = {s: ("train" if k < a else "calib" if k < b else "holdout") for k, s in enumerate(uniq)}
    ny_tr = [i for i in trusted if part[shas[i]] == "train"]
    ny_ca = [i for i in trusted if part[shas[i]] == "calib"]
    ny_ho = [i for i in trusted if part[shas[i]] == "holdout"]
    train_shas = {shas[i] for i in ny_tr}
    n1_check = [i for i in n1 if shas[i] not in train_shas]

    # ---- 4. train
    X = np.concatenate([Ftr, Fny[ny_tr]])
    y = np.r_[np.zeros(len(Ftr)), np.ones(len(ny_tr))]
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(scaler.transform(X), y)

    def rejected(F):
        return clf.predict_proba(scaler.transform(F))[:, 1] >= THRESHOLD

    # ---- 5. measure with the fixed threshold
    rej_val = rejected(Fva)
    ho, ca, n1r = rejected(Fny[ny_ho]), rejected(Fny[ny_ca]), rejected(Fny[n1_check])
    ho_external = [k for k, i in enumerate(ny_ho) if neysan[i].name.startswith("external_")]
    correct_kept = int((~wrong & ~rej_val).sum())
    max_val_rejected = int(0.01 * len(val))
    passed = ho.mean() >= 0.80 and rej_val.sum() <= max_val_rejected

    print("\n===== RESULT =====")
    print(f"trusted Neysan train / calib / holdout : {len(ny_tr)} / {len(ny_ca)} / {len(ny_ho)}")
    print(f"holdout Neysan rejected   : {ho.mean():.3f} ({int(ho.sum())}/{len(ho)})   rule >= 0.80")
    print(f"  of which external images  : {int(ho[ho_external].sum())}/{len(ho_external)}")
    print(f"calibration rejected      : {ca.mean():.3f} ({int(ca.sum())}/{len(ca)})")
    print(f"N1 (not reviewed) rejected: {n1r.mean():.3f} (n={len(n1_check)})")
    print(f"validation rejected       : {int(rej_val.sum())}/{len(val)}   rule <= {max_val_rejected}")
    for i in np.where(rej_val)[0]:
        print(f"    rejected: {val[i][1]}/{val[i][0].name}")
    print(f"validation correct after rejection: {correct_kept}/{len(val)}")
    print("DECISION:", "PASS" if passed else "FAIL")

    det = {"type": "logistic_regression_on_swin_head_input", "threshold": THRESHOLD, "feature_dim": int(Ftr.shape[1]),
           "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(), "coef": clf.coef_[0].tolist(),
           "intercept": float(clf.intercept_[0]), "checkpoint_sha256": ckpt_sha,
           "needs_review_threshold": review,
           "trained_on": {"negatives": f"dataset v3 train {len(Ftr)}", "positives": f"trusted Neysan {len(ny_tr)} (seed 42, 70%)"},
           "results": {"holdout_rejected": float(ho.mean()), "calib_rejected": float(ca.mean()), "n1_rejected": float(n1r.mean()),
                       "val_rejected": int(rej_val.sum()), "val_correct_8class": int((~wrong).sum()),
                       "val_correct_after_rejection": correct_kept},
           "decision": "PASS" if passed else "FAIL"}
    text = json.dumps(det)
    OUTPUT.write_text(text)
    print(f"saved: {OUTPUT}  sha256 {hashlib.sha256(text.encode()).hexdigest()}")
    print("===== END =====")


if __name__ == "__main__":
    main()
