"""Build the frozen Test set: 366 test candidates + 34 approved vanet replacements = 400.

- Test candidates: every image with status test_candidate (v1/test minus the 34 images confirmed
  as Neysan in decisions/neysan_test_review.csv). Label = folder_label.
- Replacements: v1/test had only 16 vanet images left, so 34 vanet images were moved from the
  train/val pool to reach 50 per class. The 34 approved images are listed below; this list is
  the decision and is used as-is.

How the 34 were chosen (recorded for audit; the list below is what counts):
    candidates = pool images with label vanet, reviewed NOT_NEYSAN, no exact duplicate anywhere,
                 no old near-duplicate evidence, excluding A017 (v1/train/vanet/214726285.jpg)
    groups     = v1/train (13), v1/unclean (16), v2/train (57), v2/unclean (75), each sorted by image_path
    picks      = 9, 9, 8, 8
    index j    = (2*j*(N-1) + (k-1)) // (2*(k-1)),  j = 0..k-1  (evenly spaced, first and last included)
At that time the candidate filter used the old relabels (v2/unclean/vanet/209939482.jpg was relabelled
vanet -> savari and therefore not a candidate), and the old near-duplicate list of an earlier dataset review.
Recomputing the formula today without those inputs can give a different list; do not.

Only reads files. Writes a CSV only when --out is given.

Usage:
    python scripts/build_test.py --overrides decisions/old_label_overrides.csv --out <file.csv>
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_statuses import DEFAULT_MANIFEST, DEFAULT_OVERRIDES, load_statuses, write_csv  # noqa: E402

CLASSES = ["ambulance", "autobus", "kamyun", "kamyunet", "minibus", "savari", "taxi", "vanet"]
PER_CLASS = 50
A017 = "v1/train/vanet/214726285.jpg"

APPROVED_REPLACEMENTS = [
    "v1/train/vanet/194285934.jpg", "v1/train/vanet/196322593.jpg", "v1/train/vanet/197838737.jpg",
    "v1/train/vanet/198628181.jpg", "v1/train/vanet/200024004.jpg", "v1/train/vanet/213788075.jpg",
    "v1/train/vanet/215738096.jpg", "v1/train/vanet/218206335.jpg", "v1/train/vanet/218514544.jpg",
    "v1/unclean/vanet/197011245.jpg", "v1/unclean/vanet/198363928.jpg", "v1/unclean/vanet/205492654.jpg",
    "v1/unclean/vanet/207074343.jpg", "v1/unclean/vanet/207849483.jpg", "v1/unclean/vanet/209913836.jpg",
    "v1/unclean/vanet/214020375.jpg", "v1/unclean/vanet/215695188.jpg", "v1/unclean/vanet/219083209.jpg",
    "v2/train/vanet/194942215.jpg", "v2/train/vanet/198100361.jpg", "v2/train/vanet/202121208.jpg",
    "v2/train/vanet/207410682.jpg", "v2/train/vanet/212755321.jpg", "v2/train/vanet/216566708.jpg",
    "v2/train/vanet/217881286.jpg", "v2/train/vanet/219127494.jpg",
    "v2/unclean/vanet/193796538.jpg", "v2/unclean/vanet/198165650.jpg", "v2/unclean/vanet/202758553.jpg",
    "v2/unclean/vanet/207534276.jpg", "v2/unclean/vanet/209060648.jpg", "v2/unclean/vanet/216463352.jpg",
    "v2/unclean/vanet/218083641.jpg", "v2/unclean/vanet/219088134.jpg",
]

OUT_COLUMNS = ["image_path", "sha256", "label", "basis"]


def evenly_spaced_indices(n, k):
    """The selection formula used for the replacements (documentation / audit only)."""
    return [(2 * j * (n - 1) + (k - 1)) // (2 * (k - 1)) for j in range(k)]


def build_test(statuses):
    """Return the 400 frozen Test rows; fails loudly if any check does not hold."""
    by_path = {r["image_path"]: r for r in statuses}
    sha_count = Counter(r["sha256"] for r in statuses)

    rows = [{"image_path": r["image_path"], "sha256": r["sha256"], "label": r["label"],
             "basis": "test_candidate"} for r in statuses if r["status"] == "test_candidate"]

    if len(set(APPROVED_REPLACEMENTS)) != 34 or A017 in APPROVED_REPLACEMENTS:
        raise ValueError("approved replacement list must hold 34 distinct images, without A017")
    for path in APPROVED_REPLACEMENTS:
        r = by_path.get(path)
        if r is None or r["status"] != "train_val_pool" or r["folder_label"] != "vanet":
            raise ValueError(f"replacement is not a vanet image of the train/val pool: {path}")
        if r["basis"] != "keep" or sha_count[r["sha256"]] != 1:
            raise ValueError(f"replacement has an old decision or an exact duplicate: {path}")
        rows.append({"image_path": path, "sha256": r["sha256"], "label": "vanet",
                     "basis": "approved_replacement"})

    counts = Counter(r["label"] for r in rows)
    if len(rows) != PER_CLASS * len(CLASSES) or any(counts[c] != PER_CLASS for c in CLASSES):
        raise ValueError(f"Test must have {PER_CLASS} images per class, got {dict(counts)}")
    return sorted(rows, key=lambda r: r["image_path"])


def main():
    parser = argparse.ArgumentParser(description="Build the frozen 400-image Test list.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--overrides", default=DEFAULT_OVERRIDES)
    parser.add_argument("--out", required=True, help="output CSV (must not exist)")
    args = parser.parse_args()

    rows = build_test(load_statuses(args.manifest, args.overrides))
    print(f"test images: {len(rows)}")
    write_csv(rows, args.out, OUT_COLUMNS)


if __name__ == "__main__":
    main()
