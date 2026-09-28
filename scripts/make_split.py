"""Build the train/val split (data/split_manifest.csv format).

Steps (same logic as the split currently in use):
1. pool = images with status train_val_pool, minus the 34 approved Test replacements;
   label = folder_label (old relabels are not applied).
2. Images with the same sha256 form one unit, so exact duplicates always land in the same split.
3. Per class (fixed class order), units are sorted by sha256, shuffled with ONE random.Random(42)
   shared across classes, and assigned to val until the class has round(0.2 * class size) images;
   the rest go to train. Stratification uses the folder label, as when the split was made.
4. The human label corrections (RELABEL_* rows of decisions/kamyun_kamyunet_review.csv and
   decisions/kamyunet_kamyun_review.csv) are then applied to the label column. They only change
   labels, never the split.

Only reads files. Writes a CSV only when --out is given.

Usage:
    python scripts/make_split.py --overrides decisions/old_label_overrides.csv --out <file.csv>
"""

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_statuses import (DECISIONS_DIR, DEFAULT_MANIFEST, DEFAULT_OVERRIDES,  # noqa: E402
                                    load_statuses, read_csv, write_csv)
from scripts.build_test import APPROVED_REPLACEMENTS, CLASSES  # noqa: E402

SEED = 42
VAL_FRACTION = 0.2
LABEL_REVIEWS = ["kamyun_kamyunet_review.csv", "kamyunet_kamyun_review.csv"]
OUT_COLUMNS = ["image_path", "sha256", "label", "split", "source", "origin_split",
               "folder_label", "dup_group"]


def label_corrections(decisions_dir=DECISIONS_DIR):
    """image_path -> (original_label, final_label) for every RELABEL_* human decision."""
    corrections = {}
    for name in LABEL_REVIEWS:
        for row in read_csv(Path(decisions_dir) / name):
            if row["decision"].startswith("RELABEL_"):
                corrections[row["image_path"]] = (row["original_label"], row["final_label"])
    return corrections


def make_split(statuses, corrections, seed=SEED, val_fraction=VAL_FRACTION):
    replacements = set(APPROVED_REPLACEMENTS)
    pool = [r for r in statuses if r["status"] == "train_val_pool" and r["image_path"] not in replacements]

    units = defaultdict(list)                       # sha256 -> images with that content
    for r in pool:
        units[r["sha256"]].append(r)
    for members in units.values():
        if len({r["folder_label"] for r in members}) != 1:
            raise ValueError(f"exact duplicates with different labels: {[r['image_path'] for r in members]}")

    rng = random.Random(seed)
    split_of = {}
    for label in CLASSES:
        class_units = sorted(s for s, m in units.items() if m[0]["folder_label"] == label)
        rng.shuffle(class_units)
        target = round(sum(len(units[s]) for s in class_units) * val_fraction)
        n_val = 0
        for s in class_units:
            if n_val + len(units[s]) <= target:
                split_of[s] = "val"
                n_val += len(units[s])
            else:
                split_of[s] = "train"

    dup_group = {s: f"DUPG{i:02d}" for i, s in
                 enumerate(sorted(s for s, m in units.items() if len(m) > 1), 1)}

    rows = []
    for r in sorted(pool, key=lambda r: r["image_path"]):
        label = r["folder_label"]
        if r["image_path"] in corrections:
            original, final = corrections[r["image_path"]]
            if original != label:
                raise ValueError(f"correction does not match folder label: {r['image_path']}")
            label = final
        rows.append({"image_path": r["image_path"], "sha256": r["sha256"], "label": label,
                     "split": split_of[r["sha256"]], "source": r["source"],
                     "origin_split": r["origin_split"], "folder_label": r["folder_label"],
                     "dup_group": dup_group.get(r["sha256"], "")})

    missing = set(corrections) - {r["image_path"] for r in rows}
    if missing:
        raise ValueError(f"label corrections for images not in the split: {sorted(missing)}")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Build the train/val split.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--overrides", default=DEFAULT_OVERRIDES)
    parser.add_argument("--out", required=True, help="output CSV (must not exist)")
    args = parser.parse_args()

    rows = make_split(load_statuses(args.manifest, args.overrides), label_corrections())
    print(f"train {sum(r['split'] == 'train' for r in rows)} | val {sum(r['split'] == 'val' for r in rows)}")
    write_csv(rows, args.out, OUT_COLUMNS)


if __name__ == "__main__":
    main()
