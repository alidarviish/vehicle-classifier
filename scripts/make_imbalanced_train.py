"""Build the simulated-imbalance training subset for the balanced-batches experiment.

Source: the "train" rows of data/split_manifest.csv (the current 2633 train images).
The imbalance is made only by removing images (docs/DATA_DECISIONS.md):
    ambulance, kamyun, minibus -> 15 images each
    the other 5 classes       -> all their train images
Selection per reduced class: rows sorted by image_path, then random.Random(42).sample(rows, 15)
(a new Random(42) for each class, so each class's choice does not depend on the others).

The train split already contains exact-duplicate pairs (same sha256); they are kept as they are.
The only sha256 repeats allowed in the output are those duplicate groups of the train split.

Only reads data/split_manifest.csv. Writes the subset (default decisions/imbalanced_train.csv),
refusing to overwrite an existing file.

Usage:
    python scripts/make_imbalanced_train.py
"""

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPLIT = REPO_ROOT / "data" / "split_manifest.csv"
DEFAULT_OUT = REPO_ROOT / "decisions" / "imbalanced_train.csv"

SEED = 42
CLASSES = ["ambulance", "autobus", "kamyun", "kamyunet", "minibus", "savari", "taxi", "vanet"]
REDUCED = {"ambulance": 15, "kamyun": 15, "minibus": 15}
OUT_COLUMNS = ["image_path", "sha256", "label", "source", "origin_split", "folder_label", "dup_group"]


def read_train_rows(split_path):
    with open(split_path, "r", encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "train"]
    bad = [r["image_path"] for r in rows if r["label"] not in CLASSES]
    if bad:
        raise ValueError(f"labels outside the 8 classes: {bad[:5]}")
    if len({r["image_path"] for r in rows}) != len(rows):
        raise ValueError("duplicate image_path in the train split")
    return rows


def build_subset(train_rows, seed=SEED):
    by_class = defaultdict(list)
    for r in train_rows:
        by_class[r["label"]].append(r)

    subset = []
    for label in CLASSES:
        rows = sorted(by_class[label], key=lambda r: r["image_path"])
        if label in REDUCED:
            if len(rows) < REDUCED[label]:
                raise ValueError(f"{label}: only {len(rows)} train images")
            rows = random.Random(seed).sample(rows, REDUCED[label])
        subset.extend(rows)
    return sorted(subset, key=lambda r: r["image_path"])


def check_subset(subset, train_rows):
    train_paths = {r["image_path"] for r in train_rows}
    train_sha_count = Counter(r["sha256"] for r in train_rows)
    full = Counter(r["label"] for r in train_rows)
    expected = {c: REDUCED.get(c, full[c]) for c in CLASSES}

    if any(r["image_path"] not in train_paths for r in subset):
        raise ValueError("subset contains an image that is not in the train split")
    if len({r["image_path"] for r in subset}) != len(subset):
        raise ValueError("duplicate image_path in the subset")
    repeated = [s for s, n in Counter(r["sha256"] for r in subset).items() if n > 1]
    if any(train_sha_count[s] < 2 for s in repeated):
        raise ValueError("sha256 repeated in the subset that is not a train duplicate group")
    counts = Counter(r["label"] for r in subset)
    if dict(counts) != {c: n for c, n in expected.items() if n}:
        raise ValueError(f"class counts {dict(counts)} != expected {expected}")
    if len(subset) != sum(expected.values()):
        raise ValueError("total does not match the expected class counts")
    return counts, repeated


def main():
    parser = argparse.ArgumentParser(description="Build the simulated-imbalance train subset.")
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    train_rows = read_train_rows(args.split)
    subset = build_subset(train_rows)
    counts, repeated = check_subset(subset, train_rows)

    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(subset)

    before = Counter(r["label"] for r in train_rows)
    for c in CLASSES:
        print(f"{c:10s} {before[c]:4d} -> {counts[c]:4d}")
    print(f"total {len(train_rows)} -> {len(subset)} | train duplicate pairs kept in subset: {len(repeated)}")


if __name__ == "__main__":
    main()
