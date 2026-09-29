"""Give every image in data/manifest.csv exactly one dataset status.

Statuses (see docs/DATA_DECISIONS.md):
    neysan_eval     CONFIRMED_NEYSAN in a Neysan review, or folder_label == neysan (policy N1)
    test_candidate  any other image from v1/test
    excluded        an old "drop" decision (train / unclean images only)
    train_val_pool  everything else

The order of the rules matters and is the same as the one used to build the current dataset:
Neysan first, then test, then drops. Old relabel decisions are read but NOT applied:
the label of a pool image is always its folder_label.

Only reads files. Writes a CSV only when --out is given.

Usage:
    python scripts/build_statuses.py --overrides decisions/old_label_overrides.csv --out <file.csv>
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MANIFEST = REPO_ROOT / "data" / "manifest.csv"
DECISIONS_DIR = REPO_ROOT / "decisions"
# In-repo copy of the old project's drop/relabel list (byte-identical, see docs/DATA_DECISIONS.md).
DEFAULT_OVERRIDES = DECISIONS_DIR / "old_label_overrides.csv"
NEYSAN_REVIEWS = ["neysan_review.csv", "neysan_test_review.csv"]

STATUSES = ("train_val_pool", "test_candidate", "neysan_eval", "excluded")
OUT_COLUMNS = ["image_path", "sha256", "source", "origin_split", "folder_label",
               "status", "label", "basis"]


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def neysan_decisions(decisions_dir=DECISIONS_DIR):
    """image_path -> decision (CONFIRMED_NEYSAN / NOT_NEYSAN) from both Neysan review files."""
    decisions = {}
    for name in NEYSAN_REVIEWS:
        for row in read_csv(Path(decisions_dir) / name):
            if row["image_path"] in decisions:
                raise ValueError(f"image reviewed twice: {row['image_path']}")
            decisions[row["image_path"]] = row["decision"]
    return decisions


def old_drops(overrides_path, manifest_rows):
    """image_path -> drop reason, from the old override list.

    The old file uses paths without the source prefix (e.g. "train/taxi/123.jpg"); each one must
    match exactly one manifest image. Relabel rows are checked but ignored (policy: not applied).
    """
    by_relative = {}
    for row in manifest_rows:
        by_relative.setdefault(row["image_path"].split("/", 1)[1], []).append(row["image_path"])
    drops = {}
    for row in read_csv(overrides_path):
        relative = row["path"].replace("\\", "/")
        matches = by_relative.get(relative, [])
        if len(matches) != 1:
            raise ValueError(f"override path matches {len(matches)} manifest images: {row['path']}")
        if row["action"] == "drop":
            drops[matches[0]] = row["reason"]
        elif row["action"] != "relabel":
            raise ValueError(f"unknown override action '{row['action']}' for {row['path']}")
    return drops


def build_statuses(manifest_rows, decisions, drops):
    """Return one dict per manifest image with its status, label and the rule that decided it."""
    result = []
    for row in manifest_rows:
        path = row["image_path"]
        decision = decisions.get(path, "")
        if decision == "CONFIRMED_NEYSAN":
            status, label, basis = "neysan_eval", "vanet", "CONFIRMED_NEYSAN"
        elif row["folder_label"] == "neysan":
            status, label, basis = "neysan_eval", "vanet", "neysan_folder_label (policy N1)"
        elif row["origin_split"] == "test":
            status, label, basis = "test_candidate", row["folder_label"], decision or "not in Neysan review"
        elif path in drops:
            status, label, basis = "excluded", "", "old drop: " + drops[path]
        else:
            status, label, basis = "train_val_pool", row["folder_label"], "keep"
        result.append({"image_path": path, "sha256": row["sha256"], "source": row["source"],
                       "origin_split": row["origin_split"], "folder_label": row["folder_label"],
                       "status": status, "label": label, "basis": basis})
    return result


def load_statuses(manifest_path=DEFAULT_MANIFEST, overrides_path=DEFAULT_OVERRIDES,
                  decisions_dir=DECISIONS_DIR):
    manifest_rows = read_csv(manifest_path)
    return build_statuses(manifest_rows, neysan_decisions(decisions_dir),
                          old_drops(overrides_path, manifest_rows))


def write_csv(rows, out_path, columns):
    out_path = Path(out_path)
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite {out_path}")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Build dataset statuses from the manifest and decisions.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--overrides", default=DEFAULT_OVERRIDES, help="in-repo old drop/relabel list")
    parser.add_argument("--out", required=True, help="output CSV (must not exist)")
    args = parser.parse_args()

    rows = load_statuses(args.manifest, args.overrides)
    print(dict(Counter(r["status"] for r in rows)))
    write_csv(rows, args.out, OUT_COLUMNS)


if __name__ == "__main__":
    main()
