"""Rebuild the dataset v3 lists from the folders as they are now.

    python scripts/refresh_split_v3.py

Run it after moving images by hand (a fixed label is just a file moved to another class folder).
The folders are the truth; this script only writes:
    data/dataset_v3/split_v3.csv     every image: path, sha256, label, split
    data/split_manifest_v3.csv       Train + Validation, in the format src/dataset.py reads
"""
import csv
import hashlib
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "data" / "dataset_v3"
CLASSES = ["ambulance", "autobus", "kamyun", "kamyunet", "minibus", "savari", "taxi", "vanet"]
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp"}


def origin(name):
    if name.startswith("mentor_"):
        return "mentor"
    if name.startswith("old_test_"):
        return "ours_old_test"
    return "ours"


def main():
    rows = []
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_TYPES:
            continue
        parts = p.relative_to(ROOT).parts
        if parts[0] in ("train", "val"):
            split, label = parts[0], parts[1]
            if label not in CLASSES:
                raise SystemExit(f"unknown class folder: {p}")
        elif parts[0] == "neysan":
            split, label = "neysan", "neysan"
        elif parts[0] == "test":
            split, label = "test_" + parts[1], parts[2]     # e.g. test_test0, folder = mentor label
        else:
            continue                                         # review/ and anything else
        rows.append({"path": "/".join(parts), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                     "label": label, "split": split, "origin": origin(p.name)})

    with open(ROOT / "split_v3.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["path", "sha256", "label", "split", "origin"])
        w.writeheader()
        w.writerows(rows)

    with open(REPO / "data" / "split_manifest_v3.csv", "w", newline="", encoding="utf-8") as f:   # no BOM: src/dataset.py reads plain utf-8
        w = csv.writer(f)
        w.writerow(["image_path", "sha256", "label", "split", "source", "origin_split", "folder_label", "dup_group"])
        for r in rows:
            if r["split"] in ("train", "val"):
                w.writerow(["v3/" + r["path"], r["sha256"], r["label"], r["split"], "v3", r["origin"], r["label"], ""])

    # checks: the test must not share an image with Train / Validation / Neysan
    train_side = {r["sha256"] for r in rows if not r["split"].startswith("test")}
    test_side = {r["sha256"] for r in rows if r["split"].startswith("test")}
    print("images shared between test and the rest:", len(train_side & test_side))
    print()
    print(f"{'split':12s}" + "".join(f"{c:>10s}" for c in CLASSES) + f"{'total':>8s}")
    for s in sorted({r["split"] for r in rows}):
        cnt = Counter(r["label"] for r in rows if r["split"] == s)
        print(f"{s:12s}" + "".join(f"{cnt[c]:10d}" for c in CLASSES) + f"{sum(cnt.values()):8d}")


if __name__ == "__main__":
    main()
