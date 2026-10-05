"""Build a manifest (one CSV row per image) from the raw datasets.

The script only READS the raw datasets. It never copies, moves, renames,
deletes or changes any file there.

Usage:
    python scripts/build_manifest.py
    python scripts/build_manifest.py --config configs/local_paths.json --out data/manifest.csv
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

COLUMNS = [
    "image_path",      # portable path: <source>/<split>/<label>/<file name>
    "source",          # dataset version name from the config, e.g. v1 or v2
    "origin_split",    # folder directly under the source root: train / test / unclean
    "folder_label",    # class folder name, exactly as on disk (not a cleaned label)
    "file_name",
    "file_size_bytes",
    "sha256",          # hash of the raw file bytes
    "readable",        # True if Pillow could fully decode the image
    "error",           # error message when readable is False
]


def load_source_roots(config_path):
    """Read {"sources": {"v1": "...", "v2": "..."}} from the local config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    roots = {}
    for source_name, folder in config["sources"].items():
        root = Path(folder)
        if not root.is_dir():
            raise FileNotFoundError(f"source '{source_name}' folder not found: {root}")
        roots[source_name] = root
    return roots


def file_sha256(path):
    """Hash the file in 1 MB chunks so large files do not fill memory."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:  # "rb" = read-only, binary
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


def check_readable(path):
    """Return (True, "") if the image decodes, otherwise (False, error text)."""
    try:
        with Image.open(path) as img:
            img.load()  # full decode, not only the header
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def scan_source(source_name, root):
    """Walk <root>/<split>/<label>/<file> and return one row per image file.

    Also returns the files that were skipped (not an image, or in an
    unexpected place) so they can be reported instead of silently ignored.
    """
    rows = []
    skipped = []
    for path in sorted(root.rglob("*")):  # sorted = same order on every run
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        parts = relative.parts  # e.g. ("train", "vanet", "123.jpg")

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            skipped.append(f"{source_name}/{relative.as_posix()} (not an image extension)")
            continue
        if len(parts) != 3:
            skipped.append(f"{source_name}/{relative.as_posix()} (not <split>/<label>/<file>)")
            continue

        split, label, file_name = parts
        readable, error = check_readable(path)
        rows.append({
            "image_path": f"{source_name}/{relative.as_posix()}",
            "source": source_name,
            "origin_split": split,
            "folder_label": label,
            "file_name": file_name,
            "file_size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            "readable": readable,
            "error": error,
        })
    return rows, skipped


def main():
    parser = argparse.ArgumentParser(description="Build a read-only manifest of the raw datasets.")
    parser.add_argument("--config", default="configs/local_paths.json",
                        help="local JSON file with the dataset folders (not committed to Git)")
    parser.add_argument("--out", default="data/manifest.csv", help="where to write the manifest CSV")
    args = parser.parse_args()

    roots = load_source_roots(args.config)
    out_path = Path(args.out)
    if out_path.exists():   # checked before the (slow) scan; nothing is read or written
        raise FileExistsError(f"refusing to overwrite {out_path}")

    # Safety check: never write the manifest inside a raw dataset folder.
    for root in roots.values():
        if root.resolve() in out_path.resolve().parents:
            raise ValueError(f"output {out_path} is inside a raw dataset folder: {root}")

    all_rows = []
    all_skipped = []
    for source_name, root in roots.items():
        print(f"scanning {source_name} ...")
        rows, skipped = scan_source(source_name, root)
        all_rows.extend(rows)
        all_skipped.extend(skipped)
        print(f"  {len(rows)} images, {len(skipped)} skipped files")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "x", newline="", encoding="utf-8") as f:   # "x": never overwrite
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    unreadable = [r for r in all_rows if not r["readable"]]
    print(f"\nwrote {len(all_rows)} rows to {out_path}")
    print(f"unreadable images: {len(unreadable)}")
    for r in unreadable:
        print(f"  {r['image_path']}: {r['error']}")
    print(f"skipped files: {len(all_skipped)}")
    for s in all_skipped:
        print(f"  {s}")


if __name__ == "__main__":
    main()
