"""Figure of labelled dataset images for the data audit (at least 12 images, all 8 classes).

Selection (deterministic, no randomness), read only from data/manifest.csv:
- rows with origin_split == "train" and folder_label in the 8 model classes; Test and unclean
  images are not used, and the `neysan` folder is not a model class;
- per class, the first PER_CLASS rows in manifest order (8 x 2 = 16 images).
The label shown under each image is the folder label from the manifest. Raw images are only read,
never copied, moved or changed; only class labels are drawn, no file paths.

The figure is written outside the repository, to the path given with --out.

Usage (from the repository root):
    python scripts/make_labelled_sample_figure.py --out ../review_package/labelled_samples.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write the file, no window
import matplotlib.pyplot as plt
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.make_contact_sheets import (check_output_folder, load_source_roots,  # noqa: E402
                                         raw_file_path, read_csv)

MANIFEST = REPO_ROOT / "data" / "manifest.csv"
CONFIG = REPO_ROOT / "configs" / "local_paths.json"
CLASSES = ["ambulance", "autobus", "kamyun", "kamyunet", "minibus", "savari", "taxi", "vanet"]
ORIGIN = "train"
PER_CLASS = 2          # 8 classes x 2 = 16 images (at least 12, every class covered)
COLUMNS = 4
TITLE = "Dataset labelled samples"


def select_rows(manifest_rows, classes=CLASSES, per_class=PER_CLASS, origin=ORIGIN):
    """First `per_class` rows of each class in manifest order; stops if a class has too few."""
    chosen = {c: [] for c in classes}
    for row in manifest_rows:
        label = row["folder_label"]
        if row["origin_split"] == origin and label in chosen and len(chosen[label]) < per_class:
            chosen[label].append(row)
    short = {c: len(rows) for c, rows in chosen.items() if len(rows) < per_class}
    if short:
        raise ValueError(f"not enough '{origin}' images for classes: {short}")
    selected = [row for c in classes for row in chosen[c]]
    if len(selected) < 12:
        raise ValueError(f"only {len(selected)} images selected; at least 12 are required")
    return selected


def load_images(rows, roots):
    """Open every selected image; stops with a clear error instead of skipping a bad file."""
    images = []
    for row in rows:
        path = raw_file_path(row["image_path"], roots)
        try:
            with Image.open(path) as img:
                images.append(img.convert("RGB"))
        except Exception as e:  # noqa: BLE001 - report which image failed, then stop
            raise RuntimeError(f"cannot read {row['image_path']}: {e}") from e
    return images


def make_figure(rows, images, out_path):
    n_rows = -(-len(rows) // COLUMNS)
    fig, axes = plt.subplots(n_rows, COLUMNS, figsize=(3 * COLUMNS, 3.2 * n_rows))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, row, img in zip(axes.ravel(), rows, images):
        ax.imshow(img)
        ax.set_title(row["folder_label"], fontsize=11)
    fig.suptitle(TITLE, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Labelled sample figure for the data audit "
                                                 "(written outside the repository).")
    parser.add_argument("--out", required=True, help="output image file outside the repository, "
                                                     "e.g. ../review_package/labelled_samples.png")
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--config", default=str(CONFIG), help="local raw-data paths (JSON)")
    args = parser.parse_args()

    out = Path(args.out)
    if out.suffix.lower() not in (".png", ".jpg", ".jpeg", ".pdf"):
        parser.error("--out must end in .png, .jpg, .jpeg or .pdf")
    if out.exists():
        parser.error(f"refusing to overwrite {out}")
    roots = load_source_roots(args.config)
    check_output_folder(out.parent, roots)   # never inside the repository or a raw dataset folder
    if not out.parent.is_dir():
        parser.error(f"output folder does not exist: {out.parent}")

    rows = select_rows(read_csv(args.manifest))
    images = load_images(rows, roots)
    make_figure(rows, images, out)
    print(f"{len(rows)} images, {len({r['folder_label'] for r in rows})} classes -> {out}")
    for row in rows:
        print(f"  {row['folder_label']:9s} {row['image_path']}")


if __name__ == "__main__":
    main()
