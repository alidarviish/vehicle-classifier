"""PyTorch Dataset for the train / val split in data/split_manifest.csv.

Each manifest row has a portable image_path such as
"v1/train/vanet/123.jpg". The first part ("v1") is the source name; its
local folder comes from configs/local_paths.json, which is not in git.

This file only READS the manifest, the config and the images. It does not
split, clean, relabel or augment anything. Transforms are passed in by
the caller.
"""

import csv
import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

# The 8 model classes, in alphabetical order. The class index is the
# position in this list, so the mapping is the same on every run.
CLASSES = sorted([
    "ambulance", "autobus", "kamyun", "kamyunet",
    "minibus", "savari", "taxi", "vanet",
])
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}

SPLITS = ("train", "val")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "data" / "split_manifest.csv"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "local_paths.json"

REQUIRED_COLUMNS = ["image_path", "label", "split"]


def load_source_roots(config_path):
    """Read {"sources": {"v1": "...", "v2": "..."}} and check each folder exists."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    roots = {}
    for source_name, folder in config["sources"].items():
        root = Path(folder)
        if not root.is_dir():
            raise FileNotFoundError(f"source '{source_name}' folder not found: {root}")
        roots[source_name] = root
    return roots


def check_row(row, line_number):
    """Stop with a clear error if a manifest row is not usable as-is."""
    where = f"split manifest line {line_number}"
    if row["split"] not in SPLITS:
        raise ValueError(f"{where}: unknown split '{row['split']}' (expected one of {SPLITS})")
    if row["label"] not in CLASS_TO_IDX:
        raise ValueError(f"{where}: label '{row['label']}' is not one of the 8 classes")
    # neysan is never a model class and never part of train/val
    if row.get("folder_label") == "neysan":
        raise ValueError(f"{where}: neysan image found in the split: {row['image_path']}")
    parts = row["image_path"].split("/")
    if len(parts) < 2 or "" in parts or ".." in parts or "\\" in row["image_path"] or ":" in row["image_path"]:
        raise ValueError(f"{where}: image_path must be relative like 'v1/train/label/file.jpg', got '{row['image_path']}'")


def read_split_rows(manifest_path, split):
    """Return the manifest rows of one split, after checking every row."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got '{split}'")
    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{manifest_path} is missing columns: {missing}")
        all_rows = list(reader)

    seen = set()
    for line_number, row in enumerate(all_rows, start=2):  # line 1 is the header
        check_row(row, line_number)
        if row["image_path"] in seen:
            raise ValueError(f"image_path listed twice in the manifest: {row['image_path']}")
        seen.add(row["image_path"])

    rows = [row for row in all_rows if row["split"] == split]
    if not rows:
        raise ValueError(f"no rows with split '{split}' in {manifest_path}")
    return rows


def resolve_image_path(image_path, source_roots):
    """Turn 'v1/train/vanet/123.jpg' into the local file path."""
    source_name, relative_path = image_path.split("/", 1)
    if source_name not in source_roots:
        raise KeyError(f"source '{source_name}' of {image_path} is not in the local config")
    return source_roots[source_name] / relative_path


class VehicleDataset(Dataset):
    """Images and integer labels for one split ("train" or "val")."""

    def __init__(self, split, transform=None,
                 manifest_path=DEFAULT_MANIFEST, config_path=DEFAULT_CONFIG):
        rows = read_split_rows(manifest_path, split)
        source_roots = load_source_roots(config_path)

        self.split = split
        self.transform = transform
        self.image_paths = [row["image_path"] for row in rows]  # portable paths, for reports
        self.files = [resolve_image_path(p, source_roots) for p in self.image_paths]
        self.labels = [CLASS_TO_IDX[row["label"]] for row in rows]

        # check every file up front, so a wrong config fails now and not mid-epoch
        missing = [p for p, f in zip(self.image_paths, self.files) if not f.is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} image files not found, first ones: {missing[:5]}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with Image.open(self.files[index]) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]
