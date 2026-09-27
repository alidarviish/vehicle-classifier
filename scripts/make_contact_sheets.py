"""Make contact sheets of the unresolved Neysan review images, for human review.

The raw images are only read. The sheets are written OUTSIDE the repository,
so no image ever ends up in Git. Nothing is drawn on the images themselves;
only the review_id and image_path are written under each thumbnail.

Usage:
    python scripts/make_contact_sheets.py --out ../review_package/neysan_review
    python scripts/make_contact_sheets.py --queue decisions/neysan_review_queue.csv \
        --decisions decisions/neysan_review.csv --config configs/local_paths.json \
        --out ../review_package/neysan_review
"""

import argparse
import csv
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent

PER_SHEET = 30        # at most 30 images per sheet
COLUMNS = 5           # 5 columns x 6 rows
THUMB = 360           # thumbnail box in pixels (the image keeps its aspect ratio)
CAPTION = 44          # space under each thumbnail for the two text lines
MARGIN = 12


def read_csv(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_source_roots(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return {name: Path(folder) for name, folder in config["sources"].items()}


def unresolved_items(queue_rows, decision_rows):
    """Queue rows (in queue order) whose decision is still empty."""
    decided = {row["review_id"] for row in decision_rows if row["decision"].strip()}
    return [row for row in queue_rows if row["review_id"] not in decided]


def raw_file_path(image_path, roots):
    """'v1/train/vanet/123.jpg' -> <v1 root>/train/vanet/123.jpg"""
    source, rest = image_path.split("/", 1)
    return roots[source] / rest


def check_output_folder(out_dir, roots):
    """Refuse to write inside the repository or inside a raw dataset folder."""
    out = out_dir.resolve()
    if out == REPO_ROOT or REPO_ROOT in out.parents:
        raise ValueError(f"output folder is inside the repository: {out_dir}")
    for root in roots.values():
        root = root.resolve()
        if out == root or root in out.parents:
            raise ValueError(f"output folder is inside a raw dataset folder: {out_dir}")


def make_sheet(items, roots, sheet_title, font):
    rows = math.ceil(len(items) / COLUMNS)
    cell_w = THUMB + MARGIN
    cell_h = THUMB + CAPTION + MARGIN
    sheet = Image.new("RGB", (COLUMNS * cell_w + MARGIN, 40 + rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((MARGIN, 10), sheet_title, fill="black", font=font)

    for i, item in enumerate(items):
        x = MARGIN + (i % COLUMNS) * cell_w
        y = 40 + (i // COLUMNS) * cell_h
        with Image.open(raw_file_path(item["image_path"], roots)) as img:
            thumb = img.convert("RGB")      # a copy in memory; the raw file is not changed
        # Scale to fill the thumbnail box (enlarges small images too), keep aspect ratio.
        scale = min(THUMB / thumb.width, THUMB / thumb.height)
        thumb = thumb.resize((round(thumb.width * scale), round(thumb.height * scale)), Image.LANCZOS)
        sheet.paste(thumb, (x + (THUMB - thumb.width) // 2, y + (THUMB - thumb.height) // 2))
        draw.rectangle([x - 1, y - 1, x + THUMB, y + THUMB], outline="#bbbbbb")
        draw.text((x, y + THUMB + 4), item["review_id"], fill="black", font=font)
        draw.text((x, y + THUMB + 22), item["image_path"], fill="black", font=font)
    return sheet


def main():
    parser = argparse.ArgumentParser(description="Contact sheets for the Neysan review (written outside the repo).")
    parser.add_argument("--queue", default="decisions/neysan_review_queue.csv")
    parser.add_argument("--decisions", default="decisions/neysan_review.csv")
    parser.add_argument("--config", default="configs/local_paths.json")
    parser.add_argument("--out", required=True, help="output folder, must be outside the repository")
    args = parser.parse_args()

    roots = load_source_roots(args.config)
    out_dir = Path(args.out)
    check_output_folder(out_dir, roots)

    items = unresolved_items(read_csv(args.queue), read_csv(args.decisions))
    print(f"unresolved images: {len(items)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=15)
    n_sheets = math.ceil(len(items) / PER_SHEET)
    map_rows = []

    for s in range(n_sheets):
        chunk = items[s * PER_SHEET:(s + 1) * PER_SHEET]
        sheet_name = f"neysan_review_{s + 1:02d}.jpg"
        title = f"Neysan review - sheet {s + 1}/{n_sheets} - {len(chunk)} images"
        make_sheet(chunk, roots, title, font).save(out_dir / sheet_name, quality=92)
        for position, item in enumerate(chunk, start=1):
            map_rows.append({"review_id": item["review_id"], "sheet_name": sheet_name, "position": position})
        print(f"  {sheet_name}: {len(chunk)} images")

    with open(out_dir / "contact_sheet_map.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["review_id", "sheet_name", "position"])
        writer.writeheader()
        writer.writerows(map_rows)
    print(f"wrote {n_sheets} sheets and contact_sheet_map.csv to {out_dir}")


if __name__ == "__main__":
    main()
