"""Contact sheets of selected baseline mistakes on the VALIDATION split.

Uses checkpoints/baseline_best.pt and the "val" rows of data/split_manifest.csv.
No training, and the test set and Neysan images are never loaded.
Raw images are only read; the sheets are written OUTSIDE the repository.

Run from the repository root:
    python -m scripts.review_baseline_errors --out "../11/New folder (2)/review_package/baseline_errors"
"""

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from scripts.make_contact_sheets import check_output_folder
from src.dataset import DEFAULT_CONFIG, load_source_roots, resolve_image_path

# (group name, true label, predicted label) - one set of sheets per direction
GROUPS = [
    ("kamyun_to_kamyunet", "kamyun", "kamyunet"),
    ("kamyunet_to_kamyun", "kamyunet", "kamyun"),
    ("vanet_to_savari", "vanet", "savari"),
    ("ambulance_to_vanet", "ambulance", "vanet"),
]

PER_SHEET = 20        # all errors are kept; a large group is split over several sheets
COLUMNS = 5
THUMB = 360           # thumbnail box in pixels (aspect ratio kept)
CAPTION = 44
MARGIN = 12


def predict_val(config_path):
    """Return one record per val image: image_path, true, pred, confidence (softmax of the predicted class)."""
    import torch
    from torch.utils.data import DataLoader

    from src.dataset import CLASS_TO_IDX, CLASSES, DEFAULT_MANIFEST, VehicleDataset
    from src.model import BaselineCNN
    from src.train import BASE_TRANSFORM, BATCH_SIZE

    checkpoint = torch.load(Path("checkpoints") / "baseline_best.pt", map_location="cpu")
    if checkpoint["class_to_idx"] != CLASS_TO_IDX:
        raise ValueError(f"class mapping in checkpoint differs: {checkpoint['class_to_idx']}")
    model = BaselineCNN(num_classes=len(CLASSES), image_size=checkpoint["image_size"],
                        dropout=checkpoint["dropout"],
                        pooling=checkpoint.get("pooling", "max"))  # older checkpoints: max
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    val_set = VehicleDataset("val", BASE_TRANSFORM, DEFAULT_MANIFEST, config_path)
    loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    records, i = [], 0
    with torch.no_grad():
        for images, labels in loader:
            probs = torch.softmax(model(images), dim=1)
            conf, pred = probs.max(dim=1)
            for t, p, c in zip(labels.tolist(), pred.tolist(), conf.tolist()):
                records.append({"image_path": val_set.image_paths[i], "true": CLASSES[t],
                                "pred": CLASSES[p], "confidence": c})
                i += 1
    return records


def select_group(records, true_label, pred_label):
    """All val images of one error type, sorted by image_path (deterministic)."""
    chosen = [r for r in records if r["true"] == true_label and r["pred"] == pred_label]
    return sorted(chosen, key=lambda r: r["image_path"])


def make_sheet(items, roots, title, font):
    rows = math.ceil(len(items) / COLUMNS)
    cell_w, cell_h = THUMB + MARGIN, THUMB + CAPTION + MARGIN
    sheet = Image.new("RGB", (COLUMNS * cell_w + MARGIN, 40 + rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((MARGIN, 10), title, fill="black", font=font)
    for n, item in enumerate(items):
        x = MARGIN + (n % COLUMNS) * cell_w
        y = 40 + (n // COLUMNS) * cell_h
        with Image.open(resolve_image_path(item["image_path"], roots)) as img:
            thumb = img.convert("RGB")      # in-memory copy; the raw file is not changed
        scale = min(THUMB / thumb.width, THUMB / thumb.height)
        thumb = thumb.resize((round(thumb.width * scale), round(thumb.height * scale)), Image.LANCZOS)
        sheet.paste(thumb, (x + (THUMB - thumb.width) // 2, y + (THUMB - thumb.height) // 2))
        draw.rectangle([x - 1, y - 1, x + THUMB, y + THUMB], outline="#bbbbbb")
        draw.text((x, y + THUMB + 4),
                  f"{n + 1}. true {item['true']} -> pred {item['pred']} ({item['confidence']:.2f})",
                  fill="black", font=font)
        draw.text((x, y + THUMB + 22), item["image_path"], fill="black", font=font)
    return sheet


def write_group_sheets(name, items, roots, out_dir, font):
    """Write every item of one group over as many sheets as needed; return the sheet paths."""
    n_sheets = math.ceil(len(items) / PER_SHEET)
    paths = []
    for s in range(n_sheets):
        chunk = items[s * PER_SHEET:(s + 1) * PER_SHEET]
        path = out_dir / f"{name}_{s + 1:02d}.jpg"
        title = f"baseline val errors: {name} - sheet {s + 1}/{n_sheets} - {len(chunk)} of {len(items)} images"
        make_sheet(chunk, roots, title, font).save(path, quality=92)
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description="Contact sheets of baseline validation errors (outside the repo).")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--out", required=True, help="output folder, must be outside the repository")
    args = parser.parse_args()

    roots = load_source_roots(args.config)
    out_dir = Path(args.out)
    check_output_folder(out_dir, roots)

    records = predict_val(args.config)
    out_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=15)

    summary = [f"baseline_best.pt, validation split ({len(records)} images); test and Neysan not used", ""]
    for name, true_label, pred_label in GROUPS:
        items = select_group(records, true_label, pred_label)
        paths = write_group_sheets(name, items, roots, out_dir, font)
        summary.append(f"{true_label} -> {pred_label}: {len(items)} images")
        summary += [f"  {p.resolve()}" for p in paths]
    (out_dir / "baseline_errors_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
