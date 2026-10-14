"""Per-test reports with pictures, written by run_test.py into the output folder.

For every test group (for example test0, test2, ...) a folder is made with:
    confusion_matrix.png / .csv     that test only
    errors_1.png, errors_2.png ...  every wrong prediction (not unknown), lowest confidence first
    unknown_1.png ...               every image the Neysan detector rejected
    needs_review_1.png ...          every image flagged for human review (right or wrong)
    wrong.csv, needs_review.csv, unknown.csv
Under each picture: file name, true folder, model output and confidence (or Neysan score);
red = wrong, green = right. Every image is shown; PER_PAGE images per picture file.
"""
from pathlib import Path

from PIL import Image

PER_PAGE = 48
COLS = 6
UNKNOWN = "unknown"


def save_image_pages(rows, input_dir, prefix, score="confidence", title=""):
    """Draw all images of rows on pages of PER_PAGE and save them as <prefix>_1.png, <prefix>_2.png, ..."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for start in range(0, len(rows), PER_PAGE):
        page = rows[start:start + PER_PAGE]
        lines = (len(page) + COLS - 1) // COLS
        fig, axes = plt.subplots(lines, COLS, figsize=(16, 3.6 * lines), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, r in zip(axes.flat, page):
            with Image.open(Path(input_dir) / r["image_path"]) as img:
                im = img.convert("RGB")
            im.thumbnail((256, 256))
            ax.imshow(im)
            value = float(r[score]) if r[score] != "" else 0.0
            ok = r["predicted_class"] == r["true_label"]
            review = "  | review" if r["needs_review"] else ""
            ax.set_title(f"{Path(r['image_path']).name}\nfolder: {r['true_label']}\noutput: {r['predicted_class']}\n"
                         f"{score} {value:.2f}{review}", fontsize=9, color="green" if ok else "red")
        fig.suptitle(f"{title}  ({start + 1}-{start + len(page)} of {len(rows)})", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.4 / (3.6 * lines)))
        fig.savefig(f"{prefix}_{start // PER_PAGE + 1}.png", dpi=90)
        plt.close(fig)


def save_group_reports(labeled, groups, input_dir, output_dir, compute_metrics, save_confusion_matrix, write_csv, fields):
    """One folder per test group with its confusion matrix, CSV lists and pictures."""
    for group, rows in sorted(groups.items()):
        name = "all" if group == "." else group.replace("/", "_")
        folder = Path(output_dir) / name
        folder.mkdir(parents=True, exist_ok=True)
        save_confusion_matrix(compute_metrics(rows), folder)

        wrong = [r for r in rows if r["predicted_class"] != r["true_label"]]
        errors = sorted([r for r in wrong if r["predicted_class"] != UNKNOWN], key=lambda r: float(r["confidence"]))
        unknown = [r for r in rows if r["predicted_class"] == UNKNOWN]
        review = sorted([r for r in rows if r["needs_review"]], key=lambda r: float(r["confidence"]))
        write_csv(folder / "wrong.csv", wrong, fields)
        write_csv(folder / "needs_review.csv", review, fields)
        write_csv(folder / "unknown.csv", unknown, fields)

        save_image_pages(errors, input_dir, folder / "errors", "confidence", f"{name} - errors")
        save_image_pages(unknown, input_dir, folder / "unknown", "neysan_score", f"{name} - unknown (Neysan)")
        save_image_pages(review, input_dir, folder / "needs_review", "confidence", f"{name} - needs_review")
        print(f"  report {name}: {len(errors)} errors, {len(unknown)} unknown, {len(review)} needs_review", flush=True)
