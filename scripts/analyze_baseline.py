"""Analyse a finished training run (default: baseline) on the VALIDATION split only.

No training happens here. The test set and Neysan images are never loaded:
the only images read are the "val" rows of data/split_manifest.csv.

Run from the repository root:
    python scripts/analyze_baseline.py                              # baseline
    python scripts/analyze_baseline.py --run-name aug_crop_jitter   # any other run
    python scripts/analyze_baseline.py --history-only               # curves only, no torch needed

Reads reports/<run_name>_history.csv and checkpoints/<run_name>_best.pt.
Evaluation always uses BASE_TRANSFORM (no augmentation) and the current labels
in data/split_manifest.csv.

Outputs (all generated, ignored by git) in reports/analysis/ for the baseline,
or reports/analysis/<run_name>/ for any other run (so baseline files are not overwritten):
    loss_curves.png, val_f1_curve.png          from the history CSV
    confusion_matrix.csv / .png                val set, best checkpoint
    per_class_metrics.csv                      precision / recall / F1 / support
    baseline_analysis.txt                      short text summary
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write PNG files, no window
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))  # so "from src..." works with "python scripts/analyze_baseline.py"

DEFAULT_RUN = "baseline"


def run_paths(run_name):
    """History, checkpoint and output folder of one run."""
    history = REPO_ROOT / "reports" / f"{run_name}_history.csv"
    checkpoint = REPO_ROOT / "checkpoints" / f"{run_name}_best.pt"
    out_dir = REPO_ROOT / "reports" / "analysis"
    if run_name != DEFAULT_RUN:
        out_dir = out_dir / run_name
    return history, checkpoint, out_dir


# ---------- part 1: training history ----------

def read_history(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [{k: (int(v) if k == "epoch" else float(v)) for k, v in row.items()} for row in rows]


def best_epoch_row(history):
    """First epoch with the highest val_f1 - same rule as the checkpoint (strictly better)."""
    best = history[0]
    for row in history[1:]:
        if row["val_f1"] > best["val_f1"]:
            best = row
    return best


def plot_loss_curves(history, out_path, run_name):
    epochs = [r["epoch"] for r in history]
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, [r["train_loss"] for r in history], marker="o", label="train loss")
    plt.plot(epochs, [r["val_loss"] for r in history], marker="o", label="validation loss")
    plt.xlabel("epoch")
    plt.ylabel("cross-entropy loss")
    plt.title(f"{run_name}: train vs validation loss")
    plt.xticks(epochs)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_val_f1(history, best, out_path, run_name):
    epochs = [r["epoch"] for r in history]
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, [r["val_f1"] for r in history], marker="o", label="validation macro F1")
    plt.axvline(best["epoch"], color="gray", linestyle="--", linewidth=1)
    plt.annotate(f"best epoch {best['epoch']}\nF1 = {best['val_f1']:.4f}",
                 xy=(best["epoch"], best["val_f1"]), xytext=(-110, -45),
                 textcoords="offset points", arrowprops={"arrowstyle": "->"})
    plt.xlabel("epoch")
    plt.ylabel("macro F1")
    plt.title(f"{run_name}: validation macro F1")
    plt.xticks(epochs)
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def history_observations(history, best):
    """Plain facts read off the history table (no interpretation of causes)."""
    last = history[-1]
    min_val = min(history, key=lambda r: r["val_loss"])
    first_perfect = next((r["epoch"] for r in history if r["train_accuracy"] == 1.0), None)
    lines = [
        f"- lowest validation loss: {min_val['val_loss']:.4f} at epoch {min_val['epoch']}",
        f"- validation loss at best-F1 epoch {best['epoch']}: {best['val_loss']:.4f}; "
        f"at last epoch {last['epoch']}: {last['val_loss']:.4f}",
        f"- train loss: {history[0]['train_loss']:.4f} at epoch 1 -> {last['train_loss']:.6f} at epoch {last['epoch']}",
        f"- train accuracy first reaches 1.0 at epoch {first_perfect}" if first_perfect
        else "- train accuracy never reaches 1.0",
        f"- at the last epoch: train F1 {last['train_f1']:.4f} vs val F1 {last['val_f1']:.4f} "
        f"(gap {last['train_f1'] - last['val_f1']:.4f}); "
        f"val loss / train loss gap {last['val_loss'] - last['train_loss']:.4f}",
        f"- after epoch {min_val['epoch']} validation loss does not return to its minimum while train "
        f"loss keeps falling; validation macro F1 stays between "
        f"{min(r['val_f1'] for r in history if r['epoch'] > min_val['epoch']):.4f} and "
        f"{max(r['val_f1'] for r in history if r['epoch'] > min_val['epoch']):.4f}. "
        f"This train/validation divergence is a typical sign of overfitting.",
    ]
    return lines


# ---------- part 2: best checkpoint on the validation set ----------

def evaluate_checkpoint(checkpoint_path, config_path):
    """Predict the val split with the best checkpoint. Returns (true labels, predictions, classes)."""
    import torch
    from torch.utils.data import DataLoader

    from src.dataset import CLASS_TO_IDX, CLASSES, DEFAULT_MANIFEST, VehicleDataset
    from src.model import BaselineCNN
    from src.train import BASE_TRANSFORM, BATCH_SIZE

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint["class_to_idx"] != CLASS_TO_IDX:
        raise ValueError(f"class mapping in checkpoint differs: {checkpoint['class_to_idx']}")
    if checkpoint["architecture"] != "BaselineCNN":
        raise ValueError(f"unexpected architecture: {checkpoint['architecture']}")

    model = BaselineCNN(num_classes=len(CLASSES), image_size=checkpoint["image_size"],
                        dropout=checkpoint["dropout"],
                        pooling=checkpoint.get("pooling", "max"))  # older checkpoints: max
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    val_set = VehicleDataset("val", BASE_TRANSFORM, DEFAULT_MANIFEST, config_path)  # no augmentation
    loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            y_pred.extend(model(images).argmax(dim=1).tolist())
            y_true.extend(labels.tolist())
    return y_true, y_pred, CLASSES, checkpoint


def save_confusion_matrix(matrix, classes, csv_path, png_path, run_name):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true \\ predicted"] + classes)
        for name, row in zip(classes, matrix):
            writer.writerow([name] + list(row))

    plt.figure(figsize=(7.5, 6.5))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar(label="images")
    ticks = range(len(classes))
    plt.xticks(ticks, classes, rotation=45, ha="right")
    plt.yticks(ticks, classes)
    limit = matrix.max() / 2
    for i in ticks:
        for j in ticks:
            plt.text(j, i, int(matrix[i][j]), ha="center", va="center",
                     color="white" if matrix[i][j] > limit else "black")
    plt.xlabel("predicted class")
    plt.ylabel("true class")
    plt.title(f"{run_name}: validation confusion matrix (counts)")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyse a training run (validation only).")
    parser.add_argument("--run-name", default=DEFAULT_RUN, help="reads reports/<run_name>_history.csv and checkpoints/<run_name>_best.pt")
    parser.add_argument("--config", default=None, help="local raw-data paths (JSON), default configs/local_paths.json")
    parser.add_argument("--history-only", action="store_true", help="only the curves from the history CSV")
    args = parser.parse_args()

    run_name = args.run_name
    history_path, checkpoint_path, OUT_DIR = run_paths(run_name)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    history = read_history(history_path)
    best = best_epoch_row(history)
    plot_loss_curves(history, OUT_DIR / "loss_curves.png", run_name)
    plot_val_f1(history, best, OUT_DIR / "val_f1_curve.png", run_name)

    report = [
        f"Run '{run_name}' analysis (validation split only; test and Neysan not used)",
        "",
        f"best epoch: {best['epoch']}",
        f"best validation macro F1: {best['val_f1']:.4f}",
        f"validation accuracy at best epoch: {best['val_accuracy']:.4f}",
        "",
        "Train / validation history:",
        *history_observations(history, best),
    ]

    if not args.history_only:
        from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
        from src.dataset import DEFAULT_CONFIG

        y_true, y_pred, classes, checkpoint = evaluate_checkpoint(checkpoint_path, args.config or DEFAULT_CONFIG)
        labels = list(range(len(classes)))
        matrix = confusion_matrix(y_true, y_pred, labels=labels)
        save_confusion_matrix(matrix, classes, OUT_DIR / "confusion_matrix.csv",
                              OUT_DIR / "confusion_matrix.png", run_name)

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=None, zero_division=0)
        with open(OUT_DIR / "per_class_metrics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["class", "precision", "recall", "f1", "support"])
            for i, name in enumerate(classes):
                writer.writerow([name, f"{precision[i]:.4f}", f"{recall[i]:.4f}", f"{f1[i]:.4f}", int(support[i])])

        macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        macro_p, macro_r, _, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average="macro", zero_division=0)
        accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
        pairs = sorted(((int(matrix[i][j]), classes[i], classes[j]) for i in labels for j in labels if i != j),
                       reverse=True)
        by_f1 = sorted(labels, key=lambda i: f1[i])

        report += [
            "",
            f"Best checkpoint re-evaluated on val ({len(y_true)} images, checkpoint epoch {checkpoint['epoch']}):",
            f"- macro F1 {macro_f1:.4f} (stored in checkpoint: {checkpoint['val_f1']:.4f}), accuracy {accuracy:.4f}",
            f"- macro precision {macro_p:.4f}, macro recall {macro_r:.4f}",
            f"- train augmentation recorded in checkpoint: {checkpoint.get('transform', {}).get('augmentation', 'not recorded')} "
            f"(evaluation uses BASE_TRANSFORM, no augmentation)",
            "",
            "Confusion matrix (rows = true, columns = predicted): " + ", ".join(classes),
            *[f"  {classes[i]:10s} " + " ".join(f"{int(v):3d}" for v in matrix[i]) for i in labels],
            "",
            "Per-class metrics (class: precision / recall / F1 / support):",
            *[f"- {classes[i]}: {precision[i]:.4f} / {recall[i]:.4f} / {f1[i]:.4f} / {int(support[i])}" for i in labels],
            "",
            f"Lowest F1: {', '.join(f'{classes[i]} ({f1[i]:.4f})' for i in by_f1[:3])}",
            f"Lowest recall: {classes[min(labels, key=lambda i: recall[i])]} ({min(recall):.4f})",
            f"Lowest precision: {classes[min(labels, key=lambda i: precision[i])]} ({min(precision):.4f})",
            "Most frequent confusions (true -> predicted: count):",
            *[f"- {t} -> {p}: {n}" for n, t, p in pairs[:5]],
        ]

    (OUT_DIR / "baseline_analysis.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    print(f"\noutputs written to {OUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
