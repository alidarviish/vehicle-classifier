"""Train the baseline CNN on the train / val split.

Run from the repository root:
    python -m src.train
    python -m src.train --config configs/local_paths.json

Only the "train" and "val" rows of data/split_manifest.csv are used.
The test set is not read anywhere in this file.

Outputs:
    reports/<run_name>_history.csv      one row per epoch
    checkpoints/<run_name>_best.pt      model with the best val macro F1
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset import CLASS_TO_IDX, CLASSES, DEFAULT_CONFIG, DEFAULT_MANIFEST, VehicleDataset
from src.model import BaselineCNN

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"

# baseline settings
SEED = 42
IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
DROPOUT = 0.0
NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD = [0.5, 0.5, 0.5]

# same transform for train and val - the baseline has no augmentation
BASE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    """Fix the random number generators used during training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(manifest_path, config_path):
    train_set = VehicleDataset("train", BASE_TRANSFORM, manifest_path, config_path)
    val_set = VehicleDataset("val", BASE_TRANSFORM, manifest_path, config_path)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader


def run_epoch(model, loader, criterion, optimizer=None):
    """One pass over a loader. Trains if an optimizer is given, otherwise only evaluates."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(len(CLASSES))),
        average="macro", zero_division=0,
    )
    accuracy = sum(p == t for p, t in zip(all_preds, all_labels)) / len(all_labels)
    return {
        "loss": total_loss / len(all_labels),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def save_history(history, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def train(run_name, manifest_path, config_path, epochs=EPOCHS):
    set_seed()
    REPORTS_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    train_loader, val_loader = build_loaders(manifest_path, config_path)
    print(f"device {DEVICE} | train {len(train_loader.dataset)} | val {len(val_loader.dataset)}")

    model = BaselineCNN(num_classes=len(CLASSES), image_size=IMAGE_SIZE, dropout=DROPOUT).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    history = []
    best_f1 = -1.0
    checkpoint_path = CHECKPOINT_DIR / f"{run_name}_best.pt"

    for epoch in range(1, epochs + 1):
        train_m = run_epoch(model, train_loader, criterion, optimizer)
        val_m = run_epoch(model, val_loader, criterion)

        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_m.items()})
        row.update({f"val_{k}": v for k, v in val_m.items()})
        row["lr"] = optimizer.param_groups[0]["lr"]
        history.append(row)
        print(f"epoch {epoch:2d} | train_loss {train_m['loss']:.4f} | val_loss {val_m['loss']:.4f} "
              f"| val_acc {val_m['accuracy']:.4f} | val_f1 {val_m['f1']:.4f}")

        # save only when val macro F1 is strictly better (ties keep the earlier epoch)
        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            torch.save({
                "model_state": model.state_dict(),
                "class_to_idx": CLASS_TO_IDX,
                "architecture": "BaselineCNN",
                "image_size": IMAGE_SIZE,
                "transform": {"resize": [IMAGE_SIZE, IMAGE_SIZE], "normalize_mean": NORM_MEAN,
                              "normalize_std": NORM_STD, "augmentation": "none"},
                "seed": SEED,
                "dropout": DROPOUT,
                "pooling": "max",
                "optimizer": "adam",
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "scheduler": "none",
                "weight_decay": WEIGHT_DECAY,
                "loss_name": "ce",
                "epoch": epoch,
                "val_f1": best_f1,
            }, checkpoint_path)
            print(f"  -> saved {checkpoint_path.name} (val_f1 {best_f1:.4f})")

    history_path = REPORTS_DIR / f"{run_name}_history.csv"
    save_history(history, history_path)
    print(f"history saved to {history_path}")
    return history


def main():
    parser = argparse.ArgumentParser(description="Train the baseline CNN (train/val only).")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="local raw-data paths (JSON)")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="split manifest CSV")
    parser.add_argument("--run-name", default="baseline")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    train(args.run_name, args.manifest, args.config, args.epochs)


if __name__ == "__main__":
    main()
