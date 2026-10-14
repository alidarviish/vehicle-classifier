"""Train Swin-Tiny on dataset v3 (data/split_manifest_v3.csv), one setting change at a time.

Run from the repository root, for example:
    python -m src.train_v3 --run-name v3_timing --timing 20
    python -m src.train_v3 --run-name v3_base
    python -m src.train_v3 --run-name v3_aug --aug strong
    python -m src.train_v3 --run-name v3_final --aug strong --unfreeze top2     (the final model)

Settings (everything else is the same as the earlier model swin_t_ft_aug):
    --aug             old    = crop 0.8-1 + flip + rotation 10 + brightness/contrast (as swin_t_ft_aug)
                      strong = crop 0.7-1 + flip + rotation 10 + brightness/contrast/saturation 0.3
                               + hue 0.02 + grayscale (20%) + blur (30%) + random erasing (30%)
    --unfreeze        top    = last stage + norm (as swin_t_ft_aug)
                      top2   = also the stage before it (used by v3_final)
    --label-smoothing 0 = off (used by v3_final), e.g. 0.1 = on (not used in the recorded runs)
    --scheduler       none = constant learning rate (used by v3_final), cosine = slowly down to ~0
    --timing N        only run N training batches, print the time and stop (to plan CPU runs)

Only Train and Validation are read; the test folders are never touched.
The history CSV has the same columns as the earlier runs (reports/*_history.csv).

Outputs:
    reports/v3/<run_name>_history.csv      one row per epoch
    checkpoints/<run_name>_best.pt         best epoch by Validation macro F1
"""
import argparse
import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src import swin
from src.dataset import CLASS_TO_IDX, CLASSES, DEFAULT_CONFIG, VehicleDataset

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "data" / "split_manifest_v3.csv"
SEED = 42
SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
BATCH_SIZE = 32
HEAD_LR, BACKBONE_LR = 1e-3, 1e-4      # same as swin_t_ft_aug
WARMUP_EPOCHS = 5                      # head only in epochs 1-5, then the backbone stages too
UNFREEZE = {"top": (6, 7), "top2": (4, 5, 6, 7)}


def train_transform(aug):
    """old = the augmentation of swin_t_ft_aug; strong = more variety for better generalization."""
    if aug == "old":
        return transforms.Compose([
            transforms.RandomResizedCrop(SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.RandomResizedCrop(SIZE, scale=(0.7, 1.0)),          # a bit more zoom / framing variety
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3,
                               hue=0.02),                               # small hue: the colour of taxi / ambulance matters
        transforms.RandomGrayscale(p=0.2),                              # like the black-and-white night camera
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.3),  # blur only some images
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),            # hide a small part of the vehicle
    ])


EVAL_TRANSFORM = transforms.Compose([transforms.Resize((SIZE, SIZE)), transforms.ToTensor(),
                                     transforms.Normalize(MEAN, STD)])


def unfreeze(model, stages):
    for i in stages:
        swin.set_trainable(model.net.features[i], True)
    swin.set_trainable(model.net.norm, True)


def backbone_params(model, stages):
    return [p for i in stages for p in model.net.features[i].parameters()] + list(model.net.norm.parameters())


def metrics(labels, preds, total_loss):
    """loss, accuracy and macro precision / recall / F1 over the 8 classes."""
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro",
                                                  labels=list(range(len(CLASSES))), zero_division=0)
    acc = float(np.mean(np.array(preds) == np.array(labels)))
    return {"loss": total_loss / len(labels), "accuracy": acc, "precision": float(p), "recall": float(r), "f1": float(f1)}


def run_epoch(model, loader, criterion, optimizer=None):
    """One pass over a loader. Trains if an optimizer is given, otherwise only evaluates."""
    model.train() if optimizer else model.eval()
    preds, labels, total_loss = [], [], 0.0
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in loader:
            logits = model(x)
            loss = criterion(logits, y)
            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            preds += logits.argmax(1).tolist()
            labels += y.tolist()
    return metrics(labels, preds, total_loss)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--aug", choices=["old", "strong"], default="old")
    p.add_argument("--unfreeze", choices=list(UNFREEZE), default="top")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--scheduler", choices=["none", "cosine"], default="none")
    p.add_argument("--timing", type=int, default=0)
    p.add_argument("--manifest", default=MANIFEST)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    a = p.parse_args()
    if not a.run_name.startswith("v3_"):
        raise SystemExit("run names must start with v3_ (so old runs are never overwritten)")
    ckpt_path = REPO / "checkpoints" / f"{a.run_name}_best.pt"
    hist_path = REPO / "reports" / "v3" / f"{a.run_name}_history.csv"
    if not a.timing and (ckpt_path.exists() or hist_path.exists()):
        raise SystemExit(f"{a.run_name} already exists; choose another run name")

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    train_set = VehicleDataset("train", train_transform(a.aug), a.manifest, a.config)
    val_set = VehicleDataset("val", EVAL_TRANSFORM, a.manifest, a.config)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    print(f"train {len(train_set)} | val {len(val_set)} | threads {torch.get_num_threads()}")

    model = swin.build_swin_tiny()                       # ImageNet weights, new 8-class head
    stages = UNFREEZE[a.unfreeze]
    optimizer = torch.optim.Adam([{"params": model.net.head.parameters(), "lr": HEAD_LR},
                                  {"params": backbone_params(model, stages), "lr": BACKBONE_LR}])
    criterion = nn.CrossEntropyLoss(label_smoothing=a.label_smoothing)
    base_lrs = [HEAD_LR, BACKBONE_LR]

    if a.timing:                                          # time a few batches with the backbone unfrozen
        unfreeze(model, stages)
        model.train()
        start = time.time()
        for i, (x, y) in enumerate(train_loader):
            if i == a.timing:
                break
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
        per_batch = (time.time() - start) / a.timing
        epoch_min = per_batch * math.ceil(len(train_set) / BATCH_SIZE) / 60
        print(f"{per_batch:.1f} s per batch -> about {epoch_min:.0f} min per training epoch "
              f"(+ validation), {a.epochs} epochs about {epoch_min * a.epochs / 60:.1f} h")
        return

    hist_path.parent.mkdir(parents=True, exist_ok=True)
    best_f1, history = -1.0, []
    for epoch in range(1, a.epochs + 1):
        if epoch == WARMUP_EPOCHS + 1:
            unfreeze(model, stages)
        if a.scheduler == "cosine":                       # one smooth curve over all epochs
            factor = 0.5 * (1 + math.cos(math.pi * (epoch - 1) / a.epochs))
            for group, lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = lr * factor

        start = time.time()
        train_m = run_epoch(model, train_loader, criterion, optimizer)
        val_m = run_epoch(model, val_loader, nn.CrossEntropyLoss())     # plain loss, no smoothing
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_m.items()},
               **{f"val_{k}": v for k, v in val_m.items()},
               "lr": optimizer.param_groups[0]["lr"], "trainable_params": swin.count_params(model)[0],
               "lr_top": optimizer.param_groups[1]["lr"]}
        history.append(row)
        print(f"epoch {epoch:2d} | train_loss {train_m['loss']:.4f} | val_loss {val_m['loss']:.4f} "
              f"| val_acc {val_m['accuracy']:.4f} | val_f1 {val_m['f1']:.4f} | {(time.time() - start) / 60:.0f} min", flush=True)

        with open(hist_path, "w", newline="", encoding="utf-8") as f:   # rewritten every epoch
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerows(history)
        if row["val_f1"] > best_f1:                       # ties keep the earlier epoch
            best_f1 = row["val_f1"]
            torch.save({"model_state": model.state_dict(), "class_to_idx": CLASS_TO_IDX,
                        "architecture": "Swin-Tiny", "image_size": SIZE,
                        "transform": {"resize": [SIZE, SIZE], "normalize_mean": MEAN, "normalize_std": STD,
                                      "augmentation": a.aug},
                        "settings": {"unfreeze": a.unfreeze, "label_smoothing": a.label_smoothing,
                                     "scheduler": a.scheduler, "epochs": a.epochs, "manifest": "split_manifest_v3.csv"},
                        "epoch": epoch, "val_f1": best_f1}, ckpt_path)
    print("best val macro F1", round(best_f1, 4), "->", ckpt_path)


if __name__ == "__main__":
    main()
