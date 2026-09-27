# Experiment: aug_full

- Run name: `aug_full`
- Command: `python -m src.train --run-name aug_full --augmentation full_aug`
- Only change vs. baseline: the train-time augmentation.

## Setup

Train augmentation (train split only):

- `RandomResizedCrop(128, scale=(0.8, 1.0))`
- `RandomHorizontalFlip()`
- `RandomRotation(10)`
- `ColorJitter(brightness=0.2, contrast=0.2)`
- then the baseline `ToTensor()` and `Normalize(mean=[0.5]*3, std=[0.5]*3)`

Validation uses `BASE_TRANSFORM` (`Resize((128, 128))`, `ToTensor()`, `Normalize`), with no augmentation.

Unchanged from the baseline: `BaselineCNN` (128x128), batch size 32, 20 epochs, Adam,
lr 1e-3, weight decay 0, CrossEntropyLoss, no scheduler, seed 42, `data/split_manifest.csv`
with the current labels, train/val = 2633 / 659, best checkpoint by validation macro F1.

## Result (validation, best checkpoint)

| Metric | Value |
|---|---|
| Best epoch | 14 |
| Macro F1 | 0.8816 |
| Accuracy | 0.8983 |
| Macro precision | 0.8936 |
| Macro recall | 0.8735 |

Comparison with the baseline (baseline values recalculated from its saved predictions
with the current validation labels):

| Metric | Baseline | aug_full |
|---|---|---|
| Macro F1 | 0.8561 | 0.8816 |
| Accuracy | 0.8801 | 0.8983 |

## Per-class F1

| Class | Baseline | aug_full |
|---|---|---|
| ambulance | 0.8633 | 0.8511 |
| autobus | 0.9362 | 0.9385 |
| kamyun | 0.8247 | 0.8865 |
| kamyunet | 0.8272 | 0.8057 |
| minibus | 0.9045 | 0.9114 |
| savari | 0.9100 | 0.9412 |
| taxi | 0.9579 | 0.9948 |
| vanet | 0.6250 | 0.7234 |

- vanet: F1 0.6250 -> 0.7234 (precision 0.8095, recall 0.6538, support 26); still the lowest F1.
- kamyun: F1 0.8247 -> 0.8865.
- kamyunet: F1 0.8272 -> 0.8057; its precision is 0.7522, the lowest of all classes.

## Main confusions (true -> predicted)

- kamyunet -> kamyun: 8
- kamyun -> kamyunet: 8
- minibus -> kamyunet: 6
- ambulance -> kamyunet: 6
- vanet -> savari: 4

## Files

- Checkpoint: `checkpoints/aug_full_best.pt`
- History: `reports/aug_full_history.csv`
- Analysis output: `reports/analysis/aug_full/`

## Scope

- Test set: not evaluated in this experiment.
- Neysan images: not evaluated in this experiment.
- Single training run with seed 42; the numbers above are the observed validation results of this run only.
