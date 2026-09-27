# Experiment: aug_crop_jitter

## Setup

- Run name: `aug_crop_jitter`
- Command: `python -m src.train --run-name aug_crop_jitter --augmentation crop_jitter`
- Architecture: `BaselineCNN`
- Input: 128x128 RGB
- Batch size: 32
- Epochs: 20
- Optimizer: Adam
- Learning rate: 1e-3
- Weight decay: 0
- Scheduler: none
- Loss: CrossEntropyLoss
- Seed: 42
- Train / validation images: 2633 / 659 (`data/split_manifest.csv`, current labels)
- Checkpoint criterion: best validation macro F1
- Only change vs. baseline: augmentation on the train split

## Augmentation

Train transform (train split only):

- `RandomResizedCrop(128, scale=(0.8, 1.0))`
- `ColorJitter(brightness=0.2, contrast=0.2)`
- then `ToTensor()` and `Normalize(mean=[0.5]*3, std=[0.5]*3)`

Validation uses `BASE_TRANSFORM` (`Resize((128, 128))`, `ToTensor()`, `Normalize`), with no augmentation.
The checkpoint records `augmentation: crop_jitter` with
`random_resized_crop_scale: [0.8, 1.0]`, `brightness: 0.2`, `contrast: 0.2`.

## Result (validation, best checkpoint)

| Metric | Value |
|---|---|
| Best epoch | 19 |
| Macro F1 | 0.8603 |
| Accuracy | 0.8741 |
| Macro precision | 0.8843 |
| Macro recall | 0.8475 |

## Comparison with final baseline

Baseline values are recalculated from its saved predictions with the current validation labels.

| Metric | Baseline | aug_crop_jitter | Observed change |
|---|---|---|---|
| Macro F1 | 0.8561 | 0.8603 | +0.0042 |
| Accuracy | 0.8801 | 0.8741 | -0.0061 (580 vs 576 correct of 659) |

## Per-class F1

| Class | F1 |
|---|---|
| ambulance | 0.8657 |
| autobus | 0.9266 |
| kamyun | 0.7700 |
| kamyunet | 0.7938 |
| minibus | 0.9020 |
| savari | 0.9289 |
| taxi | 0.9843 |
| vanet | 0.7111 |

- vanet: F1 0.6250 (baseline) -> 0.7111; still the lowest F1 (recall 0.6154, support 26).
- kamyun (F1 0.7700, baseline 0.8247) and kamyunet (F1 0.7938, baseline 0.8272) still show substantial
  confusion with each other (kamyunet -> kamyun: 18, kamyun -> kamyunet: 9).
- These are the results of a single run with seed 42.

## Main confusions (true -> predicted)

- kamyunet -> kamyun: 18
- kamyun -> kamyunet: 9
- autobus -> kamyun: 8
- vanet -> savari: 6
- ambulance -> savari: 5

## Files

- Checkpoint: `checkpoints/aug_crop_jitter_best.pt`
- History: `reports/aug_crop_jitter_history.csv`
- Analysis output: `reports/analysis/aug_crop_jitter/`

## Scope

- Test set: not evaluated.
- Neysan images: not evaluated.
- No data, label or split was changed for this experiment.
- Single training run with seed 42.
