# Reports index

Validation results of every experiment run so far (split `data/split_manifest.csv`,
train / validation = 2633 / 659, seed 42, one run per setting). Metrics are for the checkpoint
with the highest validation macro F1. The test set and the Neysan images have not been evaluated.

| Experiment | Run | Best epoch | Val accuracy | Macro precision | Macro recall | Macro F1 | Note |
|---|---|---|---|---|---|---|---|
| [Baseline](experiments/00_baseline.md) | `baseline` | 18 | 0.8801 | 0.8646 | 0.8509 | 0.8561 | Reference run; metrics recalculated with the current validation labels |
| [Augmentation](experiments/01_augmentation.md) | `aug_crop_jitter` | 19 | 0.8741 | 0.8843 | 0.8475 | 0.8603 | RandomResizedCrop + ColorJitter (train only) |
| [Augmentation](experiments/01_augmentation.md) | `aug_full` | 14 | 0.8983 | 0.8936 | 0.8735 | 0.8816 | + RandomHorizontalFlip + RandomRotation(10) |
| [Dropout](experiments/02_dropout.md) | `dropout_03` | 20 | 0.8862 | 0.8739 | 0.8477 | 0.8562 | dropout 0.3 |
| [Dropout](experiments/02_dropout.md) | `dropout_05` | 17 | 0.8862 | 0.9032 | 0.8450 | 0.8592 | dropout 0.5 |
| [Pooling](experiments/03_pooling.md) | `pooling_average` | 15 | 0.8558 | 0.8468 | 0.8171 | 0.8267 | AvgPool2d instead of MaxPool2d |
| [Weight decay](experiments/04_weight_decay.md) | `adamw_wd0` | 18 | 0.8801 | 0.8646 | 0.8509 | 0.8561 | Optimizer control (AdamW, wd 0); same history as the baseline |
| [Weight decay](experiments/04_weight_decay.md) | `adamw_wd1e4` | 12 | 0.8756 | 0.8619 | 0.8566 | 0.8578 | AdamW, weight decay 1e-4 |
| [Scheduler](experiments/05_scheduler.md) | `scheduler_step` | 5 | 0.8756 | 0.8548 | 0.8461 | 0.8494 | StepLR (8, 0.5); best epoch is before the first lr change |
| [Scheduler](experiments/05_scheduler.md) | `scheduler_plateau` | 5 | 0.8756 | 0.8548 | 0.8461 | 0.8494 | ReduceLROnPlateau (0.5, 3); same selected weights as `scheduler_step` |

Each ablation changes one factor relative to the baseline; see the linked report for the
exact settings, per-class results and limitations.

Other folders:

- `audit/`: initial dataset audit (counts and exact duplicates).
- `analysis/`: generated analysis output per run (ignored by Git).
