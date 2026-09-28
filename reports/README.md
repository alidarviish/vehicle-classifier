# Reports index

Validation results of every experiment run so far (split `data/split_manifest.csv`,
train / validation = 2633 / 659 unless noted, seed 42; each run was trained once). Metrics are for
the checkpoint with the highest validation macro F1. The test set and the Neysan images have not
been evaluated.

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
| [Balanced batches](experiments/06_balanced_batches.md) | `balanced_standard` | 15 | 0.6707 | 0.7336 | 0.6568 | 0.6120 | Simulated-imbalance training subset (1673 images); shuffled batches |
| [Balanced batches](experiments/06_balanced_batches.md) | `balanced_sampler` | 10 | 0.6525 | 0.7055 | 0.6372 | 0.5871 | Same subset; `BalancedBatchSampler`, 4 images per class per batch |
| [CE vs BCE](experiments/07_ce_vs_bce.md) | `loss_ce` | 18 | 0.8801 | 0.8646 | 0.8509 | 0.8561 | CrossEntropyLoss; same history as `adamw_wd0` |
| [CE vs BCE](experiments/07_ce_vs_bce.md) | `loss_bce` | 19 | 0.8816 | 0.8747 | 0.8482 | 0.8568 | BCEWithLogitsLoss on one-hot targets |
| [ResNet18](experiments/08_resnet.md) | `resnet224_feature_extraction` | 19 | 0.9181 | 0.9081 | 0.9038 | 0.9055 | Pretrained ResNet18, 224x224, backbone frozen, only the head trained |
| [ResNet18](experiments/08_resnet.md) | `resnet224_fine_tuning` | 8 | 0.9530 | 0.9448 | 0.9421 | 0.9432 | Head only in epochs 1-5, then `layer4` + head |
| [ResNet18](experiments/08_resnet.md) | `resnet_feature_extraction` | 7 | 0.8771 | 0.8781 | 0.8592 | 0.8666 | Exploratory run at 128x128 input |
| [ResNet18](experiments/08_resnet.md) | `resnet_finetuning` | 7 | 0.9241 | 0.9173 | 0.9168 | 0.9165 | Exploratory run at 128x128 input |

Experiments 01-05 and 07 each change one factor relative to the baseline setup. Experiment 06
trains the baseline model on a simulated-imbalance subset of the training split and compares its two
batch modes with each other. Experiment 08 uses a pretrained ResNet18 with 224x224 input and
ImageNet normalization and compares feature extraction with fine-tuning. See the linked report for
the exact settings, per-class results and limitations.

Other folders:

- `audit/`: initial dataset audit (counts and exact duplicates).
- `analysis/`: generated analysis output (ignored by Git), one folder per analysed run:
  `reports/analysis/<run_name>/`, including `reports/analysis/baseline/`.
