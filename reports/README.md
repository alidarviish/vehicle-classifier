# Reports index

Validation results of every experiment run so far (split `data/split_manifest.csv`,
train / validation = 2633 / 659 unless noted, seed 42; each run was trained once). Metrics are for
the checkpoint with the highest validation macro F1. All metrics in this index are validation
results. The test set was evaluated once, only after the final model, the `needs_review` threshold
and the inference protocol were fixed (see [Final Test evaluation](#final-test-evaluation)). The
Neysan images have not been evaluated.

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
| [ResNet18](experiments/08_resnet.md) | `resnet224_ft_aug` | 20 | 0.9605 | 0.9456 | 0.9523 | 0.9482 | Fine-tuning + train-only `full_aug` (224, ImageNet normalization) |
| [ResNet18](experiments/08_resnet.md) | `resnet224_ft_wd1e4` | 8 | 0.9514 | 0.9435 | 0.9408 | 0.9419 | Fine-tuning + AdamW weight decay 1e-4 |
| [ResNet18](experiments/08_resnet.md) | `resnet224_ft_plateau` | 8 | 0.9530 | 0.9448 | 0.9421 | 0.9432 | Fine-tuning + ReduceLROnPlateau; lr halved after epochs 11, 15, 19; selected weights identical to `resnet224_fine_tuning` |
| [ResNet18](experiments/08_resnet.md) | `resnet224_ft_aug_wd1e4_plateau` | 20 | 0.9560 | 0.9421 | 0.9487 | 0.9442 | Fine-tuning + `full_aug` + AdamW weight decay 1e-4 + ReduceLROnPlateau; lr halved after epochs 11 and 17 |
| [ResNet18](experiments/08_resnet.md) | `resnet_feature_extraction` | 7 | 0.8771 | 0.8781 | 0.8592 | 0.8666 | Exploratory run at 128x128 input |
| [ResNet18](experiments/08_resnet.md) | `resnet_finetuning` | 7 | 0.9241 | 0.9173 | 0.9168 | 0.9165 | Exploratory run at 128x128 input |
| [Combined](experiments/09_combined_aug_plateau.md) | `combo_aug_plateau` | 18 | 0.8983 | 0.8875 | 0.8844 | 0.8853 | `full_aug` + ReduceLROnPlateau (0.5, 3); lr halved after epochs 13 and 18 |
| [Combined](experiments/09_combined_aug_plateau.md) | `combo_aug_do05_wd1e4_plateau` | 14 | 0.8847 | 0.8682 | 0.8552 | 0.8590 | As above + dropout 0.5 + AdamW weight decay 1e-4; lr never reduced |

Experiments 01-05 and 07 each change one factor relative to the baseline setup. Experiment 06
trains the baseline model on a simulated-imbalance subset of the training split and compares its two
batch modes with each other. Experiment 08 uses a pretrained ResNet18 with 224x224 input and
ImageNet normalization and compares feature extraction with fine-tuning; it also records fine-tuning
with train-only augmentation (`resnet224_ft_aug`), with weight decay (`resnet224_ft_wd1e4`) and with
a scheduler (`resnet224_ft_plateau`), and one combination of all three
(`resnet224_ft_aug_wd1e4_plateau`). Experiment 09 tests two
combinations on the baseline CNN (augmentation + scheduler, then also dropout + weight decay). See
the linked report for the exact settings, per-class results and limitations.

## Validation comparison table

Main experiments of the assignment, compared on validation only: 659 images of
`data/split_manifest.csv` with the current labels, seed 42. Each row is the checkpoint with the highest
validation macro F1 of that run. The lowest-recall and lowest-precision classes come from the
per-class metrics of the same checkpoint (no ties in any row). The test set and the Neysan images
were not used in this comparison.

| Experiment | Run | Accuracy | Macro Precision | Macro Recall | Macro F1 | Lowest-Recall Class | Lowest-Precision Class |
|---|---|---:|---:|---:|---:|---|---|
| CNN baseline [1] | `baseline` | 0.8801 | 0.8646 | 0.8509 | 0.8561 | vanet (0.5769) | vanet (0.6818) |
| Balanced batches: standard [2] | `balanced_standard` | 0.6707 | 0.7336 | 0.6568 | 0.6120 | kamyun (0.1398) | vanet (0.4615) |
| Balanced batches: balanced [2] | `balanced_sampler` | 0.6525 | 0.7055 | 0.6372 | 0.5871 | kamyun (0.0968) | vanet (0.3103) |
| CE [3] | `loss_ce` | 0.8801 | 0.8646 | 0.8509 | 0.8561 | vanet (0.5769) | vanet (0.6818) |
| BCE [3] | `loss_bce` | 0.8816 | 0.8747 | 0.8482 | 0.8568 | vanet (0.5385) | vanet (0.7778) |
| Best regularized + scheduled [4] | `combo_aug_plateau` | 0.8983 | 0.8875 | 0.8844 | 0.8853 | vanet (0.7692) | vanet (0.7407) |
| ResNet18 feature extraction | `resnet224_feature_extraction` | 0.9181 | 0.9081 | 0.9038 | 0.9055 | vanet (0.7692) | vanet (0.8000) |
| ResNet18 fine-tuning | `resnet224_fine_tuning` | 0.9530 | 0.9448 | 0.9421 | 0.9432 | vanet (0.8462) | vanet (0.8462) |
| ResNet18 fine-tuning + augmentation [5] | `resnet224_ft_aug` | 0.9605 | 0.9456 | 0.9523 | 0.9482 | vanet (0.8846) | vanet (0.7667) |
| ResNet18 fine-tuning + weight decay [6] | `resnet224_ft_wd1e4` | 0.9514 | 0.9435 | 0.9408 | 0.9419 | vanet (0.8462) | vanet (0.8462) |
| ResNet18 fine-tuning + scheduler [6] | `resnet224_ft_plateau` | 0.9530 | 0.9448 | 0.9421 | 0.9432 | vanet (0.8462) | vanet (0.8462) |
| ResNet18 fine-tuning + augmentation + weight decay + scheduler [7] | `resnet224_ft_aug_wd1e4_plateau` | 0.9560 | 0.9421 | 0.9487 | 0.9442 | kamyunet (0.8673) | vanet (0.7667) |

[1] The baseline is reported with the current validation labels. These include 8 label
corrections: 7 kamyun -> kamyunet and 1 kamyunet -> kamyun. The metrics were recalculated from
the saved predictions of the same checkpoint (epoch 18). With the original labels the same
checkpoint gives accuracy 0.8680, macro precision 0.8539, macro recall 0.8402 and macro F1 0.8456.
Its lowest-recall and lowest-precision class is also vanet, with the same values (0.5769 and
0.6818). `loss_ce` and `adamw_wd0` have the same configuration and the same metrics with the
current labels.

[2] The balanced runs were trained on the simulated-imbalance subset
`decisions/imbalanced_train.csv` (1673 images), not on the full 2633-image training split. The
validation set is the same. These two rows compare with each other, not directly with the other
rows.

[3] CE and BCE form one controlled comparison: same configuration, only the loss differs. They are
not two independent candidates for the final model. The production model is a CrossEntropy
(softmax) model, and BCE is reported for comparison only.

[4] Selected on validation. Only two runs combine regularization with a scheduler:
`combo_aug_plateau` (macro F1 0.8853) and `combo_aug_do05_wd1e4_plateau` (macro F1 0.8590). In the
selected run, the regularization is augmentation (`full_aug`), combined with ReduceLROnPlateau,
which halved the learning rate after epochs 13 and 18. Each run was trained once with seed 42. See
`experiments/09_combined_aug_plateau.md`.

[5] Additional ResNet run, not one of the six rows the assignment requires. It is the same as the
fine-tuning row except for train-only augmentation `full_aug`, adapted to 224x224 and ImageNet
normalization. Validation is unchanged (`RESNET_TRANSFORM`, no augmentation). It is recorded
regardless of its result; no run is removed from this table because it scored better or worse.
Single run, seed 42. See `experiments/08_resnet.md`.

[6] Additional ResNet runs, not among the six required rows. Each is the same as the fine-tuning
row except for one factor: AdamW with weight decay 1e-4, or ReduceLROnPlateau. In
`resnet224_ft_plateau` the learning rate was first reduced after epoch 11. Its best checkpoint
(epoch 8) is from before that reduction, and its weights are byte-identical to those of
`resnet224_fine_tuning`, so its row repeats the fine-tuning values. Both runs are recorded
regardless of their result; no run is removed from this table because it scored better or worse.
Single runs, seed 42. See `experiments/08_resnet.md`.

[7] Additional combination run, not among the six required rows. It is the same as the
fine-tuning row plus three changes: train-only `full_aug`, AdamW with weight decay 1e-4, and
ReduceLROnPlateau. Validation is unchanged (`RESNET_TRANSFORM`, no augmentation). Its lowest-recall
class is kamyunet, due to 12 kamyunet -> kamyun errors. It is kept regardless of its result. It is a
single run with seed 42 and is not general evidence about the effect of this combination. See
`experiments/08_resnet.md`.

## Final model selection

Selected on validation only (see [10_final_model_selection.md](experiments/10_final_model_selection.md)):
**`resnet224_ft_aug`**, checkpoint `checkpoints/resnet224_ft_aug_best.pt` (epoch 20). Its validation
results are accuracy 0.9605, macro precision 0.9456, macro recall 0.9523 and macro F1 0.9482; it has
the highest macro F1 of all runs. Known weaknesses are vanet (F1 0.8214, precision 0.7667) and the
kamyun / kamyunet confusion. The test set was not used for this selection; it was evaluated once
afterwards (see below).

Chosen `needs_review` threshold (validation only, see
[11_needs_review_threshold.md](experiments/11_needs_review_threshold.md)): **0.90**. A prediction
whose confidence is below 0.90 gets a human-review flag; nothing is rejected. On validation this
flags 35 of 659 images, 13 of the 26 errors and 22 correct predictions, and leaves coverage at 94.7%.
The test set was not used to choose it.

## Final Test evaluation

[12_final_test_evaluation.md](experiments/12_final_test_evaluation.md) (per-image predictions:
[12_final_test_predictions.csv](experiments/12_final_test_predictions.csv)). Status: completed, run
once (`final_test_20260929T080908Z`) with `scripts/evaluate_test.py` at commit `b02782f`, after the
final model, the threshold and the inference protocol (`src/predict.py`) were fixed. The frozen test
set (`decisions/test_frozen.csv`, 400 images, 50 per class) and the frozen checkpoint
`checkpoints/resnet224_ft_aug_best.pt` were used unchanged; Neysan images were not part of this run.

| Test (400 images) | Value |
|---|---|
| Correct / incorrect | 379 / 21 |
| Accuracy | 0.9475 |
| Macro precision | 0.9515 |
| Macro recall | 0.9475 |
| Macro F1 | 0.9473 |
| `needs_review` (confidence < 0.90) | 29 (17 correct, 12 incorrect); 9 errors not flagged |

## Other folders

- `audit/`: initial dataset audit (counts and exact duplicates).
- `analysis/`: generated analysis output (ignored by Git), one folder per analysed run:
  `reports/analysis/<run_name>/`, including `reports/analysis/baseline/`.
