# Swin-Tiny feature extraction

First Swin-Tiny run in the architecture comparison: how far do the frozen ImageNet features of
Swin-Tiny get on this task, with only a new head trained and no augmentation. It follows the same
protocol as the earlier feature-extraction runs (`resnet224_feature_extraction`, `effnet_b0_fe_none`,
`convnext_t_fe_none`). This is a comparison experiment only; the final model of the project stays
`resnet224_ft_aug`.

## Setup

- Run name: `swin_t_fe_none`
- Command: `python -m src.train --run-name swin_t_fe_none --model swin_tiny`
- Code: `src/swin.py`, `src/train.py` (Swin-Tiny added in commit `e6f158e`)
- Model: Swin-Tiny (torchvision `swin_t`), ImageNet weights `IMAGENET1K_V1`; the 1000-class head is
  replaced by `Linear(768, 8)`
- Mode: feature extraction (head only). `net.features` and `net.norm` are frozen and stay in eval mode,
  so stochastic depth is off; only the head is trained.
- Parameters (checkpoint metadata): 6,152 trainable (`head`), 27,519,354 frozen
- Optimizer: Adam, one parameter group (`head`, learning rate 1e-3), weight decay 0, no scheduler;
  CrossEntropyLoss
- Augmentation: `none` (train and validation: Resize 224x224 -> ToTensor -> ImageNet normalization, the
  same project transform as the other pretrained runs, not the preset transform of `Swin_T_Weights`)
- 20 epochs, batch size 32, seed 42; train 2633 / validation 659 images
- Evaluation: validation split only, best checkpoint, transform recorded in the checkpoint
- Checkpoint `checkpoints/swin_t_fe_none_best.pt` (sha256
  `f750a3b9bc1c242c6131da0a4f4d5cb0209b02cf1191d9d789876642d4f5c9e4`), history
  `reports/swin_t_fe_none_history.csv`, analysis `reports/analysis/swin_t_fe_none/`
  (all not tracked)

## Validation results (best checkpoint, epoch 19)

The checkpoint is the epoch with the highest validation macro F1: epoch 19.

| Metric | Value |
|---|---|
| Accuracy | 0.9423 (621/659) |
| Macro precision | 0.9447 |
| Macro recall | 0.9298 |
| Macro F1 | 0.9363 |
| Validation loss | 0.1948 |

Best-F1 checkpoint and last epoch are different epochs:

| Epoch | Val accuracy | Val macro F1 | Val loss |
|---|---|---|---|
| 19 (best F1, saved checkpoint) | 0.9423 | 0.9363 | 0.1948 |
| 20 (last epoch) | 0.9393 | 0.9331 | 0.1925 (lowest of the run) |

Per-class metrics:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ambulance | 0.9857 | 0.9324 | 0.9583 | 74 |
| autobus | 0.9783 | 0.9783 | 0.9783 | 92 |
| kamyun | 0.8936 | 0.9032 | 0.8984 | 93 |
| kamyunet | 0.8788 | 0.8878 | 0.8832 | 98 |
| minibus | 0.9146 | 0.9494 | 0.9317 | 79 |
| savari | 0.9524 | 1.0000 | 0.9756 | 100 |
| taxi | 1.0000 | 0.9794 | 0.9896 | 97 |
| vanet | 0.9545 | 0.8077 | 0.8750 | 26 |

- Lowest F1: vanet 0.8750, kamyunet 0.8832, kamyun 0.8984.
- Lowest recall: vanet 0.8077 (21 of 26 images).
- Highest F1: taxi 0.9896, autobus 0.9783, savari 0.9756.

Confusion matrix (rows = true class, columns = predicted class):

| true \ predicted | ambulance | autobus | kamyun | kamyunet | minibus | savari | taxi | vanet |
|---|---|---|---|---|---|---|---|---|
| ambulance | 69 | 0 | 0 | 2 | 1 | 1 | 0 | 1 |
| autobus | 0 | 90 | 0 | 0 | 2 | 0 | 0 | 0 |
| kamyun | 0 | 1 | 84 | 8 | 0 | 0 | 0 | 0 |
| kamyunet | 0 | 1 | 7 | 87 | 3 | 0 | 0 | 0 |
| minibus | 0 | 0 | 2 | 2 | 75 | 0 | 0 | 0 |
| savari | 0 | 0 | 0 | 0 | 0 | 100 | 0 | 0 |
| taxi | 0 | 0 | 1 | 0 | 1 | 0 | 95 | 0 |
| vanet | 1 | 0 | 0 | 0 | 0 | 4 | 0 | 21 |

## Training behaviour

- Train loss falls from 0.9173 at epoch 1 to 0.082057 at epoch 20; train accuracy never reaches 1.0.
- At epoch 20: train macro F1 0.9810, validation macro F1 0.9331, a gap of 0.0478.
- Validation loss is lowest at epoch 20 (0.1925), the last epoch of the run. Because there are no later
  epochs, this run cannot show whether validation loss would rise after its minimum.

## Main errors

38 of 659 validation images are wrong. The most frequent confusions:

| True -> predicted | Images |
|---|---|
| kamyun -> kamyunet | 8 |
| kamyunet -> kamyun | 7 |
| vanet -> savari | 4 |
| kamyunet -> minibus | 3 |
| minibus -> kamyunet | 2 |

The kamyun / kamyunet pair accounts for 15 of the 38 errors. 4 of the 5 vanet errors are predicted as
savari.

## Comparison (validation, factual)

| Run | Mode | Augmentation | Accuracy | Macro F1 |
|---|---|---|---|---|
| `resnet224_ft_aug` (final model) | partial fine-tuning | full_aug | 0.9605 | 0.9482 |
| `convnext_t_ft_none` | partial fine-tuning | none | 0.9590 | 0.9538 |
| `effnet_b0_ft_none` | partial fine-tuning | none | 0.9560 | 0.9485 |
| `swin_t_fe_none` | feature extraction | none | 0.9423 | 0.9363 |

These are validation results of single runs. `swin_t_fe_none` trains only the head, while the other
three runs fine-tune part of the backbone, so the table describes the observed results of these runs and
is not a verdict on the architectures.

## Conclusion

With frozen ImageNet features and no augmentation, Swin-Tiny reaches validation macro F1 0.9363 and
accuracy 0.9423 (best epoch 19). kamyun / kamyunet remains one of the main confusions, and vanet has the
lowest recall (0.8077), mostly through vanet -> savari.

## Limitations

- Validation only (659 images), one run with seed 42 and one validation split; small differences are
  not conclusive.
- The Final Test and the Neysan evaluation were not used in this experiment, and no statement about Test
  performance follows from it.
- No label, split or data was changed.

## Next question

If the comparison is continued, Swin-Tiny could be tested with partial fine-tuning (a warm-up with the
head only, then the last stage unfrozen), under the same protocol as `convnext_t_ft_none`. No result for
that setting is assumed here.
