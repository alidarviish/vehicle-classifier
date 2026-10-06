# Swin-Tiny partial fine-tuning with full_aug

Third Swin-Tiny run in the architecture comparison. One factor changes against `swin_t_ft_none`
(`21_swin_tiny_ft_none.md`): the train images use the project `full_aug` augmentation. Everything else
is the same. This is an architecture comparison only; it does not change the final model of the
project (`resnet224_ft_aug`).

Current status: after this report, the final model selection was reopened and `swin_t_ft_aug` was
selected on validation (`decisions/REOPEN_FINAL_MODEL_SELECTION.md`, commit `76669f4`). The
statements in this report that the final model stays `resnet224_ft_aug` describe the state when it
was written.

## Setup

- Run name: `swin_t_ft_aug`
- Command: `python -m src.train --run-name swin_t_ft_aug --model swin_tiny --swin-mode fine_tuning --augmentation full_aug`
- Code: `src/swin.py`, `src/train.py` (unchanged since commit `8b034a3`)
- Model: Swin-Tiny (torchvision `swin_t`), ImageNet weights `IMAGENET1K_V1`; head `Linear(768, 8)`
- Mode: partial fine-tuning
  - epochs 1-5 (warm-up): only the head is trained (6,152 trainable parameters)
  - from epoch 6: `features[6]`, `features[7]` and `norm` are unfrozen as well (15,374,264 trainable,
    12,151,242 frozen); `features[0:6]` stay frozen and in eval mode
- Parameter groups: `head` 1e-3, `features.6-7+norm` 1e-4
- Optimizer: Adam, weight decay 0, no scheduler; CrossEntropyLoss
- Augmentation (train only): `full_aug` - RandomResizedCrop(224, scale 0.8-1.0), horizontal flip
  (p 0.5), rotation 10 degrees, ColorJitter (brightness 0.2, contrast 0.2), ImageNet normalization
- Validation: Resize 224x224 -> ToTensor -> ImageNet normalization, no augmentation
- 20 epochs, batch size 32, seed 42; train 2633 / validation 659 images
- Checkpoint `checkpoints/swin_t_ft_aug_best.pt` (sha256
  `f03bacde0a155461ea4aede0b813b8688d6ed17973d5ad19b0d2c1b913ff64db`), history
  `reports/swin_t_ft_aug_history.csv`, analysis `reports/analysis/swin_t_ft_aug/` (all not tracked)

Because the train transform differs, the warm-up epochs 1-5 are not identical to the earlier Swin-Tiny runs.

## Validation results (best checkpoint)

The checkpoint is the epoch with the highest validation macro F1: epoch 17.

| Metric | Value |
|---|---|
| Accuracy | 0.9712 (640/659) |
| Macro precision | 0.9626 |
| Macro recall | 0.9698 |
| Macro F1 | 0.9658 |
| Validation loss | 0.1268 |

Per-class metrics:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ambulance | 1.0000 | 0.9595 | 0.9793 | 74 |
| autobus | 1.0000 | 1.0000 | 1.0000 | 92 |
| kamyun | 0.9570 | 0.9570 | 0.9570 | 93 |
| kamyunet | 0.9300 | 0.9490 | 0.9394 | 98 |
| minibus | 0.9620 | 0.9620 | 0.9620 | 79 |
| savari | 0.9899 | 0.9800 | 0.9849 | 100 |
| taxi | 1.0000 | 0.9897 | 0.9948 | 97 |
| vanet | 0.8621 | 0.9615 | 0.9091 | 26 |

Confusion matrix (rows = true class, columns = predicted class):

| true \ predicted | ambulance | autobus | kamyun | kamyunet | minibus | savari | taxi | vanet |
|---|---|---|---|---|---|---|---|---|
| ambulance | 71 | 0 | 0 | 0 | 1 | 0 | 0 | 2 |
| autobus | 0 | 92 | 0 | 0 | 0 | 0 | 0 | 0 |
| kamyun | 0 | 0 | 89 | 4 | 0 | 0 | 0 | 0 |
| kamyunet | 0 | 0 | 4 | 93 | 1 | 0 | 0 | 0 |
| minibus | 0 | 0 | 0 | 3 | 76 | 0 | 0 | 0 |
| savari | 0 | 0 | 0 | 0 | 0 | 98 | 0 | 2 |
| taxi | 0 | 0 | 0 | 0 | 1 | 0 | 96 | 0 |
| vanet | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 25 |

19 of 659 images are wrong. Main confusions: kamyunet -> kamyun 4, kamyun -> kamyunet 4,
minibus -> kamyunet 3, savari -> vanet 2, ambulance -> vanet 2. The lowest F1 values are vanet
(0.9091), kamyunet (0.9394) and kamyun (0.9570).

## Training behaviour

- Validation loss is lowest at epoch 11 (0.1033). Afterwards it stays above that value: 0.1268 at the
  best epoch 17 and 0.1800 at epoch 20.
- Train loss falls from 1.0060 (epoch 1) to 0.0192 (epoch 20); train accuracy reaches 0.9932 but never 1.0.
- After epoch 11 validation macro F1 stays between 0.9499 and 0.9658.
- At epoch 20: train macro F1 0.9934, validation macro F1 0.9546, a gap of 0.0388
  (`swin_t_ft_none`: 0.9979 vs 0.9541, gap 0.0438).
- Train and validation diverge after epoch 11, so some overfitting is present in the last epochs,
  less pronounced than in `swin_t_ft_none` (lowest validation loss 0.1514 there, 0.2426 at epoch 20).

## Comparison with swin_t_ft_none (validation)

| Metric | `swin_t_ft_none` | `swin_t_ft_aug` | Change |
|---|---|---|---|
| Best epoch | 13 | 17 | |
| Accuracy | 0.9621 (634/659) | 0.9712 (640/659) | +0.0091 (+6 images) |
| Macro F1 | 0.9582 | 0.9658 | +0.0076 |
| Validation loss at best epoch | 0.2219 | 0.1268 | -0.0951 |
| kamyun -> kamyunet | 10 | 4 | -6 |
| kamyunet -> kamyun | 2 | 4 | +2 |
| kamyun F1 | 0.9266 | 0.9570 | +0.0304 |
| kamyunet F1 | 0.9223 | 0.9394 | +0.0171 |
| vanet precision / recall / F1 | 0.8621 / 0.9615 / 0.9091 | 0.8621 / 0.9615 / 0.9091 | none |

- The kamyun / kamyunet confusions fall from 12 to 8 in total; the errors are now split evenly
  between the two directions instead of mostly kamyun -> kamyunet.
- vanet is unchanged: in both runs 2 ambulance and 2 savari images are predicted as vanet, and 1
  vanet image is predicted as savari.
- minibus -> kamyunet rises from 2 to 3; minibus F1 drops from 0.9747 to 0.9620.

## Comparison with the other fine-tuned runs (validation)

| Run | Augmentation | Accuracy | Macro F1 |
|---|---|---|---|
| `resnet224_ft_aug` (final model) | full_aug | 0.9605 | 0.9482 |
| `effnet_b0_ft_none` | none | 0.9560 | 0.9485 |
| `convnext_t_ft_none` | none | 0.9590 | 0.9538 |
| `convnext_t_ft_aug` | full_aug | 0.9590 | 0.9557 |
| `swin_t_ft_none` | none | 0.9621 | 0.9582 |
| `swin_t_ft_aug` | full_aug | 0.9712 | 0.9658 |

All rows are single runs (seed 42) on the same 659 validation images; the gaps between them are a few
images.

## Status

- `full_aug` improved Swin-Tiny over `swin_t_ft_none` on validation (+0.0076 macro F1, +0.0091 accuracy).
- No further augmentation change is proposed on the basis of this experiment alone.
- Final model selection is left for the next milestone; the final model of the project stays
  `resnet224_ft_aug` until then.

## Limitations

- Validation only (659 images), one run with seed 42; small differences are not conclusive.
- The Final Test was not run again and the Neysan evaluation was not run again.
- No label, split or data was changed.
- This result does not by itself determine the final model selection.
