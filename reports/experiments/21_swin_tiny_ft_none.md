# Swin-Tiny partial fine-tuning

Second Swin-Tiny run in the architecture comparison. One factor changes against `swin_t_fe_none`
(`20_swin_tiny_fe_none.md`): after a head-only warm-up, the last patch-merging layer, the last stage
and the final LayerNorm are fine-tuned as well. This is an architecture comparison only; it does not
change the final model of the project (`resnet224_ft_aug`).

## Setup

- Run name: `swin_t_ft_none`
- Command: `python -m src.train --run-name swin_t_ft_none --model swin_tiny --swin-mode fine_tuning`
- Code: `src/swin.py`, `src/train.py` (Swin-Tiny fine-tuning added in commit `8b034a3`)
- Model: Swin-Tiny (torchvision `swin_t`), ImageNet weights `IMAGENET1K_V1`; head `Linear(768, 8)`
- Mode: partial fine-tuning
  - epochs 1-5 (warm-up): only the head is trained (6,152 trainable parameters, 27,519,354 frozen)
  - from epoch 6: `features[6]`, `features[7]` and `norm` are unfrozen as well (15,374,264 trainable,
    12,151,242 frozen); `features[0:6]` stay frozen and in eval mode
- Parameter groups: `head` 1e-3, `features.6-7+norm` 1e-4 (both groups in the optimizer from epoch 1)
- Optimizer: Adam, weight decay 0, no scheduler; CrossEntropyLoss
- Augmentation: `none` (train and validation: Resize 224x224 -> ToTensor -> ImageNet normalization)
- 20 epochs, batch size 32, seed 42; train 2633 / validation 659 images
- Checkpoint `checkpoints/swin_t_ft_none_best.pt` (sha256
  `19d9bbfca8713409df51f89ee494208d0a883502234c94f97d81fe5cc2c6fd35`), history
  `reports/swin_t_ft_none_history.csv`, analysis `reports/analysis/swin_t_ft_none/` (all not tracked)

Epochs 1-5 of the history are identical to `swin_t_fe_none` (the same head-only warm-up).

## Validation results (best checkpoint)

The checkpoint is the epoch with the highest validation macro F1: epoch 13.

| Metric | Value |
|---|---|
| Accuracy | 0.9621 (634/659) |
| Macro precision | 0.9565 |
| Macro recall | 0.9619 |
| Macro F1 | 0.9582 |
| Validation loss | 0.2219 |

Epochs 16 and 17 reach almost the same macro F1 (0.9581 and 0.9580); epoch 13 is kept because it is
the first strictly highest value.

Per-class metrics:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ambulance | 1.0000 | 0.9595 | 0.9793 | 74 |
| autobus | 0.9891 | 0.9891 | 0.9891 | 92 |
| kamyun | 0.9762 | 0.8817 | 0.9266 | 93 |
| kamyunet | 0.8796 | 0.9694 | 0.9223 | 98 |
| minibus | 0.9747 | 0.9747 | 0.9747 | 79 |
| savari | 0.9703 | 0.9800 | 0.9751 | 100 |
| taxi | 1.0000 | 0.9794 | 0.9896 | 97 |
| vanet | 0.8621 | 0.9615 | 0.9091 | 26 |

Confusion matrix (rows = true class, columns = predicted class):

| true \ predicted | ambulance | autobus | kamyun | kamyunet | minibus | savari | taxi | vanet |
|---|---|---|---|---|---|---|---|---|
| ambulance | 71 | 0 | 0 | 0 | 0 | 1 | 0 | 2 |
| autobus | 0 | 91 | 0 | 0 | 1 | 0 | 0 | 0 |
| kamyun | 0 | 1 | 82 | 10 | 0 | 0 | 0 | 0 |
| kamyunet | 0 | 0 | 2 | 95 | 1 | 0 | 0 | 0 |
| minibus | 0 | 0 | 0 | 2 | 77 | 0 | 0 | 0 |
| savari | 0 | 0 | 0 | 0 | 0 | 98 | 0 | 2 |
| taxi | 0 | 0 | 0 | 1 | 0 | 1 | 95 | 0 |
| vanet | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 25 |

25 of 659 images are wrong; 12 of them are kamyun / kamyunet confusions (kamyun -> kamyunet 10,
kamyunet -> kamyun 2). The lowest F1 values are vanet (0.9091), kamyunet (0.9223) and kamyun
(0.9266); vanet has the lowest precision (0.8621: 2 ambulance and 2 savari images predicted as vanet).

## Training behaviour

- Validation loss is lowest at epoch 6 (0.1514), the first epoch after the unfreeze. It does not
  return to that value afterwards: 0.2219 at the best epoch 13 and 0.2426 at epoch 20.
- Train loss falls from 0.9173 (epoch 1) to 0.0065 (epoch 20); train accuracy reaches 0.9989 but never 1.0.
- From epoch 7 to 20 validation macro F1 stays between 0.9347 and 0.9582.
- At epoch 20: train macro F1 0.9979, validation macro F1 0.9541, a gap of 0.0438.
- This train / validation divergence after epoch 6 indicates overfitting during the fine-tuning epochs.

## Comparison (validation, factual)

| Run | Mode | Augmentation | Accuracy | Macro F1 |
|---|---|---|---|---|
| `resnet224_ft_aug` (final model) | partial fine-tuning | full_aug | 0.9605 | 0.9482 |
| `effnet_b0_ft_none` | partial fine-tuning | none | 0.9560 | 0.9485 |
| `convnext_t_ft_none` | partial fine-tuning | none | 0.9590 | 0.9538 |
| `convnext_t_ft_aug` | partial fine-tuning | full_aug | 0.9590 | 0.9557 |
| `swin_t_fe_none` | feature extraction | none | 0.9423 | 0.9363 |
| `swin_t_ft_none` | partial fine-tuning | none | 0.9621 | 0.9582 |

- Against `swin_t_fe_none`: +0.0219 macro F1 and +0.0198 accuracy (634 vs 621 correct).
- kamyun -> kamyunet errors: 10 here, 8 in `swin_t_fe_none`; kamyunet -> kamyun: 2 here, 7 in
  `swin_t_fe_none`. The same 10 / 2 split occurs in `convnext_t_ft_none`.
- The differences to the other fine-tuned runs are a few images on 659 and come from single runs.

## Conclusion

Unfreezing `features[6:8]` and `norm` after a 5-epoch warm-up raised Swin-Tiny from 0.9363 to 0.9582
validation macro F1 (best epoch 13), with overfitting visible after epoch 6. kamyun -> kamyunet is
still the most frequent error. This is a validation-only architecture comparison and does not change
the final model of the project.

## Limitations

- Validation only (659 images), one run with seed 42; small differences are not conclusive.
- The Final Test and the Neysan evaluation were not used in this experiment and were not run again.
- No label, split or data was changed.
