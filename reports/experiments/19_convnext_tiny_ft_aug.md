# ConvNeXt-Tiny partial fine-tuning with full_aug

Third ConvNeXt-Tiny run in the architecture comparison. One factor changes against
`convnext_t_ft_none` (`18_convnext_tiny_ft_none.md`): the train augmentation, `none` -> `full_aug`.
This is an architecture comparison only; it does not change the final model of the project
(`resnet224_ft_aug`).

## Setup

- Run name: `convnext_t_ft_aug`
- Command: `python -m src.train --run-name convnext_t_ft_aug --model convnext_tiny --convnext-mode fine_tuning --augmentation full_aug`
- Code: `src/convnext.py`, `src/train.py` (ConvNeXt fine-tuning added in commit `cbc0eec`)
- Model: ConvNeXt-Tiny, ImageNet weights `IMAGENET1K_V1`; classifier
  `LayerNorm2d(768) -> Flatten -> Linear(768, 8)`
- Mode: partial fine-tuning
  - epochs 1-5 (warm-up): only the classifier is trained (7,688 trainable parameters)
  - from epoch 6: `features[6:8]` are unfrozen as well (15,478,280 trainable parameters);
    `features[0:6]` stay frozen and in eval mode
- Learning rates: classifier 1e-3, `features[6:8]` 1e-4
- Optimizer: Adam, weight decay 0, no scheduler; CrossEntropyLoss
- Augmentation (train only): `full_aug` = RandomResizedCrop(224, scale 0.8-1.0), RandomHorizontalFlip(0.5),
  RandomRotation(10), ColorJitter(brightness 0.2, contrast 0.2), ImageNet normalization; validation
  unchanged (Resize 224x224 -> ToTensor -> ImageNet normalization)
- 20 epochs, batch size 32, seed 42; train 2633 / validation 659 images
- Checkpoint `checkpoints/convnext_t_ft_aug_best.pt` (sha256
  `b6dc7aa67d5436d14136dbf57fac151b4b62eabba1d09c90095214fc9de34856`), history
  `reports/convnext_t_ft_aug_history.csv`, analysis `reports/analysis/convnext_t_ft_aug/`
  (all not tracked)

## Validation results (best checkpoint)

The checkpoint is the epoch with the highest validation macro F1: epoch 11.

| Metric | Value |
|---|---|
| Accuracy | 0.9590 (632/659) |
| Macro precision | 0.9559 |
| Macro recall | 0.9557 |
| Macro F1 | 0.9557 |

Per-class metrics:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ambulance | 0.9861 | 0.9595 | 0.9726 | 74 |
| autobus | 1.0000 | 0.9891 | 0.9945 | 92 |
| kamyun | 0.9341 | 0.9140 | 0.9239 | 93 |
| kamyunet | 0.9082 | 0.9082 | 0.9082 | 98 |
| minibus | 0.9157 | 0.9620 | 0.9383 | 79 |
| savari | 0.9804 | 1.0000 | 0.9901 | 100 |
| taxi | 1.0000 | 0.9897 | 0.9948 | 97 |
| vanet | 0.9231 | 0.9231 | 0.9231 | 26 |

Confusion matrix (rows = true class, columns = predicted class):

| true \ predicted | ambulance | autobus | kamyun | kamyunet | minibus | savari | taxi | vanet |
|---|---|---|---|---|---|---|---|---|
| ambulance | 71 | 0 | 0 | 0 | 1 | 0 | 0 | 2 |
| autobus | 0 | 91 | 0 | 0 | 1 | 0 | 0 | 0 |
| kamyun | 0 | 0 | 85 | 7 | 1 | 0 | 0 | 0 |
| kamyunet | 0 | 0 | 6 | 89 | 3 | 0 | 0 | 0 |
| minibus | 1 | 0 | 0 | 2 | 76 | 0 | 0 | 0 |
| savari | 0 | 0 | 0 | 0 | 0 | 100 | 0 | 0 |
| taxi | 0 | 0 | 0 | 0 | 1 | 0 | 96 | 0 |
| vanet | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 24 |

Main confusions: kamyun -> kamyunet 7; kamyunet -> kamyun 6; kamyunet -> minibus 3; vanet -> savari 2,
minibus -> kamyunet 2.

## Training behaviour

- The lowest validation loss is 0.1480 at epoch 10; it is 0.1647 at the best epoch 11 and 0.2496 at
  epoch 20, while train loss keeps falling.
- Train accuracy never reaches 1.0.
- At epoch 20: train macro F1 0.9990, validation macro F1 0.9431, a gap of 0.0559.
- The train/validation divergence is consistent with overfitting in the later epochs.

## Comparison (validation)

| Run | Accuracy | Macro F1 |
|---|---|---|
| `convnext_t_fe_none` (feature extraction, no augmentation) | 0.9393 | 0.9358 |
| `convnext_t_ft_none` (partial fine-tuning, no augmentation) | 0.9590 | 0.9538 |
| `convnext_t_ft_aug` (partial fine-tuning, `full_aug`) | 0.9590 | 0.9557 |

- Against `convnext_t_ft_none`: +0.0019 macro F1, with the same best validation accuracy (0.9590).
  The difference is small: both runs have 27 errors on the 659 validation images.
- Against `convnext_t_fe_none`: +0.0199 macro F1.

## Conclusion

With `full_aug`, ConvNeXt-Tiny partial fine-tuning reached validation macro F1 0.9557 at epoch 11. The gain
over the same setup without augmentation is small (+0.0019, same accuracy and the same number of errors),
so this validation comparison does not show a clear effect of the augmentation. The kamyun / kamyunet
boundary remains the main source of errors, and signs of overfitting remain in the later epochs. The
result is a validation-only architecture comparison and does not change the final model of the project.

## Limitations

- Validation only (659 images), one run with seed 42; small differences are not conclusive.
- The Final Test and the Neysan evaluation were not used in this experiment. The Test must not be run
  again for these comparison experiments.
- No label, split or data was changed.
