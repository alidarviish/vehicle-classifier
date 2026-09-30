# EfficientNet-B0 fine-tuning with full_aug: results and visual error review

Closes the EfficientNet-B0 milestone. One-factor experiment against the fine-tuning baseline
`effnet_b0_ft_none` (`14_efficientnet_b0_ft_baseline.md`): only the augmentation changes, `none` -> `full_aug`.

## Experiment

- Run: `effnet_b0_ft_aug`
- Command: `python -m src.train --run-name effnet_b0_ft_aug --model efficientnet_b0 --effnet-mode fine_tuning --augmentation full_aug`
- Architecture: EfficientNet-B0 (`IMAGENET1K_V1`), partial fine-tuning (classifier only in epochs 1-5,
  `features[6:9]` unfrozen from epoch 6; learning rates 1e-3 / 1e-4)
- Augmentation (train only): `full_aug` = RandomResizedCrop(224, scale 0.8-1.0), RandomHorizontalFlip(0.5),
  RandomRotation(10), ColorJitter(brightness 0.2, contrast 0.2), ImageNet normalization; validation unchanged
  (Resize 224x224, no augmentation)
- Everything else as the baseline: Adam, weight decay 0, no scheduler, CrossEntropyLoss, 20 epochs, seed 42
- Validation: 659 images. Test and Neysan were not used.

## Validation results

| Metric | `effnet_b0_ft_none` (baseline) | `effnet_b0_ft_aug` | Difference |
|---|---|---|---|
| Best epoch | 19 | 19 | |
| Accuracy | 0.9560 | 0.9530 | -0.0030 (-0.30 percentage points) |
| Macro F1 | 0.9485 | 0.9466 | -0.0019 (-0.19 percentage points) |
| Macro precision | 0.9469 | 0.9425 | -0.0044 |
| Macro recall | 0.9518 | 0.9531 | +0.0013 |
| Errors (of 659) | 29 | 31 | +2 |

## Per-image comparison

The validation errors of both best checkpoints were extracted and joined by exact image path
(review packages kept outside the repository).

| Group | Images |
|---|---|
| Wrong in both runs | 23 |
| Baseline wrong -> `full_aug` correct | 6 |
| Baseline correct -> `full_aug` wrong | 8 |
| Net change | +2 errors |

The 8 new errors of `full_aug` (IDs of the `effnet_b0_ft_aug` review):

| ID | Image | True -> predicted |
|---|---|---|
| E02 | `v2/train/kamyun/198712812.jpg` | kamyun -> kamyunet |
| E06 | `v2/unclean/kamyun/216109678.jpg` | kamyun -> kamyunet |
| E07 | `v2/unclean/kamyun/216231496.jpg` | kamyun -> kamyunet |
| E10 | `v2/train/kamyun/218231967.jpg` | kamyunet -> kamyun |
| E09 | `v1/unclean/kamyun/217998174.jpg` | kamyunet -> kamyun |
| E16 | `v2/unclean/kamyunet/218970114.jpg` | kamyunet -> kamyun |
| E24 | `v2/unclean/kamyunet/216430042.jpg` | kamyunet -> minibus |
| E26 | `v2/unclean/savari/201705208.jpg` | savari -> vanet |

## Visual review of the 8 new errors

| Finding | Images |
|---|---|
| Crop-related evidence | E16, E24 |
| Augmentation-related evidence unclear | E07 |
| Not obviously augmentation-related | E02, E06, E10, E09, E26 |

- **E16, E24 (crop-related evidence).** Both images are already tightly cropped in the dataset: only the
  front of the cab is visible and no load area. The remaining cues (roof line and width of the cab, external
  mirrors) lie at the image edges, which is the part a random 0.8-1.0 crop removes during training.
- **E07 (unclear).** Box truck fully in frame; the length of the box reaches the top edge, but the full image
  is visible at evaluation.
- **E02, E06, E10, E09, E26 (not obviously augmentation-related).** The vehicle is fully in frame and the cab
  or body is visible; E10 and E09 are night images and are two of the 8 validation images whose label was
  corrected by human review (kamyun -> kamyunet); E26 is an SUV fully in frame.
- 5 of the 8 images (E06, E07, E10, E09, E16) were also errors of the feature-extraction run
  `effnet_b0_fe_none`, so they are borderline images that change between runs and are not necessarily caused
  by the augmentation.
- Among the 6 errors fixed by `full_aug` are tightly cropped images of the baseline review (for example
  `v1/train/kamyunet/218583813.jpg`, `v2/train/kamyunet/215286925.jpg`, `v2/train/vanet/203739270.jpg`).

## Decision

**`full_aug` is not changed for now.**

- Only 2 of the 8 new errors show specific crop-related evidence.
- The augmentation also fixed several tightly cropped images.
- The net difference is 2 errors on 659 validation images.
- The experiment was run once with seed 42; this is not enough to attribute the errors to RandomResizedCrop.

## Limitations

- Only the validation split was reviewed.
- The Final Test was not run again.
- The Neysan evaluation was not run again.
- No label or data was changed.

## Next architectural question

With this milestone closed, a next experiment could test a different architecture, for example ConvNeXt-Tiny,
under the same split, seed and validation protocol. This report does not assume anything about the result of
that experiment.
