# EfficientNet-B0 fine-tuning: top-stage learning rate 5e-5

One-factor experiment against the EfficientNet-B0 fine-tuning baseline `effnet_b0_ft_none`
(`14_efficientnet_b0_ft_baseline.md`). Only the learning rate of `features[6:9]` changes: 1e-4 -> 5e-5.
Validation only; Test and Neysan are not part of this experiment.

## Run

- Run name: `effnet_b0_ft_lr5e5_none`
- Command: `python -m src.train --run-name effnet_b0_ft_lr5e5_none --model efficientnet_b0 --effnet-mode fine_tuning --effnet-top-lr 5e-5`
- Checkpoint `checkpoints/effnet_b0_ft_lr5e5_none_best.pt`, history `reports/effnet_b0_ft_lr5e5_none_history.csv`,
  analysis `reports/analysis/effnet_b0_ft_lr5e5_none/` (all not tracked)

## Settings

Same as the baseline except the top-stage learning rate:

- EfficientNet-B0, ImageNet weights `IMAGENET1K_V1`, partial fine-tuning
- epochs 1-5: classifier only; from epoch 6: `features[6:9]` unfrozen as well
- learning rates: classifier 1e-3, `features[6:9]` **5e-5** (baseline 1e-4); recorded in the history
  (`lr_top` = 5e-5 in every epoch) and in the checkpoint (`param_group_lrs`)
- Adam, weight decay 0, no scheduler, CrossEntropyLoss, 20 epochs, seed 42, augmentation `none`
- Validation: 659 images

Epochs 1-5 of the history are identical to the baseline (the same classifier-only warm-up).

## Validation results

| Metric | `effnet_b0_ft_none` (top lr 1e-4) | `effnet_b0_ft_lr5e5_none` (top lr 5e-5) | Difference |
|---|---|---|---|
| Best epoch | 19 | 15 | |
| Accuracy | 0.9560 (630/659) | 0.9469 (624/659) | -0.0091 (-0.91 percentage points) |
| Macro precision | 0.9469 | 0.9339 | -0.0130 |
| Macro recall | 0.9518 | 0.9399 | -0.0119 |
| Macro F1 | 0.9485 | 0.9356 | -0.0129 |
| Lowest validation loss (epoch) | 0.1638 (11) | 0.1802 (11) | +0.0164 |

Run `effnet_b0_ft_lr5e5_none`:

- validation loss at the best epoch 15: 0.1896
- last epoch 20: train macro F1 0.9990, validation macro F1 0.9350 (gap 0.0640)
- most frequent confusions: kamyunet -> kamyun 11, ambulance -> vanet 6, kamyun -> kamyunet 5,
  vanet -> savari 2, minibus -> kamyun 2
- lowest per-class F1: vanet 0.8214, kamyun 0.9072, kamyunet 0.9158

## Conclusion

In this experiment, a top-stage learning rate of 5e-5 was **not better** than the baseline 1e-4: accuracy,
macro precision, macro recall and macro F1 are all lower, and the lowest validation loss is higher.
The baseline `effnet_b0_ft_none` stays the reference for the next EfficientNet-B0 experiments.

This is a single run per setting (seed 42) on 659 validation images; the result applies to these two
values only and is not a general statement about the best learning rate.
