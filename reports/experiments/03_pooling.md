# Experiment: pooling ablation

## Setup

This is a one-factor ablation: only the pooling type in the three conv blocks of
`BaselineCNN` changes. Both use a 2x2 kernel with stride 2.

| Run | Pooling | Command |
|---|---|---|
| `baseline` | `max` (`MaxPool2d(2)`) | `python -m src.train` |
| `pooling_average` | `average` (`AvgPool2d(2)`) | `python -m src.train --run-name pooling_average --pooling average` |

Fixed for both runs:

- model: `BaselineCNN` (same conv layers, channels, ReLU, 128x128 RGB input, classifier head)
- augmentation: `none` (train and validation both use `BASE_TRANSFORM`)
- dropout: 0.0
- optimizer: Adam, learning rate 1e-3, weight decay 0, no scheduler
- loss: CrossEntropyLoss
- epochs: 20, batch size: 32
- seed: 42; each setting was trained once
- split: `data/split_manifest.csv`, train / validation = 2633 / 659
- checkpoint selection: best validation macro F1

The test set and the Neysan images were not used in this experiment.

## Results (validation, best checkpoint)

| Run | Pooling | Best epoch | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---|---|---|---|---|---|
| `baseline` | max | 18 | 0.8801 | 0.8646 | 0.8509 | 0.8561 |
| `pooling_average` | average | 15 | 0.8558 | 0.8468 | 0.8171 | 0.8267 |

Difference (average - max):

| Metric | Difference |
|---|---|
| Accuracy | -0.0243 (564 vs 580 correct of 659) |
| Macro precision | -0.0178 |
| Macro recall | -0.0339 |
| Macro F1 | -0.0294 |

Sources:

- `pooling_average`: best checkpoint re-evaluated on validation with `scripts/analyze_baseline.py`,
  which builds the model with the pooling recorded in the checkpoint (`reports/analysis/pooling_average/`).
  The re-evaluated macro F1 matches the value stored in the checkpoint and in the history.
- `baseline`: its checkpoint (epoch 18) was chosen during training with the original folder labels;
  its metrics were later recalculated from its saved predictions with the current validation labels
  (`reports/experiments/00_baseline.md`). `pooling_average` chose its checkpoint with the current labels.

## Per-class metrics for `average` (validation)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ambulance | 0.8636 | 0.7703 | 0.8143 | 74 |
| autobus | 0.9457 | 0.9457 | 0.9457 | 92 |
| kamyun | 0.7629 | 0.7957 | 0.7789 | 93 |
| kamyunet | 0.7677 | 0.7755 | 0.7716 | 98 |
| minibus | 0.8816 | 0.8481 | 0.8645 | 79 |
| savari | 0.8333 | 0.9500 | 0.8879 | 100 |
| taxi | 0.9697 | 0.9897 | 0.9796 | 97 |
| vanet | 0.7500 | 0.4615 | 0.5714 | 26 |

## Main confusions for `average` (true -> predicted)

- kamyunet -> kamyun: 14
- kamyun -> kamyunet: 11
- vanet -> savari: 10
- ambulance -> savari: 7
- minibus -> kamyun: 6

## Training behavior (`average`)

- Lowest validation loss: 0.5667 at epoch 5.
- Best validation macro F1: epoch 15.
- Final (epoch 20) train loss: 0.041905; train F1 0.9839; validation F1 0.8112.
- After epoch 5, validation loss does not return to its minimum while train loss keeps falling;
  this train/validation divergence was observed as a sign of overfitting.

## Interpretation

In this controlled run, `average` pooling had lower validation performance than the `max`
pooling baseline on all four reported metrics. This describes this single-seed experiment on
this validation split only; it is not a general statement about max versus average pooling,
and it does not describe performance on the test set or on Neysan images.

## Files

- Checkpoint: `checkpoints/pooling_average_best.pt` (baseline: `checkpoints/baseline_best.pt`)
- History: `reports/pooling_average_history.csv` (baseline: `reports/baseline_history.csv`)
- Analysis output: `reports/analysis/pooling_average/`
