# Experiment: learning-rate scheduler

## Setup

One-factor ablation: only the learning-rate schedule changes.

| Run | Scheduler | Settings | Command |
|---|---|---|---|
| `baseline` | none (fixed lr 1e-3) | - | `python -m src.train` |
| `scheduler_step` | `StepLR` | step_size 8, gamma 0.5 | `python -m src.train --run-name scheduler_step --scheduler step` |
| `scheduler_plateau` | `ReduceLROnPlateau` on validation loss | mode min, factor 0.5, patience 3 | `python -m src.train --run-name scheduler_plateau --scheduler plateau` |

Fixed for all three runs (as recorded in each checkpoint): `BaselineCNN`, 128x128 RGB, max pooling,
dropout 0.0, augmentation `none`, Adam, initial learning rate 1e-3, weight decay 0,
CrossEntropyLoss, 20 epochs, batch size 32, seed 42, `data/split_manifest.csv`
(train / validation = 2633 / 659), checkpoint selection by best validation macro F1.
Each setting was trained once. This is a validation-only experiment; the test set and the
Neysan images were not used.

## The scheduler was applied

The `lr` column of each history shows the learning rate used in each epoch:

| Epochs | `scheduler_step` | `scheduler_plateau` |
|---|---|---|
| 1-8 | 1e-3 | 1e-3 |
| 9 | 5e-4 | 1e-3 |
| 10-13 | 5e-4 | 5e-4 |
| 14-16 | 5e-4 | 2.5e-4 |
| 17 | 2.5e-4 | 2.5e-4 |
| 18-20 | 2.5e-4 | 1.25e-4 |

- `scheduler_step`: the rate halves every 8 epochs (epochs 9 and 17).
- `scheduler_plateau`: the lowest validation loss is at epoch 5 (0.4400); after no lower value
  in the following epochs, the rate halves at epochs 10, 14 and 18.
- Both checkpoints record the scheduler and its settings (`scheduler`, `scheduler_params`).

## Results (validation, best checkpoint)

| Run | Best epoch | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---|---|---|---|---|
| `baseline` | 18 | 0.8801 | 0.8646 | 0.8509 | 0.8561 |
| `scheduler_step` | 5 | 0.8756 | 0.8548 | 0.8461 | 0.8494 |
| `scheduler_plateau` | 5 | 0.8756 | 0.8548 | 0.8461 | 0.8494 |

Difference from the baseline (both scheduler runs):

| Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---|---|---|
| -0.0046 (577 vs 580 correct of 659) | -0.0098 | -0.0049 | -0.0067 |

## Why both scheduler runs give the same result

- The best validation macro F1 in both scheduler runs is at epoch 5, before either scheduler
  changed the learning rate.
- Until the first rate change, all runs follow the same training: the full history rows of
  `scheduler_step` are identical to those of the fixed-rate control run `adamw_wd0` for epochs 1-8,
  and those of `scheduler_plateau` for epochs 1-9 (`adamw_wd0` reproduced the baseline exactly,
  see `04_weight_decay.md`). The train metrics of both runs also equal the baseline history for epochs 1-8.
- The saved weights in `scheduler_step_best.pt` and `scheduler_plateau_best.pt` are byte-identical.
- The selected checkpoint of both runs is therefore the epoch-5 model of the same training path
  that the baseline also followed. After the rate was reduced, neither run exceeded the epoch-5
  macro F1 (0.8494); the fixed-rate baseline reached 0.8561 at epoch 18.

## Behavior after the rate change

| | `baseline` (fixed) | `scheduler_step` | `scheduler_plateau` |
|---|---|---|---|
| Lowest validation loss (epoch) | 0.4400 (5)* | 0.4400 (5) | 0.4400 (5) |
| Validation macro F1 over epochs 9-20 | - | 0.8378 - 0.8470 | 0.8014 - 0.8439 |
| Validation macro F1 at epoch 20 | 0.8518* | 0.8390 | 0.8314 |
| Validation loss at epoch 20 | - | 0.7720 | 0.7497 |
| Train loss at epoch 20 | 0.0001 | 0.000246 | 0.000357 |
| Train accuracy first reaches 1.0 | epoch 15 | epoch 10 | epoch 11 |

\* With the current validation labels, from the `adamw_wd0` control run, whose history matches the baseline.

- In both scheduler runs, validation loss rises after epoch 5 while train loss keeps falling and
  train accuracy reaches 1.0; this train/validation divergence is a sign of overfitting.
- In `scheduler_plateau`, validation macro F1 is 0.8314 in every epoch from 14 to 20.

## Validation details of the selected checkpoint (epoch 5)

Source: `reports/analysis/scheduler_step/`. `scheduler_plateau` was not analysed separately;
its checkpoint weights are byte-identical to `scheduler_step`, so its predictions are the same.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ambulance | 0.9062 | 0.7838 | 0.8406 | 74 |
| autobus | 0.9462 | 0.9565 | 0.9514 | 92 |
| kamyun | 0.8081 | 0.8602 | 0.8333 | 93 |
| kamyunet | 0.8125 | 0.7959 | 0.8041 | 98 |
| minibus | 0.8861 | 0.8861 | 0.8861 | 79 |
| savari | 0.8545 | 0.9400 | 0.8952 | 100 |
| taxi | 1.0000 | 0.9691 | 0.9843 | 97 |
| vanet | 0.6250 | 0.5769 | 0.6000 | 26 |

Main confusions (true -> predicted): kamyunet -> kamyun 11, vanet -> savari 8,
kamyun -> kamyunet 7, kamyunet -> minibus 6, ambulance -> savari 6.

## Scope and limitations

- Single run per setting, seed 42; validation only.
- The observations above describe these runs only and do not establish a general effect of
  learning-rate scheduling.
- The baseline checkpoint was chosen with the original folder labels and its metrics were later
  recalculated with the current labels; both scheduler runs chose their checkpoint with the current labels.
- Test set and Neysan images: not used.

## Files

- Checkpoints: `checkpoints/scheduler_step_best.pt`, `checkpoints/scheduler_plateau_best.pt`
- Histories: `reports/scheduler_step_history.csv`, `reports/scheduler_plateau_history.csv`
- Analysis output: `reports/analysis/scheduler_step/`
