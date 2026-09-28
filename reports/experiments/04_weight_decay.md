# Experiment: optimizer control and weight decay

## Setup

Three runs that differ only in the optimizer and/or its weight decay:

| Run | Optimizer | Weight decay | Command |
|---|---|---|---|
| `baseline` | Adam (`torch.optim.Adam`) | 0 | `python -m src.train` |
| `adamw_wd0` | AdamW (`torch.optim.AdamW`) | 0 | `python -m src.train --run-name adamw_wd0 --optimizer adamw --weight-decay 0` |
| `adamw_wd1e4` | AdamW (`torch.optim.AdamW`) | 1e-4 | `python -m src.train --run-name adamw_wd1e4 --optimizer adamw --weight-decay 1e-4` |

`adamw_wd0` is an optimizer control: it changes only the optimizer (Adam -> AdamW) and keeps
weight decay at 0. Comparing `baseline` with `adamw_wd0` isolates the optimizer change;
comparing `adamw_wd0` with `adamw_wd1e4` isolates the weight decay change.

Fixed for all three runs:

- model: `BaselineCNN`, 128x128 RGB input, max pooling, dropout 0.0
- augmentation: `none` (train and validation both use `BASE_TRANSFORM`)
- learning rate 1e-3, no scheduler, CrossEntropyLoss
- epochs 20, batch size 32, seed 42; each setting was trained once
- split: `data/split_manifest.csv`, train / validation = 2633 / 659
- checkpoint selection: best validation macro F1

This is a single-seed, validation-only experiment. The test set and the Neysan images were
not used.

## Results (validation, best checkpoint)

| Run | Optimizer | Weight decay | Best epoch | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---|---|---|---|---|---|---|
| `baseline` | Adam | 0 | 18 | 0.8801 | 0.8646 | 0.8509 | 0.8561 |
| `adamw_wd0` | AdamW | 0 | 18 | 0.8801 | 0.8646 | 0.8509 | 0.8561 |
| `adamw_wd1e4` | AdamW | 1e-4 | 12 | 0.8756 | 0.8619 | 0.8566 | 0.8578 |

Differences:

| Comparison | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---|---|---|---|
| `adamw_wd0` - `baseline` (optimizer only) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `adamw_wd1e4` - `adamw_wd0` (weight decay only) | -0.0046 (577 vs 580 correct of 659) | -0.0027 | +0.0057 | +0.0017 |

Notes on the optimizer control:

- The per-epoch training metrics of `adamw_wd0` (train loss, accuracy, precision, recall, F1)
  are identical to those of `baseline` for all 20 epochs in the two history files.
- The validation metrics of `adamw_wd0` at epoch 18 are identical to the baseline metrics that
  were recalculated from its saved predictions with the current labels (macro F1 0.856094).
- The validation loss columns differ between the two history files because `baseline` was
  logged with the original folder labels and `adamw_wd0` with the current labels.

Sources:

- `adamw_wd1e4`: best checkpoint re-evaluated with `scripts/analyze_baseline.py`
  (`reports/analysis/adamw_wd1e4/`); the re-evaluated macro F1 matches the checkpoint and history.
- `adamw_wd0`: `reports/adamw_wd0_history.csv` at the best epoch (no separate analysis output).
- `baseline`: its checkpoint (epoch 18) was chosen during training with the original folder
  labels; its metrics were later recalculated from its saved predictions with the current labels
  (`reports/experiments/00_baseline.md`). The two AdamW runs chose their checkpoint with the current labels.

## `adamw_wd1e4` details

Per-class metrics (validation):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ambulance | 0.8923 | 0.7838 | 0.8345 | 74 |
| autobus | 0.9175 | 0.9674 | 0.9418 | 92 |
| kamyun | 0.7745 | 0.8495 | 0.8103 | 93 |
| kamyunet | 0.8182 | 0.7347 | 0.7742 | 98 |
| minibus | 0.9459 | 0.8861 | 0.9150 | 79 |
| savari | 0.8649 | 0.9600 | 0.9100 | 100 |
| taxi | 0.9896 | 0.9794 | 0.9845 | 97 |
| vanet | 0.6923 | 0.6923 | 0.6923 | 26 |

Lowest per-class F1: vanet 0.6923, kamyunet 0.7742, kamyun 0.8103.

Main confusions (true -> predicted):

- kamyunet -> kamyun: 17
- kamyun -> kamyunet: 8
- vanet -> savari: 6
- ambulance -> vanet: 6
- ambulance -> savari: 6

Confusion matrix (rows = true, columns = predicted; order: ambulance, autobus, kamyun,
kamyunet, minibus, savari, taxi, vanet):

```
ambulance   58   0   1   2   1   6   0   6
autobus      0  89   1   0   0   2   0   0
kamyun       2   3  79   8   0   0   1   0
kamyunet     3   2  17  72   3   1   0   0
minibus      0   1   3   5  70   0   0   0
savari       1   0   1   0   0  96   0   2
taxi         1   1   0   0   0   0  95   0
vanet        0   1   0   1   0   6   0  18
```

Training behavior:

- Lowest validation loss: 0.4376 at epoch 5.
- Validation loss at the best-F1 epoch (12): 0.7045.
- Last epoch (20): train F1 1.0000, validation F1 0.8527; train accuracy reaches 1.0 from epoch 15.
- After epoch 5, validation loss does not return to its minimum while train loss keeps falling.

## Observations

- In this run, replacing Adam with AdamW at weight decay 0 produced the same training history
  and the same validation results as the baseline.
- Adding weight decay 1e-4 to AdamW changed validation macro F1 by +0.0017 and accuracy by
  -0.0046, with higher macro recall and lower macro precision; the best epoch moved from 18 to 12.
- These differences come from one seed on one validation split. They are observations only and
  do not establish a cause or a general effect of weight decay.

## Files

- Checkpoints: `checkpoints/adamw_wd0_best.pt`, `checkpoints/adamw_wd1e4_best.pt`
  (baseline: `checkpoints/baseline_best.pt`)
- Histories: `reports/adamw_wd0_history.csv`, `reports/adamw_wd1e4_history.csv`
  (baseline: `reports/baseline_history.csv`)
- Analysis output: `reports/analysis/adamw_wd1e4/`
