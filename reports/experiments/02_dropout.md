# Experiment: dropout ablation

## Setup

This is a one-factor ablation: dropout is the only factor changed between the three runs.
It sets the dropout probability in the classifier head of `BaselineCNN`
(`Flatten -> Dropout(p) -> Linear -> ReLU -> Dropout(p) -> Linear`).

| Run | Dropout | Command |
|---|---|---|
| `baseline` | 0.0 | `python -m src.train` |
| `dropout_03` | 0.3 | `python -m src.train --run-name dropout_03 --dropout 0.3` |
| `dropout_05` | 0.5 | `python -m src.train --run-name dropout_05 --dropout 0.5` |

Fixed for all three runs (as recorded in each checkpoint):

- augmentation: `none` (train and validation both use `BASE_TRANSFORM`)
- architecture `BaselineCNN`, input 128x128 RGB, max pooling
- batch size 32, 20 epochs, Adam, learning rate 1e-3, weight decay 0, no scheduler
- loss: CrossEntropyLoss
- seed 42; each setting was trained once
- `data/split_manifest.csv` with the current labels, train / validation = 2633 / 659
- checkpoint criterion: best validation macro F1

## Results (validation, best checkpoint of each run)

| Run | Dropout | Best epoch | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---|---|---|---|---|---|
| `baseline` | 0.0 | 18 | 0.8801 | 0.8646 | 0.8509 | 0.8561 |
| `dropout_03` | 0.3 | 20 | 0.8862 | 0.8739 | 0.8477 | 0.8562 |
| `dropout_05` | 0.5 | 17 | 0.8862 | 0.9032 | 0.8450 | 0.8592 |

Difference from the baseline:

| Run | Macro F1 | Accuracy |
|---|---|---|
| `dropout_03` | +0.0001 (0.85620 vs 0.85609) | +0.0061 (584 vs 580 correct of 659) |
| `dropout_05` | +0.0031 (0.85920 vs 0.85609) | +0.0061 (584 vs 580 correct of 659) |

Sources:

- `dropout_03` and `dropout_05`: best checkpoint re-evaluated on validation with
  `scripts/analyze_baseline.py` (`reports/analysis/dropout_03/`, `reports/analysis/dropout_05/`);
  the re-evaluated macro F1 matches the value stored in each checkpoint and in the history.
- `baseline`: its checkpoint (epoch 18) was chosen during training with the original folder labels.
  Its final metrics were later recalculated from its saved predictions with the current validation
  labels (`reports/experiments/00_baseline.md`). The two dropout runs chose their checkpoint with the
  current labels.

## Per-class metrics (precision / recall / F1, validation)

| Class | Support | Baseline (0.0) | dropout_03 (0.3) | dropout_05 (0.5) |
|---|---|---|---|---|
| ambulance | 74 | 0.9231 / 0.8108 / 0.8633 | 0.8971 / 0.8243 / 0.8592 | 0.9500 / 0.7703 / 0.8507 |
| autobus | 92 | 0.9167 / 0.9565 / 0.9362 | 0.8990 / 0.9674 / 0.9319 | 0.9468 / 0.9674 / 0.9570 |
| kamyun | 93 | 0.7921 / 0.8602 / 0.8247 | 0.9241 / 0.7849 / 0.8488 | 0.8200 / 0.8817 / 0.8497 |
| kamyunet | 98 | 0.8495 / 0.8061 / 0.8272 | 0.7739 / 0.9082 / 0.8357 | 0.8352 / 0.7755 / 0.8042 |
| minibus | 79 | 0.9103 / 0.8987 / 0.9045 | 0.9710 / 0.8481 / 0.9054 | 0.8690 / 0.9241 / 0.8957 |
| savari | 100 | 0.8649 / 0.9600 / 0.9100 | 0.8522 / 0.9800 / 0.9116 | 0.8250 / 0.9900 / 0.9000 |
| taxi | 97 | 0.9785 / 0.9381 / 0.9579 | 0.9895 / 0.9691 / 0.9792 | 0.9796 / 0.9897 / 0.9846 |
| vanet | 26 | 0.6818 / 0.5769 / 0.6250 | 0.6842 / 0.5000 / 0.5778 | 1.0000 / 0.4615 / 0.6316 |

## Confusion matrices (validation; rows = true, columns = predicted)

Column order: ambulance, autobus, kamyun, kamyunet, minibus, savari, taxi, vanet.

Baseline (dropout 0.0):

```
ambulance   60   0   0   3   1   5   0   5
autobus      0  88   1   0   1   1   1   0
kamyun       2   4  80   7   0   0   0   0
kamyunet     1   1  14  79   3   0   0   0
minibus      0   1   5   1  71   0   1   0
savari       0   0   0   0   2  96   0   2
taxi         0   1   1   2   0   2  91   0
vanet        2   1   0   1   0   7   0  15
```

dropout_03 (dropout 0.3):

```
ambulance   61   0   1   3   1   5   0   3
autobus      0  89   1   0   0   1   1   0
kamyun       3   4  73  12   1   0   0   0
kamyunet     3   1   4  89   0   0   0   1
minibus      0   3   0   9  67   0   0   0
savari       0   1   0   0   0  98   0   1
taxi         0   1   0   0   0   1  94   1
vanet        1   0   0   2   0  10   0  13
```

dropout_05 (dropout 0.5):

```
ambulance   57   0   1   5   1  10   0   0
autobus      0  89   1   0   1   0   1   0
kamyun       0   1  82   8   1   0   1   0
kamyunet     1   2  12  76   7   0   0   0
minibus      0   2   3   1  73   0   0   0
savari       0   0   1   0   0  99   0   0
taxi         0   0   0   0   1   0  96   0
vanet        2   0   0   1   0  11   0  12
```

Most frequent confusions (true -> predicted):

| Baseline (0.0) | dropout_03 (0.3) | dropout_05 (0.5) |
|---|---|---|
| kamyunet -> kamyun: 14 | kamyun -> kamyunet: 12 | kamyunet -> kamyun: 12 |
| vanet -> savari: 7 | vanet -> savari: 10 | vanet -> savari: 11 |
| kamyun -> kamyunet: 7 | minibus -> kamyunet: 9 | ambulance -> savari: 10 |
| minibus -> kamyun: 5 | ambulance -> savari: 5 | kamyun -> kamyunet: 8 |
| ambulance -> vanet: 5 | kamyunet -> kamyun: 4 | kamyunet -> minibus: 7 |

## Observed behavior

From the training histories:

| | Baseline (0.0) | dropout_03 (0.3) | dropout_05 (0.5) |
|---|---|---|---|
| Lowest validation loss (epoch) | 0.4921 (5) | 0.3865 (10) | 0.4153 (8) |
| Validation loss at best epoch | 0.7991 | 0.6247 | 0.4575 |
| Train loss at epoch 20 | 0.0001 | 0.0582 | 0.1328 |
| Train accuracy reaches 1.0 | from epoch 15 | no | no |
| Train F1 - val F1 at epoch 20 | 0.1587 | 0.1228 | 0.1033 |

- With higher dropout, train loss at the end of training is higher and the gap between train
  and validation metrics is smaller.
- Validation macro F1 moves from epoch to epoch in all runs (e.g. `dropout_03`: 0.8161 to 0.8562
  over epochs 16-20; `dropout_05`: 0.8364 to 0.8592 over epochs 15-20). The macro F1 differences
  from the baseline (+0.0001, +0.0031) are smaller than this epoch-to-epoch variation.
- For `dropout_03` the highest validation macro F1 is at epoch 20, the last epoch of the budget.
- Both dropout runs have higher macro precision and slightly lower macro recall than the baseline.
- vanet: F1 0.6250 (baseline), 0.5778 (0.3), 0.6316 (0.5); recall 0.5769, 0.5000, 0.4615;
  vanet -> savari: 7, 10, 11. In `dropout_05` every image predicted as vanet is correct
  (precision 1.0000), with 12 of 26 vanet images recognised.
- kamyun / kamyunet: F1 0.8247 / 0.8272 (baseline), 0.8488 / 0.8357 (0.3), 0.8497 / 0.8042 (0.5).
  The main direction of the kamyun/kamyunet confusion differs between runs
  (baseline and 0.5: kamyunet -> kamyun is larger; 0.3: kamyun -> kamyunet is larger).
- ambulance -> savari: 5 (baseline), 5 (0.3), 10 (0.5).

These are observations only; this experiment does not establish their cause.

## Scope and limitations

- Single training run per setting, seed 42 only.
- The baseline checkpoint was selected with the original folder labels; its final metrics were
  recalculated afterwards from saved predictions with the current labels. The dropout runs
  selected their checkpoint with the current labels.
- Test set: not evaluated.
- Neysan images: not evaluated.
- No data, label or split was changed for this experiment.

## Files

- Checkpoints: `checkpoints/dropout_03_best.pt`, `checkpoints/dropout_05_best.pt`
  (baseline: `checkpoints/baseline_best.pt`)
- Histories: `reports/dropout_03_history.csv`, `reports/dropout_05_history.csv`
  (baseline: `reports/baseline_history.csv`)
- Analysis output: `reports/analysis/dropout_03/`, `reports/analysis/dropout_05/`
  (baseline, current labels: `reports/analysis/baseline_eval_final_labels.txt`)
