# Experiment: Cross-Entropy vs Binary Cross-Entropy

Status: completed; validation results recorded.

## Goal

Compare Cross-Entropy (CE) and Binary Cross-Entropy (BCE) as the training loss under the same
standard settings. Only the loss function changes.

## Setup

| Run | Loss | Command |
|---|---|---|
| `loss_ce` | `CrossEntropyLoss` | `python -m src.train --run-name loss_ce --loss ce` |
| `loss_bce` | `BCEWithLogitsLoss` | `python -m src.train --run-name loss_bce --loss bce` |

Fixed for both runs (as recorded in each checkpoint): `BaselineCNN`, 128x128 RGB, max pooling,
dropout 0.0, augmentation `none`, Adam, learning rate 1e-3, weight decay 0, no scheduler,
20 epochs, batch size 32, batch mode `standard`, seed 42, `data/split_manifest.csv`
(train / validation = 2633 / 659), checkpoint selection by best validation macro F1.
Each setting was trained once. This is a validation-only experiment; the test set and the
Neysan images were not used.

BCE method (`src/train.py`):

- The model outputs the same 8 logits in both runs.
- Targets are one-hot float vectors (`F.one_hot(labels, 8).float()`), passed to `BCEWithLogitsLoss`.
- No manual sigmoid; `BCEWithLogitsLoss` applies it inside the loss.
- Predictions and metrics use `argmax(logits, dim=1)` in both runs.

`loss_ce` reproduces the baseline: its history is byte-identical to that of the control run
`adamw_wd0` (see `04_weight_decay.md`), and its metrics equal the baseline with the current labels.

## Results (validation, best checkpoint)

| Run | Best epoch | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---|---|---|---|---|
| `loss_ce` | 18 | 0.8801 | 0.8646 | 0.8509 | 0.8561 |
| `loss_bce` | 19 | 0.8816 | 0.8747 | 0.8482 | 0.8568 |

Per-class recall:

| Class | CE recall | BCE recall |
|---|---:|---:|
| ambulance | 0.8108 | 0.7973 |
| autobus | 0.9565 | 0.9783 |
| kamyun | 0.8602 | 0.8817 |
| kamyunet | 0.8061 | 0.7551 |
| minibus | 0.8987 | 0.8861 |
| savari | 0.9600 | 0.9800 |
| taxi | 0.9381 | 0.9691 |
| vanet | 0.5769 | 0.5385 |

## Observations

- Aggregate accuracy and macro F1 are very close between the two runs.
- BCE recorded slightly higher accuracy (+0.0015) and macro F1 (+0.0007), and slightly lower
  macro recall (-0.0027).
- The per-class changes are not uniform: with BCE, recall is higher for autobus, kamyun, savari
  and taxi, and lower for ambulance, kamyunet, minibus and vanet.
- vanet has the lowest recall in both runs.
- Both runs show train/validation divergence: validation loss is lowest at epoch 5 and does not
  return to that minimum while train loss keeps falling; train accuracy first reaches 1.0 at epoch 15
  in both runs.
- Validation loss values are not compared between the two runs: CE and BCE losses are on
  different scales (BCE averages over 8 binary outputs per image).

## Conclusion

On this validation split, CE and BCE gave very similar aggregate results, with small class-level
differences in both directions. The differences come from one run per loss with seed 42 and are
not interpreted as a causal effect or as a result that generalizes beyond this validation split.
This experiment used validation only; the test set and the Neysan images were not used, and no
Test result is reported for it.

## Files

- Checkpoints: `checkpoints/loss_ce_best.pt`, `checkpoints/loss_bce_best.pt`
- Histories: `reports/loss_ce_history.csv`, `reports/loss_bce_history.csv`
- Analysis output: `reports/analysis/loss_ce/`, `reports/analysis/loss_bce/`
