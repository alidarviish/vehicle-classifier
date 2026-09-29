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

## Output scores: softmax (CE) vs sigmoid (BCE)

- **CE.** The 8 logits go through one softmax. The 8 scores are coupled: raising one lowers the
  others, and they always sum to 1. They form one distribution over the 8 mutually exclusive
  classes.
- **BCE.** `BCEWithLogitsLoss` with one-hot targets treats the task as 8 independent yes/no
  questions ("is it class k?"). Each score is `sigmoid(logit_k)` on its own; no term ties the 8
  scores together.
  - For one image, several scores can be high, or all can be low.
  - Their sum is not constrained to 1 and in general is not 1.
- **How BCE scores should be read.** They are per-class scores, not a probability distribution
  over the 8 classes. They should not be read like softmax outputs, and a BCE "confidence" (the
  highest sigmoid score) is not on the same scale as a CE confidence (the highest softmax output).
- **What this means for the project.** This is why the prediction output uses the selected CE model
  with softmax, and why BCE scores are kept to this comparison. The `needs_review` threshold
  (0.90) is defined on the softmax confidence of the final CE model (`11_needs_review_threshold.md`);
  it does not transfer to BCE scores.

## Confidence behaviour

No per-image scores were saved for `loss_ce` or `loss_bce`. The analysis outputs in
`reports/analysis/loss_ce/` and `reports/analysis/loss_bce/` contain metrics and confusion matrices
only, and no separate confidence audit was run for this experiment. A numerical comparison of
confidence (for example score distributions or how many predictions fall below a threshold) is
therefore not reported here.

The comparison above is qualitative and follows from how the two losses define their outputs. The
available results compare only metrics based on the argmax prediction (see Results); they do not
show how confident either model is.

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
