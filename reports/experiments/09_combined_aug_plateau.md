# Experiment: combined experiments (C1, C2)

Two combination runs on `BaselineCNN`. The goal of each is explicitly to test a combination,
not a single factor. Both use the same split, labels, seed and 20-epoch budget as the
one-factor experiments.

| Run | Label | Augmentation | Dropout | Optimizer / weight decay | Scheduler |
|---|---|---|---|---|---|
| `combo_aug_plateau` | C1 | `full_aug` | 0.0 | Adam / 0 | `ReduceLROnPlateau` |
| `combo_aug_do05_wd1e4_plateau` | C2 | `full_aug` | 0.5 | AdamW / 1e-4 | `ReduceLROnPlateau` |

C2 = C1 + dropout 0.5 + AdamW with weight decay 1e-4. The two added factors are changed
together.

## Part A. C1 `combo_aug_plateau`

### 1. Setup

Combination experiment: the goal is explicitly to test a combination. It combines the
augmentation of `aug_full` with the `ReduceLROnPlateau` scheduler of `scheduler_plateau`.
All other settings match the baseline.

| Run | Augmentation | Scheduler | Command |
|---|---|---|---|
| `combo_aug_plateau` | `full_aug` | `ReduceLROnPlateau` on validation loss (mode min, factor 0.5, patience 3) | `python -m src.train --run-name combo_aug_plateau --augmentation full_aug --scheduler plateau` |

Recorded in the checkpoint:

- `BaselineCNN`, 128x128 RGB, normalization mean/std 0.5
- `full_aug`: random resized crop scale 0.8-1.0, horizontal flip p 0.5, rotation 10 degrees,
  brightness 0.2, contrast 0.2
- max pooling, dropout 0.0
- Adam, initial learning rate 1e-3, weight decay 0
- CrossEntropyLoss, batch size 32 (standard batches, no train subset), seed 42
- train 2633 images

Also fixed for this run:

- 20 epochs
- `data/split_manifest.csv` with the current labels (train / validation = 2633 / 659)
- checkpoint selection by best validation macro F1

Trained once. This is a validation-only experiment; the test set and the Neysan images were
not used.

### 2. Result (validation, best checkpoint)

| Metric | Value |
|---|---|
| Best epoch | 18 |
| Accuracy | 0.8983 |
| Macro precision | 0.8875 |
| Macro recall | 0.8844 |
| Macro F1 | 0.8853 |
| Validation loss at best epoch | 0.3804 |

The checkpoint stores epoch 18 and validation F1 0.8853. Re-evaluating it on the 659
validation images gives the same values.

### 3. Per-class metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ambulance | 0.9394 | 0.8378 | 0.8857 | 74 |
| autobus | 0.9545 | 0.9130 | 0.9333 | 92 |
| kamyun | 0.8495 | 0.8495 | 0.8495 | 93 |
| kamyunet | 0.7830 | 0.8469 | 0.8137 | 98 |
| minibus | 0.9103 | 0.8987 | 0.9045 | 79 |
| savari | 0.9327 | 0.9700 | 0.9510 | 100 |
| taxi | 0.9897 | 0.9897 | 0.9897 | 97 |
| vanet | 0.7407 | 0.7692 | 0.7547 | 26 |

### 4. Confusion matrix

Rows = true class, columns = predicted class (validation, 659 images, 67 errors):

| true \ predicted | ambulance | autobus | kamyun | kamyunet | minibus | savari | taxi | vanet |
|---|---|---|---|---|---|---|---|---|
| ambulance | 62 | 1 | 0 | 2 | 1 | 2 | 0 | 6 |
| autobus | 0 | 84 | 1 | 3 | 2 | 1 | 1 | 0 |
| kamyun | 0 | 3 | 79 | 10 | 1 | 0 | 0 | 0 |
| kamyunet | 4 | 0 | 10 | 83 | 1 | 0 | 0 | 0 |
| minibus | 0 | 0 | 2 | 6 | 71 | 0 | 0 | 0 |
| savari | 0 | 0 | 1 | 0 | 1 | 97 | 0 | 1 |
| taxi | 0 | 0 | 0 | 1 | 0 | 0 | 96 | 0 |
| vanet | 0 | 0 | 0 | 1 | 1 | 4 | 0 | 20 |

Row-normalized reading (each count divided by the true-class support):

- The diagonal equals per-class recall. It ranges from 0.7692 (vanet) to 0.9897 (taxi).
- The largest off-diagonal rates are:
  - vanet -> savari 0.1538 (4/26)
  - kamyun -> kamyunet 0.1075 (10/93)
  - kamyunet -> kamyun 0.1020 (10/98)
  - ambulance -> vanet 0.0811 (6/74)
  - minibus -> kamyunet 0.0759 (6/79)
- vanet has the highest rate for a single error direction despite only 4 errors, because its
  support is 26.

### 5. Pair confusion

Most frequent confusions (true -> predicted: count):

- kamyunet -> kamyun: 10
- kamyun -> kamyunet: 10
- minibus -> kamyunet: 6
- ambulance -> vanet: 6
- vanet -> savari: 4

Each pair's score is the sum of the two row-normalized rates:

| Pair | Counts (a->b, b->a) | Pair score |
|---|---|---|
| kamyun / kamyunet | 10, 10 | 0.2096 |
| savari / vanet | 1, 4 | 0.1638 |
| kamyunet / minibus | 1, 6 | 0.0862 |
| ambulance / vanet | 6, 0 | 0.0811 |
| ambulance / kamyunet | 2, 4 | 0.0678 |

- kamyun / kamyunet account for 20 of the 67 validation errors (29.9%).
- This report makes no merge or relabel decision about these two classes.

### 6. Lowest precision, recall and F1

- Lowest recall: vanet, 0.7692 (20 of 26 correct; 4 predicted as savari).
- Lowest precision: vanet, 0.7407 (20 of 27 predictions correct; 6 of the 7 wrong ones are
  true ambulance).
- Lowest F1: vanet, 0.7547. Next lowest: kamyunet 0.8137, kamyun 0.8495.
- vanet has the smallest support (26), so each vanet image changes its recall by about 0.038.

### 7. Training behavior

- **Learning rate.** The `lr` column shows 1e-3 for epochs 1-13, 5e-4 for epochs 14-18 and
  2.5e-4 for epochs 19-20.
  - The scheduler halved the rate after epoch 13, after four epochs without a new lowest
    validation loss since epoch 9 (0.3794).
  - It halved it again after epoch 18, after four epochs without a new lowest since
    epoch 14 (0.3592).
  - Replaying the scheduler rule on the logged validation losses gives the same rates.
- **Match with `aug_full`.** Epochs 1-13 of the history are byte-identical to `aug_full`: same
  seed and settings until the first learning-rate change. The two runs differ only from
  epoch 14 on.
- **Where the best epoch falls.** The best epoch (18) is at learning rate 5e-4, after the first
  reduction. In `scheduler_plateau` the best epoch (5) came before any reduction.
- **Validation loss.** Lowest 0.3592 at epoch 14; 0.3804 at the best-F1 epoch 18; 0.4108 at
  epoch 20.
- **Train loss.** 1.5956 at epoch 1, 0.1092 at epoch 20. Train accuracy never reaches 1.0.
- **Last epoch.** Train F1 0.9521 vs validation F1 0.8712, a gap of 0.0809.
- **Overfitting.** After epoch 14 the validation loss does not return to its minimum while the
  train loss keeps falling, and validation macro F1 in epochs 15-20 is between 0.8557 and
  0.8853. This is a sign of mild overfitting in the last epochs.
- **Compared with the baseline.** The divergence is much smaller than the baseline's: its
  validation loss was 0.7340 at epoch 20 and the gap 0.1482.

### 8. Comparison with the baseline and `aug_full`

Same split, same current validation labels, best checkpoint of each run:

| Run | Best epoch | Accuracy | Macro precision | Macro recall | Macro F1 | Lowest val loss (epoch) | Train-val F1 gap @20 |
|---|---|---|---|---|---|---|---|
| baseline (current labels) | 18 | 0.8801 | 0.8646 | 0.8509 | 0.8561 | 0.4400 (5) | 0.1482 |
| `aug_full` | 14 | 0.8983 | 0.8936 | 0.8735 | 0.8816 | 0.3682 (14) | 0.0468 |
| `combo_aug_plateau` | 18 | 0.8983 | 0.8875 | 0.8844 | 0.8853 | 0.3592 (14) | 0.0809 |

- Macro F1: C1 vs baseline +0.0292; C1 vs `aug_full` +0.0037.
- Accuracy equals `aug_full`: 592 of 659 correct in both.
- Macro recall is higher than `aug_full` (+0.0109); macro precision is lower (-0.0061).

Per-class F1 (baseline values from `01_augmentation.md`, recalculated with the current
validation labels):

| Class | Baseline | `aug_full` | `combo_aug_plateau` |
|---|---|---|---|
| ambulance | 0.8633 | 0.8511 | 0.8857 |
| autobus | 0.9362 | 0.9385 | 0.9333 |
| kamyun | 0.8247 | 0.8865 | 0.8495 |
| kamyunet | 0.8272 | 0.8057 | 0.8137 |
| minibus | 0.9045 | 0.9114 | 0.9045 |
| savari | 0.9100 | 0.9412 | 0.9510 |
| taxi | 0.9579 | 0.9948 | 0.9897 |
| vanet | 0.6250 | 0.7234 | 0.7547 |

Against `aug_full`, per class:

- **vanet:** recall 0.6538 -> 0.7692, precision 0.8095 -> 0.7407.
- **kamyun:** F1 is lower (0.8865 -> 0.8495). kamyun <-> kamyunet errors rise from 8 + 8 to
  10 + 10.
- **ambulance -> vanet:** rises from 4 to 6.

### 9. Summary

- **What C1 achieved.** Combining `full_aug` with `ReduceLROnPlateau` gave validation macro F1
  0.8853 at epoch 18. This is the first run in this project in which the scheduler changed the
  learning rate before the selected epoch.
- **Against the baseline (+0.0292).** The gain comes mainly from augmentation, which the
  combination shares with `aug_full`.
- **Against `aug_full` (+0.0037).** The difference is small: accuracy is identical, and some
  classes improve while others get worse (vanet and ambulance up, kamyun down). From a single
  run on 659 validation images, this difference cannot be separated from run-to-run variation.
- **Remaining weaknesses.** vanet is still the weakest class, and kamyun / kamyunet is still
  the most confused pair.

## Part B. C2 `combo_aug_do05_wd1e4_plateau`

### 1. Setup

| Run | Command |
|---|---|
| `combo_aug_do05_wd1e4_plateau` | `python -m src.train --run-name combo_aug_do05_wd1e4_plateau --augmentation full_aug --scheduler plateau --dropout 0.5 --optimizer adamw --weight-decay 1e-4` |

Recorded in the checkpoint:

- `BaselineCNN`, 128x128 RGB, normalization mean/std 0.5
- `full_aug` (same parameters as C1)
- max pooling, dropout 0.5
- AdamW, initial learning rate 1e-3, weight decay 1e-4
- `ReduceLROnPlateau` on validation loss (mode min, factor 0.5, patience 3)
- CrossEntropyLoss, batch size 32 (standard batches, no train subset), seed 42
- train 2633 images

Also fixed for this run:

- 20 epochs
- `data/split_manifest.csv` with the current labels (train / validation = 2633 / 659)
- checkpoint selection by best validation macro F1

Trained once. The test set and the Neysan images were not used.

### 2. Result (validation, best checkpoint)

| Metric | Value |
|---|---|
| Best epoch | 14 |
| Accuracy | 0.8847 |
| Macro precision | 0.8682 |
| Macro recall | 0.8552 |
| Macro F1 | 0.8590 |
| Validation loss at best epoch | 0.3836 |

The checkpoint stores epoch 14 and validation F1 0.8590. Re-evaluating it on the 659
validation images gives the same values.

### 3. Per-class metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ambulance | 0.9649 | 0.7432 | 0.8397 | 74 |
| autobus | 0.9278 | 0.9783 | 0.9524 | 92 |
| kamyun | 0.8602 | 0.8602 | 0.8602 | 93 |
| kamyunet | 0.7455 | 0.8367 | 0.7885 | 98 |
| minibus | 0.9437 | 0.8481 | 0.8933 | 79 |
| savari | 0.8981 | 0.9700 | 0.9327 | 100 |
| taxi | 0.9897 | 0.9897 | 0.9897 | 97 |
| vanet | 0.6154 | 0.6154 | 0.6154 | 26 |

### 4. Confusion matrix

Rows = true class, columns = predicted class (validation, 659 images, 76 errors):

| true \ predicted | ambulance | autobus | kamyun | kamyunet | minibus | savari | taxi | vanet |
|---|---|---|---|---|---|---|---|---|
| ambulance | 55 | 0 | 0 | 7 | 1 | 4 | 0 | 7 |
| autobus | 0 | 90 | 1 | 0 | 1 | 0 | 0 | 0 |
| kamyun | 0 | 4 | 80 | 8 | 0 | 0 | 1 | 0 |
| kamyunet | 2 | 1 | 11 | 82 | 2 | 0 | 0 | 0 |
| minibus | 0 | 1 | 1 | 10 | 67 | 0 | 0 | 0 |
| savari | 0 | 0 | 0 | 0 | 0 | 97 | 0 | 3 |
| taxi | 0 | 0 | 0 | 1 | 0 | 0 | 96 | 0 |
| vanet | 0 | 1 | 0 | 2 | 0 | 7 | 0 | 16 |

Row-normalized reading (each count divided by the true-class support):

- The diagonal equals per-class recall. It ranges from 0.6154 (vanet) to 0.9897 (taxi).
- The largest off-diagonal rates are:
  - vanet -> savari 0.2692 (7/26)
  - minibus -> kamyunet 0.1266 (10/79)
  - kamyunet -> kamyun 0.1122 (11/98)
  - ambulance -> vanet 0.0946 (7/74)
  - ambulance -> kamyunet 0.0946 (7/74)
  - kamyun -> kamyunet 0.0860 (8/93)

### 5. Pair confusion

Most frequent confusions (true -> predicted: count):

- kamyunet -> kamyun: 11
- minibus -> kamyunet: 10
- kamyun -> kamyunet: 8
- vanet -> savari: 7
- ambulance -> vanet: 7
- ambulance -> kamyunet: 7 (same count as ambulance -> vanet; not in the top-5 list of the
  analysis output)

Each pair's score is the sum of the two row-normalized rates:

| Pair | Counts (a->b, b->a) | Pair score |
|---|---|---|
| savari / vanet | 3, 7 | 0.2992 |
| kamyun / kamyunet | 8, 11 | 0.1983 |
| kamyunet / minibus | 2, 10 | 0.1470 |
| ambulance / kamyunet | 7, 2 | 0.1150 |
| ambulance / vanet | 7, 0 | 0.0946 |

- kamyun / kamyunet account for 19 of the 76 validation errors (25.0%).
- This report makes no merge or relabel decision about these two classes.

### 6. Lowest precision, recall and F1

- Lowest recall: vanet, 0.6154 (16 of 26 correct; 7 predicted as savari).
- Lowest precision: vanet, 0.6154 (16 of 26 predictions correct; the 10 wrong ones are
  7 true ambulance and 3 true savari).
- Lowest F1: vanet, 0.6154. Next lowest: kamyunet 0.7885, ambulance 0.8397.

### 7. Training behavior and scheduler

- **Scheduler.** Plateau never reduced the learning rate: the `lr` column is 1e-3 for all
  20 epochs.
  - Validation loss kept reaching new lowest values through epoch 19: 0.3836 at epoch 14,
    0.3795 at epoch 16, 0.3625 at epoch 17 and 0.3538 at epoch 19.
  - So the patience-3 condition was never met. Replaying the scheduler rule on the logged
    validation losses gives the same rates.
- **Validation loss.** Lowest 0.3538 at epoch 19; 0.3836 at the best-F1 epoch 14; 0.4052 at
  epoch 20.
- **Validation F1 after the best epoch.** In epochs 15-20 it varies between 0.8152 and 0.8559
  and does not reach 0.8590 again. Epoch 20 is 0.8422.
- **Train loss.** 1.7310 at epoch 1, 0.424016 at epoch 20. Train accuracy never reaches 1.0.
- **Train vs validation F1.** At epoch 20, train F1 is 0.8223 and validation F1 0.8422. Train
  F1 is below validation F1 in every epoch.
  - Train metrics are measured during the training pass, with augmentation and dropout
    active. Validation is measured in eval mode without augmentation.
  - So the two numbers are not measured under the same conditions, and this comparison alone
    is not evidence of overfitting or of its absence.
- **Overfitting.** C2 does not show clear overfitting.
  - Train loss is still high at epoch 20 (0.4240 vs 0.1092 in C1).
  - Validation loss reaches its lowest value at epoch 19.
  - A cautious reading is slower or incomplete fitting within the 20-epoch budget.
- **Note on the generated analysis text.** The automatic overfitting sentence on line 13 of
  `reports/analysis/combo_aug_do05_wd1e4_plateau/baseline_analysis.txt` covers only the
  single epoch after the lowest validation loss (epoch 20). It does not describe this run
  correctly and is not used here; the description above replaces it.
- **Two factors at once.** Dropout and weight decay were changed together, so this run cannot
  show which of the two caused any difference from C1.

## C1 vs C2

Same split, same current validation labels, best checkpoint of each run:

| Metric | C1 `combo_aug_plateau` | C2 `combo_aug_do05_wd1e4_plateau` | C2 - C1 |
|---|---|---|---|
| Best epoch | 18 | 14 | |
| Accuracy | 0.8983 | 0.8847 | -0.0136 |
| Macro precision | 0.8875 | 0.8682 | -0.0193 |
| Macro recall | 0.8844 | 0.8552 | -0.0292 |
| Macro F1 | 0.8853 | 0.8590 | -0.0263 |
| Correct / 659 | 592 | 583 | -9 |
| Lowest val loss (epoch) | 0.3592 (14) | 0.3538 (19) | |
| LR reductions | after epochs 13 and 18 | none | |
| Train loss at epoch 20 | 0.1092 | 0.4240 | |

Per-class F1:

| Class | C1 | C2 | C2 - C1 |
|---|---|---|---|
| ambulance | 0.8857 | 0.8397 | -0.0460 |
| autobus | 0.9333 | 0.9524 | +0.0191 |
| kamyun | 0.8495 | 0.8602 | +0.0107 |
| kamyunet | 0.8137 | 0.7885 | -0.0252 |
| minibus | 0.9045 | 0.8933 | -0.0112 |
| savari | 0.9510 | 0.9327 | -0.0183 |
| taxi | 0.9897 | 0.9897 | 0.0000 |
| vanet | 0.7547 | 0.6154 | -0.1393 |

- In this run, C2 did not reach C1: macro F1 0.8590 vs 0.8853.
- The largest per-class drop is vanet (-0.1393): its errors to savari rose from 4 to 7.
  ambulance also drops (recall 0.8378 -> 0.7432). autobus and kamyun are slightly higher.
- What the two runs differ in:
  - C2 adds dropout 0.5 and weight decay 1e-4 (AdamW instead of Adam).
  - C2's scheduler never reduced the learning rate, while C1's did twice.
  - C2's train loss stayed much higher.
- This result holds only for these two runs, one seed each. It does not separate the effects
  of dropout, weight decay and the unused scheduler, and it does not establish that the
  combination is inferior in general.

## Summary

- **C1 is the stronger of the two.** C1 (`full_aug` + Plateau) reached validation macro F1
  0.8853. C2 (C1 + dropout 0.5 + weight decay 1e-4) reached 0.8590.
- **C2 fell back to baseline level.** C2's F1 is close to the baseline's (0.8561) and below
  `aug_full`'s (0.8816).
- **Common weaknesses in both runs.** vanet is the weakest class, and kamyun / kamyunet and
  savari / vanet are among the most confused pairs.

## Files

- C1 checkpoint: `checkpoints/combo_aug_plateau_best.pt`
- C1 history: `reports/combo_aug_plateau_history.csv`
- C1 analysis output: `reports/analysis/combo_aug_plateau/`
- C2 checkpoint: `checkpoints/combo_aug_do05_wd1e4_plateau_best.pt`
- C2 history: `reports/combo_aug_do05_wd1e4_plateau_history.csv`
- C2 analysis output: `reports/analysis/combo_aug_do05_wd1e4_plateau/`

## Scope

- Test set: not evaluated in these experiments.
- Neysan images: not evaluated in these experiments.
- One training run per configuration with seed 42; the numbers above are the observed
  validation results of these runs only.
- No merge or relabel decision is made here.
