# Experiment: balanced batches

Status: set up, not yet trained. No results are recorded here yet.

## Setup

One-factor comparison on the same simulated-imbalance training set: only the way batches are
built changes.

| Run | Batch mode | Command |
|---|---|---|
| `balanced_standard` | `standard` (shuffled batches) | `python -m src.train --run-name balanced_standard --train-subset decisions/imbalanced_train.csv --batch-mode standard` |
| `balanced_sampler` | `balanced` (`BalancedBatchSampler`) | `python -m src.train --run-name balanced_sampler --train-subset decisions/imbalanced_train.csv --batch-mode balanced` |

- Training set: `decisions/imbalanced_train.csv` (1673 images), built by
  `scripts/make_imbalanced_train.py` from the 2633 train images of `data/split_manifest.csv`:
  ambulance, kamyun and minibus reduced to 15 images each (seed 42), the other 5 classes complete.
  See `docs/DATA_DECISIONS.md` for why the current train split is the cleaned training data.
- Unchanged from the baseline: `BaselineCNN`, 128x128 RGB, max pooling, dropout 0.0, augmentation
  `none`, Adam, learning rate 1e-3, weight decay 0, no scheduler, CrossEntropyLoss, 20 epochs,
  batch size 32, seed 42, checkpoint by best validation macro F1.
- Validation (the current 659 images) and Test (`decisions/test_frozen.csv`) do not change in this
  experiment; Test is not used.

## Sampling details

- Both modes have exactly 53 batches, i.e. 53 optimizer steps, per epoch (`ceil(1673 / 32)`).
- `standard`: 1673 training exposures per epoch - 52 full batches of 32 and a last batch of 9.
- `balanced`: 53 x 32 = 1696 exposures per epoch, 23 more than `standard`. Every batch holds exactly
  4 images of each of the 8 classes, so each class has exactly 212 exposures per epoch.
- ambulance, kamyun and minibus (15 images each) are reused several times per epoch to reach
  212 exposures. vanet (103 images) is also reused, since it needs 212 exposures per epoch.
- autobus, kamyunet, savari and taxi have more than 212 images, so only 212 of their images
  are used per epoch.
- An image can appear more than once in the same batch when a small class is refilled mid-batch.
  This is the allowed replacement behavior, not a bug.
- The sampler (`src/balanced_sampler.py`) uses its own random generator, `random.Random(seed + epoch)`
  with seed 42; `src/train.py` calls `set_epoch(epoch)` before each epoch, so each epoch is
  deterministic and different from the others.
