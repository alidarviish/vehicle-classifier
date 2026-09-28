# Experiment: baseline

## Setup

- Run: baseline
- Command: `python -m src.train`
- Seed: 42
- Epochs: 20
- Train / validation images: 2633 / 659
- Checkpoint criterion: best validation macro F1

## Result (validation, best checkpoint)

| Metric | Value |
|---|---|
| Best epoch | 18 |
| Accuracy | 0.8801 |
| Macro precision | 0.8646 |
| Macro recall | 0.8509 |
| Macro F1 | 0.8561 |
| Train loss at best epoch | 0.0001 |

The validation metrics above use the current labels in `data/split_manifest.csv`
(after the human-reviewed kamyun/kamyunet corrections, all in the validation split).
They were recalculated from the saved predictions of the same checkpoint; the baseline
was not retrained for this.

## Validation loss

| | Original folder labels | Current labels |
|---|---|---|
| Lowest validation loss | 0.4921 at epoch 5 | 0.4400 at epoch 5 |
| Validation loss at the best-F1 epoch (18) | 0.7991 | 0.7191 |

- Original-label values: `reports/baseline_history.csv`, logged during the baseline run before
  the 8 validation label corrections.
- Current-label values: the history of `adamw_wd0` (`reports/adamw_wd0_history.csv`, byte-identical
  to `reports/loss_ce_history.csv`). That run follows the same training path as the baseline:
  its training columns are identical to `baseline_history.csv` for all 20 epochs, and its best
  checkpoint (epoch 18) has the same weights as `checkpoints/baseline_best.pt`. Only its validation
  columns were computed with the current labels.

## Scope

The baseline was trained on the frozen train/validation split with the
8-class taxonomy. Neysan images were excluded from train/validation.

## Files

The full epoch-by-epoch history remains local in
`reports/baseline_history.csv` and is intentionally ignored by Git.
The best checkpoint remains local in `checkpoints/baseline_best.pt`.
