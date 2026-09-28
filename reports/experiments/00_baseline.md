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
They were recalculated from the saved predictions of the same checkpoint; the model
was not retrained. The training history itself (including the validation loss of
0.7991 at epoch 18) was logged with the original folder labels.

## Scope

The baseline was trained on the frozen train/validation split with the
8-class taxonomy. Neysan images were excluded from train/validation.

## Files

The full epoch-by-epoch history remains local in
`reports/baseline_history.csv` and is intentionally ignored by Git.
The best checkpoint remains local in `checkpoints/baseline_best.pt`.
