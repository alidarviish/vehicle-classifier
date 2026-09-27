# Baseline Result

- Run: baseline
- Seed: 42
- Epochs: 20
- Best epoch: 18
- Best validation macro F1: 0.8456
- Validation accuracy at best epoch: 0.8680
- Train loss at best epoch: 0.0001
- Validation loss at best epoch: 0.7991

The baseline was trained on the frozen train/validation split with the
8-class taxonomy. Neysan images were excluded from train/validation.

The full epoch-by-epoch history remains local in
`reports/baseline_history.csv` and is intentionally ignored by Git.
The best checkpoint remains local in `checkpoints/baseline_best.pt`.