# Final Test evaluation: `swin_t_ft_aug` (second use of the frozen Test set)

Written by `scripts/evaluate_test_swin.py`.

## Protocol status

- The frozen Test set was first evaluated once for the previous final model `resnet224_ft_aug`
  (`12_final_test_evaluation.md`). Those results are historical and unchanged.
- The final model selection was reopened on validation only and `swin_t_ft_aug` was selected
  (`decisions/REOPEN_FINAL_MODEL_SELECTION.md`). The `needs_review` threshold 0.95 was chosen on
  validation only (`23_swin_needs_review_threshold.md`).
- This run is the second use of the same frozen Test set. It is a documented protocol deviation:
  the result is not a pristine held-out estimate.
- This Test result is not used for model selection or for the threshold.

## Run

- Run identifier: `swin_final_test_20261004T163946Z`
- Timestamp (UTC): 2026-10-04 16:39:46
- Code: git commit `817b61e`
- Checkpoint: `checkpoints/swin_t_ft_aug_best.pt` (Swin-Tiny)
- Checkpoint SHA256: `f03bacde0a155461ea4aede0b813b8688d6ed17973d5ad19b0d2c1b913ff64db` (verified before prediction)
- Checkpoint epoch: 17 (stored validation macro F1 0.9658)
- Inference: `src/predict.py` (`load_model`, `predict_image`)
- Inference transform: Resize(size=(224, 224), interpolation=bilinear, max_size=None, antialias=True) -> ToTensor() -> Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) (no training augmentation)
- `needs_review` threshold: 0.95 (confidence < threshold -> needs_review; flag only, nothing rejected)
- Test source: `decisions/test_frozen.csv` only (400 rows, SHA256 of every file verified before prediction)
- Test manifest SHA256: `e1727eea750e73a4924d79fbb5fb810af8f8a05d822754c51ade8376e0622540`
## Results

| Metric | Value |
|---|---|
| Total | 400 |
| Correct | 382 |
| Incorrect | 18 |
| Accuracy | 0.9550 |
| Macro precision | 0.9558 |
| Macro recall | 0.9550 |
| Macro F1 | 0.9545 |

Per-class metrics:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ambulance | 1.0000 | 0.9800 | 0.9899 | 50 |
| autobus | 0.9615 | 1.0000 | 0.9804 | 50 |
| kamyun | 0.9535 | 0.8200 | 0.8817 | 50 |
| kamyunet | 0.8868 | 0.9400 | 0.9126 | 50 |
| minibus | 0.9608 | 0.9800 | 0.9703 | 50 |
| savari | 0.9423 | 0.9800 | 0.9608 | 50 |
| taxi | 1.0000 | 0.9800 | 0.9899 | 50 |
| vanet | 0.9412 | 0.9600 | 0.9505 | 50 |

Confusion matrix (rows = true class, columns = predicted class):

| true \ predicted | ambulance | autobus | kamyun | kamyunet | minibus | savari | taxi | vanet |
|---|---|---|---|---|---|---|---|---|
| ambulance | 49 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| autobus | 0 | 50 | 0 | 0 | 0 | 0 | 0 | 0 |
| kamyun | 0 | 1 | 41 | 6 | 2 | 0 | 0 | 0 |
| kamyunet | 0 | 0 | 2 | 47 | 0 | 0 | 0 | 1 |
| minibus | 0 | 1 | 0 | 0 | 49 | 0 | 0 | 0 |
| savari | 0 | 0 | 0 | 0 | 0 | 49 | 0 | 1 |
| taxi | 0 | 0 | 0 | 0 | 0 | 1 | 49 | 0 |
| vanet | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 48 |

## needs_review

| | Count |
|---|---|
| needs_review (confidence < 0.95) | 40 |
| correct predictions flagged | 28 |
| incorrect predictions flagged | 12 |
| errors not flagged | 6 |

## Integrity checks

- every manifest row has exactly one prediction: PASS
- 400 rows evaluated: PASS
- no duplicate path in the prediction CSV: PASS
- prediction order / path mapping matches the manifest: PASS
- Test manifest SHA256 unchanged after prediction: PASS
- Swin checkpoint SHA256 unchanged after prediction: PASS
- historical ResNet Test outputs (12_*) unchanged: PASS

## Scope

- The Test manifest `decisions/test_frozen.csv` was the only Test source; no folder was scanned.
- The historical outputs `12_final_test_predictions.csv` and `12_final_test_evaluation.md` were
  only hashed, before and after this run.
- Neysan evaluation was not part of this run.
- Per-image predictions: `reports/experiments/24_swin_final_test_predictions.csv`.
