# Final Test evaluation

One-time evaluation of the selected model on the frozen Test set, written by
`scripts/evaluate_test.py`.

## Run

- Run identifier: `final_test_20260929T080908Z`
- Timestamp (UTC): 2026-09-29 08:09:08
- Code: git commit `b02782f`
- Checkpoint: `checkpoints/resnet224_ft_aug_best.pt`
- Checkpoint SHA256: `c26280e97f7324ee1ec85d7bf78a58ea0b4d2acb4cf3e975dbd52316366281a1` (verified before prediction)
- Checkpoint epoch: 20 (stored validation macro F1 0.9482)
- Inference: `src/predict.py` (`load_model`, `predict_image`), unchanged
- Inference transform: Resize(size=(224, 224), interpolation=bilinear, max_size=None, antialias=True) -> ToTensor() -> Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) (no training augmentation)
- `needs_review` threshold: 0.90 (confidence < threshold -> needs_review; flag only, nothing rejected)
- Test source: `decisions/test_frozen.csv` only (400 rows, SHA256 of every file verified before prediction)
- Test manifest SHA256: `e1727eea750e73a4924d79fbb5fb810af8f8a05d822754c51ade8376e0622540`

## Results

| Metric | Value |
|---|---|
| Total | 400 |
| Correct | 379 |
| Incorrect | 21 |
| Accuracy | 0.9475 |
| Macro precision | 0.9515 |
| Macro recall | 0.9475 |
| Macro F1 | 0.9473 |

Per-class metrics:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ambulance | 1.0000 | 0.9400 | 0.9691 | 50 |
| autobus | 0.9615 | 1.0000 | 0.9804 | 50 |
| kamyun | 0.9756 | 0.8000 | 0.8791 | 50 |
| kamyunet | 0.8276 | 0.9600 | 0.8889 | 50 |
| minibus | 0.9800 | 0.9800 | 0.9800 | 50 |
| savari | 0.9091 | 1.0000 | 0.9524 | 50 |
| taxi | 1.0000 | 0.9800 | 0.9899 | 50 |
| vanet | 0.9583 | 0.9200 | 0.9388 | 50 |

Confusion matrix (rows = true class, columns = predicted class):

| true \ predicted | ambulance | autobus | kamyun | kamyunet | minibus | savari | taxi | vanet |
|---|---|---|---|---|---|---|---|---|
| ambulance | 47 | 0 | 0 | 1 | 0 | 0 | 0 | 2 |
| autobus | 0 | 50 | 0 | 0 | 0 | 0 | 0 | 0 |
| kamyun | 0 | 1 | 40 | 8 | 1 | 0 | 0 | 0 |
| kamyunet | 0 | 0 | 1 | 48 | 0 | 1 | 0 | 0 |
| minibus | 0 | 0 | 0 | 1 | 49 | 0 | 0 | 0 |
| savari | 0 | 0 | 0 | 0 | 0 | 50 | 0 | 0 |
| taxi | 0 | 1 | 0 | 0 | 0 | 0 | 49 | 0 |
| vanet | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 46 |

## needs_review

| | Count |
|---|---|
| needs_review (confidence < 0.90) | 29 |
| correct predictions flagged | 17 |
| incorrect predictions flagged | 12 |
| errors not flagged | 9 |

## Integrity checks

- every manifest row has exactly one prediction: PASS
- 400 rows evaluated: PASS
- no duplicate path in the prediction CSV: PASS
- prediction order / path mapping matches the manifest: PASS

## Scope

- The Test manifest `decisions/test_frozen.csv` was the only Test source; no folder was scanned.
- Test was not used for model selection (`10_final_model_selection.md`) or for the
  `needs_review` threshold (`11_needs_review_threshold.md`); both were decided on validation.
- Neysan evaluation was not part of this run.
- Per-image predictions: `reports/experiments/12_final_test_predictions.csv`.
