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

Row-normalized confusion matrix (each row divided by the number of true images of that class,
50 per class; values in %, each row sums to 100%). Added after the run from
`12_final_test_predictions.csv`; Test was not re-run.

| true \ predicted | ambulance | autobus | kamyun | kamyunet | minibus | savari | taxi | vanet |
|---|---|---|---|---|---|---|---|---|
| ambulance | 94.0 | 0.0 | 0.0 | 2.0 | 0.0 | 0.0 | 0.0 | 4.0 |
| autobus | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| kamyun | 0.0 | 2.0 | 80.0 | 16.0 | 2.0 | 0.0 | 0.0 | 0.0 |
| kamyunet | 0.0 | 0.0 | 2.0 | 96.0 | 0.0 | 2.0 | 0.0 | 0.0 |
| minibus | 0.0 | 0.0 | 0.0 | 2.0 | 98.0 | 0.0 | 0.0 | 0.0 |
| savari | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 |
| taxi | 0.0 | 2.0 | 0.0 | 0.0 | 0.0 | 0.0 | 98.0 | 0.0 |
| vanet | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 8.0 | 0.0 | 92.0 |

## needs_review

| | Count |
|---|---|
| needs_review (confidence < 0.90) | 29 |
| correct predictions flagged | 17 |
| incorrect predictions flagged | 12 |
| errors not flagged | 9 |

## Test error analysis

Added after the run from `12_final_test_predictions.csv` (documentation only; Test was not re-run,
nothing was re-predicted). All 21 errors, sorted by true class, then by confidence:

| # | Path | True label | Predicted | Confidence | needs_review | Basis |
|---|---|---|---|---|---|---|
| 20 | `v1/test/ambulance/215936132.jpg` | ambulance | vanet | 0.9994 | False | test_candidate |
| 1 | `v1/test/ambulance/197398569.jpg` | ambulance | vanet | 0.9726 | False | test_candidate |
| 41 | `v1/test/ambulance/218107250.jpg` | ambulance | kamyunet | 0.4128 | True | test_candidate |
| 144 | `v1/test/kamyun/218618273.jpg` | kamyun | kamyunet | 0.9984 | False | test_candidate |
| 113 | `v1/test/kamyun/215619169.jpg` | kamyun | kamyunet | 0.9513 | False | test_candidate |
| 102 | `v1/test/kamyun/195970927.jpg` | kamyun | kamyunet | 0.9129 | False | test_candidate |
| 109 | `v1/test/kamyun/212881811.jpg` | kamyun | kamyunet | 0.8841 | True | test_candidate |
| 148 | `v1/test/kamyun/218815698.jpg` | kamyun | minibus | 0.5611 | True | test_candidate |
| 128 | `v1/test/kamyun/216852176.jpg` | kamyun | kamyunet | 0.5054 | True | test_candidate |
| 130 | `v1/test/kamyun/216957598.jpg` | kamyun | kamyunet | 0.5038 | True | test_candidate |
| 114 | `v1/test/kamyun/215662509.jpg` | kamyun | kamyunet | 0.4849 | True | test_candidate |
| 135 | `v1/test/kamyun/217258940.jpg` | kamyun | kamyunet | 0.4633 | True | test_candidate |
| 134 | `v1/test/kamyun/217129195.jpg` | kamyun | autobus | 0.4498 | True | test_candidate |
| 191 | `v1/test/kamyunet/218055101.jpg` | kamyunet | kamyun | 0.5571 | True | test_candidate |
| 155 | `v1/test/kamyunet/198434168.jpg` | kamyunet | savari | 0.3883 | True | test_candidate |
| 231 | `v1/test/minibus/216335737.jpg` | minibus | kamyunet | 0.9761 | False | test_candidate |
| 326 | `v1/test/taxi/216649147.jpg` | taxi | autobus | 0.5335 | True | test_candidate |
| 398 | `v2/unclean/vanet/216463352.jpg` | vanet | savari | 0.9696 | False | approved_replacement |
| 379 | `v1/unclean/vanet/207074343.jpg` | vanet | savari | 0.9196 | False | approved_replacement |
| 367 | `v1/train/vanet/194285934.jpg` | vanet | savari | 0.9116 | False | approved_replacement |
| 352 | `v1/test/vanet/196511721.jpg` | vanet | savari | 0.5686 | True | test_candidate |

Observations (from the CSV only):

- **Most frequent confusions:** kamyun -> kamyunet 8, vanet -> savari 4, ambulance -> vanet 2; 7 other pairs occur once each.
- **needs_review:** 12 of the 21 errors have `needs_review = True` and 9 are not flagged (confidence >= 0.90): kamyun -> kamyunet 3, vanet -> savari 3, ambulance -> vanet 2, minibus -> kamyunet 1.
- **Errors by true class:** kamyun 10, vanet 4, ambulance 3, kamyunet 2, minibus 1, taxi 1; autobus, savari have none. By predicted class, kamyunet receives 10 of the 21 wrong predictions.
- **kamyun / kamyunet:** 8 kamyun -> kamyunet and 1 kamyunet -> kamyun, 9 of 21 errors. The pair is also the most confused one on validation (6 each way for this model, `10_final_model_selection.md`); on Test the confusion is mostly in one direction.
- **vanet -> savari:** 4 errors, 3 of them on approved replacement images (`basis = approved_replacement`).

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

## Later note (documentation only)

The final model selection was later reopened on validation only
(`decisions/REOPEN_FINAL_MODEL_SELECTION.md`), and `swin_t_ft_aug` became the current final model.
This report remains the historical first evaluation of the frozen Test set, for `resnet224_ft_aug`;
its results are unchanged. The second evaluation, for `swin_t_ft_aug`, is in
`24_swin_final_test_evaluation.md`. `src/predict.py` has changed since; the "Inference" line above
describes the code at commit `b02782f`.
