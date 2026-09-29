# `needs_review` threshold (validation only)

Status: threshold decided on validation (0.90) and used in `src/predict.py`. The smoke test of
`src/predict.py` passed 27 of 27 checks; it used a synthetic image and one validation image
(read-only), not the test set. The test set was not used to choose the threshold; it was evaluated
once afterwards with this threshold (see `12_final_test_evaluation.md`). The Neysan images were not
used.

## Purpose

`needs_review` is a human-review flag only. A prediction whose top softmax probability (confidence)
is below the threshold is marked `needs_review: true`. The predicted class and the probabilities are
still returned; nothing is rejected automatically, and the current code has no automatic reject.

## Data and method

- **Model:** `resnet224_ft_aug`, checkpoint `checkpoints/resnet224_ft_aug_best.pt` (epoch 20,
  sha256 `c26280e97f7324ee1ec85d7bf78a58ea0b4d2acb4cf3e975dbd52316366281a1`). This is the model
  selected in `10_final_model_selection.md`.
- **Data:** the 659 validation images only, with the unchanged `RESNET_TRANSFORM` (no augmentation).
- **Script:** `val_confidence_audit.py`, kept outside the repository. It is read-only, writes no
  file, and checks that no validation image overlaps with Test or Neysan.
- **Consistency check:** 633 correct and 26 errors, the same as the saved confusion matrix of this
  run.
- **Confidence:** the maximum softmax probability of each prediction.

No model, checkpoint or data was changed.

## Validation evidence

- **Correct predictions (633):** mean confidence 0.9864.
- **Wrong predictions (26):** mean confidence 0.8703.
- **Confidently wrong:** some wrong predictions still have confidence 1.0000 at the printed
  precision. No threshold can flag those errors.

## Threshold comparison

"Caught", "correct flagged" and "coverage" are from the script output. The other columns are
derived from those numbers: flagged = caught + correct flagged, and accuracy on not-flagged =
(633 - correct flagged) / not-flagged. The script also printed rows for 0.50 and 0.60; they are not
reproduced in this report.

| Threshold | Flagged | Errors caught (of 26) | Correct flagged | Coverage (not flagged) | Accuracy on not-flagged | Errors not flagged |
|---|---|---|---|---|---|---|
| 0.70 | 13 (2.0%) | 3 (11.5%) | 10 | 98.0% (646) | 0.9644 | 23 |
| 0.80 | 23 (3.5%) | 8 (30.8%) | 15 | 96.5% (636) | 0.9717 | 18 |
| **0.90** | **35 (5.3%)** | **13 (50.0%)** | **22** | **94.7% (624)** | **0.9792** | **13** |
| 0.95 | 47 (7.1%) | 14 (53.8%) | 33 | 92.9% (612) | 0.9804 | 12 |

## Decision

**Decided: `needs_review_threshold = 0.90`**

- **Why 0.90.** It is the lowest threshold examined that flags at least 50% of the validation
  errors: 13 of 26.
- **Cost.** 22 correct predictions are also flagged, so 35 of 659 images (5.3%) go to review and
  coverage stays at 94.7%.
- **Why not 0.95.** It catches only 1 more error (14 instead of 13) but flags 11 more correct
  predictions (33 instead of 22).
- **Why not 0.70 or 0.80.** They catch 3 and 8 errors, below half of the errors.
- **Where the 50% criterion comes from.** The earlier protocol only said that the threshold should
  catch a "substantial" share of the errors while flagging a small share of correct predictions.
  "At least 50% of the errors" is how "substantial" is made concrete at this stage. It was **not**
  fixed as a number in advance, and this report does not present it as if it had been.

## Limitations

- **Errors the flag cannot catch.** 13 of the 26 validation errors keep a confidence of at least
  0.90 and are not flagged; some are wrong with confidence 1.0000.
- **Small and reused sample.** 26 errors is a small sample. The threshold is chosen on the same
  validation set that was used to select the model, so its behaviour on new data may differ.
- **Not calibrated.** Softmax confidence of this model is not calibrated; no calibration was done.
- **Not an automatic filter.** The flag is only a review aid. It does not reject or correct any
  prediction.

## Where the threshold is stored (protocol decision)

- **Single source of truth.** The final threshold 0.90 is defined once, as
  `NEEDS_REVIEW_THRESHOLD = 0.90` in `src/predict.py`. `predict.py` uses it at inference:
  `needs_review` is true when the confidence is below this value.
- **Not in the checkpoint.** The threshold is not stored in the checkpoint metadata. The selected
  checkpoint `checkpoints/resnet224_ft_aug_best.pt` remains frozen with SHA256
  `c26280e97f7324ee1ec85d7bf78a58ea0b4d2acb4cf3e975dbd52316366281a1`, as recorded in
  `10_final_model_selection.md`.
- **Why.** Re-saving the checkpoint only to add this metadata would create a different checkpoint
  file with a different hash, without changing the learned weights.
- **Status.** This is an intentional protocol decision, not a pending task. The assignment asks for
  the selected threshold in the checkpoint metadata; this project keeps it in `src/predict.py`
  instead, next to the frozen checkpoint it is used with.

The frozen test set was then evaluated once with this checkpoint and this threshold
(`12_final_test_evaluation.md`).

## Test set

No test image, prediction or metric was used to choose this threshold. The test set was evaluated
only afterwards, once, with the threshold unchanged at 0.90 (`12_final_test_evaluation.md`).
