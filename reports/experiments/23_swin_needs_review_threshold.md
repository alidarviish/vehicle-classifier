# `needs_review` threshold for `swin_t_ft_aug` (validation only)

Status: threshold decided on validation (0.95) for the new final model `swin_t_ft_aug`
(`decisions/REOPEN_FINAL_MODEL_SELECTION.md`). It is not yet applied in `src/predict.py`.
Test and Neysan were not used. The ResNet threshold (0.90, `11_needs_review_threshold.md`)
is unchanged as the historical decision for `resnet224_ft_aug`.

Current status: the threshold is now applied in `src/predict.py` as `NEEDS_REVIEW_THRESHOLD = 0.95`
(commit `a81184a`). The statements in this report that it is "not yet applied" or "will be defined" describe the
state when this report was written; the current code differs.

## Purpose

`needs_review` is a human-review flag only. A prediction whose top softmax probability
(confidence) is below the threshold is marked `needs_review: true`. The predicted class and
the probabilities are still returned; nothing is rejected or replaced.

## Data and method

- **Model:** `swin_t_ft_aug`, checkpoint `checkpoints/swin_t_ft_aug_best.pt` (epoch 17,
  sha256 `f03bacde0a155461ea4aede0b813b8688d6ed17973d5ad19b0d2c1b913ff64db`).
- **Data:** the 659 validation images only, with the unchanged evaluation transform
  (Resize 224x224, ImageNet normalization, no augmentation).
- **Predictions:** per-image validation predictions with all 8 class probabilities, produced
  by a read-only script kept outside the repository. Its checks: 659 rows in manifest order,
  checkpoint epoch and SHA256 verified, the saved confusion matrix and metrics reproduced,
  no Test or Neysan file opened.
- **Consistency check:** 640 correct and 19 errors, the same as `22_swin_tiny_ft_aug.md`.
- **Confidence:** the maximum softmax probability of each prediction. Values are stored
  with 6 decimals; no confidence lies within 1e-6 of any threshold below, so rounding does
  not change any count.
- **Selection rule (fixed before the table was computed):** the same rule as for ResNet in
  `11_needs_review_threshold.md` - the lowest threshold examined that flags at least 50% of
  the validation errors. Thresholds examined: 0.70, 0.80, 0.90, 0.95.

## Threshold comparison

| Threshold | Flagged (of 659) | Errors caught (of 19) | Correct flagged (of 640) | Errors not flagged |
|---|---|---|---|---|
| 0.70 | 12 (1.8%) | 5 (26.3%) | 7 (1.09%) | 14 (73.7%) |
| 0.80 | 21 (3.2%) | 6 (31.6%) | 15 (2.34%) | 13 (68.4%) |
| 0.90 | 37 (5.6%) | 9 (47.4%) | 28 (4.38%) | 10 (52.6%) |
| **0.95** | **48 (7.3%)** | **11 (57.9%)** | **37 (5.78%)** | **8 (42.1%)** |

## Decision

**Decided: `needs_review_threshold = 0.95` for `swin_t_ft_aug`**

- **Why 0.95.** It is the lowest threshold examined that flags at least 50% of the validation
  errors: 11 of 19. 0.90 flags 9 of 19 (47.4%), below the rule.
- **Cost.** 37 correct predictions are also flagged, so 48 of 659 images (7.3%) go to review.

## Higher thresholds (checked after the decision, not part of the rule)

- **0.99:** 77 of 659 flagged (11.7%), 14 of 19 errors (73.7%), 63 correct flagged.
  The review load rises from 7.3% to 11.7% for 3 more errors.
- **All 19 errors:** the highest error confidence is 0.999996, so a threshold of about
  0.999997 would be needed. It flags 359 of 659 images, including 340 correct ones.
  Full error coverage is therefore not a practical policy for this flag.

These rows do not change the decision.

## Limitations

- **Errors the flag cannot catch.** 8 of the 19 validation errors keep a confidence of at
  least 0.95 and are not flagged.
- **Small and reused sample.** 19 errors is a small sample. The threshold is chosen on the
  same validation set that was used to select the model.
- **Not calibrated.** No calibration was done.
- **Not an automatic filter.** The flag is only a review aid.

## Where the threshold is stored

As for ResNet (`11_needs_review_threshold.md`), the threshold is not stored in the
checkpoint; the frozen checkpoint keeps its SHA256. It will be defined in `src/predict.py`
when that file is updated for `swin_t_ft_aug`.

## Test set

No Test image, prediction or metric and no Neysan image was used to choose this threshold.
