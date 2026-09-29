# Neysan / unclean analysis (inference only)

Written by `scripts/evaluate_neysan.py`. Descriptive analysis of the final model on the separate
Neysan evaluation set. It makes no decision about the model, the threshold, labels or Test.

## Scope

- Neysan is an unseen subtype of `vanet`, not a ninth class. The model has 8 classes and never saw
  Neysan images in training, validation or Test.
- `expected_model_label` = `vanet` for every row: the existing taxonomy mapping
  (Neysan -> vanet), not a new label.
- The test set was not used in this analysis; no Test image is in the Neysan set (checked by path
  and SHA256 before prediction).
- The `needs_review` threshold (0.90) was chosen on validation and was not tuned on Neysan.
- The 250 `neysan_folder_label` images (policy N1) are not human-confirmed; only the 371
  `CONFIRMED_NEYSAN` images were individually reviewed.
- No precision, F1, macro metric or confusion matrix is reported: the set has a single expected
  class and no negatives, so those numbers would be trivial or undefined. The share predicted as
  `vanet` (vanet recall on Neysan) is the only supervised number, and it is not comparable with the
  validation or Test macro metrics.

## Data provenance and isolation

Sources: `decisions/neysan_eval.csv`, `decisions/neysan_review.csv`, `decisions/neysan_test_review.csv`,
`decisions/test_frozen.csv`, `data/split_manifest.csv`, `reports/audit/exact_duplicates.csv` and
`docs/DATA_DECISIONS.md`. These sections were added after the run from the existing files; inference
was not re-run.

- **Neysan set:** 621 rows: 371 `CONFIRMED_NEYSAN` (human review) and 250 `neysan_folder_label`
  (policy N1, folder name `neysan`, no individual review).
- **Origin totals:** unclean 408, train 179, test 34. **Folder labels:** vanet 371, neysan 250.
- **The 34 images from the original Test:** the 34 rows with origin `test` are exactly the 34
  `CONFIRMED_NEYSAN` images in `decisions/neysan_test_review.csv`. That review covered 50 v1/test vanet
  images: 34 `CONFIRMED_NEYSAN` and 16 `NOT_NEYSAN`.
- **The 34 replacements:** the 34 images were moved out of Test into the Neysan set. To keep 50 vanet
  images in Test, 34 approved vanet images from the ordinary pool replaced them (17 of origin `train`,
  17 of origin `unclean`; `basis = approved_replacement` in `decisions/test_frozen.csv`).
- **A017** (`v1/train/vanet/214726285.jpg`): reviewed `NOT_NEYSAN`. It remains an ordinary `vanet`
  training image; it is not in Test and not in the Neysan set.
- **`NOT_NEYSAN` images:** all 163 `NOT_NEYSAN` images of `decisions/neysan_review.csv` are in the
  ordinary train/val pool; none is in the Neysan set.
- **Duplicate pair:** DUP014 in `reports/audit/exact_duplicates.csv`, `v1/train/vanet/214844236.jpg` and
  `v1/unclean/neysan/214844236.jpg`. It is the only one of the 24 exact-duplicate groups with
  conflicting folder labels (vanet / neysan). Both copies are in the Neysan set.
- **Isolation:** by path and by SHA256, the Neysan set has no overlap with the train/val split
  (3292 images), the excluded images (47), the frozen Test set or the `NOT_NEYSAN` images. The
  same checks were run by `scripts/evaluate_neysan.py` before prediction.

## Unclean-origin accounting

- **Total:** 2028 images have `origin_split = unclean`.
- **Ordinary train/val/Test pool:** 1590 of them.
  - 1258 are in train.
  - 315 are in validation.
  - 17 are Test replacements.
- **Neysan set:** 408 are in the Neysan set.
- **Excluded:** 30 are excluded.

Unclean-origin images entered the ordinary pool only after the project's cleaning steps:

1. exact-duplicate and Test-leakage checks by SHA256;
2. the recorded exclusions;
3. the Neysan review with policy N1;
4. the label review.

The policy is described in `docs/DATA_DECISIONS.md` ("Images from `unclean` folders in the training
data").

## Run

- Run identifier: `neysan_eval_20260929T083557Z`
- Timestamp (UTC): 2026-09-29 08:35:57
- Code: git commit `8b17c5f`
- Checkpoint: `checkpoints/resnet224_ft_aug_best.pt` (SHA256 verified: `c26280e97f7324ee1ec85d7bf78a58ea0b4d2acb4cf3e975dbd52316366281a1`)
- Inference: `src/predict.py` (`load_model`, `predict_image`), unchanged; transform: Resize(size=(224, 224), interpolation=bilinear, max_size=None, antialias=True) -> ToTensor() -> Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Input: `decisions/neysan_eval.csv` only (621 rows, every file's SHA256 verified)

## Predicted vanet vs non-vanet

| Group | Images | Predicted vanet | Predicted non-vanet | Vanet recall | needs_review |
|---|---|---|---|---|---|
| All 621 rows | 621 | 529 | 92 | 0.8519 | 185 (29.8%) |
| CONFIRMED_NEYSAN (371, human-confirmed) | 371 | 317 | 54 | 0.8544 | 102 (27.5%) |
| neysan_folder_label (250, policy N1, not human-confirmed) | 250 | 212 | 38 | 0.8480 | 83 (33.2%) |
| Unique SHA256 images (620) | 620 | 528 | 92 | 0.8516 | 185 (29.8%) |

The N1 row is a policy-based, non-human-confirmed result. The duplicate pair
`v1/train/vanet/214844236.jpg` and `v1/unclean/neysan/214844236.jpg` is byte-identical and therefore gets the same
prediction; the manifest keeps both rows (621), and the unique-SHA256 row counts it once (620).

## Prediction distribution over the 8 classes

| Group | ambulance | autobus | kamyun | kamyunet | minibus | savari | taxi | vanet |
|---|---|---|---|---|---|---|---|---|
| All 621 rows | 2 | 2 | 19 | 14 | 19 | 33 | 3 | 529 |
| CONFIRMED_NEYSAN (371, human-confirmed) | 1 | 2 | 13 | 5 | 11 | 19 | 3 | 317 |
| neysan_folder_label (250, policy N1, not human-confirmed) | 1 | 0 | 6 | 9 | 8 | 14 | 0 | 212 |
| Unique SHA256 images (620) | 2 | 2 | 19 | 14 | 19 | 33 | 3 | 528 |

## needs_review (confidence < 0.90) by prediction

| Group | vanet, flagged | vanet, not flagged | non-vanet, flagged | non-vanet, not flagged |
|---|---|---|---|---|
| All 621 rows | 116 | 413 | 69 | 23 |
| CONFIRMED_NEYSAN (371, human-confirmed) | 59 | 258 | 43 | 11 |
| neysan_folder_label (250, policy N1, not human-confirmed) | 57 | 155 | 26 | 12 |
| Unique SHA256 images (620) | 116 | 412 | 69 | 23 |

## Confidence (highest softmax probability)

| Group | Images | Mean | Median | Q1 | Q3 |
|---|---|---|---|---|---|
| All 621 rows | 621 | 0.8976 | 0.9815 | 0.8509 | 0.9980 |
| Predicted vanet | 529 | 0.9272 | 0.9876 | 0.9303 | 0.9985 |
| Predicted non-vanet | 92 | 0.7270 | 0.7681 | 0.5570 | 0.8929 |
| CONFIRMED_NEYSAN | 371 | 0.8994 | 0.9816 | 0.8731 | 0.9983 |
| neysan_folder_label (N1) | 250 | 0.8948 | 0.9798 | 0.8341 | 0.9971 |

Quartiles: `statistics.quantiles(n=4, method='inclusive')` on the 4-decimal confidences.

## By source / origin

| Source / origin | Basis | Images | Predicted vanet | Predicted non-vanet | needs_review |
|---|---|---|---|---|---|
| v1/test | CONFIRMED_NEYSAN | 34 | 28 | 6 | 11 |
| v1/train | CONFIRMED_NEYSAN | 36 | 29 | 7 | 11 |
| v1/unclean | CONFIRMED_NEYSAN | 34 | 31 | 3 | 7 |
| v1/unclean | neysan_folder_label | 50 | 43 | 7 | 12 |
| v2/train | CONFIRMED_NEYSAN | 143 | 120 | 23 | 41 |
| v2/unclean | CONFIRMED_NEYSAN | 124 | 109 | 15 | 32 |
| v2/unclean | neysan_folder_label | 200 | 169 | 31 | 71 |

## Probability analysis

Computed from `reports/experiments/13_neysan_predictions.csv` (the `p_<class>` columns are the model's
softmax outputs, rounded to 4 decimals). These are model probability outputs, not calibrated
real-world probabilities.

Mean softmax output per class:

| Group | ambulance | autobus | kamyun | kamyunet | minibus | savari | taxi | vanet |
|---|---|---|---|---|---|---|---|---|
| All 621 | 0.0053 | 0.0040 | 0.0461 | 0.0354 | 0.0383 | 0.0528 | 0.0087 | 0.8095 |
| CONFIRMED_NEYSAN 371 | 0.0033 | 0.0050 | 0.0505 | 0.0253 | 0.0373 | 0.0508 | 0.0087 | 0.8191 |
| N1 (`neysan_folder_label`) 250 | 0.0081 | 0.0025 | 0.0395 | 0.0505 | 0.0397 | 0.0557 | 0.0087 | 0.7953 |

`p_vanet` bins:

| Group | < 0.50 | 0.50–<0.70 | 0.70–<0.90 | 0.90–<0.95 | 0.95–1.00 |
|---|---|---|---|---|---|
| All 621 | 102 (16.4%) | 34 (5.5%) | 72 (11.6%) | 40 (6.4%) | 373 (60.1%) |
| CONFIRMED_NEYSAN 371 | 61 (16.4%) | 18 (4.9%) | 34 (9.2%) | 27 (7.3%) | 231 (62.3%) |
| N1 (`neysan_folder_label`) 250 | 41 (16.4%) | 16 (6.4%) | 38 (15.2%) | 13 (5.2%) | 142 (56.8%) |

413 of 621 images (66.5%) have `p_vanet >= 0.90`.

Non-vanet predictions (92), by predicted class. The values are the softmax output of the predicted
class (= confidence):

| Predicted class | Count | Mean | Median | Min | Max | needs_review |
|---|---|---|---|---|---|---|
| savari | 33 | 0.8143 | 0.8751 | 0.3461 | 0.9994 | 18 |
| kamyun | 19 | 0.7143 | 0.7457 | 0.3765 | 0.9785 | 16 |
| minibus | 19 | 0.6518 | 0.5937 | 0.4029 | 0.9921 | 17 |
| kamyunet | 14 | 0.7529 | 0.8220 | 0.3888 | 0.9883 | 11 |
| taxi | 3 | 0.4754 | 0.4748 | 0.3056 | 0.6457 | 3 |
| ambulance | 2 | 0.6390 | 0.6390 | 0.3936 | 0.8843 | 2 |
| autobus | 2 | 0.4068 | 0.4068 | 0.3858 | 0.4277 | 2 |

- 16 of the 92 non-vanet predictions have confidence >= 0.95.
- `vanet` is the second-highest class in 57 of the 92 non-vanet predictions.
- 23 of the 92 non-vanet predictions are not flagged `needs_review` (confidence >= 0.90).

The 10 non-vanet predictions with the highest confidence:

| Path | Predicted | Confidence | `p_vanet` | needs_review |
|---|---|---|---|---|
| `v2/unclean/vanet/212397930.jpg` | savari | 0.9994 | 0.0001 | False |
| `v2/train/vanet/218044166.jpg` | savari | 0.9988 | 0.0007 | False |
| `v2/unclean/vanet/212462069.jpg` | savari | 0.9985 | 0.0013 | False |
| `v1/unclean/neysan/212400936.jpg` | savari | 0.9960 | 0.0004 | False |
| `v2/train/vanet/216608800.jpg` | savari | 0.9929 | 0.0048 | False |
| `v2/unclean/neysan/218042482.jpg` | savari | 0.9925 | 0.0039 | False |
| `v2/train/vanet/212665899.jpg` | savari | 0.9923 | 0.0001 | False |
| `v2/unclean/neysan/208096370.jpg` | minibus | 0.9921 | 0.0070 | False |
| `v2/unclean/neysan/197741309.jpg` | kamyunet | 0.9883 | 0.0004 | False |
| `v1/unclean/neysan/212188576.jpg` | savari | 0.9797 | 0.0018 | False |

The 10 predictions with the lowest confidence:

| Path | Predicted | Confidence | `p_vanet` | needs_review |
|---|---|---|---|---|
| `v2/unclean/vanet/210819242.jpg` | taxi | 0.3056 | 0.2757 | True |
| `v2/unclean/neysan/205272178.jpg` | savari | 0.3461 | 0.1875 | True |
| `v2/unclean/vanet/206651635.jpg` | kamyun | 0.3765 | 0.1988 | True |
| `v1/unclean/vanet/215996114.jpg` | autobus | 0.3858 | 0.3394 | True |
| `v2/unclean/vanet/216896507.jpg` | kamyunet | 0.3888 | 0.3062 | True |
| `v2/unclean/vanet/215976038.jpg` | ambulance | 0.3936 | 0.3849 | True |
| `v2/unclean/vanet/208091182.jpg` | vanet | 0.3991 | 0.3991 | True |
| `v1/unclean/neysan/195728454.jpg` | vanet | 0.4027 | 0.4027 | True |
| `v2/train/vanet/208422550.jpg` | minibus | 0.4029 | 0.2257 | True |
| `v2/unclean/neysan/201982839.jpg` | kamyun | 0.4044 | 0.3602 | True |

Unique images: the two files of the duplicate pair are byte-identical and received identical
predictions (vanet, confidence 0.9950, needs_review False). The 621-row and the
620-unique results above therefore differ only by this one duplicated row.


## Interpretation and caveats

- **Not a class.** Neysan is treated as an unseen subtype of `vanet`, not as a ninth model class. The
  model has no Neysan class, and this analysis does not claim that it recognizes Neysan.
- **Mostly high `p_vanet`.** Most Neysan images receive a high `p_vanet` softmax output:
  - 60.1% are at or above 0.95;
  - 66.5% are at or above 0.90.
- **But not always vanet.** 92 of 621 images are predicted as another class, most often savari (33).
- **Some errors are confident.** Some non-vanet predictions have high confidence: 16 of 92 are at or
  above 0.95.
- **The review rule misses some.** 23 of the 92 non-vanet predictions are not flagged.
- **What `needs_review` is.** It is a human-review rule based on the 0.90 confidence threshold chosen
  on validation. It is not an accuracy metric and not a Neysan detector.
- **Not calibrated.** The softmax outputs are model scores. They are not calibrated real-world
  probabilities.
- **N1 is not confirmed.** The 250 N1 (`neysan_folder_label`) images are not equivalent to
  human-confirmed Neysan.
- **No decisions from this analysis.** No label, taxonomy, Test or threshold decision was made from
  it. Test was not used for threshold tuning, and Neysan was not used for any tuning.

## Integrity

- 621 predictions, one per manifest row: PASS
- prediction order / path mapping matches the manifest: PASS
- predicted_class is one of the 8 classes: PASS
- confidence in [0, 1]: PASS
- needs_review == (confidence < 0.90): PASS
- checkpoint, test_frozen.csv, split_manifest.csv, neysan_eval.csv unchanged: PASS
- only the two outputs are new (CSV now, report next): PASS

SHA256 of the guarded files before and after inference:

| File | Before | After |
|---|---|---|
| checkpoint | `c26280e97f7324ee1ec85d7bf78a58ea0b4d2acb4cf3e975dbd52316366281a1` | `c26280e97f7324ee1ec85d7bf78a58ea0b4d2acb4cf3e975dbd52316366281a1` |
| test_frozen.csv | `e1727eea750e73a4924d79fbb5fb810af8f8a05d822754c51ade8376e0622540` | `e1727eea750e73a4924d79fbb5fb810af8f8a05d822754c51ade8376e0622540` |
| split_manifest.csv | `4251558b6e2c89f065f05926869210a4ee896033e63b3d182e4c30ac56b1acba` | `4251558b6e2c89f065f05926869210a4ee896033e63b3d182e4c30ac56b1acba` |
| neysan_eval.csv | `1e8cad9bbca206740f78dda72da6318c8851d6f56ec17281317048308576086a` | `1e8cad9bbca206740f78dda72da6318c8851d6f56ec17281317048308576086a` |

- Files created by this run: `reports/experiments/13_neysan_predictions.csv` and this report.
