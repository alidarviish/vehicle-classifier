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
