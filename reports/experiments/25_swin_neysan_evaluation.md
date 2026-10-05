# Neysan evaluation: `swin_t_ft_aug` (inference only)

Written by `scripts/evaluate_neysan_swin.py`. Neysan is a subtype of `vanet`, not a ninth class;
the expected 8-class output for every image is `vanet`. Nothing in this report was used to change
the model, the threshold or the final Test result.

## Run

- Run identifier: `swin_neysan_20261005T143443Z`
- Timestamp (UTC): 2026-10-05 14:34:43
- Code: git commit `bdbe67c`
- Checkpoint: `checkpoints/swin_t_ft_aug_best.pt` (Swin-Tiny, epoch 17), SHA256 `f03bacde0a155461ea4aede0b813b8688d6ed17973d5ad19b0d2c1b913ff64db`
- Inference: `src/predict.py` (`load_model`, `predict_image`), one call per row; full-precision
  probabilities taken from the logits of that same call (forward hook)
- Inference transform: Resize(size=(224, 224), interpolation=bilinear, max_size=None, antialias=True) -> ToTensor() -> Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) (no training augmentation)
- `needs_review` threshold: 0.95 (confidence < threshold; flag only, nothing rejected)
- Neysan manifest: `decisions/neysan_eval.csv`, SHA256 `1e8cad9bbca206740f78dda72da6318c8851d6f56ec17281317048308576086a`
- Visibility tags (Gate-1 visibility / lighting annotation, not part of the repository): SHA256 `7f5b5fc2eabb7dcd31b1b0a9d1a06840f33a65f8fee3b5ae580f33c9e1526022`
- Validation baseline (existing per-image `swin_t_ft_aug` validation predictions, not part of the repository): SHA256 `87cf759b1d27770c876fb3748599551d32a789dd953ee8c774333253c18e956f`

## A. Subgroups

Reported separately; not pooled unless a row says so.

- Confirmed Neysan, not from v1/test: 337 rows (expected 337)
- Confirmed Neysan from v1/test: 34 rows (expected 34)
- N1 (folder label neysan, not individually reviewed): 250 rows (expected 250)
- Known duplicate pair (same SHA256): `v1/train/vanet/214844236.jpg` (confirmed, not v1/test) and `v1/unclean/neysan/214844236.jpg` (N1).
  Each copy is counted in its own subgroup; pooled figures cover 621 rows = 620 unique images.

## B. Predictions per subgroup

| Subgroup | n | ambulance | autobus | kamyun | kamyunet | minibus | savari | taxi | vanet | vanet rate | conf. median | Q1 | Q3 | min | needs_review |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Confirmed Neysan, not from v1/test | 337 | 1 | 0 | 8 | 7 | 7 | 5 | 7 | 302 | 302/337 (89.6%) | 0.9902 | 0.9213 | 0.9988 | 0.4218 | 101/337 (30.0%) |
| Confirmed Neysan from v1/test | 34 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | 31 | 31/34 (91.2%) | 0.9935 | 0.9285 | 0.9987 | 0.5767 | 9/34 (26.5%) |
| N1 (folder label neysan, not individually reviewed) | 250 | 0 | 0 | 4 | 14 | 7 | 4 | 0 | 221 | 221/250 (88.4%) | 0.9857 | 0.8975 | 0.9979 | 0.4081 | 89/250 (35.6%) |

## C. Review outcomes

| Subgroup | n | predicted vanet, not flagged (silent acceptance) | predicted vanet, flagged (sent to review) | predicted non-vanet, not flagged (confident misclassification) | predicted non-vanet, flagged (error caught by review) |
|---|---|---|---|---|---|
| Confirmed Neysan, not from v1/test | 337 | 233/337 (69.1%) | 69/337 (20.5%) | 3/337 (0.9%) | 32/337 (9.5%) |
| Confirmed Neysan from v1/test | 34 | 24/34 (70.6%) | 7/34 (20.6%) | 1/34 (2.9%) | 2/34 (5.9%) |
| N1 (folder label neysan, not individually reviewed) | 250 | 158/250 (63.2%) | 63/250 (25.2%) | 3/250 (1.2%) | 26/250 (10.4%) |
| Pooled, all 621 rows (620 unique images) | 621 | 415/621 (66.8%) | 139/621 (22.4%) | 7/621 (1.1%) | 60/621 (9.7%) |

Silent acceptance is the correct 8-class answer (`vanet`) given without a review flag; it is not a
classification error. It means the Neysan image is not distinguishable from an ordinary vanet output.

## D. Confirmed Neysan, not from v1/test, by Gate-1 visibility / lighting

| Group | n | vanet rate | conf. median | min | needs_review | predicted vanet, not flagged (silent acceptance) | predicted vanet, flagged (sent to review) | predicted non-vanet, not flagged (confident misclassification) | predicted non-vanet, flagged (error caught by review) |
|---|---|---|---|---|---|---|---|---|---|
| DAY / CLEAR | 276 | 252/276 (91.3%) | 0.9924 | 0.4380 | 76/276 (27.5%) | 198/276 (71.7%) | 54/276 (19.6%) | 2/276 (0.7%) | 22/276 (8.0%) |
| Day, partial / not observable | 17 | 16/17 (94.1%) | 0.9809 | 0.4218 | 4/17 (23.5%) | 13/17 (76.5%) | 3/17 (17.6%) | 0/17 (0.0%) | 1/17 (5.9%) |
| Night | 35 | 27/35 (77.1%) | 0.9544 | 0.4744 | 17/35 (48.6%) | 17/35 (48.6%) | 10/35 (28.6%) | 1/35 (2.9%) | 7/35 (20.0%) |
| Infrared | 5 | 4/5 (80.0%) | 0.7643 | 0.4255 | 3/5 (60.0%) | 2/5 (40.0%) | 2/5 (40.0%) | 0/5 (0.0%) | 1/5 (20.0%) |
| Low light / unclear | 4 | 3/4 (75.0%) | 0.9929 | 0.7943 | 1/4 (25.0%) | 3/4 (75.0%) | 0/4 (0.0%) | 0/4 (0.0%) | 1/4 (25.0%) |

## E. Ordinary-vanet baseline

| Set | n | vanet rate | conf. median | Q1 | Q3 | min | needs_review |
|---|---|---|---|---|---|---|---|
| Validation vanet (main baseline) | 26 | 25/26 (96.2%) | 0.9998 | 0.9976 | 1.0000 | 0.6506 | 4/26 (15.4%) |
| Test vanet (non-pristine, descriptive only) | 50 | 48/50 (96.0%) | 0.9996 | 0.9933 | 1.0000 | 0.6623 | 7/50 (14.0%) |

Validation confidences are stored with 6 decimals and Test confidences with 4; their flags are
recomputed (validation) or taken as recorded (Test). Validation rows within 1e-6 of 0.95: 0.

Flag rate: Neysan subgroup vs validation vanet (raw counts; difference in percentage points):

- Confirmed Neysan, not from v1/test: 101/337 vs 4/26 -> +14.6 points
- Confirmed Neysan from v1/test: 9/34 vs 4/26 -> +11.1 points
- N1 (folder label neysan, not individually reviewed): 89/250 vs 4/26 -> +20.2 points

## F. Integrity checks

- one prediction per manifest row (621): PASS
- no duplicate inference rows (unique paths): PASS
- prediction order / path mapping matches the manifest: PASS
- predicted_class is one of the 8 classes: PASS
- probabilities in [0, 1] and sum to 1 (+-1e-5): PASS
- argmax of probabilities equals predicted_class: PASS
- confidence_full equals the maximum probability: PASS
- rounded confidence equals predict_image output: PASS
- rounded full probabilities equal predict_image probabilities: PASS
- needs_review exactly equals confidence_full < 0.95: PASS
- known duplicate pair got identical predictions: PASS
- visibility group present for all 337 confirmed-not-v1/test rows: PASS
- checkpoint, manifests and historical outputs (12_*, 13_*, 24_*) unchanged: PASS

## G. Limitations

- Neysan is a vanet subtype, not a ninth class. The model cannot identify the Neysan subtype.
- Silent acceptance (predicted `vanet`, not flagged) is not an 8-class classification error.
- N1 images were not individually reviewed; their labels may be noisy.
- The validation baseline has 26 images, so the flag-rate difference is imprecise.
- This evaluation cannot predict the result on the mentor-held Test set.
- The set is 100% Neysan, so it cannot estimate Neysan prevalence, or the share of flagged images that
  are Neysan, in a mixed stream.
- It does not establish general calibration or generalisation to other cameras or conditions.
- Per-image predictions: `reports/experiments/25_swin_neysan_predictions.csv`.
