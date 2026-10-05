# kamyun / kamyunet: ambiguity and label-consistency evidence

This file records the evidence collected about the boundary between `kamyun` and `kamyunet`. It is
documentation only. No label, split, image, checkpoint, Test or Neysan file was changed, and no
training, inference or new experiment was run for these audits. The validation predictions used
below are the existing ones of the final model `resnet224_ft_aug`
(`reports/analysis/resnet224_ft_aug/val_audit/val_predictions.csv`, generated, not tracked).

## 1. Status

**PARTIAL DISTINCTION ONLY.**

- This is not a working definition of the two classes.
- No explicit, recoverable definition of the boundary was found in the project records or in the
  records of the earlier dataset review
  (`docs/DATA_DECISIONS.md`, `decisions/`, the review files, the old relabel list and its history).
- The earlier reviews record decisions per image ("manual visual review") without the criterion used.

## 2. Observed distinction (dataset audit)

Scope: the `kamyun` and `kamyunet` rows of the train + validation pool (490 kamyun, 465 kamyunet).
A deterministic sample of 60 images per class (seed 42) was coded from contact sheets.

| Cab configuration | kamyun (n=60) | kamyunet (n=60) |
|---|---|---|
| wide cab-over | 37 | 0 |
| bonneted | 13 | 0 |
| narrow cab-over | 9 | 60 |
| not observable | 1 | 0 |

- Body / load types (open bed, box or refrigerated box, tank, covered load) occur in both classes.
- Axles and wheel count are not observable in almost all images (front view from above); they
  cannot be used here.
- Crop size differs between the classes on average, but it depends on camera distance and cropping.
  It was not accepted as a criterion, and no size threshold was derived.
- Exact duplicates: 4 groups, all within `kamyunet`; none with different labels.
- Near duplicates (pHash distance ≤ 5, the rule of the earlier dataset review): one cross-label pair, two
  different JAC cab-over trucks of the same visible type: `v1/train/kamyun/216040476.jpg` (kamyun)
  and `v2/unclean/kamyunet/214105935.jpg` (kamyunet).

## 3. Narrow-cab review

The 9 sampled `kamyun` images with a narrow cab-over were each compared with 3 visually close
`kamyunet` images from the same sample.

| Status | Count |
|---|---|
| CLEARLY_DISTINCT | 0 |
| POTENTIALLY_DISTINCT | 2 |
| VISUALLY_OVERLAPPING | 6 |
| INSUFFICIENT_EVIDENCE | 1 |

- The two `POTENTIALLY_DISTINCT` cases differ from their comparisons only in the cab front design;
  this was not checked against the `kamyunet` class as a whole.
- K09 = review KK01 (`v1/train/kamyun/218936367.jpg`) showed no visible feature that separates it
  from several similar `kamyunet` images (same cab front, same open-bed configuration).
- No recurring visible feature other than narrow / wide cab was found.

## 4. Existing validation errors

- Of the 9 narrow-cab cases, only K06 and K09 are validation images; the other 7 are train images
  with no validation prediction.
- K06: predicted `kamyun`, correct.
- K09 / KK01: predicted `kamyunet` (confidence 0.7809), a `kamyun → kamyunet` error (validation
  index 25).
- No new inference was run; the existing validation predictions were joined.

## 5. Conflicting-neighbour evidence

For each of the 12 `kamyun ↔ kamyunet` validation errors of the final model, images of the opposite
label with the same appearance were searched in the audit sample and the recorded duplicates.

| Status | Count | Validation indices |
|---|---|---|
| CONFLICTING_NEIGHBOR_FOUND | 7 | 25, 226, 240, 482, 84, 512, 519 |
| POSSIBLE_CONFLICT_BUT_INSUFFICIENT_EVIDENCE | 5 | 230, 501, 273, 527, 542 |
| NO_CONFLICTING_NEIGHBOR_FOUND | 0 | – |

- A conflicting neighbour has the same cab front / configuration **and** the same body / load
  configuration as the error image. Brand alone or body type alone was not enough.
- For 6 of the 7, the conflicting neighbour is a train image.
- For index 482, the only conflicting neighbour is validation index 84 (label `kamyunet`).
- No error image is part of a recorded exact or near-duplicate group.

## 6. Provenance and label consistency (facts only)

The old relabel list (`decisions/old_label_overrides.csv`, from an earlier dataset review, not applied)
and the current labels are recorded separately; the old recommendation is not the current ground
truth.

- Validation indices 25, 230, 240 (reviews KK01, KK04, KK06):
  - the old list recommended `kamyunet`;
  - the current label is `kamyun`, and the 2026-09-27 review decided `KEEP_KAMYUN`.
- Validation index 84 (review KK02):
  - folder label `kamyun`, current label `kamyunet` (`RELABEL_KAMYUNET`; the old list also says `kamyunet`);
  - the existing prediction of the final model is `kamyun`.
- Sample images K32 and K40 (train, used as neighbours):
  - current label `kamyun`;
  - the old list recommended `kamyunet`.

## 7. Interpretation

- The evidence is consistent with label inconsistency and an ambiguous boundary inside the group of
  narrow cab-over trucks.
- It does not show which label is correct in the conflicting cases.
- The hard-example explanation is not ruled out.
- None of these audits on its own justifies relabelling the dataset.
- No working definition was derived from this evidence.

These audits have **not** shown that:

- all 12 errors are label noise;
- all narrow-cab `kamyun` images should be `kamyunet`;
- `kamyun = wide cab` is a complete definition;
- `kamyunet = narrow cab` is a complete definition;
- a numeric size threshold exists.

## 8. Limitations

- The visual reviews were done by one reviewer, not blind (the label was visible).
- Comparison images were chosen from existing audit evidence by the same reviewer.
- Some similarities are only at the level of visual family / configuration (for example, an older
  or newer Isuzu front design), not a recorded model identification.
- The conflicting-neighbour search used the audit sample (60 + 60 images) and the recorded
  duplicates; it was not an exhaustive search of the dataset.
- There is no independent ground truth to resolve the ambiguity.

## 9. Sources

The audits were run on the private dataset; their outputs (including contact sheets) are retained
outside the repository and are not part of the repository.

Repository inputs: `data/split_manifest.csv` (not tracked),
`reports/analysis/resnet224_ft_aug/val_audit/val_predictions.csv` (not tracked),
`decisions/old_label_overrides.csv`, `decisions/kamyun_kamyunet_review.csv`,
`decisions/kamyunet_kamyun_review.csv`.
