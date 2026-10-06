# Dataset decisions

This file records the dataset decisions that were confirmed during the project, with the
file that holds the evidence where there is one. Raw images and the dataset itself stay
outside the repository; only relative paths (`<source>/<split>/<folder>/<file>`) and SHA256
hashes are recorded here and in `decisions/`.

## Classes and Neysan

- The model has exactly 8 classes: ambulance, autobus, kamyun, kamyunet, minibus, savari, taxi, vanet
  (`src/dataset.py`, `CLASSES`).
- Neysan is not a ninth class. It is a subtype of `vanet` (model label `vanet`, subtype `neysan`),
  confirmed by the professor (`decisions/REVIEW_PROTOCOL.md`).
- Neysan images are never used for training, validation or Test. They form a separate
  evaluation set, `neysan_eval`, used only after the model is frozen.
- A017 (`v1/train/vanet/214726285.jpg`, a modern Nissan / Navara-style pickup) is `vanet`,
  subtype other, `NOT_NEYSAN` (`decisions/neysan_review.csv`, `decisions/REVIEW_PROTOCOL.md`).

## Dataset statuses

Every image in `data/manifest.csv` (4360 images, v1 + v2) gets exactly one status, applied in this
order (`scripts/build_statuses.py`):

| Order | Rule | Status | Count |
|---|---|---|---|
| 1 | `CONFIRMED_NEYSAN` in `decisions/neysan_review.csv` or `decisions/neysan_test_review.csv` | `neysan_eval` | 371 |
| 2 | `folder_label == neysan` (policy N1) | `neysan_eval` | 250 |
| 3 | any other image from `v1/test` | `test_candidate` | 366 |
| 4 | old drop decision | `excluded` | 47 |
| 5 | everything else | `train_val_pool` | 3326 |

## Old label_overrides policy

An earlier dataset review recorded a list of 112 decisions (48 `drop`, 64 `relabel`), kept in
this repository as `decisions/old_label_overrides.csv` (columns `path, action, new_label, reason`;
paths without the source prefix, each matching exactly one manifest image).

- Drops are kept: 47 of the 48 dropped images are `excluded`:
  38 exact or near-duplicates of a v1/test image (Test leakage) and 9 quality / wrong-class images.
  The 48th drop (`v1/train/vanet/214844236.jpg`, an exact duplicate of
  `v1/unclean/neysan/214844236.jpg`) is `CONFIRMED_NEYSAN` and goes to `neysan_eval` (rule 1 comes first).
- The 64 old relabels are NOT applied; pool images keep their `folder_label`
  (59 kamyun -> kamyunet, 2 savari -> taxi, 1 taxi -> savari, 1 vanet -> savari, 1 ambulance -> vanet).

Source: the list comes from an earlier dataset review. `decisions/old_label_overrides.csv` is a
byte-for-byte copy of it (SHA256 verified against the original), and `scripts/build_statuses.py`
reads it from there; no other copy is needed.

## Human label corrections

After a visual review of baseline validation errors, 8 validation images were relabelled
(`decisions/kamyun_kamyunet_review.csv`, `decisions/kamyunet_kamyun_review.csv`, rows with a
`RELABEL_*` decision). `folder_label` is never changed; only the `label` column of the split is.

| Review | Image | folder_label | Final label |
|---|---|---|---|
| KK02 | `v1/unclean/kamyun/205437377.jpg` | kamyun | kamyunet |
| KK03 | `v1/unclean/kamyun/217998174.jpg` | kamyun | kamyunet |
| KK05 | `v2/train/kamyun/218231967.jpg` | kamyun | kamyunet |
| KK07 | `v2/train/kamyun/219057756.jpg` | kamyun | kamyunet |
| KK08 | `v2/unclean/kamyun/213234314.jpg` | kamyun | kamyunet |
| KK09 | `v2/unclean/kamyun/216456908.jpg` | kamyun | kamyunet |
| KK10 | `v2/unclean/kamyun/218307800.jpg` | kamyun | kamyunet |
| KN10 | `v2/train/kamyunet/219017445.jpg` | kamyunet | kamyun |

`KEEP_*` and `UNCERTAIN` decisions (including KN04, still `UNCERTAIN`) do not change labels.

- Later audits of the kamyun / kamyunet boundary (`decisions/KAMYUN_KAMYUNET_AMBIGUITY.md`,
  `decisions/KAMYUN_KAMYUNET_SHARED_CONFIGURATIONS.md`) recorded the status PARTIAL DISTINCTION ONLY,
  DEFINITION NOT RECOVERED. The four blind-review candidates (C05, C06, C14, C15) were not
  relabelled; no label changed after the 8 corrections above. The two classes are kept separate
  (`reports/experiments/10_final_model_selection.md`).

## Test (frozen)

- 400 images, 50 per class (`scripts/build_test.py`).
- 366 `test_candidate` images: all of v1/test except the 34 images confirmed as Neysan in
  `decisions/neysan_test_review.csv`, which moved to `neysan_eval`.
- That left 16 vanet images, so 34 approved vanet replacements were taken from the train/val pool.
  The approved list and the rule used to choose it are in `scripts/build_test.py`
  (`APPROVED_REPLACEMENTS`). A017 was deliberately not chosen.
- Checks at freeze time: no `CONFIRMED_NEYSAN`, no exact duplicate between Test and train/val,
  no exact duplicate inside Test, no excluded image. The 8 excluded unclean images that are exact
  copies of Test images stay excluded.
- Test was used once for the final evaluation of `resnet224_ft_aug`, and a second time for
  `swin_t_ft_aug` after the reopened selection, a documented protocol deviation
  (`reports/experiments/24_swin_final_test_evaluation.md`). It is not in `data/split_manifest.csv`.

## Train / validation split

- Pool: the 3326 `train_val_pool` images minus the 34 Test replacements = 3292 images
  (`scripts/make_split.py`).
- Seed 42, 20% validation, stratified per class (the folder label, as when the split was made),
  exact-duplicate groups (same SHA256) kept together in one split (15 groups).
- Result: train 2633, validation 659 (`data/split_manifest.csv`, not tracked).
- The 8 human label corrections are applied to the `label` column after the split; all 8 are in
  validation.

Class counts of the final split (`label` column of `data/split_manifest.csv`, after the 8 label
corrections):

| Class | Train | Validation |
|---|---:|---:|
| ambulance | 294 | 74 |
| autobus | 370 | 92 |
| kamyun | 397 | 93 |
| kamyunet | 367 | 98 |
| minibus | 314 | 79 |
| savari | 399 | 100 |
| taxi | 389 | 97 |
| vanet | 103 | 26 |
| **Total** | **2633** | **659** |

Train + validation = 3292 images.

## Images from `unclean` folders in the training data

- Images with `origin_split=unclean` enter `train_val_pool` only after the project's cleaning steps:
  exact-duplicate and Test-leakage checks by SHA256 (`reports/audit/exact_duplicates.csv`), the
  recorded exclusions (`decisions/old_label_overrides.csv`, 47 excluded), the Neysan review
  (`decisions/neysan_review.csv`, `decisions/neysan_test_review.csv`, policy N1) and the recorded
  label review and corrections (`decisions/kamyun_kamyunet_review.csv`,
  `decisions/kamyunet_kamyun_review.csv`).
- Once in `train_val_pool` and then in `data/split_manifest.csv`, these images are cleaned training
  data. `origin_split=unclean` only records where an image originally came from.
- This follows the project decision to merge the valid data of both sources (v1 and v2) for the
  main experiment. In the current split, 1258 of the 2633 train images and 315 of the 659
  validation images have `origin_split=unclean`.
- Balanced-batches experiment: the simulated imbalance is built only by removing images from the
  current 2633 train images; no image from `unclean` (or anywhere else) is added to train.
  Validation stays the current 659 images and Test stays the frozen Test (`decisions/test_frozen.csv`).
- This records how the existing data is interpreted; it does not change the dataset, the statuses
  or the split.

## Neysan evaluation (`neysan_eval`)

- 621 images: 371 human-confirmed `CONFIRMED_NEYSAN` (337 from `decisions/neysan_review.csv`,
  34 from `decisions/neysan_test_review.csv`) and 250 images with `folder_label == neysan`
  (50 `v1/unclean`, 200 `v2/unclean`), included by policy N1 without an individual human review.
- By source: v1/test 34, v1/train 36, v1/unclean 84, v2/train 143, v2/unclean 324.
- One exact-duplicate pair is inside the set: `v1/train/vanet/214844236.jpg` and
  `v1/unclean/neysan/214844236.jpg`.
- No `NOT_NEYSAN` and no excluded image is in the set, and none of its images is in train, val or Test.
- The images are not copied: `decisions/neysan_eval.csv` points to them in their original folders
  (`vanet` for the confirmed images, `unclean/neysan` for N1) and is the only Neysan input of
  `scripts/evaluate_neysan.py` and `scripts/evaluate_neysan_swin.py`.

## Data location

- Raw datasets (v1 `dataset/`, v2 `datasetv2_TrainUclean/`) and all images stay outside the
  repository; their local paths are in `configs/local_paths.json` (not tracked).
- `data/manifest.csv` and `data/split_manifest.csv` are generated and not tracked.
- Labelled sample figure (data audit): 16 dataset images, 2 per class for all 8 classes, each
  shown with its class label. The images are the first two of each class from `v1/train` in
  `data/manifest.csv`. Because the dataset is private, the figure is kept outside the repository and
  is not committed to Git (it must not be). It can be reproduced with
  `scripts/make_labelled_sample_figure.py`.

## Known reproducibility gaps

The inputs and lists that were missing are now in the repository:
`decisions/old_label_overrides.csv`, `decisions/test_frozen.csv` (400 images) and
`decisions/neysan_eval.csv` (621 images). The one remaining gap is outside the repository:

- The local snapshot outside the repository (folder and zip) no longer contains `snapshot_manifest.csv`,
  so its files are only traceable through their names
  (`<source>_<origin_split>_<file>`).
