# Neysan rejection (`unknown` output)

Added 2026-10-08. The 8-class model is unchanged; this is one extra step in the test runner.

## Why

The mentors' test rule is that a Neysan pickup must **not** be treated as a `vanet`. The project's
earlier taxonomy (`decisions/REVIEW_PROTOCOL.md`, `reports/experiments/25_swin_neysan_evaluation.md`)
treated Neysan as a subtype of `vanet`, and the 8-class model predicts `vanet` for about 89% of
Neysan images. To follow the mentors' rule without retraining, the runner reports such images as
`unknown` instead of one of the 8 classes.

## How it works

- **Features:** for every image, the runner takes the 768-number feature vector that the frozen model
  feeds into its last layer (`model.net.head`).
- **Detector:** a logistic regression ("Neysan / not Neysan") with standardization is applied to that
  vector. The file is `checkpoints/neysan_detector.json`, SHA256
  `1f4106cbcef2c9ad3d7b5b64e017ede1f4ce03ecbdc3fb9527af0a26a845e3c7`. It is built for checkpoint
  SHA256 `f03bacde...64db` only; the runner checks both values.
- **Rule:** if p(Neysan) ≥ 0.5, the output is `unknown`, whatever the 8-class prediction was. The 0.5
  threshold was fixed in advance, not tuned. Otherwise the 8-class prediction is kept unchanged.
- **`needs_review`:** still the 8-class confidence < 0.95, as before.
- **Turning it off:** `--no-neysan-detector` gives the plain 8-class output. If the default detector
  file is missing, the runner prints a warning and runs the 8-class model only.

## How it was built

Notebook: `colab/neysan_detector_colab.ipynb`. It was run once in Colab on a GPU, with the
acceptance rules written down before the run.

- **Negatives:** the 2633 training-split images (all 8 classes).
- **Positives:** confirmed Neysan images from `decisions/neysan_eval.csv` (`basis = CONFIRMED_NEYSAN`),
  split by SHA256 with seed 42 into train / calibration / holdout = 259 / 56 / 56.
- **Model:** `StandardScaler` + `LogisticRegression(C=1.0, class_weight="balanced")`.
- **Pipeline check first:** the frozen model reproduced its recorded validation result, 640 / 659.
- **Data not used:** the historical 400-image Test set was not used.

## Results (from the Colab run)

| Check | Result | Acceptance rule |
|---|---|---|
| Holdout confirmed Neysan (never seen) rejected | 55 / 56 (0.982) | ≥ 0.80 |
| Calibration confirmed Neysan rejected | 54 / 56 (0.964) | - |
| N1 Neysan (folder label only, not reviewed; copies of training images excluded) rejected | 0.944 (n = 249) | - |
| Validation images (659 ordinary images) rejected | 1 / 659 (one `vanet`) | ≤ 6 |
| 8-class validation correct after rejection | 639 / 659 (was 640) | - |
| **Decision** | **PASS** | |

## How it shows in the results

- **`predictions.csv`:**
  - `predicted_class` is the final output: one of the 8 classes, or `unknown`;
  - `model_class` is the 8-class prediction;
  - `neysan_score` is p(Neysan).
- **`unknown_predictions.csv`** lists the images that were rejected.
- **Metrics:**
  - an `unknown` output counts as wrong for the image's folder class;
  - macro scores still average over the 8 classes;
  - the confusion matrix gets a 9th column, `unknown`.

  So on a test where Neysan images are placed in the `vanet` folder (or in other class folders), each
  rejected Neysan lowers accuracy. This is intended: the model does not accept a Neysan as a `vanet`.

## Limits

- About 2% of the held-out Neysan images, and about 6% of the unreviewed N1 images, were not
  rejected. They keep their 8-class prediction.
- On ordinary images, about 1 in 659 may be rejected by mistake.
- GPU and CPU can differ in the last decimals, so an image with a score extremely close to 0.5 can
  differ between machines.
