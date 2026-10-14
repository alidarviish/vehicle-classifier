# Neysan rejection (`unknown` output)

The 8-class model has no Neysan class. The test runner adds one step on top of it, so that a Neysan
pickup is reported as `unknown` instead of `vanet`. The 8-class model itself is not changed.

## Why

The mentors' rule is that a Neysan (the common Iranian Nissan pickup) must **not** be accepted as a
`vanet`. Neysan was never one of the 8 classes; without this step the model predicts `vanet` for most
Neysan images.

## How it works

- **Features:** for every image, the runner takes the 768-number feature vector that the model feeds
  into its last layer (`model.net.head`).
- **Detector:** a logistic regression ("Neysan / not Neysan") with standardization is applied to that
  vector. File: `checkpoints/neysan_detector_v3.json`, SHA256
  `69bc046a9c4655d8790e128afa60e8446c255a922523833a5cf03c1d3f57cf33`. It belongs to the model
  `v3_final_best.pt` (SHA256 `bb8b159a...3d8848`) only; the runner checks both files.
- **Rule:** if p(Neysan) >= 0.5, the output is `unknown`, whatever the 8-class prediction was. The 0.5
  threshold was fixed in advance, not tuned. Otherwise the 8-class prediction is kept.
- **`needs_review`:** independent of the detector: the 8-class confidence < 0.90.
- **Turning it off:** `--no-neysan-detector` gives the plain 8-class output.

## How it was built

```bash
python scripts/build_neysan_detector_v3.py
```

The script never reads the test folders. The acceptance rules were written down before the run.

- **Negatives:** the 3418 training images of dataset v3 (all 8 classes).
- **Positives:** trusted Neysan images (confirmed Neysan from our `vanet` folders and Neysan images from
  the mentors' data), split by SHA256 with seed 42 into train / calibration / holdout = 70 / 15 / 15%
  (315 / 68 / 68).
- **Extra check only:** the N1 images (Neysan by folder name, never reviewed one by one).
- **Model:** `StandardScaler` + `LogisticRegression(C=1.0, class_weight="balanced")`.

## Results

| Check | Result | Acceptance rule |
|---|---|---|
| Holdout trusted Neysan rejected | 67 / 68 (0.985) | >= 0.80 |
| Calibration trusted Neysan rejected | 65 / 68 (0.956) | - |
| N1 Neysan rejected | 0.980 | - |
| Validation images (755 ordinary images) rejected | 2 / 755 | <= 7 |
| 8-class validation correct after rejection | 742 / 755 (was 744) | - |
| **Decision** | **PASS** | |

On the mentors' test scenarios ([report 27](../reports/experiments/27_v3_dataset_and_final_model.md)):
186 of 213 Neysan images (0.873) were rejected, and no ordinary test image was reported as `unknown`.

## How it shows in the results

- **`predictions.csv`:** `predicted_class` is the final output (one of the 8 classes, or `unknown`),
  `model_class` is the 8-class prediction and `neysan_score` is p(Neysan).
- **`unknown_predictions.csv`** and the `unknown_*.png` pages list the rejected images.
- **Metrics:** an `unknown` output counts as wrong for the image's folder class; macro scores still
  average over the 8 classes; the confusion matrix gets a 9th column, `unknown`.

## Limits

- About 13% of the Neysan images in the mentors' scenarios were not rejected, against 1.5% of our own
  held-out Neysan images: the mentors' Neysan images are harder for the detector.
- On ordinary images, about 2 in 755 validation images were rejected by mistake.
- GPU and CPU can differ in the last decimals, so an image with a score extremely close to 0.5 can
  differ between machines.

The first detector, built for the earlier model `swin_t_ft_aug`, is described in
[HISTORY.md](HISTORY.md#first-neysan-detector-swin_t_ft_aug).
