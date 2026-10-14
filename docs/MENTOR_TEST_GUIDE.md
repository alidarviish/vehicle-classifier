# Testing the final model on new test images

This guide runs the finished Vehicle Classifier on a new, unseen test set. The model is final:
running a test does not train or change anything, and the results are not used to change the model,
the thresholds or the preprocessing.

## The model

| Item | Value |
|---|---|
| Model | Swin-Tiny (torchvision `swin_t`, ImageNet weights), fine-tuned on dataset v3 (run `v3_final`), epoch 19 |
| Model file | `v3_final_best.pt` (about 105 MB), SHA256 `bb8b159a2b1c42ab92cd1b1c33020ac32550baebe5200a20071525e3279d3848` |
| Classes | ambulance, autobus, kamyun, kamyunet, minibus, savari, taxi, vanet |
| Preprocessing | whole image resized to 224x224 -> tensor -> ImageNet normalization; no augmentation |
| `needs_review` | `True` when the model's confidence is below 0.90; only a flag for a human check, the prediction is kept |
| Neysan -> `unknown` | a Neysan is not accepted as a `vanet`: if the Neysan detector (`neysan_detector_v3.json`, SHA256 `69bc046a...`) gives p(Neysan) >= 0.5, the output is `unknown`. Details: [NEYSAN_REJECTION.md](NEYSAN_REJECTION.md) |

The runner checks the SHA256 of the model file and of the detector before predicting.

## What you need

- **The project code:** this repository.
- **The model and the detector:**
  - `v3_final_best.pt` and `neysan_detector_v3.json` are not in Git; you receive them separately.
  - Put both in `checkpoints/`, or pass their paths with `--checkpoint` and `--neysan-detector`.
- **Python packages:**
  - Python 3.9+ with `torch`, `torchvision`, `Pillow`, `numpy`, `matplotlib`.
  - Install them with `pip install -r requirements.txt`.

## Test folder layout

One folder per class; the folder name is the true label:

```
TEST/
├── ambulance/
├── autobus/
├── ...
└── vanet/
```

- **Several tests at once:** pass the parent folder, e.g. `TESTS/test0/...`, `TESTS/test2/...`.
  The summary shows the overall result and one line per test (`PER TEST GROUP`).
- **Folder names:**
  - Class folder names are not case-sensitive.
  - Deeper folders are fine; the nearest class folder above an image is its label.
- **Images outside class folders** are predicted without a label and are not in the metrics.
- **Formats:**
  - Images: `.jpg .jpeg .png .webp .bmp .tif .tiff`.
  - Other files are listed in `skipped_files.csv`.
  - Images that cannot be opened are listed in `failed_images.csv`.

## Run on a computer

```bash
python run_test.py --input path/to/TEST --output results_test
```

On Windows, put paths with spaces in quotes: `--input "D:\mentor data\TEST"`.

| Option | Meaning |
|---|---|
| `--input` | test folder (required) |
| `--output` | result folder (default `results`); must be new or empty, unless `--overwrite` |
| `--checkpoint` | model file (default `checkpoints/v3_final_best.pt`) |
| `--neysan-detector` | detector file (default `checkpoints/neysan_detector_v3.json`) |
| `--no-neysan-detector` | plain 8-class output, never `unknown` |
| `--batch-size` | images per step (default 32); lower it if the GPU runs out of memory |
| `--device` | `auto` (GPU if available), `cpu` or `cuda` |
| `--overwrite` | allow a non-empty output folder |
| `--no-report-images` | skip the per-test picture pages (faster) |
| `--review-threshold` | another `needs_review` threshold (default 0.90; not for the official test) |

## Result files

| File | Content |
|---|---|
| `summary.txt` | the summary that is also printed on screen |
| `metrics.json` | all numbers: overall, per class, per test group, confusion matrix |
| `predictions.csv` | one row per image |
| `wrong_predictions.csv` | images whose output differs from their folder |
| `unknown_predictions.csv` | images reported as `unknown` (Neysan) |
| `needs_review.csv` | images with confidence below 0.90 |
| `confusion_matrix.png` / `.csv` | rows = true class, columns = predicted class plus `unknown` |
| `failed_images.csv` | images that could not be opened (with the error) |
| `skipped_files.csv` | files that are not images |
| `<test>/` | one folder per test group (`all` when there are none), see below |

Each test-group folder holds that group's `confusion_matrix.png` / `.csv`, `wrong.csv`,
`needs_review.csv` and `unknown.csv`, plus picture pages for a quick visual check: `errors_N.png`,
`unknown_N.png` and `needs_review_N.png`. Each page shows 48 thumbnails, titled with the file name,
the folder class, the output and the score. These pages show test images, so keep result folders out
of Git (`results/` is ignored).

Columns of `predictions.csv`:

| Column | Meaning |
|---|---|
| `image_path` | path inside `--input` |
| `group` | the folder above the class folder (e.g. `test2`; `.` if none) |
| `true_label` | class from the folder name (empty if none) |
| `predicted_class` | final output: one of the 8 classes or `unknown` |
| `model_class` | the 8-class model's prediction (before the Neysan rule) |
| `neysan_score` | p(Neysan) |
| `confidence` | the 8-class model's highest probability |
| `needs_review` | `True` if confidence < 0.90 |
| `correct` | `True` / `False`; `unknown` always counts as `False` |
| `p_ambulance` ... `p_vanet` | the 8 class probabilities |

CSV files are UTF-8 with BOM, so Excel opens them correctly.

**Metrics:**
- They are computed in the same way as the project's own evaluation, on the images with a class
  folder that were read successfully.
- Macro precision / recall / F1 average over all 8 classes.
- An `unknown` output counts as wrong for the image's folder class. So Neysan images placed in a
  class folder lower the score. This is intended.

## Troubleshooting

| Problem | What to do |
|---|---|
| `checkpoint not found` / `Neysan detector not found` | put the file in `checkpoints/` or pass its path |
| `SHA256 ... is not the final model` / `not the accepted file` | the file is damaged or different; copy it again |
| `output folder is not empty` | choose a new `--output`, or add `--overwrite` |
| `no GPU is available` | use `--device auto` or `cpu` |
| GPU out of memory | lower `--batch-size`, e.g. 8 |
| `No module named 'src'` | run `run_test.py` from the complete project folder |
| `No module named torch` | `pip install -r requirements.txt` |
| everything is "without class folder" | check the folder spelling (`kamyunet`, not `kamionet`) |

GPU and CPU can differ in the last decimals of a probability. This can only change an image whose
confidence is extremely close to 0.90, or whose Neysan score is extremely close to 0.5.
