# Testing the final model on your own test images

This guide is for running the finished Vehicle Classifier on a new, unseen test set. The model is
final: running this test does not train, tune or change anything, and its results are not used to
change the model, the threshold or the preprocessing.

## What the model is

| Item | Value |
|---|---|
| Model | Swin-Tiny (torchvision `swin_t`), ImageNet weights `IMAGENET1K_V1`, fine-tuned |
| Model file | `swin_t_ft_aug_best.pt` (about 105 MB), SHA256 `f03bacde0a155461ea4aede0b813b8688d6ed17973d5ad19b0d2c1b913ff64db` |
| Classes (index order) | 0 ambulance, 1 autobus, 2 kamyun, 3 kamyunet, 4 minibus, 5 savari, 6 taxi, 7 vanet |
| Preprocessing | whole image resized to 224x224 (no crop) -> tensor -> ImageNet normalization; no augmentation |
| `needs_review` | `true` when the confidence (highest class probability) is below 0.95. It is only a flag for human review; the prediction is always kept |
| Neysan → `unknown` | a Neysan pickup is not accepted as a `vanet`: if the Neysan detector (`neysan_detector.json`) gives p(Neysan) ≥ 0.5, the output is `unknown` instead of one of the 8 classes. Details: [NEYSAN_REJECTION.md](NEYSAN_REJECTION.md) |

The runner checks the SHA256 of the model file and the class order stored inside it, and the SHA256
of the Neysan detector, before predicting anything.

## What you need

- The project code (this repository).
- The model file `swin_t_ft_aug_best.pt`. It is not in Git; you receive it separately. Put it in
  `checkpoints/` or pass its path with `--checkpoint`.
- The Neysan detector `neysan_detector.json` (48 KB), received with the model file. Put it in
  `checkpoints/` or pass its path with `--neysan-detector`. Without it the runner warns and gives
  the plain 8-class output.
- Python 3.9+ with `torch`, `torchvision`, `Pillow`, `numpy`, `scikit-learn`, `matplotlib`
  (`pip install -r requirements.txt`; Google Colab already has all of them).
- No training data is needed.

## Test folder layout

Put the images in one folder per class. The folder names are the ground truth:

```
TEST/
├── ambulance/
├── autobus/
├── kamyun/
├── kamyunet/
├── minibus/
├── savari/
├── taxi/
└── vanet/
```

Other layouts also work:

- **Several test groups:** `TESTS/test1/ambulance/...`, `TESTS/test2/ambulance/...`. Pass `TESTS`;
  metrics are reported overall and per group (`test1`, `test2`, ...). You can also run each group
  separately.
- **Deeper folders:** the label is the nearest folder above the image whose name is a class, so
  `test1/vanet/day/img.jpg` is labelled `vanet`. Class folder names are not case-sensitive (`Taxi/` works).
- **No class folders:** images directly in a folder (or in folders with other names) are predicted
  without a label (prediction-only mode). If some images have class folders and some do not, metrics
  use only the labelled images and the summary says so.

Image formats: `.jpg .jpeg .png .webp .bmp .tif .tiff` (any letter case). RGB, RGBA, grayscale,
palette and CMYK images are converted to RGB exactly as during the project. Every other file is
listed in `skipped_files.csv`; nothing is ignored silently.

## Run it locally

From the repository folder:

```bash
python run_test.py --input path/to/TEST
```

With all options:

```bash
python run_test.py --input path/to/TEST --checkpoint path/to/swin_t_ft_aug_best.pt --output results_test1 --batch-size 32 --device auto
```

On Windows, quote paths that contain spaces:

```bat
python run_test.py --input "D:\mentor data\TEST" --checkpoint "D:\models\swin_t_ft_aug_best.pt" --output "D:\mentor data\results"
```

| Option | Meaning |
|---|---|
| `--input` | test folder (searched recursively); required |
| `--output` | result folder (default `results`); it must not exist yet or be empty, unless `--overwrite` is given |
| `--checkpoint` | model file (default `checkpoints/swin_t_ft_aug_best.pt`) |
| `--batch-size` | images per forward pass (default 32); lower it if the GPU runs out of memory |
| `--device` | `auto` (GPU if available), `cpu` or `cuda` |
| `--neysan-detector` | Neysan detector file (default `checkpoints/neysan_detector.json`) |
| `--no-neysan-detector` | plain 8-class output, without the Neysan → `unknown` rule |

## Run it in Google Colab

1. Upload to Google Drive: the model file, `neysan_detector.json`, your test folder (or a `.zip` of it) and, if the
   GitHub repository is not accessible to you, a `.zip` of the project code.
2. Open `colab/mentor_test_colab.ipynb` in Colab (*File → Upload notebook*).
3. In the first code cell, set `REPO_SOURCE`, `CHECKPOINT`, `NEYSAN_DETECTOR`, `TEST_INPUT` and `OUTPUT`.
4. *Runtime → Run all*. The last cell prints the summary and shows the confusion matrix. The
   result files are written to `OUTPUT` on Drive.

The notebook only prepares the files and then runs the same `run_test.py` command.

## Output files

Evaluation mode (class folders found):

```
results/
├── summary.txt               human-readable summary (also printed on screen)
├── metrics.json              all numbers: accuracy, macro precision/recall/F1, per class, per group, confusion matrix, run info
├── classification_report.txt per-class precision / recall / F1 / support
├── confusion_matrix.png      counts and row-normalized (recall on the diagonal)
├── confusion_matrix.csv      the count matrix (rows = true class, columns = predicted class, plus `unknown` if any)
├── predictions.csv           one row per image
├── wrong_predictions.csv     images whose prediction differs from their folder
├── needs_review.csv          images with confidence below 0.95
├── unknown_predictions.csv   images reported as unknown (Neysan)
├── failed_images.csv         images that could not be opened or predicted (with the error)
└── skipped_files.csv         files that are not images (with the reason)
```

Prediction-only mode (no class folders): `summary.txt`, `run_info.json`, `predictions.csv`,
`needs_review.csv`, `unknown_predictions.csv`, `failed_images.csv`, `skipped_files.csv`.

`predictions.csv` columns:

| Column | Meaning |
|---|---|
| `image_path` | path relative to `--input` |
| `group` | the folder part above the class folder (`.` for a flat `TEST/` layout) |
| `true_label` | class from the folder name (empty if there is none) |
| `predicted_class` | final output: one of the 8 classes, or `unknown` (Neysan) |
| `model_class` | the 8-class model prediction (before the Neysan rule) |
| `neysan_score` | p(Neysan) from the detector (empty when the detector is off) |
| `confidence` | highest class probability of the 8-class model |
| `needs_review` | `True` if confidence < 0.95 |
| `correct` | `True`/`False` (empty without ground truth); `unknown` is always `False` |
| `image_mode`, `width`, `height` | the original image |
| `p_ambulance` ... `p_vanet` | probabilities of all 8 classes |

CSV files are UTF-8 with BOM, so they open correctly in Excel, also with non-English file names.

About the metrics: they are computed only on images that were read successfully; failed images are
counted separately and named in the summary. Macro precision / recall / F1 average over all 8
classes, as in the project's own evaluation; if a class has no test images, an extra macro average
over the classes that are present is also reported. An `unknown` output counts as wrong for the
image's folder class (so Neysan images placed in a class folder lower the score; this is intended).

## Example of the screen output (format)

```
found 403 files: 400 images, 3 other files

VEHICLE CLASSIFIER - TEST RUN SUMMARY
============================================================
mode               : evaluation
model              : Swin-Tiny (epoch 17), checkpoint sha256 f03bacde0a155461...
preprocessing      : Resize (224, 224) -> ToTensor -> ImageNet normalization (no augmentation)
needs_review rule  : confidence < 0.95 (a flag for human review; the prediction is kept)
neysan detector    : ON - p(Neysan) >= 0.5 -> unknown (any 8-class prediction) (sha256 1f4106cbcef2c9ad...)
device             : cuda

FILES
------------------------------------------------------------
files found        : 403
images predicted   : 400
images failed      : 0   (see failed_images.csv)
non-image files    : 3   (see skipped_files.csv)
...
RESULTS (images with ground truth that were read successfully)
------------------------------------------------------------
images             : 400
correct / wrong    : <n> / <n>   (see wrong_predictions.csv)
accuracy           : <value>
macro F1           : <value>
...
```

## Troubleshooting

| Problem | What to do |
|---|---|
| `checkpoint not found` | pass the model file with `--checkpoint "<path>"` |
| `Neysan detector not found` (warning) | put `neysan_detector.json` in `checkpoints/` or pass `--neysan-detector "<path>"`; without it the output is 8-class only |
| `Neysan detector SHA256 ... is not the accepted file` | the detector file is damaged or different; copy it again |
| `checkpoint SHA256 ... is not the final model` | the model file is incomplete or a different file; download it again (compare the SHA256 above) |
| `output folder is not empty` | choose a new `--output` folder, or add `--overwrite` |
| `--device cuda was requested but no CUDA GPU is available` | use `--device auto` or `cpu` (CPU is slower but gives the same predictions) |
| GPU out of memory | lower `--batch-size` (e.g. 8); the runner also retries a failed batch one image at a time |
| `ModuleNotFoundError: No module named 'src'` | use `run_test.py` from the complete repository folder (it needs `src/` and `scripts/` next to it) |
| `ModuleNotFoundError` for torch / torchvision / sklearn | `pip install -r requirements.txt` (not needed in Colab) |
| All images are "without label" | the folders are not named after the 8 classes; check spelling (`kamyunet`, not `kamionet`) |
| An image is in `failed_images.csv` | the file is damaged, empty or not really an image; the error message says which |
| `.webp` images fail | the installed Pillow has no WEBP support; convert them to `.png` or upgrade Pillow |
| `BadZipFile` when extracting the test zip | the zip was damaged during upload; upload it again and check its SHA256 |

Notes: photos are used as stored in the file (EXIF rotation is not applied), the same way the model
was trained. GPU and CPU can differ in the last decimals of a probability; this can only change
`needs_review` for an image whose confidence is extremely close to 0.95.

## Checklist for mentors

- [ ] Model file is `swin_t_ft_aug_best.pt` with the SHA256 above (the runner checks it).
- [ ] `summary.txt` shows `neysan detector    : ON`.
- [ ] Test images are in folders named exactly after the 8 classes (or no folders for prediction only).
- [ ] `--output` points to a new, empty folder outside the test folder.
- [ ] After the run: `images failed` is 0, or the failed files in `failed_images.csv` are understood.
- [ ] `skipped_files.csv` contains only non-image files you expected.
- [ ] Report the numbers from `summary.txt` / `metrics.json`; keep `predictions.csv` for reference.
- [ ] The test results are not used to change the model, threshold or preprocessing.
