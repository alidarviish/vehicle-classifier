# 27 - Dataset v3 and the final model `v3_final`

Goal: better generalization to the external images while keeping Swin-Tiny. The earlier final model
`swin_t_ft_aug` (report 22) reached 0.9478 on the external test0; most errors were `kamyun` /
`kamyunet` and images whose style differs from our own data.

## 1. Dataset v3

Built once with `scripts/archive/build_dataset_v3.py`. The folder `data/dataset_v3/` (not in Git) holds
one sub-folder per split and class, and `split_v3.csv` with one row per image (path, SHA256, label,
split). `scripts/refresh_split_v3.py` rebuilds that CSV and `data/split_manifest_v3.csv` from the
folders after manual changes, and checks that no test image is also in train, validation or Neysan.

- **External data split in two.** The external test0 (961 images) was split 50/50 per class. One half is
  the new final test; the other half went 80/20 to train / validation.
- **Near duplicates stay together.** Images with a perceptual-hash (pHash) distance <= 5 were always
  put on the same side, and a test0 image that looks like one of our own images was never put in the
  test half.
- **Our old frozen Test (400 images)** was moved into train; it is no longer a test set.
- **The other external scenarios** (test2, test3, test4) were kept only with images of the test half,
  so none of them was seen in training.

| Split | Images | Content |
|---|---:|---|
| train | 3418 | our train 2630 + our old Test 400 + external 388 |
| validation | 755 | our validation 659 + external 96 |
| Neysan (detector only) | 701 | our 624 + external 77 |
| test0 | 479 | held-out half of the external test0 |
| test2 / test3 / test4 | 557 / 537 / 556 | the same 479 images in the other external scenarios, plus Neysan images |

What the test scenarios contain (measured by SHA256 against test0):

| Scenario | The 479 test0 images | Neysan images added |
|---|---|---|
| test0 | same labels | none |
| test2 | same labels | 78, all in the `vanet` folder |
| test3 | same labels | 58, spread over `vanet`, `minibus`, `ambulance`, `autobus`, `taxi`, `savari` |
| test4 | 76 labels swapped between `kamyun` and `kamyunet` | 77, mostly in `vanet` |

So test0 is the clean measure of the 8-class model; test2 and test3 add the Neysan rule; test4 also
contains labels that disagree with test0.

## 2. Training runs

`python -m src.train_v3 --run-name <name> [options]`. Every run: Swin-Tiny (`IMAGENET1K_V1`), 224x224,
ImageNet normalization, Adam (head 1e-3, unfrozen stages 1e-4), batch 32, seed 42, head only in
epochs 1-5, best epoch by validation macro F1. Only train and validation are read.

| Run | Augmentation | Unfrozen from epoch 6 | Other | Labels | Best epoch | Val accuracy | Val macro F1 |
|---|---|---|---|---|---:|---:|---:|
| `v3_base` | old | stage 7 + norm | - | v3 | 20 | 0.9603 | 0.9539 |
| `v3_aug` | strong | stage 7 + norm | - | v3 | 19 | 0.9642 | 0.9606 |
| `v3_aug_top2` | strong | stages 4-7 + norm | - | v3 | 18 | 0.9775 | 0.9769 |
| `v3_aug_top2_cos` | strong | stages 4-7 + norm | cosine LR, 30 epochs | v3 | 26 | 0.9735 | 0.9738 |
| `v3_aug_top2_rl` | strong | stages 4-7 + norm | - | relabel round 1 | 17 | 0.9854 | 0.9803 |
| **`v3_final`** | strong | stages 4-7 + norm | - | relabel rounds 1+2 | 19 | **0.9854** | **0.9850** |

- **old** augmentation = RandomResizedCrop (0.8-1), flip, rotation 10, brightness/contrast 0.2 (as
  `swin_t_ft_aug`).
- **strong** = RandomResizedCrop (0.7-1), flip, rotation 10, ColorJitter (0.3 / 0.3 / 0.3, hue 0.02),
  RandomGrayscale 0.2, GaussianBlur (p 0.3), RandomErasing (p 0.3).
- **What helped:** stronger augmentation (+0.007 F1) and unfreezing two more stages (+0.016 F1). The
  cosine schedule did not help in this setup.
- **Comparability:** the last two runs use corrected labels (section 3), so their validation labels
  differ slightly from the first four. Histories: `reports/v3/<run>_history.csv`.

![v3_final training history](../../docs/figures/v3_final_training.png)

Validation loss is lowest around epoch 10 and rises slightly afterwards while F1 stays flat, a mild
sign of overfitting.

## 3. `kamyun` / `kamyunet` relabeling (external labeling rule)

The remaining validation errors were mostly `kamyun` <-> `kamyunet`. A review showed that our own data
labeled light cab-over trucks inconsistently, while the external data follows one rule:

- **`kamyunet`:** light cab-over trucks such as Isuzu, JAC, Hyundai and Foton.
- **`kamyun`:** heavy trucks such as Scania, Volvo FH, heavy or bonneted Mercedes-Benz, FAW, dump
  trucks and tractor units.

Train and validation images were checked against this rule in two rounds. Only train and validation
were changed; the test folders were not opened for this.

| Round | Decision list | kamyun -> kamyunet | kamyunet -> kamyun | train / val |
|---|---|---:|---:|---|
| 1 | `decisions/v3_kamyun_relabel_round1.csv` | 39 | 2 | 31 / 10 |
| 2 | `decisions/v3_kamyun_relabel_round2.csv` | 18 | 0 | 16 / 2 |

Validation errors went from 17 (`v3_aug_top2`) to 11 (`v3_final`).

## 4. `needs_review` threshold and Neysan detector

Built with `scripts/build_neysan_detector_v3.py`, which never reads the test folders. Same method as
before ([HISTORY](../../docs/HISTORY.md#first-neysan-detector-swin_t_ft_aug)); the acceptance rules
were fixed before the run.

- **`needs_review` threshold:** the lowest of 0.70 / 0.80 / 0.90 / 0.95 that flags at least half of the
  validation errors -> **0.90**.
- **Detector:** StandardScaler + logistic regression on the 768-number head input of `v3_final`.
  Negatives: the 3418 train images. Positives: 70% of the trusted Neysan images (315), split by SHA256
  with seed 42; 15% calibration, 15% holdout. Fixed threshold p(Neysan) >= 0.5 -> `unknown`.

| Check | Result | Rule |
|---|---|---|
| Holdout trusted Neysan rejected | 67 / 68 (0.985) | >= 0.80 |
| Calibration trusted Neysan rejected | 65 / 68 (0.956) | - |
| N1 Neysan (folder label only, not reviewed) rejected | 0.980 | - |
| Validation images rejected | 2 / 755 | <= 7 |
| 8-class validation correct, before / after rejection | 744 / 742 of 755 | - |
| **Decision** | **PASS** | |

Files: `checkpoints/v3_final_best.pt` (SHA256 `bb8b159a2b1c42ab92cd1b1c33020ac32550baebe5200a20071525e3279d3848`)
and `checkpoints/neysan_detector_v3.json`
(SHA256 `69bc046a9c4655d8790e128afa60e8446c255a922523833a5cf03c1d3f57cf33`). Both are checked by the
test runner and are not in Git.

## 5. Final test

`python run_test.py --input data/dataset_v3/test --output <folder>`, run once for `v3_final` after the
model, threshold and detector were fixed. For comparison the earlier model `swin_t_ft_aug` (with its own
detector and threshold 0.95) was run once on the same folders. The test was not used for any decision.

| Scenario | `swin_t_ft_aug` accuracy | `v3_final` accuracy | `v3_final` macro F1 |
|---|---:|---:|---:|
| **test0** (479) | 0.9478 | **0.9666** | **0.9720** |
| test2 (557) | 0.8259 | 0.8456 | 0.9045 |
| test3 (537) | 0.8566 | 0.8752 | 0.8517 |
| test4 (556) | 0.7230 | 0.7356 | 0.8056 |
| all (2129) | - | 0.8516 | 0.8618 |

An `unknown` output counts as wrong, so every rejected Neysan in a class folder lowers accuracy in
test2-test4; this is intended by the labeling rule.

![test0 confusion matrix](../../docs/figures/test0_confusion_matrix.png)

Split by image type:

| | `swin_t_ft_aug` | `v3_final` |
|---|---:|---:|
| test0 images correct | 454 / 479 | **463 / 479** |
| ordinary images wrongly reported as `unknown` | 3 | **0** |
| Neysan images rejected (test2-test4) | 192 / 213 (0.901) | 186 / 213 (0.873) |
| test0 images flagged `needs_review` | 49 (threshold 0.95) | 16 (threshold 0.90) |

- **test0 errors of `v3_final` (16):** 7 kamyun -> kamyunet, 4 kamyunet -> kamyun, 2 savari -> taxi,
  1 taxi -> savari, 1 savari -> vanet, 1 autobus -> minibus.
- **Neysan per scenario (`v3_final`):** test2 69 / 78 (0.885), test3 49 / 58, test4 68 / 77.
- **test4:** 401 of its 479 test0 images are correct; the difference to test0 comes from the 76 swapped
  `kamyun` / `kamyunet` labels.

## Conclusion and limits

`v3_final` is the new final model. It improves the clean external test from 0.9478 to 0.9666 (macro F1
0.9361 -> 0.9720), never rejects an ordinary test image, and needs fewer human reviews.

- **Target not reached:** the goal of 0.98-0.99 on the external test was not reached; most remaining
  errors are `kamyun` / `kamyunet`.
- **Neysan:** the v3 detector rejects 98.5% of our held-out Neysan images, but only 87.3% of the
  Neysan images in the external scenarios (the earlier detector: 90.1%).
- **Single run:** one seed per setting; the differences between the best runs are a few validation
  images, and validation was used both for selection and for the threshold.
- **Training data from the external set:** half of test0 is in train / validation, so test0 measures images
  of the same source and style, not a completely new camera.
