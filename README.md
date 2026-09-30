# Vehicle Classifier

8-class image classifier for vehicles cropped from traffic-camera images:
`ambulance`, `autobus`, `kamyun`, `kamyunet`, `minibus`, `savari`, `taxi`, `vanet`.

The dataset is private and is not part of this repository. Only relative image paths and SHA256
hashes are recorded (`decisions/`, reports).

## Pipeline

1. **Dataset audit.** Build a manifest of the raw images, then count images and find exact
   duplicates by SHA256 (`reports/audit/`).
2. **Decisions.** Apply the decisions in `decisions/` to the manifest, so that every image gets
   exactly one status: train/val pool, Test candidate, Neysan evaluation or excluded. The rules are
   in [`docs/DATA_DECISIONS.md`](docs/DATA_DECISIONS.md).
3. **Train/validation split.** Seed 42, 20% validation, stratified per class, exact duplicates kept
   together: 2633 train / 659 validation.
4. **Experiments.** A CNN baseline and ablations, balanced batches, CE vs BCE, then ResNet18
   transfer learning and combinations. All are compared on validation only.
5. **Final model.** Selected on validation. The `needs_review` threshold is also chosen on validation.
6. **Test.** The frozen 400-image Test set is evaluated once.
7. **Neysan evaluation.** A separate, inference-only analysis of 621 Neysan images.

## Setup

```
pip install -r requirements.txt
```

Copy `configs/local_paths.example.json` to `configs/local_paths.json` and set the two local
dataset folders. `local_paths.json` is ignored by Git.

```json
{"sources": {"v1": "PATH/TO/dataset", "v2": "PATH/TO/datasetv2_TrainUclean"}}
```

Image paths in all CSV files start with the source name (`v1/...`, `v2/...`). The paths in
`local_paths.json` resolve them to local files. All commands run from the repository root.

## Commands

The data scripts only read the raw images.

| Step | Command |
|---|---|
| Manifest | `python scripts/build_manifest.py` (writes `data/manifest.csv`) |
| Dataset audit | `python scripts/audit_dataset.py` (writes `reports/audit/`) |
| Statuses (inspection) | `python scripts/build_statuses.py --overrides decisions/old_label_overrides.csv --out <file.csv>` |
| Train/validation split | `python scripts/make_split.py --overrides decisions/old_label_overrides.csv --out data/split_manifest.csv` |
| Rebuild the Test list (check only) | `python scripts/build_test.py --overrides decisions/old_label_overrides.csv --out <file.csv>` |
| Imbalanced train subset (balanced-batches experiment) | `python scripts/make_imbalanced_train.py` |
| Training (baseline) | `python -m src.train` |
| Training (final model) | `python -m src.train --run-name resnet224_ft_aug --model resnet18 --resnet-mode fine_tuning --augmentation full_aug` |
| Validation analysis of a run | `python scripts/analyze_baseline.py --run-name <run_name>` |
| ResNet setup checks (no data) | `python scripts/verify_resnet.py` |
| Test evaluation (done once) | `python scripts/evaluate_test.py` |
| Neysan evaluation (done once) | `python scripts/evaluate_neysan.py` |
| Prediction (JSON) | `python -m src.predict path/to/image.jpg [more images]` |

Notes on the commands:

- **Output files.** `--out` files must not exist beforehand.
- **Committed decision lists.** `decisions/test_frozen.csv` and `decisions/neysan_eval.csv` are
  the committed lists; `build_test.py` output is only for comparison.
- **Experiment commands.** The command for every experiment is in its report in
  `reports/experiments/`.
- **Generated files kept out of Git.** Training writes `checkpoints/<run_name>_best.pt` and
  `reports/<run_name>_history.csv`; the validation analysis writes `reports/analysis/<run_name>/`.
- **One-time scripts.** `evaluate_test.py` and `evaluate_neysan.py` refuse to run again once their
  outputs exist.

## Final model

- **Run:** `resnet224_ft_aug`. ResNet18 with `IMAGENET1K_V1` weights, fine-tuned: head only in
  epochs 1-5, then `layer4` + head.
- **Input and augmentation:** 224x224 input with ImageNet normalization. The train-only augmentation
  is `full_aug`; there is no augmentation at inference.
- **Training settings:** Adam, CrossEntropyLoss, 20 epochs, seed 42.
- **Checkpoint:** `checkpoints/resnet224_ft_aug_best.pt` (epoch 20, not in Git).
  SHA256 `c26280e97f7324ee1ec85d7bf78a58ea0b4d2acb4cf3e975dbd52316366281a1`.
- **Validation (659 images):** accuracy 0.9605, macro precision 0.9456, macro recall 0.9523,
  macro F1 0.9482. This is the highest validation macro F1 of all runs.
- **`needs_review`:** set when the confidence (the highest softmax probability) is below 0.90. The
  threshold was chosen on validation and is defined in `src/predict.py`. It is only a flag for human
  review; no prediction is rejected. The threshold is not stored in the frozen checkpoint.

## Frozen Test

- **Contents:** 400 images, 50 per class, listed in `decisions/test_frozen.csv`
  (SHA256 `e1727eea750e73a4924d79fbb5fb810af8f8a05d822754c51ade8376e0622540`).
- **Kept separate:** no image overlaps train, validation or Neysan, checked by path and SHA256.
- **Used once:** Test was evaluated once, after the model, threshold and inference protocol were
  fixed. It was not used for any decision.
- **Results:** 379 correct / 21 incorrect.
  - Accuracy 0.9475, macro precision 0.9515, macro recall 0.9475, macro F1 0.9473.
  - `needs_review`: 29 flagged, including 12 of the 21 errors.

## Neysan

- **Not a class.** Neysan (the common Iranian Nissan pickup) is a subtype of `vanet`, not a ninth
  class. Neysan images were kept out of training, validation and Test.
- **Evaluation set:** `decisions/neysan_eval.csv`, 621 images. 371 are human-confirmed; 250 come from
  the `neysan` folders (policy N1) and were not individually reviewed.
- **Inference only:** the set was evaluated with the final model, and each image is expected to be
  predicted as `vanet`.
- **Results:** 529 of 621 were predicted `vanet` (0.8519; 0.8544 on the 371 confirmed images), and
  185 were flagged `needs_review`.
- **Per-image output:** predictions and all 8 class probabilities are in
  `reports/experiments/13_neysan_predictions.csv`.
- **Descriptive only.** No precision, F1 or macro metric is reported for this set.

## EfficientNet-B0 comparison

A separate architecture comparison, run after the final model was fixed. The final model stays
`resnet224_ft_aug`; nothing here changes it, the threshold or `src/predict.py`.

- **Tried:** feature extraction, then partial fine-tuning (the last three feature stages unfrozen from
  epoch 6), a lower learning rate for those stages, and `full_aug`.
- **Best run so far:** `effnet_b0_ft_none` (partial fine-tuning, no augmentation), validation accuracy
  0.9560 and macro F1 0.9485.
- **Learning rate:** 5e-5 for the fine-tuned stages was not better than the baseline 1e-4 in this setup.
- **Augmentation:** `full_aug` gave macro F1 0.9466, with no overall improvement over the baseline, so the
  augmentation is not changed for now.
- **Scope:** validation only, one seed; the Final Test and the Neysan evaluation were not run again.
- **Details:** [baseline](reports/experiments/14_efficientnet_b0_ft_baseline.md),
  [learning rate](reports/experiments/15_efficientnet_b0_ft_lr5e5.md),
  [`full_aug` review](reports/experiments/16_efficientnet_b0_ft_aug_review.md).
- **Next question:** a different architecture, such as ConvNeXt-Tiny, could be compared under the same
  protocol. No result is assumed.

## Reports

- [Reports index and validation comparison table](reports/README.md)
- [Final model selection](reports/experiments/10_final_model_selection.md)
- [`needs_review` threshold](reports/experiments/11_needs_review_threshold.md)
- [CE vs BCE](reports/experiments/07_ce_vs_bce.md)
- [ResNet18 experiments](reports/experiments/08_resnet.md)
- [Final Test evaluation](reports/experiments/12_final_test_evaluation.md)
- [Neysan / unclean analysis](reports/experiments/13_neysan_unclean_analysis.md)
- [Dataset decisions](docs/DATA_DECISIONS.md)

## Reproducibility

- **Seed and data files.** Seed 42 in every run. `data/manifest.csv` and `data/split_manifest.csv`
  are generated and not tracked. `make_split.py` rebuilds the split from the manifest and
  `decisions/`.
- **Checkpoint check.** The final checkpoint SHA256 is recorded above and in
  `reports/experiments/10_final_model_selection.md`. `evaluate_test.py` and `evaluate_neysan.py`
  check it before predicting.
- **Before/after hashes.** The Neysan report records the SHA256 of the checkpoint,
  `test_frozen.csv`, `split_manifest.csv` and `neysan_eval.csv`, before and after inference.
- **Not in Git:** raw images, checkpoints, histories, analysis output and local paths.

## Limitations

- **Single seed.** Every configuration was trained once with seed 42. The differences between the
  top runs are a few validation images.
- **Validation reuse.** The same 659 validation images were used to compare runs, select the model
  and choose the threshold. 8 validation labels were corrected after reviewing baseline errors.
  Validation metrics are therefore not an unbiased estimate.
- **Small dataset.** 2633 train / 659 validation / 400 Test images; for example, only 26 validation
  images are `vanet`.
- **Overfitting.** The final model shows signs of overfitting in the last epochs: validation loss
  is lowest at epoch 17 while train loss keeps falling. The selected epoch is the last of the budget.
- **One Test run.** Test was run once; its result is a single estimate without a confidence
  interval.
