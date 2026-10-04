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
   transfer learning and combinations. Later, EfficientNet-B0, ConvNeXt-Tiny and Swin-Tiny were
   compared under the same protocol. All are compared on validation only.
5. **Final model.** Selected on validation. The project was first closed with `resnet224_ft_aug`;
   the selection was later reopened and `swin_t_ft_aug` was selected
   ([decision](decisions/REOPEN_FINAL_MODEL_SELECTION.md)). The `needs_review` threshold is also
   chosen on validation.
6. **Test.** The frozen 400-image Test set was evaluated once for `resnet224_ft_aug`. A second
   evaluation for `swin_t_ft_aug` is a documented protocol deviation and has not been run yet.
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
| Training (final model) | `python -m src.train --run-name swin_t_ft_aug --model swin_tiny --swin-mode fine_tuning --augmentation full_aug` |
| Training (previous final model, historical) | `python -m src.train --run-name resnet224_ft_aug --model resnet18 --resnet-mode fine_tuning --augmentation full_aug` |
| Validation analysis of a run | `python scripts/analyze_baseline.py --run-name <run_name>` |
| ResNet setup checks (no data) | `python scripts/verify_resnet.py` |
| Test evaluation (done once, historical `resnet224_ft_aug`) | `python scripts/evaluate_test.py` |
| Neysan evaluation (done once, historical `resnet224_ft_aug`) | `python scripts/evaluate_neysan.py` |
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
  outputs exist. Both are fixed to the previous final model `resnet224_ft_aug` (checkpoint SHA256
  and threshold 0.90) and are kept unchanged as historical scripts; they do not evaluate
  `swin_t_ft_aug`.

## Final model

Selected after the project was reopened; see
[`decisions/REOPEN_FINAL_MODEL_SELECTION.md`](decisions/REOPEN_FINAL_MODEL_SELECTION.md).

- **Run:** `swin_t_ft_aug`. Swin-Tiny with `IMAGENET1K_V1` weights, partial fine-tuning: head only
  in epochs 1-5, then `features[6]`, `features[7]`, `norm` + head.
- **Input and augmentation:** 224x224 input with ImageNet normalization. The train-only augmentation
  is `full_aug`; there is no augmentation at inference.
- **Training settings:** Adam (head 1e-3, unfrozen stages 1e-4), CrossEntropyLoss, 20 epochs, seed 42.
- **Checkpoint:** `checkpoints/swin_t_ft_aug_best.pt` (epoch 17, not in Git).
  SHA256 `f03bacde0a155461ea4aede0b813b8688d6ed17973d5ad19b0d2c1b913ff64db`.
- **Validation (659 images):** accuracy 0.9712, macro precision 0.9626, macro recall 0.9698,
  macro F1 0.9658. This is the highest validation macro F1 of the runs documented in
  `reports/experiments/`.
- **`needs_review`:** set when the confidence (the highest softmax probability) is below 0.95. The
  threshold was chosen on validation
  ([report](reports/experiments/23_swin_needs_review_threshold.md)) and is defined in
  `src/predict.py`. It is only a flag for human review; no prediction is rejected. The threshold is
  not stored in the frozen checkpoint.
- **Final Test:** not run yet for this model.

### Previous final model (historical)

- **Run:** `resnet224_ft_aug`, ResNet18 fine-tuned (head only in epochs 1-5, then `layer4` + head).
- **Checkpoint:** `checkpoints/resnet224_ft_aug_best.pt` (epoch 20), SHA256
  `c26280e97f7324ee1ec85d7bf78a58ea0b4d2acb4cf3e975dbd52316366281a1`.
- **Validation:** accuracy 0.9605, macro precision 0.9456, macro recall 0.9523, macro F1 0.9482.
- **`needs_review` threshold:** 0.90
  ([report](reports/experiments/11_needs_review_threshold.md)).
- The Test and Neysan results below belong to this model.

## Frozen Test (previous final model, `resnet224_ft_aug`)

- **Contents:** 400 images, 50 per class, listed in `decisions/test_frozen.csv`
  (SHA256 `e1727eea750e73a4924d79fbb5fb810af8f8a05d822754c51ade8376e0622540`).
- **Kept separate:** no image overlaps train, validation or Neysan, checked by path and SHA256.
- **Used once:** Test was evaluated once for `resnet224_ft_aug`, after its model, threshold (0.90)
  and inference protocol were fixed. It was not used for any decision, including the later
  selection of `swin_t_ft_aug`. These results are historical and unchanged.
- **Results:** 379 correct / 21 incorrect.
  - Accuracy 0.9475, macro precision 0.9515, macro recall 0.9475, macro F1 0.9473.
  - `needs_review`: 29 flagged, including 12 of the 21 errors.

## Neysan

- **Not a class.** Neysan (the common Iranian Nissan pickup) is a subtype of `vanet`, not a ninth
  class. Neysan images were kept out of training, validation and Test.
- **Evaluation set:** `decisions/neysan_eval.csv`, 621 images. 371 are human-confirmed; 250 come from
  the `neysan` folders (policy N1) and were not individually reviewed.
- **Inference only:** the set was evaluated with the previous final model `resnet224_ft_aug`
  (threshold 0.90), and each image is expected to be predicted as `vanet`. It has not been
  evaluated with `swin_t_ft_aug`.
- **Results:** 529 of 621 were predicted `vanet` (0.8519; 0.8544 on the 371 confirmed images), and
  185 were flagged `needs_review`.
- **Per-image output:** predictions and all 8 class probabilities are in
  `reports/experiments/13_neysan_predictions.csv`.
- **Descriptive only.** No precision, F1 or macro metric is reported for this set.

## Architecture comparisons

Run after the original final model was fixed. The EfficientNet-B0 comparison did not change
`resnet224_ft_aug`, its threshold or `src/predict.py`.

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
- **Later comparisons:** ConvNeXt-Tiny ([17](reports/experiments/17_convnext_tiny_fe_none.md),
  [18](reports/experiments/18_convnext_tiny_ft_none.md),
  [19](reports/experiments/19_convnext_tiny_ft_aug.md)) and Swin-Tiny
  ([20](reports/experiments/20_swin_tiny_fe_none.md),
  [21](reports/experiments/21_swin_tiny_ft_none.md),
  [22](reports/experiments/22_swin_tiny_ft_aug.md)) were compared under the same protocol;
  `swin_t_ft_aug` led to the reopened final model selection.

## Reports

- [Reports index and validation comparison table](reports/README.md)
- [Final model selection (original, ResNet)](reports/experiments/10_final_model_selection.md)
- [Reopened final model selection (Swin-Tiny)](decisions/REOPEN_FINAL_MODEL_SELECTION.md)
- [`needs_review` threshold, ResNet (historical)](reports/experiments/11_needs_review_threshold.md)
- [`needs_review` threshold, Swin-Tiny](reports/experiments/23_swin_needs_review_threshold.md)
- [Swin-Tiny fine-tuning with `full_aug`](reports/experiments/22_swin_tiny_ft_aug.md)
- [CE vs BCE](reports/experiments/07_ce_vs_bce.md)
- [ResNet18 experiments](reports/experiments/08_resnet.md)
- [Final Test evaluation (ResNet, historical)](reports/experiments/12_final_test_evaluation.md)
- [Neysan / unclean analysis (ResNet, historical)](reports/experiments/13_neysan_unclean_analysis.md)
- [Dataset decisions](docs/DATA_DECISIONS.md)

## Reproducibility

- **Seed and data files.** Seed 42 in every run. `data/manifest.csv` and `data/split_manifest.csv`
  are generated and not tracked. `make_split.py` rebuilds the split from the manifest and
  `decisions/`.
- **Checkpoint check.** The final checkpoint SHA256 is recorded above and in
  `reports/experiments/22_swin_tiny_ft_aug.md`. The previous final checkpoint SHA256 is recorded in
  `reports/experiments/10_final_model_selection.md`; `evaluate_test.py` and `evaluate_neysan.py`
  check that one before predicting.
- **Before/after hashes.** The Neysan report records the SHA256 of the ResNet checkpoint,
  `test_frozen.csv`, `split_manifest.csv` and `neysan_eval.csv`, before and after inference.
- **Not in Git:** raw images, checkpoints, histories, analysis output and local paths.

## Limitations

- **Single seed.** Every configuration was trained once with seed 42. The differences between the
  top runs are a few validation images.
- **Validation reuse.** The same 659 validation images were used to compare runs, select the model
  (twice, including the reopened selection) and choose both thresholds. 8 validation labels were
  corrected after reviewing baseline errors.
  Validation metrics are therefore not an unbiased estimate.
- **Small dataset.** 2633 train / 659 validation / 400 Test images; for example, only 26 validation
  images are `vanet`.
- **Overfitting.** The final model shows signs of overfitting in the last epochs: validation loss
  is lowest at epoch 11 while train loss keeps falling; the selected epoch is 17 of 20.
- **Test.** Test was run once, for the previous final model `resnet224_ft_aug`; its result is a
  single estimate without a confidence interval. The final model `swin_t_ft_aug` has no Test
  result yet; a second Test evaluation would be a documented protocol deviation, not an
  untouched held-out estimate.
