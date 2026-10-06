# Final model selection (validation only)

Status: model selected on validation. The test set played no part in this selection. It was
evaluated once afterwards, after this selection, the `needs_review` threshold and the inference
protocol were fixed (see `12_final_test_evaluation.md`). The Neysan images were not used.

Current status: this selection is historical. The final model selection was later reopened on
validation only, and `swin_t_ft_aug` is the current final model
(`decisions/REOPEN_FINAL_MODEL_SELECTION.md`, commit `76669f4`; used by `src/predict.py` since commit
`a81184a`). The selection, results and checkpoint of `resnet224_ft_aug` in this report are unchanged.
Statements about `src/predict.py` below describe the code at the time of this selection.

## Candidates and rule

The candidates are the six 224x224 ResNet18 runs of `08_resnet.md`, the best-performing family of
this project.

- **Other runs:** every CNN run and the exploratory 128x128 ResNet runs have a lower validation
  macro F1. The best of them is `resnet_finetuning` (0.9165, the 128x128 fine-tuning run), and the
  best CNN run is `combo_aug_plateau` (0.8853).
- **Data:** all six candidates use the same split, the current validation labels, 659 validation
  images and seed 42. Each was trained once.
- **Main criterion:** validation macro F1 of the best checkpoint, which is the checkpoint each run
  saved by the same criterion.
- **Supporting criteria:** accuracy, macro precision and macro recall.
- **Per-class behaviour:** checked as well, in particular vanet and kamyun / kamyunet.
- **Final model:** a CrossEntropy (softmax) model, as the assignment requires for the prediction
  output. All six candidates use CrossEntropyLoss.

No model or data was changed for this selection. It uses only the existing checkpoints, histories
and validation analysis files (`reports/analysis/<run>/`).

## Comparison (validation, best checkpoint)

| Run | Best epoch | Macro F1 | Accuracy | Correct / 659 | Macro precision | Macro recall | Lowest val loss (epoch) |
|---|---|---|---|---|---|---|---|
| `resnet224_feature_extraction` | 19 | 0.9055 | 0.9181 | 605 | 0.9081 | 0.9038 | 0.2923 (19) |
| `resnet224_fine_tuning` | 8 | 0.9432 | 0.9530 | 628 | 0.9448 | 0.9421 | 0.1533 (7) |
| `resnet224_ft_wd1e4` | 8 | 0.9419 | 0.9514 | 627 | 0.9435 | 0.9408 | 0.1534 (7) |
| `resnet224_ft_plateau` | 8 | 0.9432 | 0.9530 | 628 | 0.9448 | 0.9421 | 0.1533 (7) |
| **`resnet224_ft_aug`** | **20** | **0.9482** | **0.9605** | **633** | **0.9456** | **0.9523** | 0.1520 (17) |
| `resnet224_ft_aug_wd1e4_plateau` | 20 | 0.9442 | 0.9560 | 630 | 0.9421 | 0.9487 | 0.1348 (13) |

`resnet224_ft_plateau` has the same selected weights as `resnet224_fine_tuning` (byte-identical,
see `08_resnet.md`), so its row is the same model.

Per-class F1:

| Class | feature extraction | fine-tuning | ft_wd1e4 | ft_plateau | **ft_aug** | ft_aug_wd1e4_plateau |
|---|---|---|---|---|---|---|
| ambulance | 0.9178 | 0.9510 | 0.9510 | 0.9510 | 0.9437 | 0.9437 |
| autobus | 0.9560 | 0.9838 | 0.9838 | 0.9838 | 1.0000 | 1.0000 |
| kamyun | 0.8681 | 0.9255 | 0.9206 | 0.9255 | 0.9355 | 0.9231 |
| kamyunet | 0.8900 | 0.9184 | 0.9128 | 0.9184 | 0.9239 | 0.9043 |
| minibus | 0.9125 | 0.9554 | 0.9554 | 0.9554 | 0.9809 | 0.9809 |
| savari | 0.9519 | 0.9756 | 0.9756 | 0.9756 | 0.9802 | 0.9802 |
| taxi | 0.9630 | 0.9896 | 0.9896 | 0.9896 | 1.0000 | 1.0000 |
| vanet | 0.7843 | 0.8462 | 0.8462 | 0.8462 | 0.8214 | 0.8214 |

Per-class recall:

| Class | feature extraction | fine-tuning | ft_wd1e4 | ft_plateau | **ft_aug** | ft_aug_wd1e4_plateau |
|---|---|---|---|---|---|---|
| ambulance | 0.9054 | 0.9189 | 0.9189 | 0.9189 | 0.9054 | 0.9054 |
| autobus | 0.9457 | 0.9891 | 0.9891 | 0.9891 | 1.0000 | 1.0000 |
| kamyun | 0.8495 | 0.9355 | 0.9355 | 0.9355 | 0.9355 | 0.9677 |
| kamyunet | 0.9082 | 0.9184 | 0.9082 | 0.9184 | 0.9286 | 0.8673 |
| minibus | 0.9241 | 0.9494 | 0.9494 | 0.9494 | 0.9747 | 0.9747 |
| savari | 0.9900 | 1.0000 | 1.0000 | 1.0000 | 0.9900 | 0.9900 |
| taxi | 0.9381 | 0.9794 | 0.9794 | 0.9794 | 1.0000 | 1.0000 |
| vanet | 0.7692 | 0.8462 | 0.8462 | 0.8462 | 0.8846 | 0.8846 |

vanet and kamyun / kamyunet (counts from the confusion matrices):

| Run | vanet correct / 26 | Predicted as vanet | vanet precision | ambulance -> vanet | kamyun -> kamyunet | kamyunet -> kamyun |
|---|---|---|---|---|---|---|
| `resnet224_feature_extraction` | 20 | 25 | 0.8000 | 5 | 9 | 5 |
| `resnet224_fine_tuning` | 22 | 26 | 0.8462 | 4 | 6 | 6 |
| `resnet224_ft_wd1e4` | 22 | 26 | 0.8462 | 4 | 6 | 7 |
| `resnet224_ft_plateau` | 22 | 26 | 0.8462 | 4 | 6 | 6 |
| `resnet224_ft_aug` | 23 | 30 | 0.7667 | 6 | 6 | 6 |
| `resnet224_ft_aug_wd1e4_plateau` | 23 | 30 | 0.7667 | 6 | 3 | 12 |

## Class relationships and merge decision

**kamyun / kamyunet: KEEP SEPARATE.** The decision is based on validation.

- **Final model, validation.**
  - Confusions: kamyun -> kamyunet 6/93, kamyunet -> kamyun 6/98.
  - These are 12 of the 26 validation errors.
  - Pair score 0.1257 (sum of the two row-normalized rates), the highest pair. Next is
    ambulance/vanet at 0.1195.
  - Most images of both classes are still classified correctly: F1 kamyun 0.9355, kamyunet 0.9239.
- **Other runs.** The direction of the confusion (kamyun -> kamyunet / kamyunet -> kamyun) varies:
  - 6/6 in `resnet224_fine_tuning`
  - 3/12 in `resnet224_ft_aug_wd1e4_plateau`
  - 10/10 in `combo_aug_plateau`
- **Human label review.** In `decisions/kamyun_kamyunet_review.csv` and
  `decisions/kamyunet_kamyun_review.csv`, 16 reviewed images were kept as labelled, 8 were relabelled
  and 1 is `UNCERTAIN`.
- **Reading.** The confusion is read as an ambiguous class boundary with some label noise, not as
  evidence that the two are one class.

Why the classes are not merged:

- **Taxonomy.** A merge would change the 8-class taxonomy.
- **Retraining.** A merge would require retraining on a new label mapping.
- **Comparability.** The results could not be compared directly with the existing experiments.
- **Test already used.** The 8-class frozen Test set has already been used once.
- **Not enough on its own.** The existing confusion alone is not a sufficient reason to remove the
  distinction.

Reported for completeness, not used for this decision: on the frozen Test, 8 kamyun images were
predicted as kamyunet and 1 kamyunet image as kamyun (`12_final_test_evaluation.md`).

## Selected model

- **Run:** `resnet224_ft_aug`
- **Checkpoint:** `checkpoints/resnet224_ft_aug_best.pt`, sha256
  `c26280e97f7324ee1ec85d7bf78a58ea0b4d2acb4cf3e975dbd52316366281a1`
- **Selected epoch:** 20 (the checkpoint stores `epoch: 20`, `val_f1: 0.9482`)
- **Settings recorded in the checkpoint:**
  - ResNet18 with `IMAGENET1K_V1` weights, mode `fine_tuning`
  - head only in epochs 1-5, `layer4` + `fc` from epoch 6
  - learning rates: `fc` 1e-3, `layer4` 1e-4
  - train-only augmentation `full_aug` (ResNet version, 224x224)
  - Adam, weight decay 0, no scheduler, CrossEntropyLoss, batch size 32, seed 42
  - trainable 8,397,832 / frozen 2,782,784
- **Inference transform:** `RESNET_TRANSFORM`, i.e. `Resize((224, 224))`, `ToTensor`, then
  ImageNet normalization, with no augmentation.
- **Validation metrics (659 images):**
  - accuracy 0.9605 (633 correct)
  - macro precision 0.9456, macro recall 0.9523, macro F1 0.9482
  - validation loss 0.1536 at the selected epoch; lowest 0.1520 at epoch 17

## Selection rationale

- **Main criterion.** `resnet224_ft_aug` has the highest validation macro F1 of the six
  candidates (0.9482).
- **Supporting criteria.** It also has the highest accuracy (0.9605), macro precision (0.9456) and
  macro recall (0.9523), so the supporting criteria agree with the main one.
- **Closest alternatives.**
  - `resnet224_ft_aug_wd1e4_plateau`: macro F1 0.9442, i.e. -0.0040 and 3 fewer correct images.
    The difference lies entirely in kamyun / kamyunet: 12 kamyunet -> kamyun errors instead of 6.
    Its lowest validation loss is lower (0.1348 vs 0.1520), but validation loss is not the
    selection criterion.
  - `resnet224_fine_tuning` (= `resnet224_ft_plateau`): macro F1 0.9432, i.e. -0.0050 and
    5 fewer correct images.
- **Per class.** The selected model has the best or equal-best F1 in 6 of 8 classes: autobus,
  kamyun, kamyunet, minibus, savari and taxi. It also has the highest vanet recall (0.8846, equal
  with the combination run). It is weaker than plain fine-tuning in two classes:
  - vanet F1 0.8214 vs 0.8462: precision drops from 0.8462 to 0.7667 because ambulance -> vanet
    errors rise from 4 to 6.
  - ambulance F1 0.9437 vs 0.9510.
- **Why it is still chosen.** This trade-off is recorded, not ignored. The selection follows the
  agreed rule: macro F1 first, supported here by all three other macro metrics.
- **Margin.** The margins over the next runs are small (0.0040 and 0.0050 macro F1, i.e. 3 and
  5 images out of 659).

## Known limitations and weaknesses

- **Single run.** Every candidate was trained once with seed 42. The differences between the top
  runs are a few validation images and are not tested for statistical significance. Another seed
  could change the ranking.
- **Checkpoint at the last epoch.** The selected epoch (20) is the last one of the 20-epoch budget.
  This run does not show whether more epochs would help or hurt.
- **Signs of overfitting.** Validation loss is lowest at epoch 17 (0.1520) and does not improve
  afterwards, while train loss keeps falling (0.016465 at epoch 20). At epoch 20 the train/validation
  F1 gap is 0.0475, measured with train metrics on augmented images.
- **vanet:**
  - It remains the weakest class: F1 0.8214, precision 0.7667.
  - 6 of the 7 wrong vanet predictions are true ambulance.
  - It has only 26 validation images, so each image changes its recall by about 0.038.
- **kamyun / kamyunet.** These remain the most confused pair: 6 errors in each direction. The final
  decision is `KEEP SEPARATE`; the classes were not merged.
- **Validation-based selection.** The model was selected on the same validation set that was used
  to compare all runs. The validation metrics above are therefore not an unbiased estimate; the
  one-time Test evaluation (`12_final_test_evaluation.md`) is the estimate on held-out data.
- **`needs_review` threshold:** it was chosen after this selection, on validation only. The final
  value is 0.90, and the test set was not used to choose it. The evidence is in
  `11_needs_review_threshold.md`.
- **Test evaluation:** done once, on the frozen test set, after this selection (see
  `12_final_test_evaluation.md`).
- **Neysan / unclean analysis:** done after this selection, inference only, with the selected
  checkpoint and threshold unchanged (see `13_neysan_unclean_analysis.md`). It was not used for this
  selection or any other decision.

## Test set

No test image, prediction or metric was used for this selection. The validation-only `needs_review`
threshold was then fixed at 0.90 (see `11_needs_review_threshold.md`), also without the test set.

Only after this selection, the threshold and the inference protocol (`src/predict.py`) were fixed
was the test set evaluated, once, with the selected checkpoint and this threshold
(`scripts/evaluate_test.py`, run `final_test_20260929T080908Z`). Results are in
`12_final_test_evaluation.md`: accuracy 0.9475, macro precision 0.9515, macro recall 0.9475,
macro F1 0.9473 on 400 images. They were not used to change the selection.

The selected checkpoint remains frozen with the SHA256 above. The threshold is stored in
`src/predict.py`, not in the checkpoint metadata. Re-saving the checkpoint only to add metadata
would create a different checkpoint file and hash without changing the learned weights, so this is
an intentional protocol decision.
