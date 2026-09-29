# Experiment: ResNet18 transfer learning

Status: completed; validation results recorded.

## Goal

Compare two ways of using an ImageNet-pretrained ResNet18 on the 8 vehicle classes:

- **Feature extraction:** the pretrained backbone is frozen; only a new 8-class head is trained.
- **Fine-tuning:** the head is trained first, then `layer4` is unfrozen and trained together with
  the head, with a smaller learning rate for the pretrained `layer4` weights.

## Setup

| Run | Mode | Command |
|---|---|---|
| `resnet224_feature_extraction` | feature extraction | `python -m src.train --run-name resnet224_feature_extraction --model resnet18 --resnet-mode feature_extraction` |
| `resnet224_fine_tuning` | fine-tuning | `python -m src.train --run-name resnet224_fine_tuning --model resnet18 --resnet-mode fine_tuning` |

Fixed for both runs (as recorded in each checkpoint): torchvision ResNet18 with `IMAGENET1K_V1`
weights, standard architecture (`conv1` and `maxpool` unchanged), new head `fc = Linear(512, 8)`,
input 224x224, ImageNet normalization, augmentation `none`, Adam, weight decay 0, no scheduler,
CrossEntropyLoss, 20 epochs, batch size 32, seed 42, `data/split_manifest.csv`
(train / validation = 2633 / 659), checkpoint selection by best validation macro F1.
Each mode was trained once. This is a validation-only experiment; the test set and the
Neysan images were not used.

Preprocessing of these two runs (`RESNET_TRANSFORM` in `src/train.py`, used for train and validation):
`Resize((224, 224))` -> `ToTensor` -> `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`.

- 224x224 because the assignment asks for ImageNet-compatible resize and normalization for the
  pretrained ResNet18, and 224 is the input size it is conventionally used with.
- The whole image is resized without cropping, so the complete vehicle stays in the frame.
- The CNN baseline and the earlier CNN experiments keep 128x128 with normalization 0.5 / 0.5.
  The earlier exploratory 128px ResNet runs (below) used the same ImageNet normalization as the
  224px runs.

## Trainable parameters and learning rates

| Run | Epochs | Trainable parts | Trainable | Frozen | Learning rate per group |
|---|---|---|---|---|---|
| feature extraction | 1-20 | `fc` | 4,104 | 11,176,512 | `fc` 1e-3 |
| fine-tuning | 1-5 | `fc` | 4,104 | 11,176,512 | `fc` 1e-3 (`layer4` group has no gradient yet) |
| fine-tuning | 6-20 | `layer4` + `fc` | 8,397,832 | 2,782,784 | `fc` 1e-3, `layer4` 1e-4 |

- `conv1`, `bn1`, `layer1`, `layer2` and `layer3` stay frozen in both runs. Frozen parts are kept in
  eval mode, so their BatchNorm running statistics stay the ImageNet values.
- Head first: the assignment asks for the head to be trained before `layer4` is unfrozen, but does
  not fix for how long. The 5 head-only epochs are a design choice (`FT_WARMUP_EPOCHS = 5`); they
  keep the total budget equal to feature extraction (20 epochs).
- Discriminative learning rates: the new head, which starts from random weights, keeps 1e-3; the
  pretrained `layer4` gets a 10x smaller rate (1e-4), so its ImageNet features are adapted
  gradually instead of being overwritten.
- The `lr` column of the histories shows the `fc` group only (constant 1e-3); the `layer4` rate is
  recorded in the checkpoint (`param_group_lrs`). The `trainable_params` column shows the unfreeze:
  4,104 in epochs 1-5 and 8,397,832 from epoch 6.
- Control: epochs 1-5 of the two histories are identical row for row (same weights, data order and
  trainable parameters); the runs differ from epoch 6 on.
- `scripts/verify_resnet.py` checks these properties (counts, freezing, BatchNorm modes, learning-rate
  groups, 224 input, checkpoint metadata) with real torch and without data. After the change to
  224px it was run in the project's Windows environment: 9 passed, 0 failed.

## Results (validation, best checkpoint)

| Run | Best epoch | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---|---|---|---|---|
| `resnet224_feature_extraction` | 19 | 0.9181 | 0.9081 | 0.9038 | 0.9055 |
| `resnet224_fine_tuning` | 8 | 0.9530 | 0.9448 | 0.9421 | 0.9432 |

Per-class F1:

| Class | Feature extraction | Fine-tuning |
|---|---:|---:|
| ambulance | 0.9178 | 0.9510 |
| autobus | 0.9560 | 0.9838 |
| kamyun | 0.8681 | 0.9255 |
| kamyunet | 0.8900 | 0.9184 |
| minibus | 0.9125 | 0.9554 |
| savari | 0.9519 | 0.9756 |
| taxi | 0.9630 | 0.9896 |
| vanet | 0.7843 | 0.8462 |

- Lowest recall and lowest precision: vanet in both runs (feature extraction: recall 0.7692,
  precision 0.8000; fine-tuning: recall 0.8462, precision 0.8462).
- Main confusions, fine-tuning (true -> predicted): kamyunet -> kamyun 6, kamyun -> kamyunet 6,
  ambulance -> vanet 4, vanet -> savari 2, minibus -> kamyunet 2.
- Main confusions, feature extraction: kamyun -> kamyunet 9, taxi -> savari 5,
  kamyunet -> kamyun 5, ambulance -> vanet 5, minibus -> kamyun 3.

## Training behavior

- Fine-tuning: the best validation macro F1 is 0.9432 at epoch 8; the lowest validation loss is at
  epoch 7 (0.1533). Train accuracy reaches 1.0 at epoch 10. At epoch 20, train F1 is 0.9990 vs
  validation F1 0.9322. After the selected checkpoint, train performance keeps improving while
  validation does not: validation macro F1 stays between 0.9205 and 0.9387 in epochs 9-20.
- Feature extraction: the best validation macro F1 (epoch 19) and the lowest validation loss
  (0.2923, epoch 19) fall almost at the end of the 20 epochs. Train accuracy does not reach 1.0;
  at epoch 20, train F1 is 0.9513 vs validation F1 0.8966.

## Fine-tuning with augmentation (`resnet224_ft_aug`)

A controlled follow-up to `resnet224_fine_tuning`. The only change is train-only augmentation
`full_aug`. It is recorded here whatever its result; no run is left out of this report because
it scored better or worse.

| Run | Command |
|---|---|
| `resnet224_ft_aug` | `python -m src.train --run-name resnet224_ft_aug --model resnet18 --resnet-mode fine_tuning --augmentation full_aug` |

### Setup

Settings as recorded in the checkpoint; all are the same as `resnet224_fine_tuning` except the
augmentation:

- ResNet18 with `IMAGENET1K_V1` weights, mode `fine_tuning`, 224x224 input, ImageNet
  normalization
- head only in epochs 1-5, `layer4` + `fc` from epoch 6
- learning rates: `fc` 1e-3, `layer4` 1e-4
- Adam, weight decay 0, no scheduler, CrossEntropyLoss, dropout 0
- 20 epochs, batch size 32 (standard batches), seed 42
- train / validation = 2633 / 659

Transforms:

- **Training transform (`full_aug`):** `resnet_full_aug_transform()` in `src/train.py`. It uses the
  same augmentations and values as the CNN `full_aug`, adapted to ResNet input:
  `RandomResizedCrop(224, scale=(0.8, 1.0))` -> `RandomHorizontalFlip()` -> `RandomRotation(10)` ->
  `ColorJitter(brightness=0.2, contrast=0.2)` -> `ToTensor` -> `Normalize` with ImageNet mean/std.
- **Validation transform:** unchanged `RESNET_TRANSFORM`, with no augmentation.
- **Checkpoint metadata:** records `augmentation: full_aug` and its parameters.
- **Crop vs resize:** `RESNET_TRANSFORM` resizes the whole image, while `RandomResizedCrop` takes a
  random crop (80-100% of the area, variable aspect ratio) before resizing.
- **Smoke test:** before the run, the new code path was checked in the Windows environment with
  a separate smoke test (29 of 29 checks passed). The test covered:
  - which augmentation values the command line accepts or rejects
  - the order of the transforms
  - batch shape `[32, 3, 224, 224]` for train and validation
  - validation being deterministic
  - no Test or Neysan image in the loaders

### Result (validation, best checkpoint)

| Run | Best epoch | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---|---|---|---|---|
| `resnet224_fine_tuning` | 8 | 0.9530 | 0.9448 | 0.9421 | 0.9432 |
| `resnet224_ft_aug` | 20 | 0.9605 | 0.9456 | 0.9523 | 0.9482 |

- Correct: 633 of 659, against 628 of 659 without augmentation.
- The checkpoint stores epoch 20 and validation F1 0.9482. Re-evaluating it on the 659
  validation images gives the same value.

Per-class metrics of `resnet224_ft_aug`:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ambulance | 0.9853 | 0.9054 | 0.9437 | 74 |
| autobus | 1.0000 | 1.0000 | 1.0000 | 92 |
| kamyun | 0.9355 | 0.9355 | 0.9355 | 93 |
| kamyunet | 0.9192 | 0.9286 | 0.9239 | 98 |
| minibus | 0.9872 | 0.9747 | 0.9809 | 79 |
| savari | 0.9706 | 0.9900 | 0.9802 | 100 |
| taxi | 1.0000 | 1.0000 | 1.0000 | 97 |
| vanet | 0.7667 | 0.8846 | 0.8214 | 26 |

Confusion matrix of `resnet224_ft_aug` (rows = true class, columns = predicted class; 26 errors):

| true \ predicted | ambulance | autobus | kamyun | kamyunet | minibus | savari | taxi | vanet |
|---|---|---|---|---|---|---|---|---|
| ambulance | 67 | 0 | 0 | 0 | 0 | 1 | 0 | 6 |
| autobus | 0 | 92 | 0 | 0 | 0 | 0 | 0 | 0 |
| kamyun | 0 | 0 | 87 | 6 | 0 | 0 | 0 | 0 |
| kamyunet | 0 | 0 | 6 | 91 | 1 | 0 | 0 | 0 |
| minibus | 0 | 0 | 0 | 2 | 77 | 0 | 0 | 0 |
| savari | 0 | 0 | 0 | 0 | 0 | 99 | 0 | 1 |
| taxi | 0 | 0 | 0 | 0 | 0 | 0 | 97 | 0 |
| vanet | 1 | 0 | 0 | 0 | 0 | 2 | 0 | 23 |

- **Lowest recall:** vanet 0.8846 (23 of 26).
- **Lowest precision:** vanet 0.7667 (23 of 30 predictions). Of the 7 wrong vanet predictions, 6 are
  true ambulance and 1 is true savari.
- **Main confusions (true -> predicted):** kamyunet -> kamyun 6, kamyun -> kamyunet 6,
  ambulance -> vanet 6, vanet -> savari 2, minibus -> kamyunet 2.

### Comparison with `resnet224_fine_tuning`

Per-class F1:

| Class | `resnet224_fine_tuning` | `resnet224_ft_aug` |
|---|---:|---:|
| ambulance | 0.9510 | 0.9437 |
| autobus | 0.9838 | 1.0000 |
| kamyun | 0.9255 | 0.9355 |
| kamyunet | 0.9184 | 0.9239 |
| minibus | 0.9554 | 0.9809 |
| savari | 0.9756 | 0.9802 |
| taxi | 0.9896 | 1.0000 |
| vanet | 0.8462 | 0.8214 |

- **Overall:** macro F1 0.9432 -> 0.9482, accuracy 0.9530 -> 0.9605, macro precision
  0.9448 -> 0.9456, macro recall 0.9421 -> 0.9523.
- **vanet:** recall 0.8462 -> 0.8846, but precision 0.8462 -> 0.7667, so vanet F1 is lower
  (0.8462 -> 0.8214). ambulance -> vanet errors rose from 4 to 6; ambulance F1 is also slightly
  lower.
- **kamyun / kamyunet:** the confusion is unchanged: 6 errors in each direction in both runs.
- **Single run:** one run per setting (seed 42). The macro-F1 difference is +0.0050, and the
  difference in correct predictions is 5 images out of 659. This describes these two runs; it is
  not a measured general effect of augmentation.
- **Not comparable epoch by epoch:** because augmentation changes the random draws from epoch 1,
  even epochs 1-5 differ from `resnet224_fine_tuning`.

### Training behavior

Train metrics are measured during training, on augmented images. Validation metrics are measured
in eval mode without augmentation.

- **Train loss** keeps falling: 1.4521 at epoch 1, 0.0310 at epoch 17, 0.016465 at epoch 20.
  Train accuracy never reaches 1.0 (0.9954 at epoch 20). Without augmentation it reached 1.0 at
  epoch 10.
- **Validation loss** is lowest at epoch 17 (0.1520). After that it does not improve further:
  - epoch 18: 0.1761
  - epoch 19: 0.1824
  - epoch 20: 0.1536 (the best-F1 epoch)
- **Validation macro F1** varies between 0.9250 and 0.9482 over epochs 10-20. Its highest value is
  at the last epoch (20), so the 20-epoch budget ends at the selected checkpoint. This run does not
  show whether further epochs would change it.
- **At epoch 20:**
  - train F1 is 0.9956 and validation F1 is 0.9482, a gap of 0.0475
  - without augmentation, at epoch 20: train F1 0.9990 vs validation F1 0.9322 (gap 0.0668)
- **Overfitting.** From epoch 17 to 20, train loss goes down (0.0310 -> 0.016465) while
  validation loss does not go below its epoch-17 minimum. This train/validation divergence is a sign
  of beginning overfitting in the last epochs. It is milder than without augmentation, where
  validation loss was lowest at epoch 7 (0.1533) and rose to 0.2403 by epoch 19.

## Fine-tuning with weight decay (`resnet224_ft_wd1e4`) and with a scheduler (`resnet224_ft_plateau`)

Two more controlled follow-ups to `resnet224_fine_tuning`. Each changes one factor. Both are
recorded here whatever their result; no run is left out of this report because it scored better
or worse.

| Run | Change against `resnet224_fine_tuning` | Command |
|---|---|---|
| `resnet224_ft_wd1e4` | AdamW with weight decay 1e-4 (instead of Adam, weight decay 0) | `python -m src.train --run-name resnet224_ft_wd1e4 --model resnet18 --resnet-mode fine_tuning --optimizer adamw --weight-decay 1e-4` |
| `resnet224_ft_plateau` | `ReduceLROnPlateau` on validation loss (mode min, factor 0.5, patience 3) | `python -m src.train --run-name resnet224_ft_plateau --model resnet18 --resnet-mode fine_tuning --scheduler plateau` |

### Setup

Recorded in both checkpoints, the same as `resnet224_fine_tuning`:

- ResNet18 with `IMAGENET1K_V1` weights, mode `fine_tuning`, 224x224 input, ImageNet
  normalization, `RESNET_TRANSFORM` for train and validation (augmentation `none`)
- head only in epochs 1-5, `layer4` + `fc` from epoch 6 (trainable 8,397,832, frozen 2,782,784)
- learning rates: `fc` 1e-3, `layer4` 1e-4
- CrossEntropyLoss, dropout 0, 20 epochs, batch size 32 (standard batches), seed 42
- train / validation = 2633 / 659

The optimizer and scheduler recorded in each checkpoint:

- `resnet224_ft_wd1e4`: `optimizer: adamw`, `weight_decay: 0.0001`, `scheduler: none`.
  AdamW applies the weight decay to both parameter groups. The `layer4` parameters have no gradient
  in epochs 1-5, so in those epochs the decay only reaches `fc`.
- `resnet224_ft_plateau`: `optimizer: adam`, `weight_decay: 0.0`, `scheduler: plateau`,
  `scheduler_params: {mode: min, factor: 0.5, patience: 3}`. The scheduler steps on the validation
  loss after each epoch and scales both parameter groups by the same factor.

### Results (validation, best checkpoint)

| Run | Best epoch | Accuracy | Macro precision | Macro recall | Macro F1 | Val loss at best epoch | Correct / 659 |
|---|---|---|---|---|---|---|---|
| `resnet224_fine_tuning` | 8 | 0.9530 | 0.9448 | 0.9421 | 0.9432 | 0.1563 | 628 |
| `resnet224_ft_wd1e4` | 8 | 0.9514 | 0.9435 | 0.9408 | 0.9419 | 0.1567 | 627 |
| `resnet224_ft_plateau` | 8 | 0.9530 | 0.9448 | 0.9421 | 0.9432 | 0.1563 | 628 |

Per-class metrics (precision / recall / F1; support in the last column):

| Class | `resnet224_ft_wd1e4` | `resnet224_ft_plateau` | Support |
|---|---|---|---|
| ambulance | 0.9855 / 0.9189 / 0.9510 | 0.9855 / 0.9189 / 0.9510 | 74 |
| autobus | 0.9785 / 0.9891 / 0.9838 | 0.9785 / 0.9891 / 0.9838 | 92 |
| kamyun | 0.9062 / 0.9355 / 0.9206 | 0.9158 / 0.9355 / 0.9255 | 93 |
| kamyunet | 0.9175 / 0.9082 / 0.9128 | 0.9184 / 0.9184 / 0.9184 | 98 |
| minibus | 0.9615 / 0.9494 / 0.9554 | 0.9615 / 0.9494 / 0.9554 | 79 |
| savari | 0.9524 / 1.0000 / 0.9756 | 0.9524 / 1.0000 / 0.9756 | 100 |
| taxi | 1.0000 / 0.9794 / 0.9896 | 1.0000 / 0.9794 / 0.9896 | 97 |
| vanet | 0.8462 / 0.8462 / 0.8462 | 0.8462 / 0.8462 / 0.8462 | 26 |

Confusion matrix of `resnet224_ft_wd1e4` (rows = true class, columns = predicted class; 32 errors):

| true \ predicted | ambulance | autobus | kamyun | kamyunet | minibus | savari | taxi | vanet |
|---|---|---|---|---|---|---|---|---|
| ambulance | 68 | 0 | 0 | 0 | 0 | 2 | 0 | 4 |
| autobus | 0 | 91 | 0 | 0 | 1 | 0 | 0 | 0 |
| kamyun | 0 | 0 | 87 | 6 | 0 | 0 | 0 | 0 |
| kamyunet | 0 | 1 | 7 | 89 | 1 | 0 | 0 | 0 |
| minibus | 0 | 0 | 2 | 2 | 75 | 0 | 0 | 0 |
| savari | 0 | 0 | 0 | 0 | 0 | 100 | 0 | 0 |
| taxi | 0 | 0 | 0 | 0 | 1 | 1 | 95 | 0 |
| vanet | 1 | 1 | 0 | 0 | 0 | 2 | 0 | 22 |

Confusion matrix of `resnet224_ft_plateau` (31 errors); it is identical to the matrix of
`resnet224_fine_tuning`:

| true \ predicted | ambulance | autobus | kamyun | kamyunet | minibus | savari | taxi | vanet |
|---|---|---|---|---|---|---|---|---|
| ambulance | 68 | 0 | 0 | 0 | 0 | 2 | 0 | 4 |
| autobus | 0 | 91 | 0 | 0 | 1 | 0 | 0 | 0 |
| kamyun | 0 | 0 | 87 | 6 | 0 | 0 | 0 | 0 |
| kamyunet | 0 | 1 | 6 | 90 | 1 | 0 | 0 | 0 |
| minibus | 0 | 0 | 2 | 2 | 75 | 0 | 0 | 0 |
| savari | 0 | 0 | 0 | 0 | 0 | 100 | 0 | 0 |
| taxi | 0 | 0 | 0 | 0 | 1 | 1 | 95 | 0 |
| vanet | 1 | 1 | 0 | 0 | 0 | 2 | 0 | 22 |

- **Lowest recall and lowest precision:** vanet in both runs (recall 0.8462, precision 0.8462).
- **Main confusions:**
  - `resnet224_ft_wd1e4`: kamyunet -> kamyun 7, kamyun -> kamyunet 6, ambulance -> vanet 4,
    vanet -> savari 2, minibus -> kamyunet 2.
  - `resnet224_ft_plateau`: the same as `resnet224_fine_tuning`, i.e. kamyunet -> kamyun 6,
    kamyun -> kamyunet 6, ambulance -> vanet 4, vanet -> savari 2, minibus -> kamyunet 2.

### `resnet224_ft_wd1e4`: comparison and training behavior

- **Overall:** against `resnet224_fine_tuning`, macro F1 0.9432 -> 0.9419 (-0.0013) and accuracy
  0.9530 -> 0.9514, i.e. 627 instead of 628 correct. The confusion matrices differ in one image:
  one more kamyunet -> kamyun error (7 instead of 6), so kamyun and kamyunet precision/recall/F1
  change slightly. All other classes are identical.
- **Histories:** the history differs from `resnet224_fine_tuning` from epoch 1, but in epochs 1-5
  the values agree to 4 decimals except the epoch-2 validation loss (0.6101 vs 0.6100). The two runs
  stay close throughout.
- **Learning rate:** unchanged at 1e-3 for `fc` in all 20 epochs (no scheduler); `layer4` 1e-4 as
  recorded in the checkpoint.
- **Overfitting, the same pattern as without weight decay:**
  - Validation loss is lowest at epoch 7 (0.1534), 0.1567 at the best-F1 epoch 8, and higher
    afterwards: 0.2943 at epoch 19, 0.2263 at epoch 20.
  - Train loss falls to 0.002945 at epoch 20, and train accuracy first reaches 1.0 at epoch 10.
  - At epoch 20, train F1 is 1.0000 vs validation F1 0.9307, a gap of 0.0693.
  - After epoch 8, validation macro F1 is between 0.9132 and 0.9398.
- **Conclusion for this run:** with this budget and learning rates, weight decay 1e-4 did not
  change the overfitting pattern of fine-tuning and gave almost the same validation result. This is
  one run (seed 42); it describes this run only.

### `resnet224_ft_plateau`: learning-rate changes and training behavior

The `lr` column of the history shows the `fc` group. The `layer4` rate is not logged; it is always
one tenth of the `fc` rate, because the scheduler scales both groups by the same factor.

| Epochs | `fc` lr (logged) | `layer4` lr (derived) |
|---|---|---|
| 1-11 | 1e-3 | 1e-4 |
| 12-15 | 5e-4 | 5e-5 |
| 16-19 | 2.5e-4 | 2.5e-5 |
| 20 | 1.25e-4 | 1.25e-5 |

- **When the scheduler acted:**
  - The lowest validation loss is 0.1533 at epoch 7. After four epochs without a new lowest value
    (8-11), the scheduler halved the rate after epoch 11.
  - It halved it again after epochs 15 and 19.
  - Replaying the rule on the logged validation losses gives the same rates.
- **The best checkpoint is from before any reduction.** It is from epoch 8, before the first
  learning-rate reduction (effective from epoch 12). Epochs 1-11 of the history are identical, row
  for row, to `resnet224_fine_tuning`.
- **Observation: same weights as `resnet224_fine_tuning`.** All 122 weight tensors in
  `checkpoints/resnet224_ft_plateau_best.pt` are byte-identical to those in
  `checkpoints/resnet224_fine_tuning_best.pt`. Only the metadata differs (`scheduler`,
  `scheduler_params`). The validation results, per-class metrics and confusion matrix are
  therefore identical. The selected checkpoint of this run does not reflect any effect of the
  scheduler. The files are kept as they are.
- **After the reductions (epochs 12-20):**
  - Validation macro F1 is between 0.9349 and 0.9400 (highest 0.9400 at epoch 13), below the
    0.9432 of epoch 8.
  - Validation loss is between 0.1692 and 0.1798. Without the scheduler, in the same epochs, it
    was between 0.1663 and 0.2403.
  - Validation F1 without the scheduler was between 0.9205 and 0.9387.
  - So the lower rate kept the later epochs steadier, but did not produce a better checkpoint.
- **Overfitting:**
  - Train loss falls to 0.000775 at epoch 20; train accuracy first reaches 1.0 at epoch 10.
  - At epoch 20, train F1 is 1.0000 vs validation F1 0.9378 (gap 0.0622).
  - Validation loss never returns to its epoch-7 minimum.

## Fine-tuning with augmentation, weight decay and a scheduler (`resnet224_ft_aug_wd1e4_plateau`)

An additional combination run on top of `resnet224_fine_tuning`: train-only `full_aug`, AdamW with
weight decay 1e-4, and `ReduceLROnPlateau`. It changes three factors at once, so it cannot show
which factor causes a difference. It is recorded here whatever its result, like every run before
it; it is not removed because it scored better or worse.

| Run | Command |
|---|---|
| `resnet224_ft_aug_wd1e4_plateau` | `python -m src.train --run-name resnet224_ft_aug_wd1e4_plateau --model resnet18 --resnet-mode fine_tuning --augmentation full_aug --optimizer adamw --weight-decay 1e-4 --scheduler plateau` |

### Setup

Recorded in the checkpoint:

- ResNet18 with `IMAGENET1K_V1` weights, mode `fine_tuning`, 224x224 input, ImageNet
  normalization
- head only in epochs 1-5, `layer4` + `fc` from epoch 6 (trainable 8,397,832, frozen 2,782,784)
- learning rates at the start: `fc` 1e-3, `layer4` 1e-4
- augmentation `full_aug` (train only, same transform and parameters as `resnet224_ft_aug`)
- AdamW, weight decay 1e-4
- `ReduceLROnPlateau` on validation loss (mode min, factor 0.5, patience 3)
- CrossEntropyLoss, dropout 0, 20 epochs, batch size 32 (standard batches), seed 42
- train / validation = 2633 / 659

Validation uses the unchanged `RESNET_TRANSFORM`, without augmentation.

### Result (validation, best checkpoint)

| Metric | Value |
|---|---|
| Best epoch | 20 |
| Accuracy | 0.9560 (630 of 659) |
| Macro precision | 0.9421 |
| Macro recall | 0.9487 |
| Macro F1 | 0.9442 |
| Validation loss at best epoch | 0.1590 |

The checkpoint stores epoch 20 and validation F1 0.9442. Re-evaluating it on the 659 validation
images gives the same value.

Per-class metrics:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ambulance | 0.9853 | 0.9054 | 0.9437 | 74 |
| autobus | 1.0000 | 1.0000 | 1.0000 | 92 |
| kamyun | 0.8824 | 0.9677 | 0.9231 | 93 |
| kamyunet | 0.9444 | 0.8673 | 0.9043 | 98 |
| minibus | 0.9872 | 0.9747 | 0.9809 | 79 |
| savari | 0.9706 | 0.9900 | 0.9802 | 100 |
| taxi | 1.0000 | 1.0000 | 1.0000 | 97 |
| vanet | 0.7667 | 0.8846 | 0.8214 | 26 |

Confusion matrix (rows = true class, columns = predicted class; 29 errors):

| true \ predicted | ambulance | autobus | kamyun | kamyunet | minibus | savari | taxi | vanet |
|---|---|---|---|---|---|---|---|---|
| ambulance | 67 | 0 | 0 | 0 | 0 | 1 | 0 | 6 |
| autobus | 0 | 92 | 0 | 0 | 0 | 0 | 0 | 0 |
| kamyun | 0 | 0 | 90 | 3 | 0 | 0 | 0 | 0 |
| kamyunet | 0 | 0 | 12 | 85 | 1 | 0 | 0 | 0 |
| minibus | 0 | 0 | 0 | 2 | 77 | 0 | 0 | 0 |
| savari | 0 | 0 | 0 | 0 | 0 | 99 | 0 | 1 |
| taxi | 0 | 0 | 0 | 0 | 0 | 0 | 97 | 0 |
| vanet | 1 | 0 | 0 | 0 | 0 | 2 | 0 | 23 |

- **Most important confusion:** kamyunet -> kamyun, with 12 errors (41% of the 29).
- **Other confusions:** ambulance -> vanet 6, kamyun -> kamyunet 3, vanet -> savari 2,
  minibus -> kamyunet 2.
- **Lowest F1 and lowest precision:** vanet (F1 0.8214, precision 0.7667; 23 of 30 vanet predictions
  correct, 6 of the 7 wrong ones are true ambulance).
- **Lowest recall:** kamyunet, 0.8673 (85 of 98), because of the 12 kamyunet -> kamyun errors.
  vanet recall is 0.8846.

### Learning rate

The `lr` column shows the `fc` group; `layer4` is always one tenth of it.

| Epochs | `fc` lr (logged) | `layer4` lr (derived) |
|---|---|---|
| 1-11 | 1e-3 | 1e-4 |
| 12-17 | 5e-4 | 5e-5 |
| 18-20 | 2.5e-4 | 2.5e-5 |

- **First reduction:** validation loss was lowest at epoch 7 (0.1661) and did not improve in
  epochs 8-11, so the rate was halved after epoch 11.
- **Second reduction:** a new lowest value followed at epoch 13 (0.1348). After four epochs
  without improvement (14-17), the rate was halved again after epoch 17.
- **Check:** replaying the rule on the logged validation losses gives the same rates.
- **Selected checkpoint:** the best epoch (20) comes after both reductions, at `fc` rate 2.5e-4.

### Training behavior

Train metrics are measured on augmented images during training. Validation is measured in eval
mode without augmentation.

- **Validation loss:** lowest 0.1348 at epoch 13; 0.1590 at the best-F1 epoch 20. After epoch 13
  it varies between 0.1398 and 0.1762 and does not return to its minimum.
- **Train loss:** keeps falling after epoch 13: 0.0328 at epoch 13, 0.012835 at epoch 20. Train
  accuracy never reaches 1.0 (0.9958 at epoch 20).
- **Overfitting:** the divergence after epoch 13, with train loss going down while validation loss
  does not improve, is a sign of overfitting in the last epochs.
- **Validation macro F1 after epoch 13:** between 0.9334 and 0.9442. The highest value is at the
  last epoch (20), so the 20-epoch budget ends at the selected checkpoint.
- **At epoch 20:**
  - train F1 is 0.9946 and validation F1 is 0.9442, a gap of 0.0504
  - validation loss minus train loss is 0.1461

### Comparison

| Run | Best epoch | Accuracy | Macro precision | Macro recall | Macro F1 | Correct / 659 |
|---|---|---|---|---|---|---|
| `resnet224_fine_tuning` | 8 | 0.9530 | 0.9448 | 0.9421 | 0.9432 | 628 |
| `resnet224_ft_aug` | 20 | 0.9605 | 0.9456 | 0.9523 | 0.9482 | 633 |
| `resnet224_ft_aug_wd1e4_plateau` | 20 | 0.9560 | 0.9421 | 0.9487 | 0.9442 | 630 |

- **Against `resnet224_fine_tuning`:** slightly better. Macro F1 is +0.0010 and accuracy
  0.9530 -> 0.9560; macro recall is higher and macro precision lower.
- **Against `resnet224_ft_aug`:** weaker. Macro F1 is -0.0040 and 3 fewer images are correct.
  - The per-class metrics are identical for ambulance, autobus, minibus, savari, taxi and vanet.
    The whole difference is in kamyun / kamyunet: kamyunet -> kamyun 6 -> 12, kamyun -> kamyunet
    6 -> 3.
  - kamyun F1 falls from 0.9355 to 0.9231 and kamyunet F1 from 0.9239 to 0.9043.
  - Up to epoch 5 the history agrees with `resnet224_ft_aug` to four decimals. The runs diverge
    after that.
- **Scope of this result:** this is an observation from a single run with seed 42. It is not
  general evidence about the effect of this combination, and the differences of 0.0010 and 0.0040
  in macro F1 are small for 659 validation images.

## Earlier 128x128 runs (exploratory)

Before the input size was set to 224, both modes were run once with the same settings at 128x128
(runs `resnet_feature_extraction` and `resnet_finetuning`). They are kept as exploratory runs; the
224 runs above are the ResNet results of this project.

| Input | Run | Best epoch | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---|---|---|---|---|---|
| 128 | `resnet_feature_extraction` | 7 | 0.8771 | 0.8781 | 0.8592 | 0.8666 |
| 224 | `resnet224_feature_extraction` | 19 | 0.9181 | 0.9081 | 0.9038 | 0.9055 |
| 128 | `resnet_finetuning` | 7 | 0.9241 | 0.9173 | 0.9168 | 0.9165 |
| 224 | `resnet224_fine_tuning` | 8 | 0.9530 | 0.9448 | 0.9421 | 0.9432 |

In both modes the 224 run recorded higher validation metrics than the 128 run. This is one run per
setting; it describes these runs and is not a measured effect of input size in general.

## Scope and limitations

- Single run per setting, seed 42; validation only.
- All ResNet runs described above are kept in this report, including `resnet224_ft_aug`,
  `resnet224_ft_wd1e4`, `resnet224_ft_plateau` and `resnet224_ft_aug_wd1e4_plateau`; none is removed
  or replaced because of its result.
- The test set and the Neysan images were not used; these are not final test results.
- The warm-up length (5 epochs) and the learning rates were fixed in advance and not tuned.
- `resnet224_ft_aug`, `resnet224_ft_wd1e4` and `resnet224_ft_plateau` each change one factor
  (augmentation, weight decay, scheduler) against `resnet224_fine_tuning`; none was tuned further.
  `resnet224_ft_aug_wd1e4_plateau` combines all three and cannot separate their effects.

## Files

- Checkpoints: `checkpoints/resnet224_feature_extraction_best.pt`, `checkpoints/resnet224_fine_tuning_best.pt`,
  `checkpoints/resnet224_ft_aug_best.pt`, `checkpoints/resnet224_ft_wd1e4_best.pt`,
  `checkpoints/resnet224_ft_plateau_best.pt`, `checkpoints/resnet224_ft_aug_wd1e4_plateau_best.pt`
  (exploratory: `checkpoints/resnet_feature_extraction_best.pt`, `checkpoints/resnet_finetuning_best.pt`)
- Histories: `reports/resnet224_feature_extraction_history.csv`, `reports/resnet224_fine_tuning_history.csv`,
  `reports/resnet224_ft_aug_history.csv`, `reports/resnet224_ft_wd1e4_history.csv`,
  `reports/resnet224_ft_plateau_history.csv`, `reports/resnet224_ft_aug_wd1e4_plateau_history.csv`
- Analysis output: `reports/analysis/resnet224_feature_extraction/`, `reports/analysis/resnet224_fine_tuning/`,
  `reports/analysis/resnet224_ft_aug/`, `reports/analysis/resnet224_ft_wd1e4/`,
  `reports/analysis/resnet224_ft_plateau/`, `reports/analysis/resnet224_ft_aug_wd1e4_plateau/`
- Code: `src/resnet.py`, `src/train.py` (`--model resnet18`; `resnet_full_aug_transform()` for
  `--augmentation full_aug`), `scripts/verify_resnet.py`
