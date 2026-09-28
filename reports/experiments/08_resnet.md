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

Preprocessing (`RESNET_TRANSFORM` in `src/train.py`, used for train and validation):
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

- Single run per mode, seed 42; validation only.
- The test set and the Neysan images were not used; these are not final test results.
- The warm-up length (5 epochs) and the learning rates were fixed in advance and not tuned.

## Files

- Checkpoints: `checkpoints/resnet224_feature_extraction_best.pt`, `checkpoints/resnet224_fine_tuning_best.pt`
  (exploratory: `checkpoints/resnet_feature_extraction_best.pt`, `checkpoints/resnet_finetuning_best.pt`)
- Histories: `reports/resnet224_feature_extraction_history.csv`, `reports/resnet224_fine_tuning_history.csv`
- Analysis output: `reports/analysis/resnet224_feature_extraction/`, `reports/analysis/resnet224_fine_tuning/`
- Code: `src/resnet.py`, `src/train.py` (`--model resnet18`), `scripts/verify_resnet.py`
