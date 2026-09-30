# ConvNeXt-Tiny feature extraction

First ConvNeXt-Tiny run in the architecture comparison: how far do the frozen ImageNet features of
ConvNeXt-Tiny get on this task, with only a new classifier trained and no augmentation. It follows
the same protocol as the earlier feature-extraction runs (`resnet224_feature_extraction`,
`effnet_b0_fe_none`). This is a comparison experiment only; the final model of the project stays
`resnet224_ft_aug`.

## Setup

- Run name: `convnext_t_fe_none`
- Command: `python -m src.train --run-name convnext_t_fe_none --model convnext_tiny`
- Code: `src/convnext.py`, `src/train.py` (ConvNeXt-Tiny added in commit `f7d5384`)
- Model: ConvNeXt-Tiny, ImageNet weights `IMAGENET1K_V1`; classifier
  `LayerNorm2d(768) -> Flatten -> Linear(768, 8)`
- Mode: feature extraction. The whole backbone (`net.features`) is frozen and stays in eval mode, so
  stochastic depth is off; the whole classifier is trained (7,688 parameters).
- Optimizer: Adam, learning rate 1e-3, weight decay 0, no scheduler; CrossEntropyLoss
- Augmentation: `none` (train and validation: Resize 224x224 -> ToTensor -> ImageNet normalization)
- 20 epochs, batch size 32, seed 42; train 2633 / validation 659 images
- Checkpoint `checkpoints/convnext_t_fe_none_best.pt` (sha256
  `4c157b3d54edceb97a60b0d7bbf28efb3064b64e1a8c8610fb2d8fee201e7dc2`), history
  `reports/convnext_t_fe_none_history.csv`, analysis `reports/analysis/convnext_t_fe_none/`
  (all not tracked)

## Validation results (best checkpoint, epoch 18)

| Metric | Value |
|---|---|
| Accuracy | 0.9393 (619/659) |
| Macro precision | 0.9358 |
| Macro recall | 0.9368 |
| Macro F1 | 0.9358 |
| Validation loss | 0.1960 (lowest of the run) |

Per-class F1:

| Class | F1 |
|---|---|
| ambulance | 0.9371 |
| autobus | 0.9890 |
| kamyun | 0.8913 |
| kamyunet | 0.8889 |
| minibus | 0.9202 |
| savari | 0.9756 |
| taxi | 0.9789 |
| vanet | 0.9057 |

## Training behaviour

- Validation loss reaches its minimum at epoch 18 (0.1960) and rises afterwards (0.2094 at epoch 20),
  while train loss keeps falling (0.0428 at epoch 20).
- At epoch 20: train macro F1 0.9937, validation macro F1 0.9329, a gap of 0.0609.
- This divergence after epoch 18 is a sign of overfitting, even with only the classifier trained.

## Main errors

40 of 659 validation images are wrong. The most frequent confusions:

| True -> predicted | Images |
|---|---|
| kamyun -> kamyunet | 10 |
| kamyunet -> kamyun | 7 |
| taxi -> savari | 3 |
| ambulance -> vanet | 3 |
| ambulance -> minibus | 3 |

The kamyun / kamyunet pair is still the main confusion (17 of 40 errors), as in all earlier runs;
kamyun and kamyunet also have the two lowest per-class F1 values.

## Comparison (validation, factual)

| Run | Mode | Augmentation | Accuracy | Macro F1 |
|---|---|---|---|---|
| `resnet224_feature_extraction` | feature extraction | none | 0.9181 | 0.9055 |
| `effnet_b0_fe_none` | feature extraction | none | 0.9363 | 0.9261 |
| `convnext_t_fe_none` | feature extraction | none | 0.9393 | 0.9358 |
| `effnet_b0_ft_none` | partial fine-tuning | none | 0.9560 | 0.9485 |
| `resnet224_ft_aug` (final model) | partial fine-tuning | full_aug | 0.9605 | 0.9482 |

The first three rows use the same protocol (frozen backbone, new classifier, no augmentation). The
fine-tuning rows are listed for reference only; they are not the same setting.

## Conclusion

With frozen ImageNet features and no augmentation, ConvNeXt-Tiny reaches validation macro F1 0.9358 at
epoch 18. Signs of overfitting appear after epoch 18, and kamyun / kamyunet remains the main source of
errors.

## Limitations

- Validation only (659 images), one run with seed 42; small differences are not conclusive.
- The Final Test was not run again and the Neysan evaluation was not run again.
- No label, split or data was changed.
