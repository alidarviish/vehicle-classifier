# ConvNeXt-Tiny partial fine-tuning

Second ConvNeXt-Tiny run in the architecture comparison. One factor changes against
`convnext_t_fe_none` (`17_convnext_tiny_fe_none.md`): after a classifier-only warm-up, the last
downsampling layer and the last stage of the backbone are fine-tuned as well. This is an architecture
comparison only; it does not change the final model of the project (`resnet224_ft_aug`).

## Setup

- Run name: `convnext_t_ft_none`
- Command: `python -m src.train --run-name convnext_t_ft_none --model convnext_tiny --convnext-mode fine_tuning`
- Code: `src/convnext.py`, `src/train.py` (ConvNeXt fine-tuning added in commit `cbc0eec`)
- Model: ConvNeXt-Tiny, ImageNet weights `IMAGENET1K_V1`; classifier
  `LayerNorm2d(768) -> Flatten -> Linear(768, 8)`
- Mode: partial fine-tuning
  - epochs 1-5 (warm-up): only the classifier is trained (7,688 trainable parameters)
  - from epoch 6: `features[6:8]` are unfrozen as well (15,478,280 trainable parameters);
    `features[0:6]` stay frozen and in eval mode
- Learning rates: classifier 1e-3, `features[6:8]` 1e-4
- Optimizer: Adam, weight decay 0, no scheduler; CrossEntropyLoss
- Augmentation: `none` (train and validation: Resize 224x224 -> ToTensor -> ImageNet normalization)
- 20 epochs, batch size 32, seed 42; train 2633 / validation 659 images
- Checkpoint `checkpoints/convnext_t_ft_none_best.pt` (sha256
  `bac7e00c83ebce683f5c078442055de1e9bc802d9b5e03682b9a374d83564d40`), history
  `reports/convnext_t_ft_none_history.csv`, analysis `reports/analysis/convnext_t_ft_none/`
  (all not tracked)

Epochs 1-5 of the history are identical to `convnext_t_fe_none` (the same classifier-only warm-up).

## Validation results (best checkpoint)

The checkpoint is the epoch with the highest validation macro F1: epoch 13.

| Metric | Value |
|---|---|
| Accuracy | 0.9590 (632/659) |
| Macro precision | 0.9550 |
| Macro recall | 0.9542 |
| Macro F1 | 0.9538 |

Per-class metrics:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ambulance | 0.9855 | 0.9189 | 0.9510 | 74 |
| autobus | 1.0000 | 0.9783 | 0.9890 | 92 |
| kamyun | 0.9765 | 0.8925 | 0.9326 | 93 |
| kamyunet | 0.8796 | 0.9694 | 0.9223 | 98 |
| minibus | 0.9383 | 0.9620 | 0.9500 | 79 |
| savari | 0.9709 | 1.0000 | 0.9852 | 100 |
| taxi | 1.0000 | 0.9897 | 0.9948 | 97 |
| vanet | 0.8889 | 0.9231 | 0.9057 | 26 |

Confusion matrix (rows = true class, columns = predicted class):

| true \ predicted | ambulance | autobus | kamyun | kamyunet | minibus | savari | taxi | vanet |
|---|---|---|---|---|---|---|---|---|
| ambulance | 68 | 0 | 0 | 1 | 1 | 1 | 0 | 3 |
| autobus | 0 | 90 | 0 | 0 | 2 | 0 | 0 | 0 |
| kamyun | 0 | 0 | 83 | 10 | 0 | 0 | 0 | 0 |
| kamyunet | 0 | 0 | 2 | 95 | 1 | 0 | 0 | 0 |
| minibus | 1 | 0 | 0 | 2 | 76 | 0 | 0 | 0 |
| savari | 0 | 0 | 0 | 0 | 0 | 100 | 0 | 0 |
| taxi | 0 | 0 | 0 | 0 | 1 | 0 | 96 | 0 |
| vanet | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 24 |

Main confusions: kamyun -> kamyunet 10; ambulance -> vanet 3; vanet -> savari 2, minibus -> kamyunet 2,
kamyunet -> kamyun 2.

## Training behaviour

- The lowest validation loss is 0.1575 at epoch 8; it does not return to that value afterwards
  (0.1825 at the best epoch 13, 0.2137 at epoch 20), while train loss keeps falling.
- Train accuracy first reaches 1.0 at epoch 17.
- At epoch 20: train macro F1 0.9990, validation macro F1 0.9484, a gap of 0.0507.
- This indicates overfitting during the later fine-tuning epochs.

## Comparison (validation)

| Run | Macro F1 |
|---|---|
| `convnext_t_fe_none` (feature extraction) | 0.9358 |
| `convnext_t_ft_none` (partial fine-tuning) | 0.9538 |
| `effnet_b0_ft_none` (EfficientNet-B0 fine-tuning baseline) | 0.9485 |

- Partial fine-tuning vs feature extraction: +0.0180 macro F1.
- Against the EfficientNet-B0 fine-tuning baseline: +0.0053 macro F1 on validation.

## Conclusion

Unfreezing `features[6:8]` after a 5-epoch warm-up raised ConvNeXt-Tiny from 0.9358 to 0.9538 validation
macro F1 (best epoch 13), with signs of overfitting during the later epochs. kamyun -> kamyunet remains the most
frequent error. The result is a validation-only architecture comparison and does not change the final
model of the project.

## Limitations

- Validation only (659 images), one run with seed 42; small differences are not conclusive.
- The Final Test and the Neysan evaluation were not used in this experiment. The Test must not be run
  again for these comparison experiments.
- No label, split or data was changed.
