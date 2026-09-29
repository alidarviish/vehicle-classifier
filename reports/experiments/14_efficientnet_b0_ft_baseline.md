# EfficientNet-B0 fine-tuning baseline

Reference run for the EfficientNet-B0 comparison experiments that follow. Every later EfficientNet-B0
experiment changes one factor relative to this run and is compared with it on validation.

This is a separate comparison experiment. The final model of the project stays `resnet224_ft_aug`
(`10_final_model_selection.md`); nothing here changes it, `src/predict.py` or the threshold.

## Run

- Run name: `effnet_b0_ft_none`
- Command: `python -m src.train --run-name effnet_b0_ft_none --model efficientnet_b0 --effnet-mode fine_tuning`
- Code: `src/efficientnet.py`, `src/train.py` (EfficientNet-B0 fine-tuning added in commit `55393dc`)
- Checkpoint: `checkpoints/effnet_b0_ft_none_best.pt` (not tracked), sha256
  `1aecfd1c38df177828b7676871e3740eb4217933dda46eeb80a5672b1b8072d1`
- History: `reports/effnet_b0_ft_none_history.csv` (not tracked)
- Analysis: `reports/analysis/effnet_b0_ft_none/` (not tracked), from `scripts/analyze_baseline.py`

## Settings

- Model: EfficientNet-B0, ImageNet weights `IMAGENET1K_V1`; classifier `Dropout(0.2) -> Linear(1280, 8)`
- Mode: `fine_tuning`
  - epochs 1-5 (warm-up): only the classifier is trained
  - from epoch 6: `features[6:9]` are unfrozen as well; `features[0:6]` stay frozen and in eval mode
  - trainable parameters: 10,248 in epochs 1-5, 3,165,988 from epoch 6
- Learning rates: classifier 1e-3, `features[6:9]` 1e-4
- Optimizer: Adam, weight decay 0, no scheduler
- Loss: CrossEntropyLoss
- Augmentation: `none` (train and validation: Resize 224x224 -> ToTensor -> ImageNet normalization)
- Epochs: 20, batch size 32, seed 42
- Data: `data/split_manifest.csv`, train 2633 / validation 659 images

## Validation results (best checkpoint)

| Metric | Value |
|---|---|
| Best epoch (highest val macro F1) | 19 |
| Accuracy | 0.9560 |
| Macro precision | 0.9469 |
| Macro recall | 0.9518 |
| Macro F1 | 0.9485 |
| Validation images | 659 |

At the last epoch (20): train macro F1 0.9986, validation macro F1 0.9466.

Epochs 1-5 of the history are identical to `effnet_b0_fe_none` (the same classifier-only warm-up).

## How to use this baseline

- Later EfficientNet-B0 experiments use the run-name prefix `effnet_b0_` and change one factor at a time
  relative to the settings above (for example the augmentation). They are compared with this run on the
  same validation set, with the same metrics (accuracy, macro precision, macro recall, macro F1,
  per-class metrics and confusion matrix).
- **The Final Test must not be run again for these experiments.** Test was evaluated once, for the
  selected final model (`12_final_test_evaluation.md`), and is not used for any comparison or decision here.
  The Neysan set is not used either.
- Each configuration is trained once with seed 42, so small validation differences are not conclusive.
