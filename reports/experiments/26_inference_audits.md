# Inference-time and boundary audits of `swin_t_ft_aug` (validation only)

Status: diagnostic only, run on 2026-10-08 after the final model was fixed. None of the audits changed
the official pipeline: inference stays single-view (Resize 224x224 -> ImageNet normalization), and the
`needs_review` threshold stays 0.95 on the single-view confidence. No model was trained, no label or
split was changed, and the Test and Neysan sets were not used.

## Purpose

- **Test-time augmentation:** could inference with extra views improve the final model without
  retraining?
- **Boundary audits:** why are `kamyun` and `kamyunet` confused? This is the main error source
  (`decisions/KAMYUN_KAMYUNET_AMBIGUITY.md`).

## Data and method

- **Model:** `swin_t_ft_aug`, checkpoint `checkpoints/swin_t_ft_aug_best.pt` (epoch 17,
  sha256 `f03bacde0a155461ea4aede0b813b8688d6ed17973d5ad19b0d2c1b913ff64db`).
- **Data:**
  - the 659 validation images;
  - the train split, read only as the feature-space neighbours of the kNN audits.
- **Scripts:** read-only scripts kept outside the repository, because they contain machine-specific
  paths and dataset file names. Their outputs list dataset file names and are also not committed.
- **Single-view reference:** on the same 659 images, 640 correct, accuracy 0.971168, macro F1
  0.965819. `kamyun` / `kamyunet` accuracy is 182 / 191 = 0.952880.

## 1. Test-time augmentation (TTA)

- **Method:** six inference modes, each averaging the softmax probabilities of its views:
  - the full image only;
  - the full image plus one crop: upper, middle or lower;
  - the full image plus its horizontal flip;
  - all five views.
- **Best mode, `full_flip`:**
  - accuracy 0.972686 (641 / 659, one image more than single view);
  - macro F1 0.967132;
  - `kamyun` / `kamyunet` accuracy 183 / 191 = 0.958115.
- **Note on the numbers:** the console output of the script was not saved; the numbers were recorded
  by hand at the time.
- **Decision:** not adopted; inference stays single-view. Full flip is kept only as an optional candidate.
  - The gain is one validation image.
  - The mode was chosen among six on the same validation set that had already been used for the
    checkpoint and threshold choices.
  - It doubles the inference cost.
  - The frozen Test set was not re-evaluated with it.

## 2. Occlusion

- **Method:** for each image, one band of the normalized input is set to 0 and the prediction is
  repeated. Five bands were tested: upper 40%, lower 40%, middle 30-70%, left 40% and right 40%.

| Occluded band | Accuracy, all 659 | Accuracy, `kamyun` + `kamyunet` (191) |
|---|---|---|
| none | 0.9712 | 0.9529 |
| upper 40% | 0.9332 | 0.8953 |
| lower 40% | 0.9074 | 0.7853 |
| middle 30-70% | 0.8756 | 0.7906 |
| left 40% | 0.9439 | 0.8691 |
| right 40% | 0.9347 | 0.8639 |

- **Finding:** the `kamyun` / `kamyunet` decision depends most on the middle and lower parts of the
  image.

## 3. kNN boundary

- **Method:** for every `kamyun` / `kamyunet` validation image, take the 10 nearest train images in the
  model's feature space and count how many have the opposite label.
- **Mean share of opposite-label neighbours:**
  - 0.087 overall;
  - 0.052 for the correctly predicted images;
  - 0.80 for the 9 misclassified images.
- **Finding:** the errors lie inside the other class's neighbourhood. The two classes overlap in
  feature space, which is consistent with `decisions/KAMYUN_KAMYUNET_AMBIGUITY.md`.

## 4. Hard boundary train samples

- **Method:** collect the train images that appear repeatedly among the 20 nearest neighbours of
  opposite-class `kamyun` / `kamyunet` validation images. Only queries whose opposite-neighbour share
  is at least 0.40 are used.
- **Result:** 229 boundary train images (106 `kamyun`, 123 `kamyunet`).
- **Use:** none. They were not used for relabelling, reweighting or hard-example training.
- **Contact sheet:** a visual sheet of selected boundary samples was made for manual inspection only. It
  contains dataset images and is not committed.

## Conclusion

| Audit | Effect on the final pipeline |
|---|---|
| Test-time augmentation | none (single view kept; full flip optional only) |
| Occlusion | none (diagnostic) |
| kNN boundary | none (diagnostic; labels unchanged) |
| Hard boundary samples | none (not used for training or labels) |

The `kamyun` / `kamyunet` errors come from a real overlap between the two classes in appearance and in
the model's feature space. They do not come from a defect in the inference pipeline. The labels stay
frozen and the two classes stay separate (`decisions/KAMYUN_KAMYUNET_SHARED_CONFIGURATIONS.md`).
