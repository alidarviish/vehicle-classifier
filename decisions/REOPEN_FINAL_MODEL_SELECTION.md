# Reopening Final Model Selection

Status: Adopted
Decision date: 2026-10-04
Decision commit: 76669f4

## Context

The project was first completed with `resnet224_ft_aug` as the final model. The model was
selected on validation (`reports/experiments/10_final_model_selection.md`, commit `74772ae`),
the `needs_review` threshold was fixed on validation
(`reports/experiments/11_needs_review_threshold.md`), and the frozen Test set was then
evaluated once (`reports/experiments/12_final_test_evaluation.md`, commit `b33acfb`).
The Neysan evaluation followed (commit `87c5f02`). The last commit of the original project
is `7b83571`.

After the project was closed, EfficientNet-B0, ConvNeXt-Tiny and Swin-Tiny were added as
post-hoc architecture comparisons (reports 14-22, from commit `394dc32` on). The README and
reports 14 and 17-22 state that the final model stays `resnet224_ft_aug`. These runs used the
same train and validation split, the same validation preprocessing and the same selection
metric (validation macro F1 of the best checkpoint).

This record reopens the final model selection in a controlled way. It does not claim that
Swin-Tiny was the final model from the start.

## Previous Final Model

- Run: `resnet224_ft_aug` (ResNet18, `IMAGENET1K_V1`, fine-tuned, `full_aug`)
- Checkpoint: `checkpoints/resnet224_ft_aug_best.pt`, epoch 20, SHA256
  `c26280e97f7324ee1ec85d7bf78a58ea0b4d2acb4cf3e975dbd52316366281a1` (not in Git)
- Validation: accuracy 0.9605, macro precision 0.9456, macro recall 0.9523, macro F1 0.9482
- `needs_review` threshold: 0.90, chosen on validation with this checkpoint
- Final Test (400 images, `decisions/test_frozen.csv`, run once): accuracy 0.9475,
  macro precision 0.9515, macro recall 0.9475, macro F1 0.9473

These results remain unchanged. They stay the historical, frozen result of the original
project.

## New Validation Evidence

Same 659 validation images, single runs, seed 42:

| Metric | `resnet224_ft_aug` | `swin_t_ft_aug` |
|---|---|---|
| Best epoch | 20 | 17 |
| Accuracy | 0.9605 (633/659) | 0.9712 (640/659) |
| Macro precision | 0.9456 | 0.9626 |
| Macro recall | 0.9523 | 0.9698 |
| Macro F1 | 0.9482 | 0.9658 |

Source for Swin-Tiny: `reports/experiments/22_swin_tiny_ft_aug.md` (commit `9072ce6`);
checkpoint `checkpoints/swin_t_ft_aug_best.pt`, SHA256
`f03bacde0a155461ea4aede0b813b8688d6ed17973d5ad19b0d2c1b913ff64db` (not in Git).

Under the main criterion of report 10 (validation macro F1 of the best checkpoint, supported
by accuracy, macro precision and macro recall, cross-entropy model required), `swin_t_ft_aug`
ranks first among the runs documented in `reports/experiments/`. Both models are
cross-entropy models.

Limitations: one seed per run and one validation split. Report 22 notes that the gaps between
runs amount to a few images. No significance test was done.

## Decision

The final model selection is reopened. `swin_t_ft_aug` is selected as the new final model on
the basis of the validation evidence above.

The selection uses validation metrics only.

## Test Integrity

- No Test metric, Test prediction or Test image was used to select `swin_t_ft_aug`.
  The selection criterion is validation macro F1 only.
- Disclosure: the ResNet Test result was recorded (`b33acfb`) before the architecture
  comparisons were designed and run (`394dc32` onward). The comparisons were therefore not
  run blind to the existence of that result.
- The ResNet Final Test result is not deleted, edited or hidden. `12_final_test_evaluation.md`
  and `12_final_test_predictions.csv` remain unchanged as the historical result.
- The project protocol evaluates Test once for final evaluation (README;
  `scripts/evaluate_test.py`; `12_final_test_evaluation.md`). A new Final Test for
  `swin_t_ft_aug` is therefore a second use of the same frozen Test set. This is a
  documented protocol deviation. The new Test result will be reported as a second evaluation
  on a previously used Test set, not as an untouched (pristine) held-out estimate.
- The new Final Test will be run once, only after the threshold and inference protocol for
  `swin_t_ft_aug` are fixed on validation. It is used only to evaluate the selected model, and
  its result will not be used to choose between ResNet and Swin or to change any setting.

## Consequences / Required Follow-up

1. **Threshold:** derive the `needs_review` threshold for `swin_t_ft_aug` on validation only.
   The value 0.90 was derived for the ResNet checkpoint and is not carried over unchecked.
2. **`src/predict.py`:** it currently builds ResNet18 (`build_resnet18`, `resnet_transform`)
   and defaults to `resnet224_ft_aug_best.pt`. It must support Swin-Tiny and the new default
   checkpoint, without changing the recorded ResNet behaviour.
3. **Checkpoint and metadata:** record the final checkpoint path, SHA256, epoch and the
   selected threshold for `swin_t_ft_aug`.
4. **New Final Test:** `scripts/evaluate_test.py` is fixed to the ResNet checkpoint, the
   threshold 0.90 and the `12_*` output files, and refuses to run again by design. A
   separate one-time evaluation for Swin must write new output files and must not overwrite
   any `12_*` file.
5. **Neysan:** evaluate `swin_t_ft_aug` in the same way. `scripts/evaluate_neysan.py` is
   fixed to the `13_*` outputs and refuses to run again by design. A separate evaluation
   must write new output files and must not overwrite any `13_*` file; the existing Neysan
   results remain unchanged.
6. **Documentation:** update the README (final model, usage, results, "used once" wording),
   and add a note to reports 10, 11 and 12 referring to this record, without changing their
   historical results.
7. **Final audit:** check the code, metadata, checksums and documentation against this
   record; confirm that `test_frozen.csv` is unchanged and that no earlier output was
   overwritten.

## Follow-up status (added later)

1. Threshold: done, 0.95 on validation (`23_swin_needs_review_threshold.md`, commit `41df725`).
2. `src/predict.py`: done, Swin-Tiny with 0.95 (commit `a81184a`); item 2 above describes the code
   at decision time.
3. Checkpoint and metadata: recorded in `README.md` and reports 22 and 23.
4. New Final Test: done, `24_swin_final_test_evaluation.md` (commits `817b61e`, `b93560d`); the
   results of `12_final_test_evaluation.md` remain unchanged.
5. Neysan: done, `25_swin_neysan_evaluation.md` (commits `bdbe67c`, `0083791`);
   `13_neysan_unclean_analysis.md` remains unchanged.
6. Documentation: README updated (`4bb39db`); status notes added to reports 10, 11, and 12.

## Timeline

| Date | Commit | Event |
|---|---|---|
| 2026-09-29 11:32 | `74772ae` | Final ResNet selection and inference protocol (reports 10, 11, `src/predict.py`) |
| 2026-09-29 11:45 | `b33acfb` | ResNet Final Test recorded (script added in `b02782f`) |
| 2026-09-29 12:10 | `87c5f02` | Neysan evaluation recorded |
| 2026-09-29 14:31 | `7b83571` | Last commit of the original project |
| 2026-09-29 22:43 | `394dc32` | Start of the post-hoc architecture comparisons (EfficientNet-B0) |
| 2026-10-02 22:28 | `e6f158e` | Swin-Tiny infrastructure (feature extraction) |
| 2026-10-03 09:29 | `39dc769` | Swin-Tiny feature-extraction result |
| 2026-10-03 09:42 | `8b034a3` | Swin-Tiny partial fine-tuning infrastructure |
| 2026-10-03 13:10 | `5ae57b6` | `swin_t_ft_none` result (report 21) |
| 2026-10-03 17:36 | `9072ce6` | `swin_t_ft_aug` result (report 22) |
| 2026-10-04 | 76669f4 | This reopening decision |

## Scope

This decision:

- does not delete, edit or replace the previous ResNet Final Test or Neysan results;
- does not use the previous Test result to select Swin-Tiny;
- does not make any EfficientNet-B0, ConvNeXt-Tiny or other ResNet run the final model;
- is based only on the validation metrics in the reports listed above; no other analysis is
  part of this decision;
- does not change labels, splits, `decisions/test_frozen.csv` or any raw data;
- does not itself run training, inference or Test.

The ResNet18 transfer-learning work required by the project definition remains documented in
full; this record only changes which model is delivered as final.
