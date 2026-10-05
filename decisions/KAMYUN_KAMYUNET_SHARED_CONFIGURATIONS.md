# kamyun / kamyunet: shared configurations (decision record)

Date: 2026-10-03. Follows `decisions/KAMYUN_KAMYUNET_AMBIGUITY.md`.

## Decision

The four images below are **not relabelled** for now. No label, split, manifest, Test or Neysan
file is changed.

| Audit ID | Image | Split | Current label |
|---|---|---|---|
| C05 | `v2/unclean/kamyun/194280486.jpg` | train | kamyun |
| C06 | `v2/train/kamyun/216158404.jpg` | val | kamyun |
| C14 | `v2/train/kamyun/198119425.jpg` | train | kamyun |
| C15 | `v2/train/kamyun/201179873.jpg` | train | kamyun |

## Context

- A candidate list of the kamyun / kamyunet boundary was built from existing evidence (old relabel
  list, the 2026-09-27 reviews, the narrow-cab and conflicting-neighbour audits, the validation
  predictions of `resnet224_ft_aug`). The 17 strongest candidates were reviewed blind.
- In a second, non-blind pass the blind verdict differed from the current label for 11 images.
  For these four the second pass recorded `CHANGE_TO_BLIND_VERDICT` (kamyun -> kamyunet).
- Before applying that, an isolation audit checked whether the same visual configuration also
  occurs elsewhere in train / validation. This record documents its result.

## Result of the isolation audit

All four were classified **SHARED_CONFIGURATION**: the same cab front and body configuration was
found with label `kamyun` and with label `kamyunet`.

| Audit ID | Configuration | Similar kamyun | Similar kamyunet |
|---|---|---|---|
| C05 | older Isuzu narrow cab-over front, open bed with rack and headboard sign | 4 | 4 |
| C14 | same front, open bed with headboard sign and covered load | 3 | 4 |
| C15 | same front, closed box body with roof unit | 4 | 4 |
| C06 | same front, box body with roof logo panel | 3 | 4 |

The four are not isolated. The same configurations are labelled `kamyun` in 13 other images:

| Audit ID | Image | Split | On old relabel list |
|---|---|---|---|
| C35 | `v1/train/kamyun/216847004.jpg` | train | yes |
| C43 | `v1/unclean/kamyun/216972904.jpg` | train | yes |
| C44 | `v1/unclean/kamyun/218562521.jpg` | train | yes |
| C50 | `v2/train/kamyun/215721968.jpg` | train | yes |
| C52 | `v2/train/kamyun/216026504.jpg` | train | yes |
| C60 | `v2/train/kamyun/218719693.jpg` | train | yes |
| C62 | `v2/train/kamyun/219002808.jpg` | train | yes |
| C76 | `v2/unclean/kamyun/217273409.jpg` | train | yes |
| C84 | `v2/unclean/kamyun/219062079.jpg` | train | yes |
| C04 | `v2/train/kamyun/218827750.jpg` | val | yes (kept as kamyun in review KK06) |
| k108 | `v2/train/kamyun/210749256.jpg` | train | no |
| k121 | `v2/train/kamyun/215099946.jpg` | train | no |
| k133 | `v2/train/kamyun/215692883.jpg` | train | no |

- 10 of the 13 are on the old relabel list (recommendation `kamyunet`, not applied).
- k108, k121 and k133 are on no list and were not candidates of the boundary audit.
- Relabelling only the four images would leave the same configuration labelled both ways.

## Status

The audit did not determine which label is correct for these shared configurations. The status in
`decisions/KAMYUN_KAMYUNET_AMBIGUITY.md` is unchanged: **PARTIAL DISTINCTION ONLY**, **DEFINITION
NOT RECOVERED**. This record does not define either class.

## Limitations

- One reviewer; the isolation audit was not blind (labels visible).
- Similarity was judged visually on thumbnails (cab front, body type, load, roof / body
  configuration, viewpoint); crop size was not used.
- The images were not coded one by one; the search is not exhaustive, so more images with the same
  configurations may exist.

## Sources

The audit outputs (candidate list, review sheets, decisions, isolation audit) are retained outside
the repository and are not part of the repository. Repository inputs: `data/split_manifest.csv`
(not tracked), `decisions/old_label_overrides.csv`, `decisions/kamyun_kamyunet_review.csv`,
`decisions/kamyunet_kamyun_review.csv`, `reports/analysis/resnet224_ft_aug/val_audit/val_predictions.csv`
(not tracked). No training or inference was run; Test and Neysan were not read.
