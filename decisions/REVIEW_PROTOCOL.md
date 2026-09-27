# Neysan review protocol

Scope: the 500 images with folder label `vanet` in the train and unclean folders
(v1 and v2) that may be Neysan. Test images are not part of this review.

Files in this folder:

- `neysan_review_queue.csv`: one row per image (review_id, image_path, sha256, screening group, evidence, contact sheet name). Read only.
- `neysan_review.csv`: the decisions. This is the only file the reviewer edits.

Images are never copied into this repository. The reviewer looks at the
contact sheets (generated outside the repository, never committed) or opens
the raw image directly. The red label under each thumbnail is the `review_id`.

## Taxonomy (confirmed by the professor)

The model has 8 classes. Neysan is not a training class; it is a subtype of `vanet`
used only for a separate evaluation after the model is frozen.

```
vanet
├── neysan   the common Iranian Nissan pickup
├── other    every other pickup, including modern Nissan pickups (e.g. Navara)
└── unknown  subtype cannot be determined
```

## What counts as Neysan

The common Iranian Nissan pickup, as in the Neysan folders: its front grille and
headlight shape, cab and bed shape. Any colour; most are blue, some are not.

Not sufficient evidence on its own:

- blue colour (the colour screen only sorted the queue),
- the Nissan badge (modern Nissan pickups have it too),
- the folder name, old project decisions, or model predictions,
- "looks like the others on the sheet".

## Decisions

Write exactly one of these three values in the `decision` column:

| decision | meaning | effect later |
|---|---|---|
| `CONFIRMED_NEYSAN` | Clearly the Iranian Nissan pickup | Removed from the 8-class training pool; kept for Neysan evaluation (model_label = vanet, subtype = neysan) |
| `NOT_NEYSAN` | Clearly a different pickup (including modern Nissan pickups) | Not Neysan; can stay in `vanet`, subject to the other cleaning rules |
| `UNCERTAIN` | Cannot decide from the image | No inclusion/exclusion decision is made yet |

If the image is not a `vanet` at all (another class, no vehicle, unusable crop),
do not invent a new decision value: use `UNCERTAIN` and describe it in `note`
(for example `not a pickup: looks like kamyunet`). Relabel or exclude decisions
are made in a later step.

Leave `decision` empty for rows not reviewed yet.

## Recording

- Fill `note` when the decision is `UNCERTAIN`, or when anything is unusual
  (`night`, `front not visible`, `bed hidden by cargo`, `not a pickup: ...`).
- Fill `reviewer` (initials) and `date` (YYYY-MM-DD) for every decision.
- Never change `review_id`, `image_path` or `sha256`.
- If you hesitate, choose `UNCERTAIN`. Do not pick a value to clear the row.

## Rows already resolved by rule

| review_id | decision | reason |
|---|---|---|
| A017 | `NOT_NEYSAN` | Modern Nissan pickup (Navara-style); professor clarification |
| A018 | `CONFIRMED_NEYSAN` | Exact duplicate (same sha256) of an image in the Neysan folder |

These rows have `reviewer` starting with `rule:` and need no review.
