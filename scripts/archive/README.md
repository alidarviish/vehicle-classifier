# Archived scripts

These scripts belong to earlier stages of the project. They are kept so that every result in
`reports/experiments/` (00-26) can be traced back to the code that produced it. They are not needed to
use the final model.

| Script | Stage | Note |
|---|---|---|
| `build_manifest.py`, `audit_dataset.py` | dataset v1/v2 | manifest of the raw images, SHA256 duplicate audit (`reports/audit/`) |
| `build_statuses.py`, `build_test.py`, `make_split.py` | dataset v1/v2 | apply `decisions/`, frozen 400-image Test list, train/validation split |
| `evaluate_test.py`, `evaluate_neysan.py` | ResNet18 final model | one-time Test and Neysan evaluations (reports 12, 13) |
| `evaluate_test_swin.py`, `evaluate_neysan_swin.py` | Swin-Tiny `swin_t_ft_aug` | one-time Test and Neysan evaluations (reports 24, 25) |
| `build_neysan_detector.py` | Swin-Tiny `swin_t_ft_aug` | first Neysan detector (`docs/HISTORY.md`) |
| `build_dataset_v3.py` | dataset v3 | built `data/dataset_v3/` once (report 27); its input folders no longer exist |

The one-time evaluation scripts refuse to run again once their outputs exist. Run any of them from the
repository root, e.g. `python scripts/archive/make_split.py --help`. Reports written before the move
name these scripts by their old path `scripts/<name>.py`.
