"""Run the frozen final model on a folder of test images.

    python run_test.py --input path/to/TEST
    python run_test.py --input path/to/TEST --checkpoint path/to/swin_t_ft_aug_best.pt --output results

If the images are inside class folders (ambulance/, autobus/, ... vanet/), the folder name is used as
ground truth and metrics are computed (evaluation mode). Otherwise only predictions are written
(prediction-only mode). An image that the Neysan detector recognises as a Neysan is reported as
"unknown" instead of one of the 8 classes (docs/NEYSAN_REJECTION.md). See docs/MENTOR_TEST_GUIDE.md.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.test_runner import DEFAULT_CHECKPOINT, DEFAULT_DETECTOR, run  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Vehicle classifier: test / inference runner for the final model.")
    p.add_argument("--input", required=True, help="folder with the test images (searched recursively)")
    p.add_argument("--output", default="results", help="folder for the result files (default: results)")
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT),
                   help="path of swin_t_ft_aug_best.pt (default: checkpoints/swin_t_ft_aug_best.pt)")
    p.add_argument("--batch-size", type=int, default=32, help="images per forward pass (default: 32)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="default: GPU if available")
    p.add_argument("--overwrite", action="store_true", help="allow writing into a non-empty output folder")
    p.add_argument("--allow-other-checkpoint", action="store_true",
                   help="skip the SHA256 check of the final checkpoint (not for the official test)")
    p.add_argument("--neysan-detector", default=str(DEFAULT_DETECTOR),
                   help="Neysan detector file (default: checkpoints/neysan_detector.json). Images the detector "
                        "says are Neysan are reported as 'unknown'")
    p.add_argument("--no-neysan-detector", action="store_true",
                   help="plain 8-class output, without the Neysan -> unknown rule")
    a = p.parse_args()
    run(a.input, a.output, checkpoint=a.checkpoint, batch_size=a.batch_size, device=a.device,
        overwrite=a.overwrite, allow_other_checkpoint=a.allow_other_checkpoint,
        neysan_detector=a.neysan_detector, use_neysan_detector=not a.no_neysan_detector)


if __name__ == "__main__":
    main()
