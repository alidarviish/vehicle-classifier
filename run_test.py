"""Test the final model on a folder of images.

    python run_test.py --input path/to/TEST

Images inside class folders (ambulance/, autobus/, ... vanet/) are scored against the folder name.
An image the Neysan detector recognises as a Neysan is reported as "unknown".
Guide: docs/TEST_GUIDE.md. The code is in src/test_runner.py.
"""

import argparse

from src.test_runner import DEFAULT_CHECKPOINT, DEFAULT_DETECTOR, run

parser = argparse.ArgumentParser(description="Test the final vehicle classifier on a folder of images.")
parser.add_argument("--input", required=True, help="folder with the test images")
parser.add_argument("--output", default="results", help="folder for the results (default: results)")
parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="model file (default: the final model)")
parser.add_argument("--neysan-detector", default=str(DEFAULT_DETECTOR), help="Neysan detector file")
parser.add_argument("--no-neysan-detector", action="store_true", help="plain 8-class output, no 'unknown'")
parser.add_argument("--batch-size", type=int, default=32, help="images per step (default: 32)")
parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="default: GPU if available")
parser.add_argument("--overwrite", action="store_true", help="allow a non-empty output folder")
parser.add_argument("--allow-other-checkpoint", action="store_true", help="skip the model SHA256 check (not for the official test)")
parser.add_argument("--allow-other-detector", action="store_true", help="skip the detector SHA256 check (e.g. the previous model's detector)")
parser.add_argument("--review-threshold", type=float, default=None, help="needs_review threshold (default: the final model's)")
parser.add_argument("--no-report-images", action="store_true", help="skip the per-test pictures (faster)")
args = parser.parse_args()

run(args.input, args.output, checkpoint=args.checkpoint, batch_size=args.batch_size, device=args.device,
    overwrite=args.overwrite, allow_other_checkpoint=args.allow_other_checkpoint,
    neysan_detector=args.neysan_detector, use_neysan_detector=not args.no_neysan_detector,
    allow_other_detector=args.allow_other_detector, review_threshold=args.review_threshold,
    report_images=not args.no_report_images)
