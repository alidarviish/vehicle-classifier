"""Predict the vehicle class of one or more images with the selected final model.

Usage (from the repository root):
    python -m src.predict path/to/image.jpg [more images ...]

Prints JSON, one object per image:
    {"path": ..., "predicted_class": ..., "confidence": ..., "probabilities": {...}, "needs_review": ...}

- Model: the checkpoint selected on validation (reports/experiments/10_final_model_selection.md).
- Preprocessing: the transform recorded in the checkpoint, i.e. RESNET_TRANSFORM:
  Resize((224, 224)) -> ToTensor -> ImageNet normalization. No training augmentation.
- needs_review: True when the confidence (highest softmax probability) is below
  NEEDS_REVIEW_THRESHOLD. It is only a flag for human review. The predicted class and the
  probabilities are always returned; nothing is rejected or replaced.

This script only reads the checkpoint and the given images; it writes nothing.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

from src.dataset import CLASS_TO_IDX, CLASSES
from src.resnet import build_resnet18
from src.train import resnet_transform

REPO_ROOT = Path(__file__).resolve().parent.parent
FINAL_CHECKPOINT = REPO_ROOT / "checkpoints" / "resnet224_ft_aug_best.pt"

# Chosen on validation only, see reports/experiments/11_needs_review_threshold.md.
# This is the single place where the threshold is defined.
NEEDS_REVIEW_THRESHOLD = 0.90


def needs_review(confidence, threshold=NEEDS_REVIEW_THRESHOLD):
    """True if the prediction should be checked by a person (confidence below the threshold)."""
    return confidence < threshold


def load_model(checkpoint_path=FINAL_CHECKPOINT):
    """Load the ResNet18 checkpoint for inference. Returns (model, eval_transform, checkpoint)."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["architecture"] != "ResNet18":
        raise ValueError(f"expected a ResNet18 checkpoint, got {checkpoint['architecture']}")
    if checkpoint["class_to_idx"] != CLASS_TO_IDX:
        raise ValueError(f"class mapping in checkpoint differs: {checkpoint['class_to_idx']}")
    model = build_resnet18(num_classes=len(CLASSES), pretrained=False)   # weights come from the checkpoint
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    # evaluation transform only (resize + normalization); the training augmentation is never used here
    recorded = checkpoint["transform"]
    eval_transform = resnet_transform(recorded["resize"][0], recorded["normalize_mean"], recorded["normalize_std"])
    return model, eval_transform, checkpoint


def predict_image(image, model, eval_transform, path=None):
    """Predict one image (a file path or a PIL image). Returns a JSON-serialisable dict."""
    if isinstance(image, (str, Path)):
        path = str(image) if path is None else path
        with Image.open(image) as img:
            image = img.convert("RGB")
    else:
        image = image.convert("RGB")
    x = eval_transform(image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    confidence, index = probs.max(dim=0)
    confidence = float(confidence)
    return {
        "path": path,
        "predicted_class": CLASSES[int(index)],
        "confidence": round(confidence, 4),
        "probabilities": {name: round(float(p), 4) for name, p in zip(CLASSES, probs)},
        "needs_review": needs_review(confidence),   # decided on the unrounded confidence
    }


def main():
    parser = argparse.ArgumentParser(description="Predict vehicle classes with the final model (JSON output).")
    parser.add_argument("images", nargs="+", help="image file(s)")
    parser.add_argument("--checkpoint", default=str(FINAL_CHECKPOINT), help="default: the selected final model")
    args = parser.parse_args()
    model, eval_transform, _ = load_model(args.checkpoint)
    results = [predict_image(p, model, eval_transform) for p in args.images]
    json.dump(results[0] if len(results) == 1 else results, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
