"""ResNet18 transfer learning for the 8 vehicle classes.

A torchvision ResNet18 with ImageNet weights (IMAGENET1K_V1) and a new 8-class head
(fc = Linear(512, 8)). The model returns raw logits, like BaselineCNN.

- feature extraction: only fc is trainable, the whole backbone is frozen.
- fine-tuning: fc first, then layer4 is unfrozen as well (src/train.py decides when).
  conv1, bn1, layer1, layer2 and layer3 always stay frozen.

Frozen parts are kept in eval mode, also when model.train() is called, so the
running statistics of their BatchNorm layers stay the pretrained ImageNet values.
"""

from torch import nn
from torchvision import models

NUM_CLASSES = 8
PRETRAINED_WEIGHTS = "IMAGENET1K_V1"
# top-level parts of torchvision's ResNet18, in forward order (fc is the head)
BACKBONE_PARTS = ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4")


def set_trainable(module, trainable):
    for param in module.parameters():
        param.requires_grad = trainable


def is_trainable(module):
    """True if at least one parameter of the module is trained."""
    return any(param.requires_grad for param in module.parameters())


class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        # pretrained=False only builds the same network without downloading weights
        # (used for evaluation, where the weights come from our own checkpoint)
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.net = models.resnet18(weights=weights)
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)  # 512 -> 8, new and random
        freeze_backbone(self)

    def forward(self, x):
        return self.net(x)

    def train(self, mode=True):
        """Like nn.Module.train, but parts without trainable parameters stay in eval mode."""
        super().train(mode)
        for name in BACKBONE_PARTS:
            part = getattr(self.net, name)
            if not is_trainable(part):
                part.eval()   # frozen BatchNorm keeps its ImageNet running statistics
        return self


def build_resnet18(num_classes=NUM_CLASSES, pretrained=True):
    return ResNet18Classifier(num_classes=num_classes, pretrained=pretrained)


def freeze_backbone(model):
    """Feature extraction: every backbone parameter frozen, only fc trainable."""
    set_trainable(model.net, False)
    set_trainable(model.net.fc, True)


def unfreeze_layer4(model):
    """Fine-tuning: layer4 becomes trainable too (conv1, bn1, layer1-3 stay frozen)."""
    set_trainable(model.net.layer4, True)


def trainable_parts(model):
    """Names of the top-level parts that are trained, e.g. ["fc"] or ["layer4", "fc"]."""
    return [name for name in BACKBONE_PARTS + ("fc",) if is_trainable(getattr(model.net, name))]


def count_params(model):
    """(trainable, frozen) number of parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen
