"""ConvNeXt-Tiny transfer learning for the 8 vehicle classes (separate comparison experiment).

A torchvision ConvNeXt-Tiny with ImageNet weights (IMAGENET1K_V1). The standard classifier
(LayerNorm2d(768) -> Flatten -> Linear(768, 1000)) is kept, only its Linear layer is replaced by a new
8-class layer: classifier = LayerNorm2d(768) -> Flatten -> Linear(768, 8). The model returns raw logits.

Only feature extraction is implemented: the whole backbone (net.features) is frozen and the whole
net.classifier (the pretrained LayerNorm2d and the new Linear) is trained.

Frozen stages are kept in eval mode, also when model.train() is called, so stochastic depth is off in
the frozen blocks (ConvNeXt uses LayerNorm, not BatchNorm; the rule is the same as in src/resnet.py
and src/efficientnet.py, checked stage by stage).

This model is not used by src/predict.py; the final model of the project stays ResNet18.
"""

from torch import nn
from torchvision import models

NUM_CLASSES = 8
PRETRAINED_WEIGHTS = "IMAGENET1K_V1"
FEATURE_DIM = 768          # output channels of the last ConvNeXt-Tiny stage


def set_trainable(module, trainable):
    for param in module.parameters():
        param.requires_grad = trainable


def is_trainable(module):
    """True if at least one parameter of the module is trained."""
    return any(param.requires_grad for param in module.parameters())


class ConvNeXtTinyClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        # pretrained=False only builds the same network without downloading weights
        # (used for evaluation, where the weights come from our own checkpoint)
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        self.net = models.convnext_tiny(weights=weights)
        head = self.net.classifier[2]
        if head.in_features != FEATURE_DIM:
            raise ValueError(f"unexpected ConvNeXt-Tiny feature size: {head.in_features}")
        self.net.classifier[2] = nn.Linear(FEATURE_DIM, num_classes)   # 768 -> 8, new and random
        freeze_backbone(self)

    def forward(self, x):
        return self.net(x)

    def train(self, mode=True):
        """Like nn.Module.train, but every stage of net.features without trainable parameters stays in eval mode."""
        super().train(mode)
        for stage in self.net.features:
            if not is_trainable(stage):
                stage.eval()   # frozen blocks: stochastic depth off
        if not is_trainable(self.net.features):
            self.net.features.eval()   # feature extraction: the whole backbone container in eval
        return self


def build_convnext_tiny(num_classes=NUM_CLASSES, pretrained=True):
    return ConvNeXtTinyClassifier(num_classes=num_classes, pretrained=pretrained)


def freeze_backbone(model):
    """Feature extraction: every backbone parameter frozen, the whole classifier trainable."""
    set_trainable(model.net, False)
    set_trainable(model.net.classifier, True)


def trainable_parts(model):
    """Names of the parts that are trained, e.g. ["classifier"]."""
    parts = [f"features.{i}" for i, stage in enumerate(model.net.features) if is_trainable(stage)]
    return parts + (["classifier"] if is_trainable(model.net.classifier) else [])


def count_params(model):
    """(trainable, frozen) number of parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen
