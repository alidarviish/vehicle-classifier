"""Swin-Tiny transfer learning for the 8 vehicle classes (separate comparison experiment).

A torchvision Swin-Tiny (swin_t) with ImageNet weights (IMAGENET1K_V1). The 1000-class head
(Linear(768, 1000)) is replaced by a new 8-class layer: head = Linear(768, 8). The model returns raw logits.

torchvision's SwinTransformer runs features -> norm -> permute -> avgpool -> flatten -> head.
net.features has 8 parts: features[0] = patch embedding, features[1, 3, 5, 7] = the four stages
(2, 2, 6, 2 Swin blocks), features[2, 4, 6] = the patch-merging layers in between; net.norm = LayerNorm(768).

Only feature extraction is supported: net.features and net.norm are frozen, only net.head is trained.
The frozen parts are kept in eval mode, also when model.train() is called, so the stochastic depth of
the Swin blocks is off (Swin uses LayerNorm, not BatchNorm; the rule is the same as in src/convnext.py).

This model is not used by src/predict.py; the final model of the project stays ResNet18.
"""

from torch import nn
from torchvision import models

NUM_CLASSES = 8
PRETRAINED_WEIGHTS = "IMAGENET1K_V1"
FEATURE_DIM = 768   # output size of the last Swin-Tiny stage (input size of the head)


def set_trainable(module, trainable):
    for param in module.parameters():
        param.requires_grad = trainable


def is_trainable(module):
    """True if at least one parameter of the module is trained."""
    return any(param.requires_grad for param in module.parameters())


class SwinTinyClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        # pretrained=False only builds the same network without downloading weights
        # (used for evaluation, where the weights come from our own checkpoint)
        weights = models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        self.net = models.swin_t(weights=weights)
        if self.net.head.in_features != FEATURE_DIM:
            raise ValueError(f"unexpected Swin-Tiny feature size: {self.net.head.in_features}")
        self.net.head = nn.Linear(FEATURE_DIM, num_classes)   # 768 -> 8, new and random
        freeze_backbone(self)

    def forward(self, x):
        return self.net(x)

    def train(self, mode=True):
        """Like nn.Module.train, but net.features and net.norm stay in eval mode while they are frozen."""
        super().train(mode)
        for stage in self.net.features:
            if not is_trainable(stage):
                stage.eval()   # frozen blocks: stochastic depth off
        if not is_trainable(self.net.features):
            self.net.features.eval()   # feature extraction: the whole backbone container in eval
        if not is_trainable(self.net.norm):
            self.net.norm.eval()
        return self


def build_swin_tiny(num_classes=NUM_CLASSES, pretrained=True):
    return SwinTinyClassifier(num_classes=num_classes, pretrained=pretrained)


def freeze_backbone(model):
    """Feature extraction: every backbone parameter (features and norm) frozen, only the head trainable."""
    set_trainable(model.net, False)
    set_trainable(model.net.head, True)


def trainable_parts(model):
    """Names of the parts that are trained, e.g. ["head"]."""
    parts = [f"features.{i}" for i, stage in enumerate(model.net.features) if is_trainable(stage)]
    parts += ["norm"] if is_trainable(model.net.norm) else []
    return parts + (["head"] if is_trainable(model.net.head) else [])


def count_params(model):
    """(trainable, frozen) number of parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen
