"""EfficientNet-B0 transfer learning for the 8 vehicle classes (separate comparison experiment).

A torchvision EfficientNet-B0 with ImageNet weights (IMAGENET1K_V1). The standard classifier
(Dropout(0.2) -> Linear(1280, 1000)) is kept, only its Linear layer is replaced by a new
8-class layer: classifier = Dropout(0.2) -> Linear(1280, 8). The model returns raw logits.

net.features has 9 stages (features[0] = stem, features[1..7] = MBConv stages, features[8] = the
1x1 conv to 1280 channels). Two modes (src/train.py decides when to unfreeze):
- feature extraction: the whole backbone (net.features) is frozen, only net.classifier is trained.
- fine-tuning: the classifier first, then the top stages features[6:9] (UNFREEZE_STAGES) as well;
  features[0:6] always stay frozen.

Frozen stages are kept in eval mode, also when model.train() is called, so their BatchNorm
running statistics stay the pretrained ImageNet values and stochastic depth is off in them
(the same rule as src/resnet.py, checked stage by stage). Unfrozen stages and the classifier
dropout follow model.train() / model.eval() as usual.

This model is not used by src/predict.py; the final model of the project stays ResNet18.
"""

from torch import nn
from torchvision import models

NUM_CLASSES = 8
PRETRAINED_WEIGHTS = "IMAGENET1K_V1"
HEAD_DROPOUT = 0.2          # torchvision default for EfficientNet-B0, kept unchanged
FEATURE_DIM = 1280          # output channels of net.features[8]
UNFREEZE_STAGES = (6, 7, 8)  # fine-tuning: the last three stages of net.features


def set_trainable(module, trainable):
    for param in module.parameters():
        param.requires_grad = trainable


def is_trainable(module):
    """True if at least one parameter of the module is trained."""
    return any(param.requires_grad for param in module.parameters())


class EfficientNetB0Classifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        # pretrained=False only builds the same network without downloading weights
        # (used for evaluation, where the weights come from our own checkpoint)
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.net = models.efficientnet_b0(weights=weights)
        head = self.net.classifier[1]
        if head.in_features != FEATURE_DIM:
            raise ValueError(f"unexpected EfficientNet-B0 feature size: {head.in_features}")
        self.net.classifier[1] = nn.Linear(FEATURE_DIM, num_classes)   # 1280 -> 8, new and random
        freeze_backbone(self)

    def forward(self, x):
        return self.net(x)

    def train(self, mode=True):
        """Like nn.Module.train, but every stage of net.features without trainable parameters stays in eval mode."""
        super().train(mode)
        for stage in self.net.features:
            if not is_trainable(stage):
                stage.eval()   # frozen BatchNorm keeps its ImageNet statistics, stochastic depth off
        if not is_trainable(self.net.features):
            self.net.features.eval()   # feature extraction: the whole backbone container in eval, as before
        return self


def build_efficientnet_b0(num_classes=NUM_CLASSES, pretrained=True):
    return EfficientNetB0Classifier(num_classes=num_classes, pretrained=pretrained)


def freeze_backbone(model):
    """Feature extraction: every backbone parameter frozen, only the classifier trainable."""
    set_trainable(model.net, False)
    set_trainable(model.net.classifier, True)


def top_stage_parameters(model):
    """Parameters of features[6:9], the stages that fine-tuning unfreezes (for the optimizer)."""
    return [p for i in UNFREEZE_STAGES for p in model.net.features[i].parameters()]


def unfreeze_top_stages(model):
    """Fine-tuning: features[6:9] become trainable too (features[0:6] stay frozen)."""
    for i in UNFREEZE_STAGES:
        set_trainable(model.net.features[i], True)


def trainable_parts(model):
    """Names of the parts that are trained, e.g. ["classifier"] or ["features.6", "features.7", "features.8", "classifier"]."""
    parts = [f"features.{i}" for i, stage in enumerate(model.net.features) if is_trainable(stage)]
    return parts + (["classifier"] if is_trainable(model.net.classifier) else [])


def count_params(model):
    """(trainable, frozen) number of parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen
