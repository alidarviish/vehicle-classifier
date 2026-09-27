"""Baseline CNN for the 8 vehicle classes.

Three conv blocks (Conv2d 3x3 -> ReLU -> 2x2 pooling), then a small
fully connected head. The model returns raw logits (no softmax), which is
what nn.CrossEntropyLoss expects.
"""

from torch import nn

NUM_CLASSES = 8
POOLING_TYPES = ("max", "average")


def pool_layer(pooling):
    """2x2 pooling with stride 2 (halves height and width): max (baseline) or average."""
    if pooling == "max":
        return nn.MaxPool2d(2)
    if pooling == "average":
        return nn.AvgPool2d(2)
    raise ValueError(f"pooling must be one of {POOLING_TYPES}, got '{pooling}'")


def conv_block(in_channels, out_channels, pooling="max"):
    """Conv 3x3 (same size) -> ReLU -> 2x2 pooling."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        pool_layer(pooling),
    )


class BaselineCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, image_size=128, dropout=0.0, pooling="max"):
        super().__init__()
        if image_size % 8 != 0:
            raise ValueError("image_size must be divisible by 8 (three 2x2 poolings)")

        self.features = nn.Sequential(
            conv_block(3, 32, pooling),    # 128 -> 64
            conv_block(32, 64, pooling),   # 64 -> 32
            conv_block(64, 128, pooling),  # 32 -> 16
        )

        side = image_size // 8   # 16 for a 128x128 input
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(128 * side * side, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),  # logits
        )

    def forward(self, x):
        return self.classifier(self.features(x))
