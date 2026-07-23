"""Model registry."""

import torch.nn as nn
from torchvision.models import resnet18

from .lenet import LeNet


def build_model(name: str, num_classes: int = 10) -> nn.Module:
    if name == "lenet":
        return LeNet(num_classes)
    if name == "resnet18":
        model = resnet18(num_classes=num_classes)
        # ImageNet stem downsamples too aggressively for 32x32 inputs
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model
    raise ValueError(f"unknown model: {name}")
