from __future__ import annotations


def resnet18(num_classes: int = 10):
    from torchvision.models import resnet18 as tv_resnet18

    model = tv_resnet18(weights=None, num_classes=num_classes)
    model.conv1 = _conv3x3(3, 64)
    model.maxpool = _identity()
    return model


def _conv3x3(in_channels: int, out_channels: int):
    import torch.nn as nn

    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )


def _identity():
    import torch.nn as nn

    return nn.Identity()
